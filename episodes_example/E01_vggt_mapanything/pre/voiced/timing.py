"""Schedule the voice-over times -> timing.json (read by the voiced render shell shots/voiced.py, mix.py and check_voiced.py).
Piecewise-linear time map: the picture speeds up or slows down between anchors, with frame holds only when slowing is not enough.
- One anchor per sentence (anchor in narration.json, pre time): the sentence starts speaking when the anchor picture appears. Between two adjacent anchors the pre
  duration is P; the voice-over needs V = sentence duration + gap between sentences (GAP_SEG when the next sentence belongs to a new segment); for the last sentence V = sentence duration + END_TAIL.
- V < P: this stretch of picture is sped up uniformly, voiced duration L = max(V, P / MAX_SPEED); whatever remains after the speed cap is silence (recorded in silence_after_s).
- V > P: first slow down uniformly, rate >= MIN_SLOW; if that is not enough, freeze a frame near HOLD_LEAD seconds before the next anchor (avoiding the cross-fade of
  +-XF around a PreShot segment boundary and the last CARD_FADE seconds of a title card's fade-out); the last sentence freezes on the final frame of the film.
- Every L is rounded to whole frames, so every shot starts on the frame grid.
- CONTENT_END: when the picture a sentence talks about ends partway through its interval (the next picture cuts in before the next anchor), the picture
  reaches CONTENT_END exactly when the sentence ends; the rest of the pre picture plays at <= MAX_SPEED.
- ANCHOR_OVERRIDE: anchors of single sentences (pre global seconds) set here, without editing narration.json.
- Hold points sit on the pre shot's frame grid; among the candidates the still frame closest to the target is used, else the frame with the least motion
  (frame-difference cache _check/motion_<shot>.npz).
Times: "pre time" = seconds in the cut pre film (t0 / t1 / anchor in narration.json); "voiced time" = seconds in the voiced film.
Each shot stores a list of knots [(voiced local seconds, pre local seconds)]; voiced.back() interpolates through them, voiced.fwd() is the inverse.
render env: python -I -X utf8 timing.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PRE = HERE.parent
FPS = 30
GAP, GAP_SEG, END_TAIL = 0.3, 0.5, 2.3     # END_TAIL: >= 2.0 s after the last word (the wav tail carries 0.03 s of padding; 0.3 s margin)
MAX_SPEED, MIN_SLOW = 2.5, 0.6
HOLD_LEAD, XF, CARD_FADE = 0.15, 0.125, 0.3
MIN_GAP = 0.25                             # minimum gap between sentences after frame rounding
# p1.10_2: Table 3 is fully lit at pre 84.9-86.6; 86.75 is a PreShot cut (cross-fade 86.625-86.875)
CONTENT_END = {"p1.10_2": 86.6}
# p2.4_1: s09's ray / depth / pose boxes appear one by one at pre 133.45-134.65 (s09 t 3.3-4.5, ss(t, 2.9 + 0.4k, 3.3 + 0.4k));
#   anchor 134.65 (narration.json: 132.95), so the boxes start to appear while p2.4_0 "The factors can also be inputs." says "inputs"
ANCHOR_OVERRIDE = {"p2.4_1": 134.65}
STILL = 0.03                               # hold candidates: mean grey difference to the neighbouring frames <= this counts as still (still frames measure 0.00-0.01; 0.5 would pick the start of p2.3_1's retract animation at 0.20)
SHOT_OF = {"c0": "c0_open", "c1": "c1_vggt", "p1": "p1_vggt", "c2": "c2_mapanything", "p2": "p2_mapanything",
           "c3": "c3_compare", "p3": "p3_compare", "c4": "c4_test", "p4": "p4_our_test"}


def shot_table():
    """[(shot, pre start, pre duration, segment boundaries in pre local seconds)] in episode.json order; durations / edit list come from the meta / check files of the pre render."""
    ep = json.loads((PRE / "episode.json").read_text(encoding="utf-8"))
    out, t = [], 0.0
    for s in ep["shots"]:
        stem = Path(s["script"]).stem
        d = PRE / "shots" / "render" / stem
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        edl = json.loads((d / "check.json").read_text(encoding="utf-8")).get("data", {}).get("edl", [])
        out.append((stem, round(t, 4), round(meta["dur"], 4), [e[3] for e in edl[1:]]))
        t += meta["dur"]
    return out


def frames(x):
    return round(x * FPS) / FPS


_MOTION = {}


def motion(name):
    """Per-frame motion of the pre shot render shots/render/<shot>/<shot>.mp4: m[i] = mean grey value of max(|f_i - f_{i-1}|, |f_{i+1} - f_i|),
    without the caption band at the bottom (y >= 0.84 H; the voiced version replaces those captions); computed at 1/4 size. Cached by mp4 size + mtime."""
    if name in _MOTION:
        return _MOTION[name]
    mp4 = PRE / "shots" / "render" / name / f"{name}.mp4"
    st = mp4.stat()
    cache = HERE / "_check" / f"motion_{name}.npz"
    if cache.exists():
        z = np.load(cache)
        if int(z["size"]) == st.st_size and float(z["mtime"]) == st.st_mtime:
            _MOTION[name] = z["m"]
            return z["m"]
    cap = cv2.VideoCapture(str(mp4))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g[:int(0.84 * g.shape[0])], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        fr.append(g.astype(np.float32))
    cap.release()
    d = np.array([np.abs(x - y).mean() for x, y in zip(fr[1:], fr[:-1])])
    m = np.maximum(np.r_[d[:1], d], np.r_[d, d[-1:]])
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, m=m, size=st.st_size, mtime=st.st_mtime)
    _MOTION[name] = m
    return m


def hold_point(a, b, shot_row):
    """Which second (pre global) inside [a, b] to freeze on: only times on the pre shot's frame grid, not inside a cross-fade / card fade-out;
    among the still ones (motion <= STILL) the one closest to b - HOLD_LEAD, else the one with the least motion. Returns (seconds, ok, motion)."""
    name, s0, dur, cuts = shot_row
    bad = [(s0 + o - XF, s0 + o + XF) for o in cuts]
    if name.startswith("c"):
        bad.append((s0 + dur - CARD_FADE - 0.05, s0 + dur + 1))
    m = motion(name)
    i0, i1 = math.ceil((a - s0) * FPS - 1e-6), math.floor((b - s0) * FPS + 1e-6) - 1
    ok = [i for i in range(i0, min(i1, len(m) - 1) + 1) if not any(w0 <= s0 + i / FPS <= w1 for w0, w1 in bad)]
    if not ok:
        return a, False, None
    target = (b - HOLD_LEAD - s0) * FPS
    still = [i for i in ok if m[i] <= STILL]
    i = min(still, key=lambda i: (abs(i - target), -i)) if still else min(ok, key=lambda i: (m[i], abs(i - target)))
    return round(s0 + i / FPS, 6), True, round(float(m[i]), 3)


def span(v, a, b, need, shot_row, last, pre_end):
    """Fit one pre stretch [a, b] into voiced time: returns (L, new knots, mode, hold seconds, hold point, ok, motion). Rules: see the file header."""
    P = b - a
    if need <= P:
        L = max(frames(need), frames(P / MAX_SPEED))
        return L, [(v + L, b)], ("speed" if L < P - 1e-9 else "normal"), 0.0, None, True, None
    if need <= P / MIN_SLOW:
        L = frames(need)
        return L, [(v + L, b)], "slow", 0.0, None, True, None
    L = frames(need)
    if last:
        h, h_ok, mv = pre_end - 1.0 / FPS, True, None
    else:
        h, h_ok, mv = hold_point(a, b, shot_row)
    hold = L - P / MIN_SLOW
    v1 = v + (h - a) / MIN_SLOW
    return L, [(v1, h), (v1 + hold, h), (v + L, b)], "slow+hold", hold, h, h_ok, mv


def main():
    narr = json.loads((PRE / "narration" / "narration.json").read_text(encoding="utf-8"))
    sel = {s["sid"]: s for s in json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))}
    shots = shot_table()
    sh = {r[0]: r for r in shots}
    pre_end = round(shots[-1][1] + shots[-1][2], 4)
    # t0 / anchor in the script and the actual shot start can differ by a small remainder: shift per shot to the actual start
    shift = {}
    for seg in narr:
        shot = SHOT_OF[seg["id"].split(".")[0]]
        shift.setdefault(shot, round(sh[shot][1] - seg["t0"], 4))
    sents = []                                  # (sentence, seg id, shot, anchor in pre global seconds)
    for seg in narr:
        shot = SHOT_OF[seg["id"].split(".")[0]]
        for k, s in enumerate(seg["sentences"]):
            assert s["anchor"] is not None, f"{seg['id']}_{k} has no anchor"
            sid = f"{seg['id']}_{k}"
            sents.append((sel[sid], seg["id"], shot, ANCHOR_OVERRIDE.get(sid, round(s["anchor"] + shift[shot], 4))))
    assert all(x[3] < y[3] for x, y in zip(sents, sents[1:])), "anchors must increase"
    starts = {r[1] for r in shots}
    assert all(any(abs(x[3] - s0) < 1e-6 for x in sents) for s0 in starts), "every shot must start on an anchor"
    knots, rows, v = [(0.0, 0.0)], [], 0.0
    assert abs(sents[0][3]) < 1e-6, "first anchor must be 0"
    for i, (s, seg_id, shot, a) in enumerate(sents):
        last = i + 1 == len(sents)
        b = pre_end if last else sents[i + 1][3]
        P = round(b - a, 4)
        assert P > 0, (s["sid"], a, b)
        gap = END_TAIL if last else (GAP_SEG if sents[i + 1][1] != seg_id else GAP)
        need = s["dur"] + gap
        c = CONTENT_END.get(s["sid"])
        if c is None:
            L, kn_, mode, hold, h, h_ok, mv = span(v, a, b, need, sh[shot], last, pre_end)
            rate = round(P / L, 3) if not hold else None
        else:
            # CONTENT_END: the picture reaches c when the sentence ends (wav end, rounded up to a frame); the rest b - c plays at <= MAX_SPEED
            assert a < c < b and not last, (s["sid"], a, c, b)
            L1, kn1, mode, hold, h, h_ok, mv = span(v, a, c, math.ceil(s["dur"] * FPS - 1e-6) / FPS, sh[shot], False, pre_end)
            L2, kn2, mode2, hold2, *_ = span(v + L1, c, b, max(gap - (L1 - s["dur"]), 1.0 / FPS), sh[shot], False, pre_end)
            assert not hold2, s["sid"]
            L, kn_ = L1 + L2, kn1 + kn2
            mode = f"{mode}|{mode2}"
            rate = [round((c - a) / L1, 3) if not hold else None, round((b - c) / L2, 3)]
        knots += kn_
        assert last or L >= s["dur"] + MIN_GAP, (s["sid"], L)
        rows.append({"sid": s["sid"], "seg": seg_id, "shot": shot, "pre_a": a, "pre_b": b, "P": P,
                     "need": round(need, 3), "L": round(L, 4), "mode": mode, "rate": rate,
                     "slow_rate": MIN_SLOW if hold else None, "hold_s": round(hold, 3), "hold_at_pre": h,
                     "hold_point_ok": h_ok, "hold_motion": mv, "content_end_pre": c,
                     "anchor_override": s["sid"] in ANCHOR_OVERRIDE, "t_start": round(v, 4), "t_end": round(v + s["dur"], 4),
                     "silence_after_s": round(L - s["dur"], 3)})
        v += L
    total = v                                       # not rounded to 4 decimals: must stay on the frame grid (last shot duration = total - its start)
    # drop duplicate knots (same instant)
    kn = [knots[0]]
    for x in knots[1:]:
        if abs(x[0] - kn[-1][0]) > 1e-9 or abs(x[1] - kn[-1][1]) > 1e-9:
            kn.append(x)

    def v_of(p):                                        # shot start pre -> voiced (every start is an anchor knot)
        return next(k[0] for k in kn if abs(k[1] - p) < 1e-6)

    shot_rows = []
    for name, p0, pdur, cuts in shots:
        v0 = v_of(p0)
        v1 = total if name == shots[-1][0] else v_of(round(p0 + pdur, 4))
        loc = [(round(x - v0, 6), round(p - p0, 6)) for x, p in kn if v0 - 1e-9 <= x <= v1 + 1e-9]
        assert abs(round((v1 - v0) * FPS) - (v1 - v0) * FPS) < 1e-6, (name, v1 - v0)
        shot_rows.append({"shot": name, "pre_t0": p0, "pre_dur": pdur, "t0": round(v0, 6), "dur": round(v1 - v0, 6),
                          "knots": loc})
    sentences = []
    for (s, seg_id, shot, a), r in zip(sents, rows):
        sr = next(x for x in shot_rows if x["shot"] == shot)
        sentences.append({"sid": s["sid"], "seg": seg_id, "shot": shot, "text": s["text"], "tts": s["tts"],
                          "wav": s["wav"], "dur": s["dur"], "anchor_pre": a, "t_start": r["t_start"],
                          "t_end": r["t_end"], "local_start": round(r["t_start"] - sr["t0"], 4),
                          "local_end": round(r["t_end"] - sr["t0"], 4)})
        assert -1e-6 <= sentences[-1]["local_start"] and sentences[-1]["local_end"] <= sr["dur"] + 1e-6, sentences[-1]
    segs = []
    for seg in narr:
        ss_ = [x for x in sentences if x["seg"] == seg["id"]]
        shot = ss_[0]["shot"]
        t1_pre = min(seg["t1"] + shift[shot], pre_end)
        nxt = [x for x in sentences if x["t_start"] > ss_[-1]["t_start"]]
        segs.append({"id": seg["id"], "shot": shot, "pre_t0": ss_[0]["anchor_pre"], "pre_t1": round(t1_pre, 4),
                     "t0": ss_[0]["t_start"], "t1": nxt[0]["t_start"] if nxt else total, "n_sent": len(ss_),
                     "last_end": ss_[-1]["t_end"]})
    doc = {"rule": "piecewise-linear map: sentence starts at its anchor; speed up <= MAX_SPEED, slow >= MIN_SLOW, "
                   "then hold before next anchor", "fps": FPS, "gap": GAP, "gap_seg": GAP_SEG, "end_tail": END_TAIL,
           "max_speed": MAX_SPEED, "min_slow": MIN_SLOW, "pre_dur": pre_end, "total_s": round(total, 6),
           "hold_total_s": round(sum(r["hold_s"] for r in rows), 3), "speech_s": round(sum(s["dur"] for s, *_ in sents), 3),
           "intervals": rows, "shots": shot_rows, "segments": segs, "sentences": sentences}
    (HERE / "timing.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['sid']:9s} pre {r['pre_a']:7.2f}+{r['P']:5.2f} need {r['need']:5.2f} -> L {r['L']:5.2f} "
              f"{r['mode']:9s} rate {str(r['rate'] if r['rate'] is not None else '-'):>5} hold {r['hold_s']:4.2f} "
              f"sil {r['silence_after_s']:4.2f}{'' if r['hold_point_ok'] else ' HOLD-IN-XFADE'}"
              + (f" hold@{r['hold_at_pre']:.4f} motion {r['hold_motion']}" if r["hold_s"] else ""))
    # report sentences that keep speaking > 0.5 s past a PreShot cut inside their interval (cut time mapped to voiced time through the knots)
    kv, kp = np.array([k[0] for k in kn]), np.array([k[1] for k in kn])
    print("sentences still speaking > 0.5 s after a PreShot cut inside their interval:")
    for r in rows:
        name, s0, _, cuts = sh[r["shot"]]
        for o in cuts:
            x = s0 + o
            if r["pre_a"] + 1e-6 < x < r["pre_b"] - 1e-6:
                vc = float(np.interp(x, kp, kv))
                if r["t_end"] - vc > 0.5:
                    print(f"  {r['sid']:9s} cut pre {x:7.2f} (local {o:6.2f}) at voiced {vc:7.2f}; "
                          f"sentence {r['t_start']:7.2f}-{r['t_end']:7.2f} -> {r['t_end'] - vc:4.2f} s after the cut")
    for g in segs:
        print(f"seg {g['id']:6s} voiced {g['t0']:7.2f}-{g['t1']:7.2f} (pre {g['pre_t0']:7.2f}-{g['pre_t1']:7.2f}) "
              f"over_pre {round((g['t1'] - g['t0']) - (g['pre_t1'] - g['pre_t0']), 2):+6.2f}")
    print(f"total {total:.3f} s = {int(total // 60)}:{total % 60:04.1f} (pre {pre_end} s); speech {doc['speech_s']} s; "
          f"holds {doc['hold_total_s']} s; max silence {max(r['silence_after_s'] for r in rows[:-1]):.2f} s")


if __name__ == "__main__":
    main()
