"""Voiced variant: per ../timing.json, replace a shot's captions (one per voice-over sentence, appearing / disappearing at the sentence times) and change the time axis.
Call voice(shot) after constructing pre_cut.PreShot and pre_card.card; titles / corner tags / picture are unchanged, only the times follow a piecewise-linear map.
Times: pre time = seconds of the pre shot; voiced time = seconds of the voiced shot. shot.knots = [(voiced, pre)], both columns monotone non-decreasing
(a stretch where the pre column stays equal = a frame hold).
"""
import json
from pathlib import Path

import numpy as np

TIMING = json.loads((Path(__file__).resolve().parents[1] / "timing.json").read_text(encoding="utf-8"))
MIN_CAP = 1.0      # short sentences keep their caption at least 1 s, but not within 0.05 s of the next sentence appearing


def back(shot, t):
    """voiced time -> which second of the pre shot the picture shows (does not move during a hold)."""
    v, p = shot._kv, shot._kp
    return float(np.interp(t, v, p))


def fwd(shot, x, side="left"):
    """pre time -> voiced time. When x falls on a hold, left takes the start of the hold and right the end (titles / corner tags use left for their start and right for their end)."""
    ks = shot.knots
    if x <= ks[0][1]:
        return ks[0][0] + (x - ks[0][1])
    if x >= ks[-1][1]:
        return ks[-1][0] + (x - ks[-1][1])
    idx = range(len(ks) - 1) if side == "left" else reversed(range(len(ks) - 1))
    for k in idx:
        (v0, p0), (v1, p1) = ks[k], ks[k + 1]
        if p0 <= x <= p1:
            if p1 - p0 < 1e-12:
                return v0 if side == "left" else v1
            return v0 + (x - p0) * (v1 - v0) / (p1 - p0)
    raise ValueError(x)


def voice(shot):
    row = next(r for r in TIMING["shots"] if r["shot"] == shot.name)
    sents = [s for s in TIMING["sentences"] if s["shot"] == shot.name]
    shot.pre_dur = shot.dur
    shot.knots = [tuple(k) for k in row["knots"]]
    shot._kv = np.array([k[0] for k in shot.knots])
    shot._kp = np.array([k[1] for k in shot.knots])
    shot.dur = row["dur"]
    shot.titles = [(text, sub, fwd(shot, t0), fwd(shot, t1, "right"), fin, fout)
                   for text, sub, t0, t1, fin, fout in shot.titles]
    shot.tags = [(text, fwd(shot, t0), fwd(shot, t1, "right")) for text, t0, t1 in shot.tags]
    shot.captions = []
    for k, s in enumerate(sents):
        a, b = s["local_start"], s["local_end"]
        nxt = sents[k + 1]["local_start"] - 0.05 if k + 1 < len(sents) else shot.dur
        shot.caption(a, min(max(b, a + MIN_CAP), max(b, nxt), shot.dur), s["text"])

    def check(ck):
        kv, kp = shot._kv, shot._kp
        ck.true("knots monotone (voiced strictly up, pre non-decreasing)",
                bool(np.all(np.diff(kv) > 0) and np.all(np.diff(kp) >= -1e-9)))
        ck.close("knots span the voiced shot", kv[-1], shot.dur, 1e-6)
        ck.close("knots span the pre shot", kp[-1], shot.pre_dur, 2e-3)
        ck.true("knots start at (0, 0)", abs(kv[0]) < 1e-9 and abs(kp[0]) < 1e-9)
        ck.eq("one caption per narration sentence", [c[2] for c in sorted(shot.captions)], [s["text"] for s in sents])
        for (a, b, text), s in zip(sorted(shot.captions), sents):
            ck.true(f"caption inside shot and starts with its sentence: {text!r}",
                    0 <= a < b <= shot.dur + 1e-6 and abs(a - s["local_start"]) < 1e-6 and b >= s["local_end"] - 1e-6)
        cs = sorted(shot.captions)
        ck.true("captions do not overlap", all(c1[0] >= c0[1] - 1e-6 for c0, c1 in zip(cs, cs[1:])))
        rates = [(p1 - p0) / (v1 - v0) for (v0, p0), (v1, p1) in zip(shot.knots, shot.knots[1:])]
        ck.true("picture rate = 0 (hold) or within [0.59, 2.6] (0.6x / 2.5x after rounding to frames)",
                all(r < 1e-9 or 0.59 <= r <= 2.6 for r in rates))
        ck.record(knots=[list(k) for k in shot.knots], pre_dur=shot.pre_dur,
                  rate_min=round(min(rates), 3), rate_max=round(max(rates), 3))

    shot._checks.append(check)
    return shot
