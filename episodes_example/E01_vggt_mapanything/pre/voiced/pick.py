"""Run whisper on every candidate of each sentence (CPU int8, default medium; small hears rows as roast and depth as death, so it is not used as the judge), then pick one:
among candidates whose keywords (numbers / terms / single-letter symbols) are all correct, whose pauses inside the sentence are <= 0.9 s, and whose speaking rate is 0.8–5 words/s (only catches abnormal dragging / swallowed words),
first take the shortest one whose transcript equals the text exactly (WER = 0); if none, take the shortest one with WER <= 0.12; within the same tier, a candidate whose transcript has no ! / ? is preferred (flat intonation).
The comparison uses the tts text of narration.json (not a tts with changed spelling).
The picked file is copied to audio/sel/<sid>.wav and the result is written to sentences.json; sentences with no qualified candidate are listed in regen.txt.
--plan <file.json> --only sid,...: the listed sentences take the candidate `proposed_k` from the plan file (a list of
{"sid", "proposed_k"}, e.g. written after a prosody check of the candidates; the candidate must be whisper-ok); all other
sentences are picked by the rules above (same result as without --plan).
render env: python -I -X utf8 pick.py [--model medium|small] [--plan <file.json> --only sid,...]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                             # -I does not add the script directory
from voicecheck import compare  # noqa: E402

WER_MAX, GAP_MAX, WPS = 0.12, 0.9, (0.8, 5.0)    # words/s counted as written on screen (VGGT and MapAnything count as 1 word each)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--plan", default=None, help="plan JSON: use its proposed_k")
    ap.add_argument("--only", default=None, help="comma-separated sids: apply --plan only to these sentences")
    a = ap.parse_args()
    plan = {}
    if a.plan:
        plan = {r["sid"]: r["proposed_k"] for r in json.loads((HERE / a.plan).read_text(encoding="utf-8"))}
        if a.only:
            only = a.only.split(",")
            assert set(only) <= set(plan), set(only) - set(plan)
            plan = {k: plan[k] for k in only}
    raw = json.loads((HERE / "gen_raw.json").read_text(encoding="utf-8"))
    cache_p = HERE / "whisper_cands.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    m = None
    sel_dir = HERE / "audio" / "sel"
    sel_dir.mkdir(parents=True, exist_ok=True)
    out, regen = [], []
    for r in raw:
        nw = len(r["text"].split())
        for c in r["cands"]:
            key = f"{a.model}|{c['file']}|{c.get('seed', r['seed'])}"
            if key not in cache:
                m = m or WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=6)
                seg, _ = m.transcribe(str(HERE / c["file"]), language="en", word_timestamps=True)
                words = [(w.word.strip(), round(float(w.start), 2), round(float(w.end), 2)) for s in seg for w in s.words]
                cache[key] = {"text": " ".join(w[0] for w in words), "words": words}
            wh = cache[key]
            cmp_ = compare(r["tts"], wh["text"])
            ws = wh["words"]
            gap = max([b[1] - x[2] for x, b in zip(ws, ws[1:])] or [0.0])
            c.update(whisper=wh["text"], wer=cmp_["wer"], missing=cmp_["missing_keys"], max_gap=round(float(gap), 2),
                     wps=round(nw / c["dur"], 2))
            c["ok"] = bool(not c["missing"] and c["wer"] <= WER_MAX and c["max_gap"] <= GAP_MAX
                           and WPS[0] <= c["wps"] <= WPS[1])
        ok = [c for c in r["cands"] if c["ok"]]
        exact = [c for c in ok if c["wer"] == 0]
        pool = exact or ok or r["cands"]
        calm = [c for c in pool if not re.search(r"[!?]", c["whisper"])]   # candidates whose whisper transcript has ! / ? (tends to sound excited) go last
        best = min(calm or pool, key=lambda c: c["dur"])
        rule_k = best["k"]
        if r["sid"] in plan:
            best = next(c for c in r["cands"] if c["k"] == plan[r["sid"]])
            assert best["ok"], (r["sid"], best["k"], "plan candidate is not whisper-ok")
        dst = sel_dir / f"{r['sid']}.wav"
        shutil.copyfile(HERE / best["file"], dst)
        out.append({"sid": r["sid"], "seg": r["seg"], "text": r["text"], "tts": r["tts"],
                    "tts_used": best.get("tts", r["tts"]), "wav": f"audio/sel/{dst.name}", "dur": best["dur"],
                    "pick": best["k"], "pick_rule_k": rule_k, "pick_by": "f0_plan" if r["sid"] in plan else "rule",
                    "seed": best.get("seed", r["seed"]), "rounds": r.get("rounds", 1),
                    "whisper_model": a.model, "whisper": best["whisper"], "wer": best["wer"], "missing": best["missing"],
                    "max_gap": best["max_gap"], "wps": best["wps"], "ok": best["ok"],
                    "cand_durs": [c["dur"] for c in r["cands"]], "cand_ok": [c["ok"] for c in r["cands"]]})
        if not ok:
            regen.append(r["sid"])
        flag = ("OK " if best["ok"] else "BAD") + ("F0" if r["sid"] in plan else "  ")
        print(f"{flag} {r['sid']:10s} {best['dur']:5.2f}s k={best['k']} (rule k={rule_k}) wer={best['wer']} miss={best['missing']} "
              f"gap={best['max_gap']} | {best['whisper']}", flush=True)
        if not ok or not exact:
            for c in r["cands"]:
                print(f"      k={c['k']} {c['dur']:.2f}s wer={c['wer']} miss={c['missing']} gap={c['max_gap']} "
                      f"wps={c['wps']} | {c['whisper']}")
        cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    (HERE / "sentences.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "regen.txt").write_text("\n".join(regen), encoding="utf-8")
    print(f"{len(out)} sentences, {len(regen)} without a passing candidate: {regen}")


if __name__ == "__main__":
    main()
