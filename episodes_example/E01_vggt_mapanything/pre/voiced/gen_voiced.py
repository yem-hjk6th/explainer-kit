"""E01 pre English version, ryan voice: generate TRIES candidates for every sentence of narration.json (including sentences in alt_sentences that differ from the main text).
tts env: python -I -X utf8 gen_voiced.py [--only ID ...]
Output: audio/cand/<sid>_<k>.wav (leading/trailing silence trimmed, normalized to -18 dBFS) + gen_raw.json. Selection / whisper is in pick.py (render env).
Same approach as the voice sample generator: numbers are already spelled out as words in the tts column of narration.json; sampled durations for the same text can differ by 2x, so sample several and pick.
Cut version, ryan-slow voice: INSTRUCT is verbatim PACE["slow"] of gen_samples.py with no suffix; as in the samples, one sentence per call (no batching),
one seed per candidate.
"""
import argparse
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

HERE = Path(__file__).resolve().parent
NARR = HERE.parent / "narration" / "narration.json"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "ryan"
INSTRUCT = "Speak at a natural, steady pace, like a lecturer explaining to graduate students. No laughter, no filler sounds."
TRIES = 3
DUTY = float(os.environ.get("VOICE_DUTY", "0"))   # after each generation sleep DUTY x generation time, to cap GPU utilization


def normalize(y, rms_db=-18.0, peak_db=-1.0):
    rms = np.sqrt((y ** 2).mean()) + 1e-9
    g = min(10 ** (rms_db / 20) / rms, 10 ** (peak_db / 20) / (np.abs(y).max() + 1e-9))
    return (y * g).astype(np.float32)


def sentences():
    """[(sid, seg_id, text, tts)]: main sentences <seg>_<k>, alternate sentences <seg>_alt<k> (an alternate whose tts equals the main sentence at the same position is not generated again)."""
    out = []
    for seg in json.loads(NARR.read_text(encoding="utf-8")):
        main = seg["sentences"]
        for k, s in enumerate(main):
            out.append((f"{seg['id']}_{k}", seg["id"], s["text"], s["tts"]))
        for k, s in enumerate(seg.get("alt_sentences", [])):
            if k < len(main) and main[k]["tts"] == s["tts"]:
                continue
            out.append((f"{seg['id']}_alt{k}", seg["id"], s["text"], s["tts"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="regenerate only these sids (new candidates are appended after the old ones, k continues numbering, nothing is overwritten)")
    ap.add_argument("--tries", type=int, default=TRIES)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tts", nargs="*", default=[], help="SID=TEXT: use a different tts spelling for this sentence (on-screen text unchanged); recorded in gen_raw.json")
    a = ap.parse_args()
    over = dict(x.split("=", 1) for x in a.tts)
    torch.cuda.set_per_process_memory_fraction(0.30)
    cand = HERE / "audio" / "cand"
    cand.mkdir(parents=True, exist_ok=True)
    raw_p = HERE / "gen_raw.json"
    res = {r["sid"]: r for r in json.loads(raw_p.read_text(encoding="utf-8"))} if raw_p.exists() else {}
    todo = [x for x in sentences() if not a.only or x[0] in a.only]
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    t_all = time.time()
    for n, (sid, seg, text, line) in enumerate(todo):
        line = over.get(sid, line)
        seed = a.seed0 + 10 * n                           # candidate j uses seed + j; change --seed0 when regenerating
        t0 = time.time()
        wavs = []
        for j in range(a.tries):                          # one call per candidate, as in voice/gen_samples.py
            torch.manual_seed(seed + j)
            w, sr = tts.generate_custom_voice(text=line, speaker=SPEAKER, language="English", instruct=INSTRUCT)
            wavs.append((seed + j, w[0]))
        g = time.time() - t0
        old = res.get(sid, {}).get("cands", []) if a.only else []
        cands = list(old)
        for k, (seed_k, w) in enumerate(wavs, start=len(old)):
            y = np.asarray(w, dtype=np.float32)
            raw_dur = len(y) / sr
            _, idx = librosa.effects.trim(y, top_db=35)
            y = y[max(0, idx[0] - int(0.03 * sr)): idx[1] + int(0.03 * sr)]
            y = normalize(y)
            f = cand / f"{sid}_{k}.wav"
            sf.write(f, y, sr)
            cands.append({"k": k, "file": f"audio/cand/{f.name}", "dur": round(len(y) / sr, 3),
                          "raw_dur": round(raw_dur, 3), "seed": seed_k, "tts": line, "instruct": INSTRUCT})
        res[sid] = {"sid": sid, "seg": seg, "text": text, "tts": res.get(sid, {}).get("tts", line), "seed": seed,
                    "sr": sr, "gen_s": round(g, 1), "rounds": res.get(sid, {}).get("rounds", 0) + 1, "cands": cands}
        print(f"[{n + 1}/{len(todo)}] {sid} {[c['dur'] for c in cands]} gen {g:.1f}s | {line}", flush=True)
        raw_p.write_text(json.dumps(list(res.values()), ensure_ascii=False, indent=1), encoding="utf-8")
        if DUTY > 0:
            time.sleep(DUTY * g)
    print(f"done {len(todo)} sentences in {time.time() - t_all:.0f}s; peak GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del tts
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
