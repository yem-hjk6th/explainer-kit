"""Generate voice samples per sentence (numbers spelled out, avoids digit misreads and drift), 3 segments x 3 speakers x 2 paces.
Run in tts env. Output tools/tts/audio/*.wav + results_raw.json. GPU freed at exit."""
import json, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, librosa
from qwen_tts import Qwen3TTSModel

torch.cuda.set_per_process_memory_fraction(0.40)
HERE = Path(__file__).parent
SEGS = json.loads((HERE / "segments.example.json").read_text(encoding="utf-8"))
SPEAKERS = ["ryan", "aiden", "serena"]
# Qwen3-TTS runs slow by default (~1.5 words/s with a "natural pace" instruct, 2-3x the speed lecturers use); "normal" asks for quick speech.
PACE = {"normal": "Speak quickly and fluently, like a confident lecturer. No laughter, no filler sounds.",
        "slow": "Speak at a natural, steady pace, like a lecturer explaining to graduate students. No laughter, no filler sounds."}
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
GAP = 0.25  # s between sentences


def normalize(y, rms_db=-18.0, peak_db=-1.0):
    rms = np.sqrt((y ** 2).mean()) + 1e-9
    g = min(10 ** (rms_db / 20) / rms, 10 ** (peak_db / 20) / (np.abs(y).max() + 1e-9))
    return (y * g).astype(np.float32)


tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
out = HERE / "audio"; out.mkdir(exist_ok=True)
res = []
for seg in SEGS:
    for sp in SPEAKERS:
        for pace, ins in PACE.items():
            name = f"{seg['id']}_{sp}_{pace}"
            t0 = time.time()
            parts, sdur = [], []
            for line in seg["tts"]:
                best = None
                for _try in range(3):  # sampling is stochastic (same text: 7 s vs 15 s); keep the fastest of up to 3
                    wavs, sr = tts.generate_custom_voice(text=line, speaker=sp, language="English", instruct=ins)
                    y = np.asarray(wavs[0], dtype=np.float32)
                    if best is None or len(y) < len(best):
                        best = y
                    if len(y) / sr < 0.5 * len(line.split()):
                        break
                y = best
                _, idx = librosa.effects.trim(y, top_db=35)
                y = y[max(0, idx[0] - int(0.03 * sr)): idx[1] + int(0.03 * sr)]
                parts.append(y); sdur.append(round(len(y) / sr, 2))
            gap = np.zeros(int(GAP * sr), dtype=np.float32)
            y = normalize(np.concatenate([p for q in parts for p in (q, gap)][:-1]))
            sf.write(out / f"{name}.wav", y, sr)
            res.append({"name": name, "seg": seg["id"], "speaker": sp, "pace": pace,
                        "dur": round(len(y) / sr, 2), "sent_dur": sdur, "gen_s": round(time.time() - t0, 1)})
            print(name, res[-1]["dur"], res[-1]["gen_s"], flush=True)
            (HERE / "results_raw.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
