"""Lay out the voice-over track + loudness + mux. render env: python -I -X utf8 mix.py
1. Lay audio/sel/*.wav into one track at the t_start values of timing.json (sample-aligned, digital silence between sentences, no music, no gate) -> audio/voice_track_raw.wav
2. ffmpeg loudnorm, two passes: I = -16 LUFS, TP = -2.0 dBTP, 48 kHz -> audio/voice_track.wav; then measure the final film with ebur128
3. Mux with render/E01_pre_en_voiced_silent.mp4 into ../../render/E01_pre_en_voice.mp4 (video stream copied, AAC 192k)
4. render/E01_pre_en_voice.srt: one entry per sentence, time = on-screen caption time (read from each shot's meta.json, so it matches the picture)
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root
from animkit.config import ffmpeg as _ffmpeg, ffprobe as _ffprobe  # noqa: E402
FF = _ffmpeg()
SILENT = HERE / "render" / "E01_pre_en_voiced_silent.mp4"
OUT = HERE.parents[1] / "render" / "E01_pre_en_voice.mp4"
I, TP, LRA = -16.0, -2.0, 11.0   # target: the final film's true peak <= -1.5 dBTP. AAC encoding raises the true peak by about 0.1 dB over the wav, so the wav target is -2.0 (0.5 dB margin)
FADE = 0.02                      # seconds of linear fade-out at the end of every sentence wav


def ff(args, **kw):
    return subprocess.run([FF, "-hide_banner", "-nostats"] + args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", **kw)


def lay(tm):
    sents = tm["sentences"]
    sr = sf.info(HERE / sents[0]["wav"]).samplerate
    y = np.zeros(int(round(tm["total_s"] * sr)), dtype=np.float32)
    last = -1
    for s in sents:
        w, r = sf.read(HERE / s["wav"], dtype="float32")
        assert r == sr
        nf = int(FADE * sr)                      # a 20 ms fade-out at the end of the sentence avoids a click or a hard-cut consonant
        w[-nf:] *= np.linspace(1.0, 0.0, nf, dtype=np.float32)
        i = int(round(s["t_start"] * sr))
        assert i >= last, f"overlap at {s['sid']}"
        assert i + len(w) <= len(y), f"{s['sid']} runs past the end"
        y[i:i + len(w)] += w
        last = i + len(w)
    raw = HERE / "audio" / "voice_track_raw.wav"
    sf.write(raw, y, sr, subtype="FLOAT")
    return raw, sr


def loudnorm(raw):
    p1 = ff(["-i", str(raw), "-af", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json", "-f", "null", "-"])
    m = json.loads(re.findall(r"\{[^{}]*\}", p1.stderr)[-1])
    af = (f"loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
          f"linear=true:print_format=json")
    out = HERE / "audio" / "voice_track.wav"
    p2 = ff(["-y", "-i", str(raw), "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", str(out)])
    m2 = json.loads(re.findall(r"\{[^{}]*\}", p2.stderr)[-1])
    return out, m, m2


def ebur128(path):
    p = ff(["-i", str(path), "-map", "0:a", "-af", "ebur128=peak=true", "-f", "null", "-"])
    tail = p.stderr[p.stderr.rfind("Summary:"):]
    return {"I_LUFS": float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1)),
            "LRA_LU": float(re.search(r"LRA:\s+(-?[\d.]+) LU", tail).group(1)),
            "TP_dBTP": float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail).group(1))}


def srt(tm):
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    cues = []
    for row in tm["shots"]:
        meta = json.loads((HERE / "shots" / "render" / row["shot"] / "meta.json").read_text(encoding="utf-8"))
        for c in meta["captions"]:
            cues.append((row["t0"] + c["t0"], row["t0"] + c["t1"], c["text"].replace("\n", " ")))
    assert [c[2] for c in cues] == [s["text"] for s in tm["sentences"]], "on-screen captions != narration sentences"
    p = HERE / "render" / "E01_pre_en_voice.srt"
    p.write_text("".join(f"{k + 1}\n{ts(a)} --> {ts(b)}\n{t}\n\n" for k, (a, b, t) in enumerate(cues)), encoding="utf-8")
    return p, len(cues)


def probe_dur(path, stream):
    p = subprocess.run([_ffprobe(FF), "-v", "error", "-select_streams", stream, "-show_entries",
                        "stream=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(p.stdout.strip().splitlines()[0])


def main():
    tm = json.loads((HERE / "timing.json").read_text(encoding="utf-8"))
    raw, sr = lay(tm)
    track, m1, m2 = loudnorm(raw)
    OUT.parent.mkdir(exist_ok=True)
    p = ff(["-y", "-i", str(SILENT), "-i", str(track), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", "-movflags", "+faststart", str(OUT)])
    assert p.returncode == 0, p.stderr[-2000:]
    s, n = srt(tm)
    rep = {"raw": str(raw), "sr_in": sr, "loudnorm_pass1": m1, "loudnorm_pass2": {k: m2[k] for k in
           ("output_i", "output_tp", "output_lra", "normalization_type")}, "measured_final_mp4": ebur128(OUT),
           "video_s": probe_dur(OUT, "v:0"), "audio_s": probe_dur(OUT, "a:0"), "timing_total_s": tm["total_s"],
           "srt": str(s), "srt_cues": n, "out": str(OUT)}
    (HERE / "mix_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
