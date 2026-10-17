"""Self-check of the voiced cut (see references/TTS.md §6): whole-track and per-sentence transcription, silence table, 8 acceptance checks. render env: python -I -X utf8 check_voiced.py
1. Run whisper over the whole audio track of the final film (CPU int8, medium, with word times), take the words inside each sentence's [t_start, t_end] from timing.json, and compare sentence by sentence with the voice-over text
   (keywords = numbers / terms / single-letter symbols must all be present; WER is also recorded) -> check_whisper.json
2. Per-sentence slices: cut the final audio track at [t_start - 0.1, t_end + 0.1] of each sentence and transcribe each slice alone (same medium model), compare the same way -> check_whisper.json slices
3. Silence table: ffmpeg silencedetect (-45 dB, >= 0.2 s) on the final audio track, per segment (title cards included; a segment = from its first sentence's onset to the next segment's first onset):
   whether there is sound, the longest silence in seconds, and where -> silence_table.json
4. Acceptance checks 1-8 -> check_whisper.json acceptance
5. Grab one frame at the midpoint of each segment (from the final film), print the segment id + its voice-over below -> _check/mid_sheet_{1,2,3}.jpg
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                             # -I does not add the script directory
from voicecheck import compare, tokens  # noqa: E402
MP4 = HERE.parents[1] / "render" / "E01_pre_en_voice.mp4"
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root
from animkit.config import ffmpeg as _ffmpeg, ffprobe as _ffprobe  # noqa: E402
from animkit.text import font as _font  # noqa: E402
FF = _ffmpeg()
NARR = HERE.parent / "narration" / "narration.json"
PAD = 0.15
SIL_DB, SIL_MIN = -45, 0.2
SLICE_PAD = 0.1


def whisper_full(m, tm):
    seg, info = m.transcribe(str(MP4), language="en", word_timestamps=True, vad_filter=False)
    words = [(w.word.strip(), w.start, w.end) for s in seg for w in s.words]
    rows = []
    for s in tm["sentences"]:
        hyp = " ".join(w for w, a, b in words if s["t_start"] - PAD <= (a + b) / 2 <= s["t_end"] + PAD)
        c = compare(s["tts"], hyp)
        rows.append({"sid": s["sid"], "seg": s["seg"], "t_start": s["t_start"], "text": s["text"], "tts": s["tts"],
                     "whisper": hyp, "wer": c["wer"], "missing_keys": c["missing_keys"]})
    return words, rows


def whisper_slices(m, tm):
    wav = HERE / "_check" / "final_16k.wav"
    subprocess.run([FF, "-y", "-v", "error", "-i", str(MP4), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    y, sr = sf.read(wav, dtype="float32")
    rows = []
    for s in tm["sentences"]:
        a, b = max(0, int((s["t_start"] - SLICE_PAD) * sr)), int((s["t_end"] + SLICE_PAD) * sr)
        seg, _ = m.transcribe(y[a:b], language="en", vad_filter=False)
        hyp = " ".join(x.text.strip() for x in seg)
        c = compare(s["tts"], hyp)
        rows.append({"sid": s["sid"], "whisper": hyp, "wer": c["wer"], "missing_keys": c["missing_keys"]})
    return rows


def silences():
    p = subprocess.run([FF, "-hide_banner", "-nostats", "-i", str(MP4), "-vn", "-af",
                        f"silencedetect=noise={SIL_DB}dB:d={SIL_MIN}", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    st = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", p.stderr)]
    en = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", p.stderr)]
    dur = float(subprocess.run([_ffprobe(FF), "-v", "error", "-show_entries",
                                "format=duration", "-of", "csv=p=0", str(MP4)], capture_output=True, text=True).stdout)
    if len(en) < len(st):
        en.append(dur)                                      # a silence at the end of the film runs to the end of the file
    return list(zip(st, en)), dur


def silence_table(tm, sil, dur):
    rows = []
    segs = tm["segments"]
    for g in segs:
        t0, t1 = g["t0"], g["t1"]
        pieces = [(max(a, t0), min(b, t1)) for a, b in sil if b > t0 and a < t1]
        pieces = [(a, b) for a, b in pieces if b - a > 1e-3]
        sil_s = sum(b - a for a, b in pieces)
        tail = [(a, b) for a, b in pieces if g is segs[-1] and abs(b - dur) < 0.05]   # the silence at the end of the film is listed separately, not counted between sentences
        inner = [x for x in pieces if x not in tail]
        lng = max(inner, key=lambda x: x[1] - x[0]) if inner else None
        rows.append({"seg": g["id"], "card": g["id"].startswith("c"), "t0": round(t0, 2), "t1": round(t1, 2),
                     "len_s": round(t1 - t0, 2), "voiced_s": round(t1 - t0 - sil_s, 2), "has_voice": t1 - t0 - sil_s > 0.3,
                     "longest_silence_s": round(lng[1] - lng[0], 2) if lng else 0.0,
                     "longest_at": [round(lng[0], 2), round(lng[1], 2)] if lng else None,
                     "end_tail_s": round(tail[0][1] - tail[0][0], 2) if tail else None})
    return rows


def acceptance(tm, words, full, slices, sil, dur):
    by_full = {r["sid"]: r for r in full}
    by_slice = {r["sid"]: r for r in slices}
    sents = tm["sentences"]

    def has(sid, phrase, which):
        r = (by_full if which == "full" else by_slice)[sid]
        return " ".join(tokens(phrase)) in " ".join(tokens(r["whisper"]))

    four = [s["sid"] for s in sents if "four factors" in s["tts"]]
    m_s = [s["sid"] for s in sents if re.search(r"\bM\b", s["tts"])]
    vg = [s["sid"] for s in sents if "VGGT" in s["text"] or "V G G T" in s["tts"]]
    narr = json.loads(NARR.read_text(encoding="utf-8"))
    p33 = [x["tts"] for g in narr if g["id"] == "p3.3" for x in g["sentences"]]
    last_word_end = max(w[2] for w in words) if words else None
    tail_sil = [b - a for a, b in sil if abs(b - dur) < 0.05]
    mid = silence_table(tm, sil, dur)
    acc = {
        "1_four_factors": {"sids": four,
                           "full": [has(s, "four factors", "full") for s in four],
                           "slice": [has(s, "four factors", "slice") for s in four],
                           "no_1.4_anywhere": not any(re.search(r"1\.4", r["whisper"]) for r in full + slices)},
        "2_M": {"sids": m_s,
                "full_no_missing_m": ["m" not in by_full[s]["missing_keys"] for s in m_s],
                "slice_no_missing_m": ["m" not in by_slice[s]["missing_keys"] for s in m_s],
                "full_text": [by_full[s]["whisper"] for s in m_s], "slice_text": [by_slice[s]["whisper"] for s in m_s]},
        "3_p3.3_calibration": {"in_tts": any("calibration" in t.lower() for t in p33),
                               "heard_full": any(has(s["sid"], "calibration", "full") for s in sents if s["seg"] == "p3.3"),
                               "heard_slice": any(has(s["sid"], "calibration", "slice") for s in sents if s["seg"] == "p3.3")},
        "4_VGGT": {"n_sentences": len(vg),
                   "tts_all_spelled": all("V G G T" in s["tts"] for s in sents if "VGGT" in s["text"]),
                   "tts_no_unspelled": not any("VGGT" in s["tts"] for s in sents),
                   "full_ok": sum("vggt" not in by_full[s]["missing_keys"] for s in vg),
                   "slice_ok": sum("vggt" not in by_slice[s]["missing_keys"] for s in vg)},
        "5_end_tail": {"last_word_end_s": round(last_word_end, 2) if last_word_end else None, "video_s": round(dur, 2),
                       "after_last_word_s": round(dur - last_word_end, 2) if last_word_end else None,
                       "final_silence_s": round(tail_sil[0], 2) if tail_sil else 0.0},
        "6_silence": {"max_mid_silence_s": max(r["longest_silence_s"] for r in mid),
                      "segments_over_1.0s": [(r["seg"], r["longest_silence_s"], r["longest_at"]) for r in mid
                                             if r["longest_silence_s"] > 1.0],
                      "segments_without_voice": [r["seg"] for r in mid if not r["has_voice"]]},
        "7_flat_text": {"tts_question_marks": sum(s["tts"].count("?") for s in sents),
                        "tts_exclamations": sum(s["tts"].count("!") for s in sents),
                        # sentences whose slice / full-track transcript has ! or ? (a hint about tone, not a pass criterion)
                        "slice_bang_or_q": [r["sid"] for r in slices if re.search(r"[!?]", r["whisper"])],
                        "full_bang_or_q": [r["sid"] for r in full if re.search(r"[!?]", r["whisper"])]},
        "8_recheck": {"p1.1_forward_full": has("p1.1_0", "forward", "full"),
                      "p1.1_whisper_full": by_full["p1.1_0"]["whisper"],
                      "p3.4_table_two_full": any(has(s["sid"], "table two", "full") for s in sents if s["seg"] == "p3.4"),
                      "p3.4_whisper_full": [by_full[s["sid"]]["whisper"] for s in sents if s["seg"] == "p3.4"]},
    }
    return acc

def mid_sheets(tm):
    d = HERE / "_check"
    d.mkdir(exist_ok=True)
    f = _font("text", 15)
    tiles = []
    for g in tm["segments"]:
        t = round((g["t0"] + g["t1"]) / 2, 2)
        p = d / f"mid_{g['id']}.jpg"
        subprocess.run([FF, "-y", "-v", "error", "-ss", str(t), "-i", str(MP4), "-frames:v", "1", "-vf", "scale=300:400",
                        str(p)], check=True)
        said = " / ".join(s["text"] for s in tm["sentences"] if s["seg"] == g["id"])
        tiles.append((g["id"], t, Image.open(p).convert("RGB"), said))
    out = []
    for k in range(0, len(tiles), 16):
        part = tiles[k:k + 16]
        sheet = Image.new("RGB", (4 * 300, 4 * 470), (24, 24, 24))
        dr = ImageDraw.Draw(sheet)
        for j, (sid, t, im, said) in enumerate(part):
            x, y = (j % 4) * 300, (j // 4) * 470
            sheet.paste(im, (x, y))
            lines, cur = [], ""
            for w in f"{sid} @{t:.1f}s: {said}".split():
                if f.getlength(cur + " " + w) > 290:
                    lines.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            lines.append(cur)
            for i, ln in enumerate(lines[:4]):
                dr.text((x + 5, y + 402 + 16 * i), ln, font=f, fill=(230, 230, 230))
        p = d / f"mid_sheet_{k // 16 + 1}.jpg"
        sheet.save(p, quality=88)
        out.append(str(p))
    return out


def main():
    tm = json.loads((HERE / "timing.json").read_text(encoding="utf-8"))
    sheets = mid_sheets(tm)
    sil, dur = silences()
    table = silence_table(tm, sil, dur)
    (HERE / "silence_table.json").write_text(json.dumps({"detector": f"silencedetect noise={SIL_DB}dB d={SIL_MIN}",
                                                         "silences": sil, "video_s": dur, "segments": table},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    m = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=6)
    words, full = whisper_full(m, tm)
    slices = whisper_slices(m, tm)
    acc = acceptance(tm, words, full, slices, sil, dur)
    out = {"model": "faster-whisper medium, cpu int8", "n_words": len(words), "full_text": " ".join(w for w, *_ in words),
           "sentences": full, "slices": slices, "acceptance": acc,
           "n_missing": sum(1 for r in full if r["missing_keys"]), "mean_wer": round(sum(r["wer"] for r in full) / len(full), 3),
           "n_missing_slices": sum(1 for r in slices if r["missing_keys"]),
           "mean_wer_slices": round(sum(r["wer"] for r in slices) / len(slices), 3)}
    (HERE / "check_whisper.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    sl = {r["sid"]: r for r in slices}
    for r in full:
        flag = "MISS" if r["missing_keys"] or sl[r["sid"]]["missing_keys"] else ("wer " if r["wer"] > 0.25 else "ok  ")
        print(f"{flag} {r['sid']:9s} wer={r['wer']:.2f}/{sl[r['sid']]['wer']:.2f} miss={r['missing_keys']}"
              f"/{sl[r['sid']]['missing_keys']} | {r['whisper']} || {sl[r['sid']]['whisper']}")
    print("\nsilence table (seg, len, voiced, longest silence @):")
    for r in table:
        print(f"  {r['seg']:6s} {r['len_s']:6.2f} {r['voiced_s']:6.2f} {r['longest_silence_s']:5.2f} {r['longest_at']}"
              f"{'' if r['end_tail_s'] is None else ' end tail %.2f' % r['end_tail_s']}")
    print(json.dumps(acc, ensure_ascii=False, indent=1))
    print("sheets", sheets, "| missing", out["n_missing"], "/", out["n_missing_slices"], "mean wer", out["mean_wer"],
          "/", out["mean_wer_slices"])


if __name__ == "__main__":
    main()
