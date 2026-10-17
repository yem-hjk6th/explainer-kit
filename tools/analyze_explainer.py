"""Explainer-video breakdown: timestamped narration (faster-whisper, CPU), on-screen text changes (RapidOCR),
cut timing, and timestamped contact sheets. Writes into <video_dir>/analysis/.
  python -I -X utf8 analyze_explainer.py <video_dir> [--sheet-step 3] [--ocr-step 0.5]
Shot-cut rule = frame-difference peaks.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROMPT = ""  # --prompt: domain terms that bias whisper toward the right spelling (e.g. "epipolar line, fundamental matrix.")


def transcribe(video, out, lang=None, prompt=PROMPT):
    from faster_whisper import WhisperModel
    m = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=8)
    segs, info = m.transcribe(str(video), language=lang, vad_filter=True, initial_prompt=prompt or None, beam_size=5)
    rows = [{"t0": round(s.start, 2), "t1": round(s.end, 2), "text": s.text.strip()} for s in segs]
    (out / "asr.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def ocr_track(video, out, step):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    events, prev = [], None
    for t in np.arange(0, dur, step):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        H, W = fr.shape[:2]
        res, _ = ocr(cv2.resize(fr, (720, int(720 * H / W))))
        boxes = []
        for box, txt, conf in res or []:
            if conf < 0.6 or not txt.strip():
                continue
            ys = [p[1] for p in box]
            boxes.append({"text": txt, "y": round(float(np.mean(ys)) / (720 * H / W), 2)})
        key = " | ".join(b["text"] for b in boxes)
        if key != prev:
            events.append({"t": round(float(t), 1), "boxes": boxes})
        prev = key
    cap.release()
    (out / "ocr.json").write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")
    return events


def cuts(video, out):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=160:-2", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.release()
    h = int(round(H * 160 / W / 2) * 2)
    g = np.frombuffer(p.stdout, np.uint8)
    g = g[: g.size // (160 * h) * 160 * h].reshape(-1, h, 160).astype(np.float32)
    diff = np.abs(np.diff(g, axis=0)).mean((1, 2))
    med = np.median(diff) + 1e-6
    cut_t, i, gap = [], 1, int(0.25 * fps)
    while i < len(diff) - 1:
        w = diff[max(0, i - 4): i + 5]
        if diff[i] == w.max() and diff[i] > max(18, 6 * med):
            cut_t.append(round((i + 1) / fps, 2))
            i += gap
            continue
        i += 1
    motion = [round(float(diff[int(k * fps):int((k + 1) * fps)].mean()), 2) for k in range(int(len(diff) / fps))]
    res = {"fps": fps, "dur": round(len(g) / fps, 2), "n_cuts": len(cut_t), "cut_t": cut_t, "motion_per_sec": motion}
    (out / "cuts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def sheets(video, out, step, cols=4, rows=5, tw=240):
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for t in np.arange(step / 2, dur, step):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        im = im.resize((tw, int(tw * im.height / im.width)))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 52, 15], fill=(0, 0, 0))
        d.text((3, 2), f"{t:6.1f}s", fill=(255, 255, 0))
        tiles.append(im)
    cap.release()
    th, per = tiles[0].height, cols * rows
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        r = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, r * th), (30, 30, 30))
        for k, t in enumerate(chunk):
            sheet.paste(t, ((k % cols) * tw, (k // cols) * th))
        sheet.save(out / f"sheet_{s // per:02d}.jpg", quality=82)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--sheet-step", type=float, default=3.0)
    ap.add_argument("--ocr-step", type=float, default=0.5)
    ap.add_argument("--lang", default=None, help="spoken language code for whisper (default: auto-detect)")
    ap.add_argument("--prompt", default=PROMPT, help="domain terms for whisper initial_prompt")
    a = ap.parse_args()
    d = Path(a.video_dir)
    video = d / "video.mp4"
    out = d / "analysis"
    out.mkdir(exist_ok=True)
    print("cuts", cuts(video, out)["n_cuts"], flush=True)
    sheets(video, out, a.sheet_step)
    print("sheets done", flush=True)
    print("ocr events", len(ocr_track(video, out, a.ocr_step)), flush=True)
    print("asr segs", len(transcribe(video, out, a.lang, a.prompt)), flush=True)
