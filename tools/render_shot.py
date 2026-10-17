"""Render one shot script.
  python -I -X utf8 render_shot.py <shot.py> [--out DIR] [--check-only] [--still T [T ...]] [--sheet-only]
                                         [--jobs N] [--lang CODE [--captions FILE]] [--collect-text [--fps 10]]
Default output: <shot dir>/render/<shot stem>/. Checks run first; a failed check stops the render (exit 2).
--jobs N splits the frames into N contiguous ranges rendered by N child processes, then joins them (stream copy).
Child processes inherit the CPU affinity of this process, so launch it under your `start /AFFINITY` mask.
Shots are written in English. --lang CODE (e.g. es): on-screen text from captions.json (next to the shot or in its
parent; --captions overrides); every output file gets the suffix _CODE (<stem>_es.mp4, sheet_es.jpg, check_es.json,
meta_es.json).
--collect-text: run the frames at --fps (no video) and write text_src.json = every source string drawn (i18n_table.py).
--still PNGs are written once and never refreshed by later renders: each run lists stills older than the shot script.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]          # repo root: animkit/, examples/
sys.path.insert(0, str(LIB))

from PIL import Image  # noqa: E402

from animkit import i18n  # noqa: E402
from animkit.config import ffmpeg  # noqa: E402
from animkit.timeline import CheckFailed, default_out, load_shot  # noqa: E402


def render_parallel(shot, script, out, jobs, lang_args):
    n = int(round(shot.dur * shot.fps))
    bounds = [round(k * n / jobs) for k in range(jobs + 1)]
    t0 = time.time()
    py = [sys.executable, "-I"] + (["-B"] if sys.flags.dont_write_bytecode else []) + ["-X", "utf8"]
    procs = [subprocess.Popen(py + [__file__, str(script), "--out", str(out), "--part", f"{a}:{b}"] + lang_args)
             for a, b in zip(bounds[:-1], bounds[1:])]
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError("a render part failed")
    sfx = i18n.suffix()
    parts = out / f"_parts{sfx}"
    lst = parts / "list.txt"
    lst.write_text("".join(f"file 'part_{a:06d}.mp4'\n" for a in bounds[:-1]), encoding="utf-8")
    mp4 = out / f"{out.name}{sfx}.mp4"
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(mp4)], check=True)
    tiles = [(t, Image.open(parts / f"tile_{i:06d}.png").convert("RGB")) for t, i in shot._times_for_sheet()]
    shot._save_sheet(out / f"sheet{sfx}.jpg", tiles)
    if shot.audio:
        shot._mux_audio(mp4)
    render_s = round(time.time() - t0, 1)
    rep = None
    if i18n.LANG != i18n.SOURCE:
        rep = i18n.report()
        for a in bounds[:-1]:
            r = json.loads((parts / f"i18n_{a:06d}.json").read_text(encoding="utf-8"))
            rep["used"].update(r["used"])
            rep["untranslated"] += [s for s in r["untranslated"] if s not in rep["untranslated"]]
            rep["off_canvas"] += [w for w in r["off_canvas"] if w["src"] not in {x["src"] for x in rep["off_canvas"]}]
    shot.write_meta(out, render_s, n, rep)
    return {"render_s": render_s, "frames": n, "jobs": jobs}


def collect_text(shot, out, fps):
    """Draw every 1/fps s (no encoding) and list the source strings that went through the text entry points."""
    n = int(round(shot.dur * fps))
    for i in range(n + 1):
        shot.render_frame(min(i / fps, shot.dur - 1e-6))
    p = out / "text_src.json"
    p.write_text(json.dumps({"shot": Path(shot.path).stem, "fps": fps, "strings": i18n.collected()}, indent=1,
                            ensure_ascii=False), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("--out")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--still", type=float, nargs="+", help="render these times (s) as PNG stills only")
    ap.add_argument("--sheet-only", action="store_true", help="render only the contact-sheet frames")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--lang", default="en", help="on-screen language: en = the source text (default), other codes "
                    "translate through captions.json")
    ap.add_argument("--captions", help="translation table (default: captions.json next to the shot or one level up)")
    ap.add_argument("--collect-text", action="store_true", help="list the source strings drawn -> text_src.json")
    ap.add_argument("--fps", type=float, default=10.0, help="sampling rate for --collect-text")
    ap.add_argument("--part", help=argparse.SUPPRESS)
    a = ap.parse_args()
    script = Path(a.shot).resolve()
    lang = i18n.SOURCE if a.collect_text else a.lang
    i18n.setup(lang, script, a.captions)
    if a.collect_text:
        i18n.start_collect()
    shot = load_shot(script)
    out = Path(a.out).resolve() if a.out else default_out(script)
    out.mkdir(parents=True, exist_ok=True)
    stale = [p.name for p in sorted(out.glob("still_*.png")) if p.stat().st_mtime < script.stat().st_mtime]
    if stale and not a.part:
        print(f"note: {len(stale)} still(s) older than {script.name}, not refreshed: {', '.join(stale[:6])}")
    if a.collect_text:
        p = collect_text(shot, out, a.fps)
        print(f"{len(i18n.collected())} source strings -> {p}")
        return
    lang_args = ["--lang", lang] + (["--captions", str(Path(a.captions).resolve())] if a.captions else [])
    try:
        if a.part:
            i0, i1 = map(int, a.part.split(":"))
            shot.render(out, frames=(i0, i1), skip_checks=True, quiet=True)
            return
        chk = shot.check_to(out)
        print(f"checks: {'OK' if chk['ok'] else 'FAILED'} ({len(chk['items'])} items) -> "
              f"{out / f'check{i18n.suffix()}.json'}")
        if not chk["ok"]:
            for i in chk["items"]:
                if not i["ok"]:
                    print("  FAIL", i)
            sys.exit(2)
        if a.check_only:
            return
        if a.still:
            for t in a.still:
                p = out / f"still_{t:06.2f}{i18n.suffix()}.png"
                Image.fromarray(shot.render_frame(t)).save(p)
                print("still", p)
            return
        if a.sheet_only:
            r = shot.render(out, sheet_only=True, skip_checks=True)
        elif a.jobs > 1:
            r = render_parallel(shot, script, out, a.jobs, lang_args)
        else:
            r = shot.render(out, skip_checks=True)
        print(f"done {r} -> {out}")
    except CheckFailed as e:
        print(e)
        sys.exit(2)


if __name__ == "__main__":
    main()
