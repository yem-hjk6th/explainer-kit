"""Join an episode's rendered shots in episode.json order.
  python -I -X utf8 concat.py <episode_dir> [--out FILE] [--allow-stale] [--lang CODE]
Writes <episode_dir>/render/<episode id>.mp4 (--lang es: <id>_es.mp4 from the shots' *_es.mp4) by stream copy
(all shots come from the same encoder settings).
Stops if a shot is missing, failed its checks, or its script changed since it was rendered (unless --allow-stale);
with --lang CODE also if a shot has untranslated text or its translations in captions.json changed since it was rendered.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]          # repo root: animkit/, examples/
sys.path.insert(0, str(LIB))

from animkit import i18n  # noqa: E402
from animkit.config import ffmpeg  # noqa: E402
from animkit.timeline import default_out  # noqa: E402


def shot_status(ep_dir, s, lang="en"):
    """(mp4 path, meta dict or None, list of problems) for one episode.json shot entry."""
    script = (ep_dir / s["script"]).resolve()
    out = default_out(script)
    sfx = "" if lang == "en" else f"_{lang}"
    mp4, meta_p, chk_p = out / f"{out.name}{sfx}.mp4", out / f"meta{sfx}.json", out / f"check{sfx}.json"
    probs = []
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else None
    if not mp4.exists() or meta is None:
        probs.append("not rendered")
    elif script.exists() and meta.get("script_sha1") != hashlib.sha1(script.read_bytes()).hexdigest()[:12]:
        probs.append("script changed since render")
    if not chk_p.exists():
        probs.append("no check report")
    elif not json.loads(chk_p.read_text(encoding="utf-8"))["ok"]:
        probs.append("checks failed")
    if meta and lang != "en":
        rep = meta.get("i18n", {})
        if rep.get("untranslated"):
            probs.append(f"{len(rep['untranslated'])} untranslated string(s)")
        now = i18n.load_table(rep.get("table"), lang, script.stem)
        if any(now.get(k) != v for k, v in rep.get("used", {}).items()):
            probs.append("translations changed since render")
    return mp4, meta, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--out")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--lang", default="en", help="en = source text; other codes = a translated render")
    a = ap.parse_args()
    ep_dir = Path(a.episode_dir).resolve()
    ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    sfx = "" if a.lang == "en" else f"_{a.lang}"
    files, total, bad = [], 0.0, []
    for s in ep["shots"]:
        mp4, meta, probs = shot_status(ep_dir, s, a.lang)
        if a.allow_stale:
            probs = [p for p in probs if p not in ("script changed since render", "translations changed since render")]
        if probs:
            bad.append(f"{s['id']}: {', '.join(probs)}")
            continue
        files.append(mp4)
        total += meta["dur"]
    if bad:
        print("cannot concat:\n  " + "\n  ".join(bad))
        sys.exit(2)
    rd = ep_dir / "render"
    rd.mkdir(exist_ok=True)
    lst = rd / f"_concat_list{sfx}.txt"
    lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in files), encoding="utf-8")
    out = Path(a.out) if a.out else rd / f"{ep.get('id', ep_dir.name)}{sfx}.mp4"
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    print(f"{len(files)} shots, {total:.1f} s -> {out}")


if __name__ == "__main__":
    main()
