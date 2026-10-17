"""Write config/animkit.json for this machine: finds fonts, ffmpeg, and a python with matplotlib.

  python -X utf8 tools/setup_config.py [--math-python PATH] [--ffmpeg PATH] [--force]

Prints what it found and what is missing. Edit the JSON by hand afterwards if you want other fonts.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "animkit.json"

# candidates per font role, first hit wins (file names are matched case-insensitively anywhere under the font dirs)
CANDIDATES = {
    "text": ["segoeui.ttf", "Arial.ttf", "DejaVuSans.ttf"],
    "bold": ["segoeuib.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"],
    "light": ["segoeuil.ttf", "HelveticaNeue.ttc", "DejaVuSans-ExtraLight.ttf", "DejaVuSans.ttf"],
    "math": ["cambriai.ttf", "Times New Roman Italic.ttf", "DejaVuSerif-Italic.ttf"],
    "math_up": ["cambria.ttc", "Times New Roman.ttf", "DejaVuSerif.ttf"],
}


def font_dirs():
    d = []
    if os.name == "nt":
        d += [Path(os.environ.get("WINDIR", "C:" + os.sep + "Windows")) / "Fonts",
              Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"]
    d += [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
          Path.home() / "Library" / "Fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
          Path.home() / ".fonts", Path.home() / ".local" / "share" / "fonts"]
    try:  # matplotlib ships DejaVu fonts: a portable fallback for the Latin / math roles
        import matplotlib
        d.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except ImportError:
        pass
    return [p for p in d if p.is_dir()]


def index_fonts():
    idx = {}
    for d in font_dirs():
        for p in d.rglob("*"):
            if p.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                idx.setdefault(p.name.lower(), p)
    return idx


def has_matplotlib(py):
    try:
        return subprocess.run([py, "-c", "import matplotlib"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_ffmpeg():
    exe = Path(sys.executable).parent
    for c in (shutil.which("ffmpeg"), exe / "Library" / "bin" / "ffmpeg.exe", exe / "ffmpeg"):
        if c and Path(c).exists():
            return str(c)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-python", help="python with matplotlib (default: this interpreter if it has matplotlib)")
    ap.add_argument("--ffmpeg")
    ap.add_argument("--force", action="store_true", help="overwrite an existing config/animkit.json")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        sys.exit(f"{OUT} exists; pass --force to overwrite")
    idx = index_fonts()
    fonts, missing = {}, []
    for role, names in CANDIDATES.items():
        hit = next((idx[n.lower()] for n in names if n.lower() in idx), None)
        if hit:
            fonts[role] = hit.as_posix()
        else:
            missing.append(role)
    math_py = a.math_python
    if math_py is None and not has_matplotlib(sys.executable):
        print("! matplotlib is not importable here: install it, or pass --math-python <python that has it>")
    if math_py and not has_matplotlib(math_py):
        print(f"! {math_py} cannot import matplotlib")
    ff = a.ffmpeg or find_ffmpeg()
    cfg = {"_note": "written by tools/setup_config.py; machine-specific, git-ignored", "math_python": math_py,
           "ffmpeg": ff, "fonts": fonts}
    OUT.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for role in CANDIDATES:
        print(f"  {role:8s} {fonts.get(role, '-- NOT FOUND --')}")
    print(f"  ffmpeg   {ff or '-- NOT FOUND (install ffmpeg or `pip install imageio-ffmpeg`) --'}")
    print(f"  math     {math_py or sys.executable}")
    if missing:
        print(f"! missing font roles {missing}: install DejaVu fonts (or `pip install matplotlib`, which ships them) "
              f"or edit {OUT}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
