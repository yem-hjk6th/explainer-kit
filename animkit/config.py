"""Machine paths. Read from config/animkit.json (override with env ANIMKIT_CONFIG); cache dir from ANIMKIT_CACHE
(default .cache/ in the repo root). Create the config with `python tools/setup_config.py` or by copying
config/animkit.example.json and editing it."""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT
_CFG_PATH = Path(os.environ.get("ANIMKIT_CONFIG", ROOT / "config" / "animkit.json"))
if not _CFG_PATH.exists():
    raise FileNotFoundError(f"{_CFG_PATH} not found: run `python tools/setup_config.py` (detects fonts / ffmpeg) or copy "
                            "config/animkit.example.json to config/animkit.json and fill in the font paths")
CFG = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
CACHE = Path(os.environ.get("ANIMKIT_CACHE", ROOT / ".cache"))


def ffmpeg():
    """ffmpeg from config, else PATH, else the conda env of the running interpreter (conda-forge ffmpeg lives in
    Library/bin on Windows, bin/ elsewhere), else the binary bundled with the imageio-ffmpeg package."""
    exe = Path(sys.executable).parent
    for c in (CFG.get("ffmpeg"), shutil.which("ffmpeg"), exe / "Library" / "bin" / "ffmpeg.exe", exe / "ffmpeg"):
        if c and Path(c).exists():
            return str(c)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                       # not installed, or its binary is missing
        pass
    raise FileNotFoundError("ffmpeg not found: set 'ffmpeg' in config/animkit.json, put it on PATH, "
                            "or pip install imageio-ffmpeg")


def ffprobe(ff=None):
    """ffprobe: next to the given ffmpeg binary (ffprobe.exe or ffprobe), else on PATH. The imageio-ffmpeg binary has
    no ffprobe next to it, so it needs ffprobe on PATH or 'ffprobe' in config/animkit.json."""
    ff = Path(ff or ffmpeg())
    for c in (CFG.get("ffprobe"), ff.with_name("ffprobe.exe"), ff.with_name("ffprobe"), shutil.which("ffprobe")):
        if c and Path(c).exists():
            return str(c)
    raise FileNotFoundError(f"ffprobe not found next to {ff} or on PATH: install a full ffmpeg build (it ships ffprobe), "
                            "or set 'ffprobe' in config/animkit.json")
