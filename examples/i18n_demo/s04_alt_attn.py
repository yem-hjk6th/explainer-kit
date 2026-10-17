"""i18n demo: the unchanged (English) E01-04 script, read from the E01 folder, rendered with this folder's
captions.json, which holds its captions translated into another language (Spanish, "es").
  render_shot.py examples/i18n_demo/s04_alt_attn.py --lang es      -> render/s04_alt_attn/s04_alt_attn_es.mp4
  concat.py examples/i18n_demo --lang es                             -> render/E01-04demo_es.mp4
Stale detection follows this wrapper, not the E01 script: re-render after E01-04 changes.
"""
import importlib.util
import sys
from pathlib import Path

E01_SHOTS = Path(__file__).resolve().parents[2] / "episodes_example" / "E01_vggt_mapanything" / "shots"
sys.path.insert(0, str(E01_SHOTS))
_spec = importlib.util.spec_from_file_location("e01_s04_alt_attn", E01_SHOTS / "s04_alt_attn.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
shot = _mod.shot
