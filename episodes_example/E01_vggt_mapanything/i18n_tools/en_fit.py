"""How short must a translation be to fit its box: estimate the width of one character from the drawn box and
compute how many characters fit.
  python -I -X utf8 en_fit.py [--lang es] [shot name ...]
Left-anchored (x0 >= 0): available width = canvas width - margin - x0; centered / right-anchored (x0 < 0):
available width = canvas width - 2 x margin. Margin: 30 px on the main canvas, 16 px on a popup's small canvas.
Multi-line strings use their widest line.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("shots", nargs="*")
ap.add_argument("--lang", default="es")
a = ap.parse_args()

EP = Path(__file__).resolve().parents[1]
R = EP / "shots" / "render"
shots = a.shots or sorted(p.name for p in R.iterdir() if p.is_dir() and p.name.startswith("s"))
for s in shots:
    m = R / s / f"meta_{a.lang}.json"
    if not m.exists():
        continue
    off = json.loads(m.read_text(encoding="utf-8")).get("i18n", {}).get("off_canvas", [])
    for o in off:
        x0, y0, x1, y1 = o["box"]
        W = o["canvas"][0]
        pad = 30 if W == 1080 else 16
        avail = (W - 2 * pad) if x0 < 0 else (W - pad - x0)
        lines = o["tr"].split("\n")
        L = max(len(x) for x in lines)
        tgt = int(L * avail / (x1 - x0))
        print(f"{s}\t{'C' if x0 < 0 else 'L'}\tnow {L}\tfit {tgt}\t{o['src']!r}\t{o['tr']!r}")