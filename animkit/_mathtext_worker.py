"""Runs under the env that has matplotlib: renders mathtext strings to alpha masks.
  python -I -X utf8 _mathtext_worker.py jobs.json out_dir
jobs.json = [{"key", "tex", "size", "fontset"}]; writes <key>.png (L mode, 255 = ink) + <key>.json, and errors.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from PIL import Image

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
parser = MathTextParser("agg")
errors = {}
for j in jobs:
    try:
        r = parser.parse(f"${j['tex']}$", dpi=72, prop=FontProperties(size=j["size"], math_fontfamily=j["fontset"]))
    except Exception as e:  # noqa: BLE001 - report every bad formula back to the caller
        errors[j["key"]] = f"{j['tex']!r}: {e}"
        continue
    im = np.asarray(r.image)
    tmp = out / f"{j['key']}.tmp.png"
    Image.fromarray(im, "L").save(tmp)
    os.replace(tmp, out / f"{j['key']}.png")
    meta = dict(tex=j["tex"], size=j["size"], fontset=j["fontset"], width=float(r.width), height=float(r.height),
                depth=float(r.depth))
    (out / f"{j['key']}.json").write_text(json.dumps(meta), encoding="utf-8")
(out / "errors.json").write_text(json.dumps(errors, ensure_ascii=False), encoding="utf-8")
