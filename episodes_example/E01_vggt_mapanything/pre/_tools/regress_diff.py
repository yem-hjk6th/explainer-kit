"""Pixel-exact regression of the source-language render: pre/_regress/before/<shot>/still_*.png vs the same-named files in after/.
  python -I -X utf8 regress_diff.py <pre dir> [after dir name, default after]"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

reg = Path(sys.argv[1]).resolve() / "_regress"
after = sys.argv[2] if len(sys.argv) > 2 else "after"
bad = 0
for b in sorted((reg / "before").rglob("still_*.png")):
    a = reg / after / b.relative_to(reg / "before")
    if not a.exists():
        print("MISSING", a)
        bad += 1
        continue
    x, y = np.asarray(Image.open(b)), np.asarray(Image.open(a))
    n = int((x != y).any(-1).sum()) if x.shape == y.shape else -1
    bad += n != 0
    print(f"{b.parent.name}/{b.name}: differing pixels = {n}")
print("ALL ZERO" if bad == 0 else f"{bad} file(s) differ")
