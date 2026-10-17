"""Shared by all E02 shots: read-only import of the E01 tools, E02 numbers, option-R measurements, series-wide corner tags.

E01's `e01_common.py` / `scene_e01.py` are imported read-only (never edited); E02's own things live here.
Shot scripts: `import e02_common as c2`, then `c2.ec.*` (E01 tools), `c2.sc.*` (scene), `c2.NB()` (e02_numbers),
`c2.metrics()` (option-R metrics).
"""
import json
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E01_SHOTS = os.path.normpath(os.path.join(HERE, "..", "..", "E01_vggt_mapanything", "shots"))
if E01_SHOTS not in sys.path:
    sys.path.insert(0, E01_SHOTS)

import e01_common as ec  # noqa: E402
import scene_e01 as sc  # noqa: E402
import e02_numbers as _num  # noqa: E402

RUNS = os.path.normpath(os.path.join(HERE, "..", "model_runs"))

# ---- series-wide corner tags (tag convention) ----
SCENE = ec.SCENE                                   # "our test scene"
SCENE_TAG = ec.SCENE_TAG                           # model output: "Measured on our test scene (our own test, not a paper experiment)"
GT_TAG = "Scene GT (own test, not model)"         # frames computed from the scene GT
FAKE_TAG = "Illustration: predictions built from GT"                    # fake predictions built from GT
CODE_TAG = "Official code"                               # read from the code, not stated in the paper
DERIV_TAG = "Derivation"
VIEW_TAG = "Our take"                                   # explanatory comparison / our reading

# Accent colors of the two methods (title bar, split-screen comparison shots, paper cards): Depth Pro teal, DA3 orange-pink. Use only where they do not clash with the meaning colors of STYLE section C.
DP_COLOR = (110, 220, 200)
DA3_COLOR = (255, 150, 120)

WHITE_PT = (255, 255, 255)
CAM2 = sc.EX_CAM                                   # camera 2 (0-based index 1)
CAM2_COLOR = ec.CAM_COLORS[CAM2]


@lru_cache(maxsize=None)
def NB():
    """`e02_numbers.numbers()`, cached (every number computed for real in a shot comes from here)."""
    return _num.numbers()


@lru_cache(maxsize=None)
def metrics():
    """Measured model metrics (`model_runs/e02_metrics.json`)."""
    with open(os.path.join(RUNS, "e02_metrics.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def run_meta():
    with open(os.path.join(RUNS, "run_meta.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def npz(name):
    """Raw option-R outputs: dp_cam2 / da3_A_nopose_camhead / da3_B_nopose_raypose / da3_C_gtpose / da3metric_cam2."""
    p = os.path.join(RUNS, name + ".npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{os.path.normpath(p)} missing: this shot draws real model output. Run model_runs/run_e02.py first (GPU; weights download from the official Hugging Face repos). See episodes_example/README.md")
    d = np.load(p)
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=None)
def cam2_depth():
    """Ground-truth z-depth map of camera 2 (518×518, background = inf)."""
    return sc.full(CAM2)[1]


def fmt(x, nd=3):
    """On-screen number: the minus sign is U+2212."""
    return f"{x:.{nd}f}".replace("-", "−")


def fmt_vec(v, nd=3):
    return ec.fmt_vec(v, nd)
