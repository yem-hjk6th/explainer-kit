"""Self-check of the depth-models env: both repos import and CUDA is available. No weights are loaded and no forward pass is run."""
import importlib
import sys

import torch

print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
for mod in ["depth_pro", "depth_anything_3.api"]:
    try:
        importlib.import_module(mod)
        print("OK  ", mod)
    except Exception as e:  # noqa: BLE001
        print("FAIL", mod, type(e).__name__, e)
