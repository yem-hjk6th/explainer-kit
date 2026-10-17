"""Real model run: MapAnything on the 3 images of the E01 toy scene (third_party/map-anything @ 3d10cf7, weights HF facebook/map-anything,
CC-BY-NC 4.0). Three input settings:
  A images only; B images + intrinsics (ground-truth K); C images + intrinsics + metric poses (ground-truth cam2world, is_metric_scale=True).
Usage (env mapanything): python -I -X utf8 run_mapanything.py <shots dir> <output dir>
Output: <output dir>/ma_<A|B|C>.npz, run_meta.json. apply_mask / mask_edges are off and the raw dense output is saved; the mask is saved separately.
"""
import json
import os
import sys
import time

import numpy as np

shots_dir, out_dir = sys.argv[1:3]
sys.path.insert(0, shots_dir)
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch  # noqa: E402

import scene_e01 as sc  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import preprocess_inputs  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(0.65)

t0 = time.time()
model = MapAnything.from_pretrained("facebook/map-anything").to(dev).eval()
t_load = time.time() - t0

cams = sc.cameras()
imgs = [torch.from_numpy(sc.full(i)[0].copy()) for i in range(3)]          # (H, W, 3) uint8, [0, 255]
K = torch.tensor(sc.K, dtype=torch.float32)


def cam2world(i):
    R, t, c = cams[i]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = c
    return torch.tensor(T, dtype=torch.float32)


CONFIGS = {
    "A": [{"img": imgs[i]} for i in range(3)],
    "B": [{"img": imgs[i], "intrinsics": K} for i in range(3)],
    "C": [{"img": imgs[i], "intrinsics": K, "camera_poses": cam2world(i), "is_metric_scale": torch.tensor([True])}
          for i in range(3)],
}
KEYS = ["pts3d", "pts3d_cam", "depth_z", "depth_along_ray", "ray_directions", "intrinsics", "camera_poses",
        "cam_trans", "cam_quats", "conf", "mask", "non_ambiguous_mask", "metric_scaling_factor"]
meta = dict(model="facebook/map-anything (CC-BY-NC 4.0)", code="facebookresearch/map-anything 3d10cf7",
            image_hw=[sc.H, sc.W], load_s=round(t_load, 2), gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__, runs={})
for name, views in CONFIGS.items():
    proc = preprocess_inputs(views)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        preds = model.infer(proc, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                            apply_mask=False, mask_edges=False, apply_confidence_mask=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    out = {}
    for k in KEYS:
        if k in preds[0]:                         # there is no "mask" when apply_mask=False
            out[k] = np.stack([p[k][0].float().cpu().numpy() for p in preds])
    np.savez_compressed(os.path.join(out_dir, f"ma_{name}.npz"), **out)
    meta["runs"][name] = dict(inputs=sorted(k for k in views[0] if k != "img"), infer_s=round(dt, 3),
                              peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                              metric_scaling_factor=out["metric_scaling_factor"].ravel().round(4).tolist())
    print(name, meta["runs"][name], flush=True)
meta["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
