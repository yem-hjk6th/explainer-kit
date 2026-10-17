"""E02 real model runs: Depth Pro / DA3-Large-1.1 / DA3-Metric-Large on our test scene (the 3 518×518 images from E01).

env: the depth-models env (torch 2.11+cu128). One forward pass on the GPU, GPU memory cap 0.65. Writes only the npz files + run_meta.json under model_runs/.
How to run (pin to the CPU cores assigned for the run):
  cmd /c start "" /B /WAIT /BELOWNORMAL /AFFINITY FF <depth-models env python> -X utf8 -B run_e02.py
Input images = E01 model_runs/vggt/inputs/cam{1,2,3}.png (E01 rendered them with the same scene_e01; read-only).
"""
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
E01 = os.path.normpath(os.path.join(HERE, "..", "..", "E01_vggt_mapanything"))
sys.path.insert(0, os.path.join(E01, "shots"))
import scene_e01 as sc  # noqa: E402

TP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "third_party"))
DEV = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.65)
IMGS = [os.path.join(E01, "model_runs", "vggt", "inputs", f"cam{i}.png") for i in (1, 2, 3)]
EX = sc.EX_CAM
meta = {"inputs": IMGS, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "runs": {}}


def commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def tic(name):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    return name, time.time()


def toc(tk, **extra):
    torch.cuda.synchronize()
    meta["runs"][tk[0]] = dict(sec=round(time.time() - tk[1], 3),
                               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), **extra)


# ---------------- Depth Pro: one image from cam 2 ----------------
import depth_pro  # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

ck = hf_hub_download("apple/DepthPro", "depth_pro.pt")
cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ck)
dp, tf = depth_pro.create_model_and_transforms(cfg, device=DEV, precision=torch.float32)
dp.eval()
x = tf(Image.open(IMGS[EX]).convert("RGB"))                       # (3, 518, 518), normalized to [-1, 1]
with torch.no_grad():
    tk = tic("depth_pro")
    x1536 = F.interpolate(x[None], size=(1536, 1536), mode="bilinear", align_corners=False)
    C1536, fov_deg = dp.forward(x1536)                             # canonical inverse depth C (1536²), horizontal FOV (degrees)
    est = dp.infer(x)                                              # with its own estimated focal length
    gtf = dp.infer(x, f_px=torch.tensor(float(sc.FX), device=DEV))  # GT focal length 448.6 given
    toc(tk, checkpoint=ck)
np.savez_compressed(os.path.join(HERE, "dp_cam2.npz"),
                    C1536=C1536.squeeze().float().cpu().numpy(), fov_deg=float(fov_deg.squeeze()),
                    depth_est=est["depth"].float().cpu().numpy(), fpx_est=float(est["focallength_px"]),
                    depth_gtf=gtf["depth"].float().cpu().numpy(), fpx_gt=float(sc.FX))
del dp
torch.cuda.empty_cache()

# ---------------- DA3-Large-1.1: 3 images ----------------
from depth_anything_3.api import DepthAnything3  # noqa: E402

da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(DEV).eval()
net = da3.model
stash = {}
_orig = net._process_depth_head


def _keep_ray(feats, H, W):
    out = _orig(feats, H, W)                                       # the ray map is dropped inside the camera-head branch, so keep a copy first
    for k in ("ray", "ray_conf"):
        if k in out:
            stash[k] = out[k].detach().float().cpu().numpy()
    return out


net._process_depth_head = _keep_ray


def pred_dict(p):
    return {k: v for k, v in vars(p).items() if isinstance(v, np.ndarray)}


cams = sc.cameras()
w2c = np.stack([np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]) for R, t, _ in cams]).astype(np.float32)
Ks = np.repeat(sc.K[None].astype(np.float32), 3, 0)
runs = {
    "A_nopose_camhead": dict(),
    "B_nopose_raypose": dict(use_ray_pose=True),
    "C_gtpose": dict(extrinsics=w2c, intrinsics=Ks),
}
for name, kw in runs.items():
    stash.clear()
    with torch.no_grad():
        tk = tic("da3_" + name)
        p = da3.inference(IMGS, process_res=504, ref_view_strategy="first", **kw)
        toc(tk)
    out = pred_dict(p)
    out.update({k: v for k, v in stash.items()})
    np.savez_compressed(os.path.join(HERE, f"da3_{name}.npz"), **out)
    meta["runs"]["da3_" + name]["arrays"] = {k: list(v.shape) for k, v in out.items()}
meta["da3_snapshot"] = snapshot_download("depth-anything/DA3-LARGE-1.1")
del da3, net
torch.cuda.empty_cache()

# ---------------- DA3-Metric-Large: one image from cam 2 ----------------
dm = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(DEV).eval()
with torch.no_grad():
    tk = tic("da3metric")
    pm = dm.inference([IMGS[EX]], process_res=504)
    toc(tk)
np.savez_compressed(os.path.join(HERE, "da3metric_cam2.npz"), **pred_dict(pm))
meta["runs"]["da3metric"]["note"] = "README: metric = focal_px * net_output / 300 (focal taken at the processing resolution); the conversion is done in the analysis script"
meta["da3metric_snapshot"] = snapshot_download("depth-anything/DA3METRIC-LARGE")
meta["commits"] = {"ml-depth-pro": commit(os.path.join(TP, "ml-depth-pro")),
                   "Depth-Anything-3": commit(os.path.join(TP, "Depth-Anything-3"))}
with open(os.path.join(HERE, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(json.dumps(meta["runs"], indent=1))
