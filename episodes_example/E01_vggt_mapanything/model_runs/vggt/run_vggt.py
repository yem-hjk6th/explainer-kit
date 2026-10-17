"""Real model run: VGGT-1B on the 3 images of the E01 toy scene (official code third_party/vggt @ a288dd0, weights from the HF cache facebook/VGGT-1B).

Usage (env vggt):
  python -I -X utf8 run_vggt.py <vggt repo dir> <shots dir> <output dir>
Output: <output dir>/vggt_out.npz (raw predictions + points unprojected from depth + camera), inputs/cam{1,2,3}.png, run_meta.json.
GPU share is capped: memory_fraction 0.65.
"""
import json
import os
import sys
import time

import numpy as np

vggt_dir, shots_dir, out_dir = sys.argv[1:4]
sys.path.insert(0, vggt_dir)
sys.path.insert(0, shots_dir)
# after the first download you can set HF_HUB_OFFLINE=1 to skip update checks

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import scene_e01 as sc  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

os.makedirs(os.path.join(out_dir, "inputs"), exist_ok=True)
paths = []
for i in range(3):
    p = os.path.join(out_dir, "inputs", f"cam{i + 1}.png")
    Image.fromarray(sc.full(i)[0]).save(p)
    paths.append(p)

dev = "cuda"
torch.cuda.set_per_process_memory_fraction(0.65)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

t0 = time.time()
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
t_load = time.time() - t0

images = load_and_preprocess_images(paths).to(dev)            # (3, 3, 518, 518), [0, 1]
assert tuple(images.shape) == (3, 3, sc.H, sc.W), images.shape
# the white dot's pixel in frame 1 is the tracking query point (the official tracker's query frame is the first frame)
q_uv = torch.tensor([[float(v) for v in sc.project(*sc.cameras()[0][:2], sc.X_STAR)[0]]], device=dev)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images, query_points=q_uv)
torch.cuda.synchronize()
t_fwd = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2 ** 30

extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
depth = pred["depth"][0].float().cpu().numpy()                # (3, H, W, 1)
extr_np, intr_np = extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
pts_dc = unproject_depth_map_to_point_map(depth, extr_np, intr_np)   # (3, H, W, 3)

np.savez_compressed(
    os.path.join(out_dir, "vggt_out.npz"),
    pose_enc=pred["pose_enc"][0].float().cpu().numpy(),         # (3, 9) = [T(3), quat xyzw(4), fov_h, fov_w]
    extrinsic=extr_np, intrinsic=intr_np,
    depth=depth[..., 0], depth_conf=pred["depth_conf"][0].float().cpu().numpy(),
    world_points=pred["world_points"][0].float().cpu().numpy(),
    world_points_conf=pred["world_points_conf"][0].float().cpu().numpy(),
    points_from_depth=pts_dc.astype(np.float32),
    track=pred["track"][0].float().cpu().numpy(), vis=pred["vis"][0].float().cpu().numpy(),
    conf=pred["conf"][0].float().cpu().numpy(), query_uv=q_uv.cpu().numpy())
meta = dict(model="facebook/VGGT-1B", code="facebookresearch/vggt a288dd0", dtype=str(dtype), n_frames=3,
            image_hw=[sc.H, sc.W], load_s=round(t_load, 2), forward_s=round(t_fwd, 3), peak_alloc_gib=round(peak, 3),
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            time=time.strftime("%Y-%m-%d %H:%M:%S"))
with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False))
