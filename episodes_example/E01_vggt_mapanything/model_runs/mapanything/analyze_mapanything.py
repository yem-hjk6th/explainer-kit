"""R2 results vs ground truth: read ma_A/B/C.npz, compare with the scene_e01 ground truth, write ma_metrics.json.
MapAnything output is metric (pts3d = m * X~); the world frame = the camera frame of view 1, the same frame as the scene ground truth, so the "metric error" uses no alignment.
"Shape error": prediction and ground truth are each divided by their own mean distance before comparing (as in MapAnything's evaluation code).
Valid pixels = pixels whose ray hits the scene (363,918 over the three images); prediction and ground truth are paired per pixel.
Usage: python -I -X utf8 analyze_mapanything.py <shots dir> <output dir>
"""
import json
import os
import sys

import numpy as np

shots_dir, run_dir = sys.argv[1:3]
sys.path.insert(0, shots_dir)
import scene_e01 as sc  # noqa: E402

NB = sc.numbers()
s_gt = NB["s"]
cams = sc.cameras()
Kinv = np.linalg.inv(sc.K)
u, v = np.meshgrid(np.arange(sc.W, dtype=float), np.arange(sc.H, dtype=float))
RAY_GT = np.stack([u, v, np.ones_like(u)], -1) @ Kinv.T
RAY_GT /= np.linalg.norm(RAY_GT, axis=-1, keepdims=True)


def rot_angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def bilinear(img, uu, vv):
    x0, y0 = int(np.floor(uu)), int(np.floor(vv))
    fx, fy = uu - x0, vv - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1] + (1 - fx) * fy * img[y0 + 1, x0]
            + fx * fy * img[y0 + 1, x0 + 1])


GT = [sc.full(i) for i in range(3)]
VALID = [np.isfinite(g[1]) for g in GT]
X_GT = np.concatenate([GT[i][2][VALID[i]] for i in range(3)])
out = {"s_gt_m": round(float(s_gt), 4), "n_valid": int(len(X_GT)), "configs": {}}
for name, desc in [("A", "images only"), ("B", "images + intrinsics"), ("C", "images + intrinsics + metric poses")]:
    P = np.load(os.path.join(run_dir, f"ma_{name}.npz"))
    m = float(P["metric_scaling_factor"].ravel()[0])
    Xp = np.concatenate([P["pts3d"][i][VALID[i]] for i in range(3)])
    s_pred = float(np.linalg.norm(Xp, axis=1).mean())
    e_metric = np.linalg.norm(Xp - X_GT, axis=1)
    Xn_p, Xn_g = Xp / s_pred, X_GT / s_gt
    e_shape = np.linalg.norm(Xn_p - Xn_g, axis=1)
    r = np.linalg.norm(Xn_p, axis=1) / np.linalg.norm(Xn_g, axis=1)
    tau = float(np.mean(np.maximum(r, 1 / r) < 1.03))
    ray_err, ray_err_full = [], []
    dr_rel = []
    for i in range(3):
        rp = P["ray_directions"][i]
        rp = rp / np.linalg.norm(rp, axis=-1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip((rp * RAY_GT).sum(-1), -1, 1)))
        ray_err.append(ang[VALID[i]].mean())          # valid pixels (same pixel set as the other metrics)
        ray_err_full.append(ang.mean())               # full 518x518 image, for reference only
        R, t, c = cams[i]
        dr_gt = np.linalg.norm(GT[i][2] - c, axis=-1)
        ok = VALID[i]
        dr_rel.append(np.mean(np.abs(P["depth_along_ray"][i][..., 0][ok] - dr_gt[ok]) / dr_gt[ok]))
    cam_rows = []
    for i in range(3):
        R, t, c = cams[i]
        Tp = P["camera_poses"][i]
        cam_rows.append(dict(cam=i + 1, rot_err_deg=round(rot_angle(Tp[:3, :3], R.T), 3),
                             center_pred_m=Tp[:3, 3].round(4).tolist(), center_gt_m=c.round(4).tolist(),
                             center_err_m=round(float(np.linalg.norm(Tp[:3, 3] - c)), 4),
                             fx_pred=round(float(P["intrinsics"][i][0, 0]), 2), fx_gt=round(float(sc.K[0, 0]), 2)))
    uu, vv = NB["uv"]
    j = sc.EX_CAM
    wr = bilinear(P["ray_directions"][j], uu, vv)
    wd = dict(ray=(wr / np.linalg.norm(wr)).round(4).tolist(), ray_gt=NB["ray"].round(4).tolist(),
              depth_along_ray_m=round(float(bilinear(P["depth_along_ray"][j][..., 0], uu, vv)), 4),
              depth_along_ray_gt_m=round(float(NB["ray_depth"]), 4),
              pts3d_m=bilinear(P["pts3d"][j], uu, vv).round(4).tolist(), gt_m=NB["X_star"].round(4).tolist())
    out["configs"][name] = dict(
        inputs=desc, m=round(m, 4), z_tilde=round(s_pred / m, 4), s_pred_m=round(s_pred, 4),
        metric_scale_rel_err=round(abs(s_pred - s_gt) / s_gt, 4),
        metric_point_err_m=dict(mean=round(float(e_metric.mean()), 4), median=round(float(np.median(e_metric)), 4)),
        shape_err_normalized=dict(mean=round(float(e_shape.mean()), 4), median=round(float(np.median(e_shape)), 4)),
        tau_1p03=round(tau, 4), ray_err_deg=[round(float(x), 3) for x in ray_err],
        ray_err_deg_full_image=[round(float(x), 3) for x in ray_err_full],
        depth_along_ray_abs_rel=[round(float(x), 4) for x in dr_rel], cameras=cam_rows, white_dot=wd)
with open(os.path.join(run_dir, "ma_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps({k: {kk: vv for kk, vv in c.items() if kk not in ("cameras", "white_dot")}
                  for k, c in out["configs"].items()}, ensure_ascii=False, indent=1))
print(json.dumps({k: c["white_dot"] for k, c in out["configs"].items()}, ensure_ascii=False))
print(json.dumps({k: c["cameras"] for k, c in out["configs"].items()}, ensure_ascii=False))
