"""E02 model-run analysis: model output vs. our test scene's ground truth. Writes e02_metrics.json (every measured number the episode shows comes from here).

Ground truth: scene_e01 ray casting, re-cast at pixel centers at each model's output resolution (no image interpolation).
Pixel-center conversion: u_518 = (u_n + 0.5) * 518 / n - 0.5 (n = 504 / 288; consistent with bilinear resizing, align_corners=False).
The alignment is noted next to each number: raw = no alignment; scale = one scalar, median(GT / prediction) over all valid pixels (shared by the 3 images).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shots"))
import e02_numbers as en  # noqa: E402

sc = en.sc
EX = sc.EX_CAM
nb = sc.numbers()
UV = nb["uv"]                                   # sub-pixel coordinates of the white dot on the 518 image


def truth_depth(i, n):
    """z-depth (m) of camera i at the pixel centers of an n×n grid (ray misses = inf)."""
    g = (np.arange(n) + 0.5) * sc.W / n - 0.5
    uu, vv = np.meshgrid(g, g)
    _, tz, _ = sc.render(i, np.stack([uu.ravel(), vv.ravel()], 1))
    return tz.reshape(n, n)


def bilinear(img, u, v):
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    a, b = u - u0, v - v0
    return float((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u0 + 1]
                 + (1 - a) * b * img[v0 + 1, u0] + a * b * img[v0 + 1, u0 + 1])


def to_n(uv, n):
    return (uv + 0.5) * n / sc.W - 0.5


def depth_stats(pred, gt):
    m = np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    p, g = pred[m], gt[m]
    ratio = np.maximum(p / g, g / p)
    return dict(n_px=int(m.sum()), absrel=float(np.mean(np.abs(p - g) / g)), delta1=float(np.mean(ratio < 1.25)))


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


cams = sc.cameras()
out = {"white_uv_518": UV.tolist(), "truth": {"D_white": float(nb["zdepth"]), "fov_deg": sc.FOV_DEG, "f_px_518": sc.FX}}

# ---------------- Depth Pro (518, cam 2) ----------------
dp = np.load(os.path.join(HERE, "dp_cam2.npz"))
gt518 = sc.full(EX)[1]
uv1536 = to_n(UV, 1536)
dpo = dict(fov_deg=float(dp["fov_deg"]), f_px_est=float(dp["fpx_est"]),
           C_white=bilinear(dp["C1536"], *uv1536),
           C_white_truth=float(sc.FX / sc.W / nb["zdepth"]),
           D_white_est=bilinear(dp["depth_est"], *UV), D_white_gtf=bilinear(dp["depth_gtf"], *UV),
           stats_est_raw=depth_stats(dp["depth_est"], gt518), stats_gtf_raw=depth_stats(dp["depth_gtf"], gt518))
m = np.isfinite(gt518)
s_dp = float(np.median(gt518[m] / dp["depth_est"][m]))
dpo["scale_est"] = s_dp
dpo["stats_est_scale"] = depth_stats(dp["depth_est"] * s_dp, gt518)
# the edge in E02-06: row 297, u 341–381, depth profile of GT and prediction (GT focal length given)
row = int(round(UV[1]))
seg = slice(341, 382)
dpo["edge_row"] = row
dpo["edge_truth"] = gt518[row, seg].tolist()
dpo["edge_pred_gtf"] = dp["depth_gtf"][row, seg].tolist()


def edge_width(prof):
    p = np.asarray(prof)
    lo, hi = np.nanmin(p), np.nanmax(p)
    a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    idx = np.where((p > a) & (p < b))[0]
    return int(len(idx))


dpo["edge_width_px_10_90"] = {"truth": edge_width(dpo["edge_truth"]), "pred_gtf": edge_width(dpo["edge_pred_gtf"])}
out["depth_pro"] = dpo

# ---------------- DA3-Large-1.1 (504) ----------------
gt504 = [truth_depth(i, 504) for i in range(3)]
uv504 = to_n(UV, 504)
r504 = 504 / sc.W
K504 = np.array([[sc.FX * r504, 0, (sc.CX + 0.5) * r504 - 0.5], [0, sc.FY * r504, (sc.CY + 0.5) * r504 - 0.5], [0, 0, 1]])
out["truth"]["K_504"] = K504.tolist()
Rt = [R for R, _, _ in cams]
ct = [c for _, _, c in cams]


def pose_errors(ext):
    """ext: (3,3,4) world->camera. Pairwise relative rotation error and relative translation-direction error (degrees); rotation angle of cam 2 relative to cam 1."""
    E = [np.vstack([e, [0, 0, 0, 1]]) for e in ext]
    res = {}
    for i in range(3):
        for j in range(i + 1, 3):
            Rp = E[j][:3, :3] @ E[i][:3, :3].T
            Rg = Rt[j] @ Rt[i].T
            cp = -E[j][:3, :3].T @ E[j][:3, 3] - (-E[i][:3, :3].T @ E[i][:3, 3])
            cg = ct[j] - ct[i]
            ang_t = np.degrees(np.arccos(np.clip(cp @ cg / np.linalg.norm(cp) / np.linalg.norm(cg), -1, 1)))
            res[f"{i + 1}-{j + 1}"] = dict(rot_err_deg=rot_angle(Rp @ Rg.T), trans_dir_err_deg=float(ang_t))
    res["cam2_rot_deg"] = rot_angle(E[1][:3, :3] @ E[0][:3, :3].T)
    return res


da3 = {}
for name in ["A_nopose_camhead", "B_nopose_raypose", "C_gtpose"]:
    z = np.load(os.path.join(HERE, f"da3_{name}.npz"))
    D = z["depth"]
    allp = np.concatenate([D[i][np.isfinite(gt504[i])] for i in range(3)])
    allg = np.concatenate([gt504[i][np.isfinite(gt504[i])] for i in range(3)])
    s = float(np.median(allg / allp))
    r = dict(scale_to_m=s, D_white_raw=bilinear(D[EX], *uv504), D_white_scaled=bilinear(D[EX], *uv504) * s,
             stats_raw=depth_stats(np.concatenate([D[i] for i in range(3)]), np.concatenate(gt504)),
             stats_scale=depth_stats(np.concatenate([D[i] for i in range(3)]) * s, np.concatenate(gt504)),
             f_504=[float(z["intrinsics"][i][0, 0]) for i in range(3)],
             c_504=[z["intrinsics"][i][:2, 2].tolist() for i in range(3)],
             pose=pose_errors(z["extrinsics"]),
             conf_white=bilinear(z["conf"][EX], *uv504))
    da3[name] = r
    if name == "A_nopose_camhead" and "ray" in z:
        ray = z["ray"][0]                                   # (3, 288, 288, 6)
        n = ray.shape[1]
        g = (np.arange(n) + 0.5) * 504 / n - 0.5            # 288-grid pixel centers → 504 pixel coordinates
        uu, vv = np.meshgrid(g, g)
        pix = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3)
        # channel order in the code: first 3 = direction (not normalized), last 3 = origin (camera center); the opposite of the paper's text on p.6
        DIR, ORG = slice(0, 3), slice(3, 6)
        rr = {"shape": list(ray.shape), "channel_order": "dir(0:3), origin(3:6)",
              "origin_mean": [ray[i, ..., ORG].reshape(-1, 3).mean(0).tolist() for i in range(3)],
              "origin_std": [ray[i, ..., ORG].reshape(-1, 3).std(0).tolist() for i in range(3)],
              "truth_center_over_s": [(c / nb["s"]).tolist() for c in ct],
              "dir_z_cam_check": None}
        sel = np.arange(0, n * n, 97)                       # about 855 pixels used for the DLT
        d2 = ray[EX, ..., DIR].reshape(-1, 3)
        Kh, Rh = en.rq(np.linalg.inv(en.dlt_rays(pix[sel], d2[sel])))
        if np.linalg.det(Rh) < 0:
            Rh, Kh = -Rh, -Kh
        Kh = Kh / Kh[2, 2]
        d1 = ray[0, ..., DIR].reshape(-1, 3)
        K1, R1 = en.rq(np.linalg.inv(en.dlt_rays(pix[sel], d1[sel])))
        if np.linalg.det(R1) < 0:
            R1, K1 = -R1, -K1
        rr.update(dlt_n=int(len(sel)), K_hat_cam2_504=Kh.tolist(), R_hat_cam2=Rh.tolist(),
                  cam2_rot_vs_cam1_deg=rot_angle(Rh @ R1.T),
                  rot_err_vs_truth_deg=rot_angle((Rh @ R1.T) @ (Rt[EX] @ Rt[0].T).T),
                  ray_conf_mean=[float(z["ray_conf"][0, i].mean()) for i in range(3)])
        # is the direction "camera-frame z = 1"? Rotate d back to the camera frame with the solved R and check the z component
        dc = d2[sel] @ Rh.T
        rr["dir_z_cam_check"] = dict(mean=float(dc[:, 2].mean()), std=float(dc[:, 2].std()))
        da3[name]["ray"] = rr
out["da3_large"] = da3

# ---------------- DA3-Metric (504, cam 2) ----------------
zm = np.load(os.path.join(HERE, "da3metric_cam2.npz"))
raw = zm["depth"][0]
f504 = float(K504[0, 0])
metric = f504 * raw / 300.0
out["da3_metric"] = dict(f_504_used=f504, D_white_m=bilinear(metric, *uv504), stats_raw=depth_stats(metric, gt504[EX]),
                         note="README: metric = focal * net_output / 300; focal is the GT focal length converted to 504 (the model does not estimate focal length itself)")

with open(os.path.join(HERE, "e02_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
