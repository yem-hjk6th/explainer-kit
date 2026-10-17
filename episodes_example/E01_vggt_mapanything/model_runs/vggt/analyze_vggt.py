"""R1 results vs ground truth: read vggt/vggt_out.npz, compare with the ground truth from scene_e01, write vggt/vggt_metrics.json.
VGGT output scale is "divided by the mean distance s" (paper Sec. 3.4), so the ground truth is divided by s before comparing; errors after a
Umeyama similarity alignment are given too (as in the paper's ETH3D point-map evaluation). Error unit = normalized units
(1 = s = 2.153 m, the mean distance from scene points to camera 1).
Usage: python -I -X utf8 analyze_vggt.py <shots dir> <vggt output dir>
"""
import json
import os
import sys

import numpy as np

shots_dir, run_dir = sys.argv[1:3]
sys.path.insert(0, shots_dir)
import scene_e01 as sc  # noqa: E402

P = np.load(os.path.join(run_dir, "vggt_out.npz"))
NB = sc.numbers()
s = NB["s"]
cams = sc.cameras()


def rot_angle(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def umeyama(A, B):
    """B ~ c R A + t (similarity transform); returns c, R, t."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    c = np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean()
    return c, R, mb - c * R @ ma


def bilinear(img, u, v):
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - x0, v - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1] + (1 - fx) * fy * img[y0 + 1, x0]
            + fx * fy * img[y0 + 1, x0 + 1])


out = {"units": "normalized: 1 = s = %.4f m (mean distance of GT points to camera 1)" % s}
# cameras
cam_rows = []
for i in range(3):
    R, t, c = cams[i]
    Rp, tp = P["extrinsic"][i][:, :3], P["extrinsic"][i][:, 3]
    fov = np.degrees(P["pose_enc"][i][7:9])
    cam_rows.append(dict(cam=i + 1, rot_err_deg=round(rot_angle(Rp, R), 3),
                         t_pred=tp.round(4).tolist(), t_gt_over_s=(t / s).round(4).tolist(),
                         t_err=round(float(np.linalg.norm(tp - t / s)), 4),
                         t_dir_err_deg=(round(float(np.degrees(np.arccos(np.clip(
                             tp @ t / (np.linalg.norm(tp) * np.linalg.norm(t)), -1, 1)))), 3) if i > 0 else None),
                         fov_pred_deg=fov.round(3).tolist(), fov_gt_deg=[60.0, 60.0],
                         q_pred_xyzw=P["pose_enc"][i][3:7].round(4).tolist()))
out["cameras"] = cam_rows

# depth, point maps
dep_rows, X_all, Pp_all, Pd_all = [], [], [], []
for i in range(3):
    _, tz, X = sc.full(i)
    ok = np.isfinite(tz)
    Dg = tz[ok] / s
    Dp = P["depth"][i][ok]
    dep_rows.append(dict(cam=i + 1, n_valid=int(ok.sum()),
                         abs_rel=round(float(np.mean(np.abs(Dp - Dg) / Dg)), 4),
                         median_ratio_pred_over_gt=round(float(np.median(Dp / Dg)), 4)))
    X_all.append(X[ok] / s)
    Pp_all.append(P["world_points"][i][ok])
    Pd_all.append(P["points_from_depth"][i][ok])
out["depth"] = dep_rows
Xg, Pp, Pd = (np.concatenate(a) for a in (X_all, Pp_all, Pd_all))


def pt_err(Pr):
    e = np.linalg.norm(Pr - Xg, axis=1)
    c, R, t = umeyama(Pr, Xg)
    ea = np.linalg.norm((c * Pr @ R.T + t) - Xg, axis=1)
    return dict(mean=round(float(e.mean()), 4), median=round(float(np.median(e)), 4),
                aligned_mean=round(float(ea.mean()), 4), aligned_median=round(float(np.median(ea)), 4),
                sim3_scale=round(float(c), 4), mean_norm_pred=round(float(np.linalg.norm(Pr, axis=1).mean()), 4))


out["points"] = {"point_head": pt_err(Pp), "depth_plus_camera": pt_err(Pd), "n_points": int(len(Xg)),
                 "mean_norm_gt": round(float(np.linalg.norm(Xg, axis=1).mean()), 4)}

# white dot (camera 2, sub-pixel, bilinear)
u, v = NB["uv"]
i = sc.EX_CAM
wd = dict(uv=[round(float(u), 3), round(float(v), 3)], depth_gt=round(NB["zdepth"] / s, 4),
          depth_pred=round(float(bilinear(P["depth"][i], u, v)), 4),
          point_head=bilinear(P["world_points"][i], u, v).round(4).tolist(),
          depth_cam=bilinear(P["points_from_depth"][i], u, v).round(4).tolist(),
          gt=(NB["X_star"] / s).round(4).tolist(), depth_conf=round(float(bilinear(P["depth_conf"][i], u, v)), 3))
out["white_dot"] = wd

# tracking (query = the white dot's pixel in frame 1)
tr = []
for j in range(3):
    gt = NB["vis"][j][0]
    pr = P["track"][j, 0]
    tr.append(dict(frame=j + 1, gt=gt.round(2).tolist(), pred=pr.round(2).tolist(),
                   err_px=round(float(np.linalg.norm(pr - gt)), 3), vis=round(float(P["vis"][j, 0]), 3)))
out["track_white_dot"] = tr

# confidence: depth error of the high / low half of the pixels
conf_rows = []
for i in range(3):
    _, tz, _ = sc.full(i)
    ok = np.isfinite(tz)
    e = np.abs(P["depth"][i][ok] - tz[ok] / s) / (tz[ok] / s)
    cf = P["depth_conf"][i][ok]
    med = np.median(cf)
    conf_rows.append(dict(cam=i + 1, conf_min=round(float(cf.min()), 3), conf_median=round(float(med), 3),
                          conf_max=round(float(cf.max()), 3),
                          abs_rel_high_conf_half=round(float(e[cf >= med].mean()), 4),
                          abs_rel_low_conf_half=round(float(e[cf < med].mean()), 4)))
out["depth_conf_vs_error"] = conf_rows

with open(os.path.join(run_dir, "vggt_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
