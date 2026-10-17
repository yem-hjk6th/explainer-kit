"""E01-05 Output heads: a camera as 9 numbers + per-pixel depth (M, 73.0 s). Storyboard storyboard.md section E01-05.
The 9 numbers are joined end to end in code order [t, q, f] (3 + 4 + 2 = 9, not a product);
popup 1: what the quaternion q is and why use it (general reasons); popup 2: how the DPT head turns tokens into per-pixel output (VGGT's layer indices, sizes, channels).
Computed / measured: output slots = VGGT-1B's real output on camera 2 + scene ground truth (t/s, radians); q's axis / angle are computed from q;
DPT sizes are computed from the official code's conv parameters, channel counts are read from the weight header; the depth map and confidence map = real VGGT output.
Illustration: the camera token's movement, how the 24-round stack is drawn, how tokens and feature maps are drawn in the DPT popup (counts and side lengths not to scale).
"""
import glob
import json
import os
import struct

import numpy as np

import e01_common as ec
import scene_e01 as sc
from e01_common import Popup, popup_check
from animkit import (CYAN, GOLD, ORANGE, WHITE, Formula, ImagePanel, OrbitCam, Shot, arrow, axes3d, bump, colorbar,
                     colorize, fade, grid_square, lerp, matrix, module_box, number, ss, style, table, token, vector)

DUR = 73.0
shot = Shot("s05_heads", DUR)
shot.title("Output Heads", "Camera head · DPT head · VGGT §3.3", 0.2, 2.6)
ec.scene_tag(shot, 2.8, 46.6, "VGGT-1B output")
ec.scene_tag(shot, 67.6, DUR, "Depth output by VGGT-1B")

NB = ec.nb()
V = ec.vggt_out()
CAM = sc.EX_CAM
S = NB["s"]
PE = V["pose_enc"][CAM]                              # code order: [t(3), q xyzw(4), fov_h, fov_w]
T_P, Q_P, FOV_P = PE[:3], PE[3:7], PE[7:9]
R_GT, T_GT, _ = sc.cameras()[CAM]
Q_GT = NB["q_xyzw"]
FOV_GT = np.radians([60.0, 60.0])
R_P = ec.quat_to_R(Q_P)                              # quat_to_R normalizes first (the predicted |q| = 0.99968)
ANG_P = float(np.degrees(np.arccos(np.clip((np.trace(R_P) - 1) / 2, -1, 1))))   # angle computed from R = 30.9°
FX_GT = (sc.W / 2) / np.tan(FOV_GT[1] / 2)
FX_P = (sc.W / 2) / np.tan(FOV_P[1] / 2)
FY_P = (sc.H / 2) / np.tan(FOV_P[0] / 2)
DEPTH_P = V["depth"][CAM]
CONF_P = V["depth_conf"][CAM]
UV = NB["uv"]
D_W = float(ec.vggt_metrics()["white_dot"]["depth_pred"])
D_W_GT = NB["zdepth"] / S
LAYERS = [4, 11, 17, 23]
SIDES = [37 * 4, 37 * 2, 37, (37 + 2 - 3) // 2 + 1]           # transposed conv k4s4, k2s2, identity, conv k3s2p1
FUSE = [SIDES[3], SIDES[2], SIDES[1], SIDES[0], SIDES[0] * 2]   # fusion chain: 19 -> 37 -> 74 -> 148 -> x2 = 296


def rot(u, th):
    """Rotation matrix for a rotation by th (radians) about the unit axis u (Rodrigues)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    Kx = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def axis_angle(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    q = q if q[3] >= 0 else -q
    th = 2 * np.arccos(np.clip(q[3], -1, 1))
    return q[:3] / np.sin(th / 2), th


def f3(v):
    return "0" if abs(v) < 5e-4 else ("{" + f"{v:.3f}" + "}")      # braces: typeset the minus sign as a unary minus


U_GT, TH_GT = axis_angle(Q_GT)
U_P, TH_P = axis_angle(Q_P)
NORM_P = float(np.linalg.norm(Q_P))
TILT = float(np.degrees(np.arccos(-U_GT[1])))       # angle between the rotation axis and "straight up" (−y)

# ---------------- Captions (all here) ----------------
T_Q, T_D = 20.2, 51.2                                # times the two popups appear
CAPS = [(0.3, 2.8, "Take the camera token after 24 rounds"), (2.8, 5.4, "Camera head reads out 9 numbers"),
        (5.4, 7.8, "3 for translation, 4 for rotation, 2 for FoV"), (7.8, 10.2, "Concatenated end to end: 3+4+2=9"),
        (10.2, 12.6, "Not 4×3×2=24, but concatenation"), (12.6, 15.0, "Paper writes [q,t,f]; code stores [t,q,f]"),
        (15.0, 17.6, "First 3 (t): camera translation"), (17.6, 20.2, "Middle 4 (q): rotation, as a quaternion"),
        # popup 1: quaternion
        (20.2, 22.8, "Quaternion: 4 numbers for one rotation"), (22.8, 25.4, "u is the axis, θ is the angle"),
        (25.4, 28.2, f"Camera 2 ground truth: rotate {np.degrees(TH_GT):.0f}° about u"),
        (28.2, 30.6, f"Measured q from VGGT: rotates {ANG_P:.1f}°"), (30.6, 33.2, "Why use it? Here is the general reasoning"),
        (33.2, 35.8, "Matrix: 9 numbers, must also be orthogonal"), (35.8, 38.4, "Euler angles: 3 numbers, but gimbal lock"),
        (38.4, 41.2, "A quaternion only needs length 1"),
        (41.2, 43.8, "Last 2 (f): field of view, in radians"), (43.8, 46.6, f"FoV 60° → focal length {FX_GT:.1f}"),
        (46.6, 48.8, "Patch tokens go to the DPT head"), (48.8, 51.2, "Take the output of 4 of the 24 rounds"),
        # popup 2: DPT
        (51.2, 53.8, "DPT: turn tokens back into a per-pixel map"), (53.8, 56.4, "1369 tokens back into 37×37"),
        (56.4, 59.2, "4 rounds become " + ", ".join(str(s) for s in SIDES)),
        (59.2, 62.0, f"{FUSE[0]} interpolated to {FUSE[1]}, add round 17's"),
        (62.0, 64.6, f"Then {FUSE[2]}, {FUSE[3]}, and on to {FUSE[4]}"),
        (64.6, 67.6, f"Conv, interpolate to {sc.W}, output 2 numbers"),
        (67.6, 70.2, "Camera 2: depth output by VGGT"), (70.2, DUR, "The second channel is confidence Σ")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

# ---------------- Output slots: code order [t, q, f] ----------------
SLOT_VALS_P = list(PE)
SLOT_VALS_G = list(T_GT / S) + list(Q_GT) + list(FOV_GT)
GROUPS = [(0, 3, "$t$ (3 numbers): translation", ORANGE, "No. 1–3"), (3, 7, "$q$ (4 numbers): rotation", GOLD, "No. 4–7"),
          (7, 9, "$f$ (2 numbers): FoV", CYAN, "No. 8–9")]
SLOT_W, SLOT_GAP, SEP0, SEP1 = 84, 6, 56, 6
Y_BAR = 820
FOCUS = [(15.0, 17.6), (17.6, 41.2), (41.2, 46.6)]       # time span in which each group is lit on its own


def slot_xs(sep):
    widths = [(b - a) * SLOT_W + (b - a - 1) * SLOT_GAP for a, b, *_ in GROUPS]
    x = 540 - (sum(widths) + 2 * sep) / 2
    xs = []
    for (a, b, *_), w in zip(GROUPS, widths):
        xs += [x + k * (SLOT_W + SLOT_GAP) for k in range(b - a)]
        x += w + sep
    return xs


_XS1 = slot_xs(SEP1)
QX = (_XS1[3] + _XS1[6] + SLOT_W) / 2                   # center of the q segment after concatenation (the popup's pointer tip aims here)

CONCAT = Formula([("g=[", "g0"), ("t", "t"), (",", "c1"), ("q", "q"), (",", "c2"), ("f", "f"), ("]", "g1")], size=44,
                 gap=0.08)
SUM = Formula([("3", "n3"), ("+", "p1"), ("4", "n4"), ("+", "p2"), ("2", "n2"), ("=9", "eq")], size=44, gap=0.12)
FX = Formula([(r"f_x=\frac{W/2}{\tan(f_w/2)}", "def"), (r"=\frac{259}{\tan 30^{\circ}}", "num"),
              (f"={FX_GT:.1f}", "res")], size=50)

# ---------------- Popup 1: quaternion ----------------
POP_Q = Popup((60, 290, 960, 480), t_in=T_Q, t_out=41.0, title="Quaternion $q$: 4 numbers for one 3D rotation", corner="bl",
              accent=GOLD)
QF = Formula([("q=(", "a"), (r"\sin\frac{\theta}{2}\,\mathbf{u}", "xyz"), (",\\ ", "c"), (r"\cos\frac{\theta}{2}", "w"),
              (")", "b")], size=44)
TH_TEX = (rf"\cos\frac{{\theta}}{{2}}=w={Q_GT[3]:.3f}\ \Rightarrow\ \theta={np.degrees(TH_GT):.1f}^{{\circ}}")
U_TEX = (rf"\mathbf{{u}}=(x,y,z)\,/\,\sin{np.degrees(TH_GT) / 2:.0f}^{{\circ}}"
         rf"=({f3(U_GT[0])},\,{f3(U_GT[1])},\,{f3(U_GT[2])})")
QV = OrbitCam(np.zeros(3), 6.0, az=30, el=38, world="cv")
QV.fit(np.r_[np.eye(3), -0.25 * np.eye(3), [1.6 * U_GT, -0.9 * U_GT], (rot(U_GT, TH_GT) @ np.eye(3)).T], 280, 250,
       center=(765, 206), az=42)
Q_TABLE = [[r"$3\times3$ matrix", "9", "$R^{\\mathsf{T}}R=I$ (6 equations)\n$\\det R=1$", "The 9 numbers a network outputs\ngenerally violate these"],
           ["Euler angles", "3", "None", "Gimbal lock: at ±90° (middle angle) the other\ntwo axes align: one DoF lost (yaw-pitch-roll)"],
           ["Quaternion $q$", "4", "$|q|=1$", "Divide by length to satisfy\n$q$ and $-q$ are the same rotation"]]

# ---------------- Popup 2: DPT ----------------
POP_D = Popup((290, 590, 760, 652), t_in=T_D, t_out=67.4, title="What is the DPT head", corner="tl", point_to=(430, 521),
              tag="Illustration", accent=CYAN)     # height 652, figure moved down 24 so column titles are not covered by the top edge of the blocks
DY = 24
DP_ROWS = [(4, SIDES[0], 100, 160 + DY), (11, SIDES[1], 66, 272 + DY), (17, SIDES[2], 44, 358 + DY),
           (23, SIDES[3], 30, 430 + DY)]
DP_OPS = ["×4", "×2", "×1", "÷2"]
XC1, XC2, XC3, XC4, XC5 = 86, 242, 355, 505, 645
D_STEPS = [(0.0, "Input: tokens from rounds 4, 11, 17, 23, each $1369\\times2048$\n"
                 "(one round = frame + global layer, outputs concatenated; 0-based rounds,\n"
                 "matching layers 5, 12, 18, 24 of ViT-L in the DPT paper, which is 1-based)\n"
                 f"Output: per-pixel values; tokens are only $37\\times37$ ({sc.W} ÷ 14), so upsample"),
           (2.6, "1. Reassemble: $1369=37\\times37$, tokens return to the grid by patch position\n$1369\\times2048\\ \\rightarrow\\ 2048\\times37\\times37$"),
           (5.2, "2. Resize: 1×1 conv to cut channels (256 / 512 / 1024 / 1024),\nthen transposed conv ×4, ×2, unchanged, stride-2 conv ÷2"),
           (8.0, "3. Fuse: 3×3 conv maps each level to 256 channels (adding needs equal ch.);\n"
                 f"the coarsest ({FUSE[0]}) is bilinearly upsampled to {FUSE[1]}, then added to round 17's"),
           (10.8, "4. Each level: “add → interpolate to the next size”:\n"
                  f"{FUSE[1]} → {FUSE[2]} → {FUSE[3]}, last level ×2 to {FUSE[4]}"),
           (13.4, f"5. At {FUSE[4]}: 3×3 conv 256→128 → interpolate to {sc.W} → conv 128→32\n"
                  "→ 1×1 conv to 2 channels: depth, confidence\n"
                  "(point head has the same structure, ending in 4 channels: $x,y,z$ + confidence)")]

_VALID = np.isfinite(sc.full(CAM)[1])                     # pixels hit by rays; background (outside the scene) is not drawn
D_LO, D_HI = float(DEPTH_P[_VALID].min()), float(DEPTH_P[_VALID].max())
DEPTH_IMG = colorize(np.where(_VALID, DEPTH_P, np.nan), D_LO, D_HI)   # near bright, far dark (CMAP_DEPTH low value = bright yellow; consistent across 01/05/07/08/09)
# Confidence uses the same colors as shot 07 (log Σ + CONF_STOPS, color scale = 2%-98% percentiles of log Σ over the three images); bright = high confidence
CONF_STOPS = [(0.0, (30, 45, 100)), (0.5, (90, 220, 230)), (1.0, (250, 250, 250))]
_OK3 = [np.isfinite(sc.full(i)[1]) for i in range(3)]
LC_LO, LC_HI = (float(v) for v in np.percentile(np.log(np.concatenate([V["depth_conf"][i][_OK3[i]] for i in range(3)])),
                                                [2, 98]))
CONF_IMG = colorize(np.where(_VALID, np.log(CONF_P), np.nan), LC_LO, LC_HI, stops=CONF_STOPS)
CONF_RAMP = colorize(np.tile(np.linspace(0, 1, 160), (14, 1)), 0.0, 1.0, stops=CONF_STOPS)   # legend: low on the left, high on the right


def _weight_shapes():
    """JSON header of the VGGT-1B safetensors (shapes only); returns None if this machine has no weights."""
    ps = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.safetensors"))
    if not ps:
        return None
    with open(ps[0], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"}


@shot.check
def check(ck):
    ck.close("q_gt recovers the ground-truth rotation", ec.quat_to_R(Q_GT), R_GT, 1e-9)
    ck.close("axis-angle: rotate by θ about u = R(q_gt)", rot(U_GT, TH_GT), R_GT, 1e-9)
    ck.close("ground-truth rotation angle 30.0°", np.degrees(TH_GT), 30.0, 1e-6)
    ck.close("ground-truth rotation axis (0, −0.966, −0.259)", U_GT, [0, -0.966, -0.259], 1e-3)
    ck.close("axis is 15° from vertical", TILT, 15.0, 1e-6)
    ck.close("camera 2's own orientation = rotate by −θ about u (R is world→camera)", R_GT.T, rot(U_GT, -TH_GT), 1e-9)
    ck.close("|q_pred| = 0.9997", NORM_P, 0.9997, 1e-4)
    ck.close("angle of q_pred's axis-angle = angle computed from R", np.degrees(TH_P), ANG_P, 1e-4)
    ck.close("on-screen angle (30.87°)", ANG_P, 30.87, 0.01)
    ck.close("q_pred rotation axis (−0.005, −0.967, −0.254)", U_P, [-0.005, -0.967, -0.254], 2e-3)
    ck.close("R(q_pred) is orthogonal", R_P @ R_P.T, np.eye(3), 1e-3)
    ck.close("R(q_pred) agrees with extrinsic", R_P, V["extrinsic"][CAM][:, :3], 2e-3)
    ck.close("code order: pose_enc[0:3] = extrinsic translation t", T_P, V["extrinsic"][CAM][:, 3], 1e-4)
    ck.close("code order: pose_enc[8] = fov_w → f_x", FX_P, V["intrinsic"][CAM][0, 0], 0.05)
    ck.close("code order: pose_enc[7] = fov_h → f_y", FY_P, V["intrinsic"][CAM][1, 1], 0.05)
    ck.eq("9 numbers = 3 + 4 + 2 (concatenation)", [len(PE), len(T_P) + len(Q_P) + len(FOV_P)], [9, 9])
    ck.close("f_x ground truth = 259 / tan 30°", FX_GT, 448.601, 1e-3)
    ck.eq("DPT's four scales (k4s4, k2s2, identity, k3s2p1)", SIDES, [148, 74, 37, 19])
    ck.eq("fusion chain 19→37→74→148→296; interpolate to 518; 37×37 = 1369", FUSE + [37 * 14, 37 * 37], [19, 37, 74, 148, 296, 518, 1369])
    ck.eq("depth map shape", list(DEPTH_P.shape), [sc.H, sc.W])
    lum = CONF_IMG[_VALID].astype(float) @ np.array([0.299, 0.587, 0.114])
    lc = np.log(CONF_P[_VALID])
    ck.true("confidence map: pixels with the top 10% of log Σ are brighter than the bottom 10% (bright = high confidence, same as 07)",
            lum[lc >= np.percentile(lc, 90)].mean() > lum[lc <= np.percentile(lc, 10)].mean() + 50)
    ck.true("legend ramp is dark on the left, bright on the right", float(CONF_RAMP[0, -1].astype(float) @ [0.299, 0.587, 0.114])
            > float(CONF_RAMP[0, 0].astype(float) @ [0.299, 0.587, 0.114]) + 150)
    w = _weight_shapes()
    ck.true("weight header readable (local VGGT-1B)", w is not None)
    if w is not None:
        ck.eq("depth DPT outputs 2 channels, point DPT 4 channels",
              [w["depth_head.scratch.output_conv2.2.weight"][0], w["point_head.scratch.output_conv2.2.weight"][0]], [2, 4])
        ck.eq("camera head MLP 2048→1024→9", [w["camera_head.pose_branch.fc1.weight"], w["camera_head.pose_branch.fc2.weight"]],
              [[1024, 2048], [9, 1024]])
        ck.eq("DPT 1×1 conv 2048 → 256/512/1024/1024", [w[f"depth_head.projects.{k}.weight"][:2] for k in range(4)],
              [[256, 2048], [512, 2048], [1024, 2048], [1024, 2048]])
        ck.eq("output_conv1 256→128, output_conv2.0 128→32",
              [w["depth_head.scratch.output_conv1.weight"][:2], w["depth_head.scratch.output_conv2.0.weight"][:2]],
              [[128, 256], [32, 128]])
    popup_check(ck, [POP_Q, POP_D], DUR)
    ck.record(pose_enc_cam2=PE.round(4), fov_pred_deg=np.degrees(FOV_P).round(2), fx_pred=round(float(FX_P), 1),
              axis_gt=U_GT.round(4), axis_pred=U_P.round(4), white_depth_pred=D_W, white_depth_gt=round(D_W_GT, 4),
              depth_conf_log_range=[LC_LO, LC_HI])


def lit_levels(t):
    base = 0.55 + 0.15 * ss(t, 7.9, 8.3)
    F = [ss(t, a, a + 0.3) * (1 - ss(t, b, b + 0.3)) for a, b in FOCUS]
    anyf = max(F)
    out = []
    for g in range(3):
        b = min(1.0, base + 0.45 * bump(t, 5.5 + 0.8 * g, 0.8))
        foc = 0.3 + 0.7 * F[g] / anyf if anyf > 1e-6 else b
        out.append((1 - anyf) * b + anyf * foc)
    return out


def slots(cv, y, alpha, lit, sep, outline):
    """The 9 output slots (code order t, q, f): top row = measured VGGT, bottom row = scene ground truth; sep = gap between the three segments."""
    xs = slot_xs(sep)
    for g, (a, b, lab, col, idx) in enumerate(GROUPS):
        la = alpha * lit[g]
        cx = (xs[a] + xs[b - 1] + SLOT_W) / 2
        cv.rich(lab, (cx, y - 18), 24, alpha=la, fill=col, anchor="mb")
        for k in range(a, b):
            cv.s.rect(xs[k], y, xs[k] + SLOT_W, y + 46, col, 0.18 * la, r=6)
            cv.s.polyline([(xs[k], y), (xs[k] + SLOT_W, y), (xs[k] + SLOT_W, y + 46), (xs[k], y + 46)], col, la, 1,
                          closed=True)
            cv.text((xs[k] + SLOT_W / 2, y + 23), f"{SLOT_VALS_P[k]:.3f}".replace("-", "−"), 23, "light", la, style.TEXT,
                    "mm")
            cv.text((xs[k] + SLOT_W / 2, y + 70), f"{SLOT_VALS_G[k]:.3f}".replace("-", "−"), 21, "light", 0.8 * la,
                    style.TEXT_DIM, "mm")
        cv.text((cx, y + 98), idx, 19, alpha=0.8 * la, fill=style.TEXT_DIM, anchor="mm")
    cv.text((xs[0] - 12, y + 23), "VGGT", 22, "light", alpha, style.TEXT, "rm")
    cv.text((xs[0] - 12, y + 70), "GT", 22, alpha=alpha, fill=style.TEXT_DIM, anchor="rm")
    if outline > 0.01:                                    # after joining: one length-9 vector
        x0, x1 = xs[0] - 6, xs[-1] + SLOT_W + 6
        cv.s.polyline([(x0, y - 6), (x1, y - 6), (x1, y + 52), (x0, y + 52)], WHITE, 0.8 * alpha * outline, 2,
                      closed=True)


def order_strip(cv, alpha, u):
    """Small strip from paper order [q, t, f] to code order [t, q, f] (width ∝ count). u: 0 = paper, 1 = code."""
    y, cell, x0 = 1078, 22, 470
    items = {"q": (4, GOLD), "t": (3, ORANGE), "f": (2, CYAN)}
    paper, code = ["q", "t", "f"], ["t", "q", "f"]

    def starts(order):
        x, out = x0, {}
        for k in order:
            out[k] = x
            x += items[k][0] * cell + 6
        return out

    sp, sc_ = starts(paper), starts(code)
    for k, (n, col) in items.items():
        x = sp[k] + (sc_[k] - sp[k]) * u
        dy = -16 * np.sin(np.pi * u) * (1 if k == "q" else (-1 if k == "t" else 0))
        cv.s.rect(x, y - 15 + dy, x + n * cell, y + 15 + dy, col, 0.3 * alpha, r=5)
        cv.s.polyline([(x, y - 15 + dy), (x + n * cell, y - 15 + dy), (x + n * cell, y + 15 + dy), (x, y + 15 + dy)],
                      col, alpha, 1, closed=True)
        cv.rich(f"${k}$", (x + n * cell / 2, y + dy), 24, alpha=alpha, fill=col, anchor="mm")
    cv.rich("Paper §3.1 writes", (x0 - 14, y), 24, alpha=alpha * (1 - ss(u, 0.4, 0.5)), anchor="rm")
    cv.rich("Code stores pose_enc as", (x0 - 14, y), 24, alpha=alpha * ss(u, 0.5, 0.6), anchor="rm")
    cv.rich("(bar above follows the code order)", (x0 + 9 * cell + 30, y), 20, alpha=alpha * u, fill=style.TEXT_DIM, anchor="lm")


@shot.frame
def frame(cv, t):
    onQ, onD = POP_Q.on(t), POP_D.on(t)
    dq = 1 - 0.55 * onQ
    # ---------- Camera head ----------
    a_cam = ss(t, 0.3, 0.8) * (1 - ss(t, 46.4, 46.9))
    if a_cam > 0:
        ac = a_cam * dq
        for r in range(3):                                   # tokens of the three frames (folded sketch), the camera token is taken out
            y = 440 + r * 46
            for k in range(1, 8):
                kind = "patch" if k > 2 else ("register1" if r == 0 else "register")
                token(cv, (170 + k * 34, y), 26, style.TOKEN[kind], ac * 0.7)
            cv.text((140, y), f"Frame {r + 1}", 22, alpha=ac, fill=ec.CAM_COLORS[r], anchor="rm")
            u = ss(t, 1.2 + 0.2 * r, 2.4 + 0.2 * r)
            p0, p1 = np.array([170.0, y]), np.array([250.0 + r * 60, 560.0])
            token(cv, p0 + (p1 - p0) * u, 30, style.TOKEN["camera1" if r == 0 else "camera"], ac,
                  glow=0.4 * bump(t, 1.6, 1.0))
        module_box(cv, (120, 590, 500, 680), "camera head", "Paper: 4 self-attention layers + linear layer", GOLD,   # wide enough that the subtitle stays inside the box
                   ac * ss(t, 2.6, 3.0), glow=0.6 * bump(t, 3.2, 1.0), title_kind="light")
        cv.rich("Code: the same 4 layers iterated 4 times; the 3 camera tokens\nattend to each other; finally a 2-layer MLP (2048→1024→9)", (124, 696), 22,
                alpha=ac * ss(t, 3.4, 3.9), fill=style.TEXT_DIM)
        # 9 numbers: first appear as three segments, then join end to end
        lit = [lv * (1 if g == 1 else dq) for g, lv in enumerate(lit_levels(t))]
        u_join = ss(t, 8.0, 9.2)
        slots(cv, Y_BAR, a_cam * ss(t, 3.8, 4.4), lit, lerp(SEP0, SEP1, u_join), ss(t, 9.0, 9.5))
        a_cc = ac * ss(t, 9.0, 9.5)
        if a_cc > 0:
            CONCAT.draw(cv, (250, 978), alpha=a_cc, part_color={"t": ORANGE, "q": GOLD, "f": CYAN})
            SUM.draw(cv, (520, 978), alpha=a_cc, part_color={"n3": ORANGE, "n4": GOLD, "n2": CYAN, "eq": style.TEXT},
                     part_glow={"eq": 0.5 * bump(t, 9.3, 1.0)})
            a_x = ac * ss(t, 10.4, 10.9)
            if a_x > 0:
                bx = cv.math(r"4\times3\times2=24", (790, 978), 36, style.GREY, a_x)
                cv.dr.line([(bx[0] - 6, (bx[1] + bx[3]) / 2 - 5), (bx[2] + 6, (bx[1] + bx[3]) / 2 - 5)],
                           fill=style.RED + (int(230 * a_x),), width=3)
                cv.rich("Not ×", (bx[2] + 16, 978), 24, alpha=a_x, fill=style.RED, anchor="lm")
            order_strip(cv, ac * ss(t, 12.7, 13.2) * (1 - ss(t, 41.0, 41.4)), ss(t, 13.4, 14.4))   # make room for the f_x formula
        # Small 3D view: camera 2's frustum first translates by the VGGT output, then rotates to the predicted pose (measured on our test scene)
        a3 = ac * ss(t, 15.0, 15.5) * (1 - onQ)            # the popup covers this area, so do not draw meanwhile
        if a3 > 0:
            P = VIEW.at(az=0)
            ec.frustum(cv, P, 0, 0.45, a3, label="Camera 1")
            ut, ur = ss(t, 15.3, 17.0), ss(t, 17.8, 19.6)
            qn = (1 - ur) * np.array([0, 0, 0, 1.0]) + ur * Q_P
            c_pred = -V["extrinsic"][CAM][:, :3].T @ V["extrinsic"][CAM][:, 3] * S        # convert back to meters so it can be drawn with the scene
            ec.frustum_pose(cv, P, ec.quat_to_R(qn), ut * c_pred, 0.45, a3, ec.CAM_COLORS[CAM], label="Camera 2")
            cv.rich("Top view: camera 2 placed by VGGT's $t$ and $q$", (815, 420), 22, alpha=a3, fill=style.TEXT_DIM, anchor="ma")
        # Note area below (y 1110-1225): t, R(q), f, f_x appear in turn
        a_t = ac * fade(t, 15.3, 17.7, 0.4, 0.3)
        cv.rich(f"Translation $t$ is divided by $s$ ($s$ = mean distance from scene points to camera 1 = {S:.3f} m, see 07)", (540, 1150), 22,
                alpha=a_t, fill=style.TEXT_DIM, anchor="mm")
        a_r = a_cam * (1 - 0.3 * onQ) * fade(t, 18.2, 41.3, 0.4, 0.3)
        if a_r > 0:
            cv.math(r"R(q)=", (330, 1160), 40, style.LABEL_GOLD, a_r, "rm")
            matrix(cv, (340, 1160), R_P.round(3).tolist(), "{:.3f}", 30, alpha=a_r, anchor="lm")
            cv.text((760, 1160), f"rotates {ANG_P:.1f}°", 30, alpha=a_r, fill=style.LABEL_GOLD, anchor="lm")
        a_fn = ac * fade(t, 41.4, 43.9, 0.4, 0.3)
        cv.rich(f"$f$ = (fov_h, fov_w): vertical, horizontal FoV in radians; {FOV_GT[0]:.3f} rad = 60°", (540, 1150), 22,
                alpha=a_fn, fill=style.TEXT_DIM, anchor="mm")
        a_f = ac * ss(t, 43.9, 44.4)
        if a_f > 0:
            FX.draw(cv, (540, 1130), alpha=a_f, state={"def": "lit", "num": "lit" if t > 44.6 else "dim",
                                                     "res": "hl" if 45.1 < t < 46.1 else ("lit" if t > 45.1 else "dim")})
            cv.rich(f"VGGT prediction on our test scene (camera 2): fov_w = {np.degrees(FOV_P[1]):.2f}° → $f_x$ = {FX_P:.1f} px"
                    f" (fov_h = {np.degrees(FOV_P[0]):.2f}°)", (540, 1200), 22,
                    alpha=a_f * ss(t, 45.3, 45.7), fill=style.TEXT_DIM, anchor="ma")

    # ---------- DPT head ----------
    a_d = ss(t, 46.7, 47.2)
    if a_d > 0:
        dd = 1 - 0.45 * onD
        for j in range(24):                                  # stack of 24 rounds, 4 of them taken
            y = 430 + j * 16
            on = j in LAYERS
            hl = ss(t, 48.9 + 0.25 * LAYERS.index(j), 49.3 + 0.25 * LAYERS.index(j)) if on else 0.0
            c = GOLD if hl > 0.5 else (ORANGE if j % 2 == 0 else CYAN)
            cv.s.rect(80, y, 230, y + 11, c, a_d * (0.35 + 0.65 * hl) * (dd if not on else 1.0), r=2)
            if on and hl > 0:
                cv.text((240, y + 6), f"{j}", 20, "light", a_d * hl, style.LABEL_GOLD, "lm")
        cv.text((155, 418), "24 rounds of output", 22, alpha=a_d, anchor="mb")
        cv.text((155, 830), "Indices 4, 11, 17, 23\n(0-based)", 20, align="center",alpha=a_d * ss(t, 49.0, 49.4), fill=style.TEXT_DIM,
                anchor="ma")
        module_box(cv, (300, 430, 560, 520), "DPT head", "Dense prediction head", CYAN, a_d, glow=0.6 * bump(t, 47.2, 1.0),
                   title_kind="light")
        cv.rich("Each token is $1024+1024=2048$ dims\n(code: frame + global outputs concatenated)", (590, 445), 22,
                alpha=a_d * dd * ss(t, 49.6, 50.1) * (1 - ss(t, 67.4, 67.8)), fill=style.TEXT_DIM)
        a_o = ss(t, 67.7, 68.3)                               # depth map (measured VGGT)
        if a_o > 0:
            pan = ImagePanel(DEPTH_IMG, (300, 600), w=420)
            pan.draw(cv, a_o, border=ec.CAM_COLORS[CAM], title="Camera 2: depth output by VGGT $D_2$")
            colorbar(cv, (735, 600, 755, 1020), D_LO, D_HI, "", "{:.2f}", alpha=a_o)      # near (bright) at top, far (dark) at bottom
            pan.mark(cv, UV[0], UV[1], alpha=a_o * ss(t, 68.6, 69.1), label_offset=(-175, 40),
                     label=f"The white dot (sub-pixel): {D_W:.3f}, GT {D_W_GT:.3f}", size=24)
            cp = ImagePanel(CONF_IMG, (830, 640), w=200)
            a_cp = a_o * ss(t, 70.3, 70.8)
            cp.draw(cv, a_cp, border=style.GREY, title=r"Confidence $\Sigma$ (measured)", title_size=22)
            cv.text((930, 852), "Bright = high confidence", 20, alpha=a_cp, fill=style.TEXT, anchor="ma")    # legend (same colors as 07)
            ec.paste_img(cv, CONF_RAMP, (850, 884), 160, a_cp)
            cv.text((842, 891), "low", 18, alpha=a_cp, fill=style.TEXT_DIM, anchor="rm")
            cv.text((1018, 891), "high", 18, alpha=a_cp, fill=style.TEXT_DIM, anchor="lm")
            cv.rich(r"Colored by $\log\Sigma$, color bar as in 07", (930, 916), 18, alpha=a_cp, fill=style.TEXT_DIM, anchor="ma")
            cv.rich(f"Depth unit: ÷ $s$ ($s$ = {S:.3f} m), near bright, far dark; background pixels outside the scene not drawn\n"
                    "Depth head outputs depth + confidence; the point map comes from another DPT head (point head)", (540, 1060), 22,
                    alpha=a_o * ss(t, 70.6, 71.0), fill=style.TEXT_DIM, anchor="ma", align="m")

    # ---------- Popups (drawn last) ----------
    POP_Q.draw(cv, t, body_q)
    if onQ > 0.8:                                             # pointer tip at the popup's bottom edge, aimed at the q segment
        a = ss(onQ, 0.8, 1.0)
        yb = POP_Q.box[1] + POP_Q.box[3] - 1
        cv.dr.polygon([(QX - 16, yb), (QX + 16, yb), (QX, yb + 9)], fill=GOLD + (int(235 * a),))
    POP_D.draw(cv, t, body_d)


def body_q(sub, lt, box):
    p1 = 1 - ss(lt, 10.2, 10.6)
    p2 = ss(lt, 10.5, 10.9)
    if p1 > 0.01:
        _q_page1(sub, lt, p1)
    if p2 > 0.01:
        _q_page2(sub, lt, p2)


def _q_page1(sub, lt, A):
    sub.rich(r"Input: rotation axis $\mathbf{u}$ (unit vector), angle $\theta$", (28, 74), 24, alpha=A * ss(lt, 0.2, 0.6), anchor="lm")
    sub.rich("Output: 4 numbers $q=(x,y,z,w)$ representing this rotation", (28, 108), 24, alpha=A * ss(lt, 0.7, 1.1), anchor="lm")
    a_f = A * ss(lt, 2.7, 3.2)
    QF.draw(sub, (290, 172), alpha=a_f,
            state={"xyz": "hl" if 3.2 < lt < 4.2 else "lit", "w": "hl" if 4.2 < lt < 5.1 else "lit"},
            notes={"xyz": "$(x,y,z)$", "w": "$w$"})
    sub.rich(r"$w$ goes last (VGGT code convention); $\mathbf{u}$ has length 1, so $|q|=1$", (28, 264), 22,
             alpha=A * ss(lt, 4.4, 4.8), fill=style.TEXT_DIM, anchor="lm")
    a_n = A * ss(lt, 5.3, 5.7)
    sub.rich("Camera 2 GT $q$ =", (28, 306), 24, alpha=a_n, anchor="lm")
    vector(sub, (214, 306), Q_GT, "{:.3f}", 26, style.MATH, a_n, anchor="lm")
    sub.math(TH_TEX, (28, 350), 30, style.MATH, A * ss(lt, 5.9, 6.3), "lm")
    sub.math(U_TEX, (28, 394), 28, style.MATH, A * ss(lt, 6.6, 7.0), "lm")
    sub.rich(f"VGGT measured: $q$ = {ec.fmt_vec(Q_P)}", (28, 438), 22,
             alpha=A * ss(lt, 8.1, 8.5), fill=CYAN, anchor="lm")
    # Right column: axes rotate by θ about u (ground truth -> measured VGGT)
    a3 = A * ss(lt, 0.5, 1.1)
    if a3 <= 0.01:
        return
    P = QV.at(az=30 + 24 * ss(lt, 0.0, 10.4))
    v = ss(lt, 8.2, 9.0)
    u = U_GT + (U_P - U_GT) * v
    u = u / np.linalg.norm(u)
    th_full = TH_GT + (TH_P - TH_GT) * v
    s = ss(lt, 1.2, 2.4) * (1 - ss(lt, 5.0, 5.35)) + ss(lt, 5.45, 7.4)
    th = th_full * min(s, 1.0)
    o = P(np.zeros(3))
    arrow(sub, P(-0.9 * u), P(1.3 * u), GOLD, a3, 3, 14)
    sub.rich("$\\mathbf{u}$", P(1.45 * u), 28, alpha=a3, fill=GOLD, anchor="mm")
    axes3d(sub, P, np.zeros(3), np.eye(3), 1.0, labels=None, alpha=0.25 * a3, th=2)
    Rc = rot(u, th)
    axes3d(sub, P, np.zeros(3), Rc, 1.0, alpha=a3, th=3, label_size=22)
    if th > 1e-3:
        for e in (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])):
            arc = np.array([P(rot(u, ph) @ e) for ph in np.linspace(0, th, 24)])
            sub.s.polyline([tuple(p) for p in arc], GOLD, 0.8 * a3, 2)
    sub.math(r"\theta=", (760, 376), 30, style.MATH, a3, "rm")
    number(sub, (764, 376), np.degrees(th), "{:.1f}", 30, CYAN if v > 0.5 else style.MATH, a3, "lm", unit="°")
    sub.rich(r"$R(q)$ rotates a vector by $\theta$ about $\mathbf{u}$", (765, 412), 20, alpha=a3, fill=style.TEXT_DIM, anchor="mm")
    sub.rich("$R$: world→cam; cam 2 turns opposite", (765, 438), 20, alpha=a3 * ss(lt, 3.0, 3.4), fill=style.TEXT_DIM,
             anchor="mm")
    del o


def _q_page2(sub, lt, A):
    sub.rich("Why 4 numbers? General reasons below (the VGGT paper doesn't say why quaternions)", (28, 74), 22, alpha=A, anchor="lm")
    rev = ss(lt, 13.0, 13.4) + ss(lt, 15.6, 16.0) + ss(lt, 18.2, 18.6)
    hl = 0 if lt < 15.6 else (1 if lt < 18.2 else 2)
    table(sub, (28, 100), [150, 80, 260, 420],["Repr.", "Count", "Constraints", "Issue"], Q_TABLE, row_h=80, size=21,
          reveal=rev, highlight=hl if rev > 0.5 else None, alpha=A)
    a_n = A * ss(lt, 18.6, 19.0)
    if a_n > 0.01:
        bx = sub.rich("Measured $q$ from VGGT: length", (28, 446), 22, alpha=a_n, anchor="lm")
        x = (bx[2] if bx is not None else 230) + 10
        val = NORM_P + (1.0 - NORM_P) * ss(lt, 19.2, 20.0)
        number(sub, (x, 446), val, "{:.4f}", 26, style.MATH, a_n, "lm")
        sub.rich("→ divide by length → 1 (code's quat→matrix does too)", (x + 92, 446), 20, alpha=a_n * ss(lt, 19.4, 19.8),
                 fill=style.TEXT_DIM, anchor="lm")


def body_d(sub, lt, box):
    sub.rich("DPT = Dense Prediction Transformer (Ranftl et al., ICCV 2021)", (28, 74), 22, alpha=ss(lt, 0.1, 0.5),
             fill=style.TEXT_DIM, anchor="lm")
    heads = [(XC1 + 52, "4 rounds", 0.4), (XC2, "To grid", 2.8), (XC3, "Resize", 5.4), (XC4, "Fuse", 8.2),
             (XC5, "Output", 12.0)]
    for x, s, t0 in heads:
        sub.text((x, 100), s, 18, alpha=ss(lt, t0, t0 + 0.4), fill=style.TEXT_DIM, anchor="mm")
    for r, (layer, side, disp, y) in enumerate(DP_ROWS):
        a1 = ss(lt, 0.4 + 0.2 * r, 0.8 + 0.2 * r)
        sub.text((18, y), f"R{layer}", 20, alpha=a1, fill=style.LABEL_GOLD, anchor="lm")
        u2 = ss(lt, 2.8 + 0.15 * r, 3.8 + 0.15 * r)
        sq2 = ss(lt, 3.9 + 0.15 * r, 4.4 + 0.15 * r)
        for k in range(9):                                   # token strip -> 3×3 small grid -> 37×37 feature map
            p0 = np.array([XC1 + k * 13.0, y])
            p1 = np.array([XC2 + (k % 3 - 1) * 14.0, y + (k // 3 - 1) * 14.0])
            token(sub, p0 + (p1 - p0) * u2, 10 + 2 * u2, GOLD, a1 * (1 - sq2))
        sub.text((XC1 + 9 * 13 - 2, y), "…", 20, alpha=a1 * (1 - u2), anchor="lm")
        grid_square(sub, (XC2, y), 44, 6, None, CYAN, sq2)
        u3 = ss(lt, 5.4 + 0.4 * r, 6.2 + 0.4 * r)            # resize
        if u3 > 0.01:
            d3 = 44 + (disp - 44) * u3
            arrow(sub, (XC2 + 26, y), (XC3 - disp / 2 - 6, y), WHITE, u3, 2, 10)
            sub.text(((XC2 + 26 + XC3 - disp / 2 - 6) / 2, y - 12), DP_OPS[r], 18, "light", u3, style.TEXT_DIM, "mb")
            grid_square(sub, (XC3, y), d3, 6, None, CYAN, u3)
            sub.text((XC3, y + disp / 2 + 4), f"{side}×{side}", 18, "light", u3, style.TEXT, "ma")
    # Fusion: from the coarsest level (row 4) upward
    t_f = [8.2, 9.1, 10.0, 11.0]                             # fusion results of rows 4, 3, 2, 1 appear
    for i, r in enumerate([3, 2, 1, 0]):
        layer, side, disp, y = DP_ROWS[r]
        u4 = ss(lt, t_f[i], t_f[i] + 0.6)
        if u4 <= 0.01:
            continue
        glow = 0.6 * bump(lt, t_f[i], 1.2)
        arrow(sub, (XC3 + disp / 2 + 4, y), (XC4 - disp / 2 - 6, y), WHITE, u4, 2, 10)
        sub.text(((XC3 + disp / 2 + 4 + XC4 - disp / 2 - 6) / 2, y + 4), "Start" if r == 3 else "+", 18 if r == 3 else 24,
                 alpha=u4, fill=style.TEXT_DIM if r == 3 else ORANGE, anchor="ma")
        grid_square(sub, (XC4, y), disp, 6, None, ORANGE, u4, glow=glow)
        if r < 3:                                            # interpolated up from the next (coarser) level
            _, _, dprev, yprev = DP_ROWS[r + 1]
            arrow(sub, (XC4, yprev - dprev / 2 - 3), (XC4, y + disp / 2 + 4), ORANGE, u4, 2, 9)
            sub.text((XC4 - 8, (yprev - dprev / 2 + y + disp / 2) / 2), f"Interp. to {side}", 17, alpha=u4,
                     fill=style.LABEL_ORANGE, anchor="rm")
    u5 = ss(lt, 12.0, 12.6)                                  # x2 -> 296
    if u5 > 0.01:
        arrow(sub, (XC4 + 54, 160 + DY), (XC5 - 56, 160 + DY), ORANGE, u5, 2, 9)
        sub.text(((XC4 + 54 + XC5 - 56) / 2, 148 + DY), "×2", 18, "light", u5, style.LABEL_ORANGE, "mb")
        grid_square(sub, (XC5, 160 + DY), 104, 8, None, ORANGE, u5, glow=0.6 * bump(lt, 12.0, 1.2))
        sub.text((XC5, 160 + DY), f"{FUSE[4]}×{FUSE[4]}", 20, "light", u5, style.TEXT, "mm")
    u6 = ss(lt, 13.8, 14.4)                                  # interpolate to 518: 2 channels (depth + confidence)
    if u6 > 0.01:
        arrow(sub, (XC5, 216 + DY), (XC5, 260 + DY), WHITE, u6, 2, 9)
        sub.text((XC5 + 8, 238 + DY), "Interp.", 17, alpha=u6, fill=style.TEXT_DIM, anchor="lm")
        ec.paste_img(sub, CONF_IMG, (XC5 - 60 - 7, 270 + DY - 7), 120, u6 * ss(lt, 14.3, 14.8))
        ec.paste_img(sub, DEPTH_IMG, (XC5 - 60 + 7, 270 + DY + 7), 120, u6)
        sub.text((XC5 + 7, 400 + DY), f"{sc.W}×{sc.W}, 2 channels", 18, alpha=u6 * ss(lt, 14.6, 15.0), fill=style.TEXT,
                 anchor="ma")
    # one explanation per step (only one at a time)
    for k, (t0, s) in enumerate(D_STEPS):
        t1 = D_STEPS[k + 1][0] if k + 1 < len(D_STEPS) else 99.0
        a = ss(lt, t0 + 0.15, t0 + 0.5) * (1 - ss(lt, t1, t1 + 0.3))
        if a > 0.01:
            sub.rich(s, (28, 478 + DY), 21, alpha=a, fill=style.TEXT_DIM, anchor="la")


VIEW = ec.viewer(az=0, el=60, r=7.0, fit_pts=np.r_[[np.zeros(3)], [sc.cameras()[CAM][2]],
                                                   ec.plane_corners(0, 0.45), ec.plane_corners(CAM, 0.45)],
                 box=(360, 240), center=(840, 560), target=0.5 * sc.cameras()[CAM][2])
