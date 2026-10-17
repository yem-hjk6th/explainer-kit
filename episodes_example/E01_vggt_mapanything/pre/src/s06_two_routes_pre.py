"""E01-06 Two ways to get a point cloud: Method A = the point head regresses the point map directly; Method B = depth head + camera head, then unproject (S, 42.5 s). A 2.4 s insert after sentence 1 explains "ground truth = known 3D points from the scene script, the reference for comparison"; 0.925 gets its source (the depth head's
predicted depth at the white dot pixel, bilinear sample); the "our test scene" wording and the tag are unified (e01_common.SCENE_TAG).
Storyboard: storyboard.md section E01-06. The two methods are called "Method A / Method B"; the two bar charts each state the metric definition, unit and
lower-is-better; the popup explains how the two errors are computed (paper Tab. 3 = Chamfer distance, nearest point; our test scene = per-pixel pairing).
Measured (R1, our test scene, 3 images of 518×518): both results for the white dot, the full-image point cloud, the errors of the two methods (unaligned / Umeyama alignment with scale),
unit = normalized units (1 = s = 2.153 m). Paper numbers: Tab. 3 (ETH3D, 10 images per scene, Chamfer Overall, lower is better).
Real computation: unproject the white dot step by step with VGGT's predicted K, R, t, D (the check asserts agreement with the official unprojection); the illustrative points in the popup are also computed from the definitions.
"""
import sys
from pathlib import Path

E01_SHOTS = Path(__file__).resolve().parents[2] / "shots"
if str(E01_SHOTS) not in sys.path:
    sys.path.insert(0, str(E01_SHOTS))

import numpy as np

PRE_HIDE = True      # pre cut: hide the VGGT numbers measured on our test scene (see the "pre cut:" comments)

import e01_common as ec
import scene_e01 as sc
from e01_common import Popup, popup_check
from animkit import (CYAN, GOLD, ORANGE, WHITE, Formula, Shot, bar_chart, bump, fade, module_box, number, split, ss,
                     style, vector)

NB = ec.nb()
V = ec.vggt_out()
M = ec.vggt_metrics()
CAM = sc.EX_CAM
S = NB["s"]
UV = NB["uv"]
E = V["extrinsic"][CAM]
R_P, T_P = E[:, :3], E[:, 3]
K_P = V["intrinsic"][CAM]
PM = M["points"]
N_PIX = int(PM["n_points"])


def bil(img, u, v):
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - x0, v - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1] + (1 - fx) * fy * img[y0 + 1, x0]
            + fx * fy * img[y0 + 1, x0 + 1])


D_W = float(bil(V["depth"][CAM], *UV))
RAY_P = np.linalg.solve(K_P, [UV[0], UV[1], 1.0])              # K^{-1}[u, v, 1]
XC_P = D_W * RAY_P                                             # camera frame
P2 = R_P.T @ (XC_P - T_P)                                       # Method B result
P1 = bil(V["world_points"][CAM], *UV)                            # Method A (point head)
GT = NB["X_star"] / S

# Introduce "ground truth" as soon as Method A appears -> insert INS seconds after sentence 1 (the original timeline stays at T_INS, t -> tm(t) in frame)
T_INS, INS = 2.5, 2.4
DUR = 40.1 + INS


def tm(t):
    """Real time -> original timeline (stays at T_INS during the insert)."""
    return t if t < T_INS else (T_INS if t < T_INS + INS else t - INS)


def sh(x):
    """Original timeline -> real time."""
    return x if x < T_INS - 1e-9 else x + INS


shot = Shot("s06_two_routes", DUR)
shot.title("Point Cloud: Two Ways", "A: point head · B: depth + camera heads · VGGT §3.1, §4.3", 0.2, 2.2)
# unified tag (e01_common.scene_tag)
ec.scene_tag(shot, 0.3, sh(13.7), "VGGT-1B output (units ÷ s)")
ec.scene_tag(shot, sh(13.7), sh(37.9), "Left: paper Tab. 3   Right: our test scene")
# no "point clouds have two routes" caption (the two big split-screen titles already say it), see storyboard section 3.
CAPS = [(0.3, 2.5, "Method A: point head outputs points"), (2.5, 4.7, "Method B: depth + camera, unproject"),
        (4.7, 6.9, "First, the direction of this pixel"), (6.9, 9.1, f"Multiply by depth {D_W:.3f} for the camera-frame point"),
        (9.1, 11.3, "Transform back to camera 1's frame"), (11.3, 13.7, "Unproject all three images into one point cloud"),
        (13.7, 16.1, "Paper, ETH3D: Method B has lower error"), (16.1, 18.7, "Our test scene: B is lower without alignment"),
        # popup: how the two errors are computed
        (18.7, 21.3, "Paper: each point finds its nearest point"), (21.3, 23.9, "Average each direction, then average both"),
        (23.9, 26.3, "Paper aligns first; no unit given"), (26.3, 28.9, "Our test scene: per-pixel pairing"),
        (28.9, 31.5, f"Averaged over {N_PIX:,} valid pixels"), (31.5, 33.9, "Both: lower is better"),
        (33.9, 36.3, "Our test scene: align rotation, translation, scale together"), (36.3, 37.9, "After alignment, A and B are nearly identical"),
        (37.9, 40.1, "Authors' view: splitting into subtasks is easier to learn")]
CAPS = CAPS[:1] + [(T_INS, T_INS + INS, "Ground truth: known scene points, as reference")] + [(sh(a), sh(b), s) for a, b, s in CAPS[1:]]
for a, b, s in CAPS:
    shot.caption(a, b, s)

EQ = Formula([(r"P_i(y)=R_i^{\mathsf{T}}(", "a"), (r"D_i(y)", "D"), (r"\,K_i^{-1}[u,v,1]^{\mathsf{T}}", "K"),
              (r"-\,t_i", "t"), (r")", "b")], size=36)
EQ_XY = (735, 405)
FOV_W = float(np.degrees(V["pose_enc"][CAM][8]))

X_ALL = np.concatenate([V["points_from_depth"][i][::4, ::4][np.isfinite(sc.full(i)[1])[::4, ::4]] for i in range(3)])
C_ALL = np.concatenate([np.tile(ec.CAM_COLORS[i], (int(np.isfinite(sc.full(i)[1])[::4, ::4].sum()), 1))
                        for i in range(3)]).astype(float)
OC = ec.viewer(az=-10, el=28, r=7.0 / S, fit_pts=np.r_[X_ALL, [c / S for _, _, c in sc.cameras()]], box=(820, 470),
               center=(540, 800), target=ec.scene_center() / S)

# ---- bar chart positions (upper half of the screen; the lower half is left for the popup) ----
BOX_L = (190, 400, 500, 600)
BOX_R = (690, 400, 1000, 490)
BOX_RA = (690, 600, 1000, 690)

# ---- popup: how the two errors are computed (illustrative points, values computed from the definitions) ----
POP_M = Popup((60, 640, 960, 500), t_in=18.7, t_out=33.7, title="How Each Side Computes Error", corner="tl", tag="Illustration",
              accent=ORANGE)
TOY_GT = np.array([[0.0, 0.90], [1.0, 0.45], [2.0, 0.15], [3.0, 0.05], [4.0, 0.15], [5.0, 0.45]])
TOY_PR = TOY_GT + np.array([[0.55, 0.12], [0.62, -0.08], [0.45, 0.14], [0.60, -0.10], [0.40, 0.12], [0.55, -0.10]])
_D = np.linalg.norm(TOY_PR[:, None] - TOY_GT[None], axis=-1)       # [prediction, GT]
NN_PG, NN_GP = _D.argmin(1), _D.argmin(0)
ACC, COMP = float(_D.min(1).mean()), float(_D.min(0).mean())
OVERALL = (ACC + COMP) / 2
PAIRED = float(np.linalg.norm(TOY_PR - TOY_GT, axis=-1).mean())


def toy_xy(p):
    p = np.asarray(p, float)
    return np.stack([50 + (p[..., 0] + 0.2) * 66, 400 - (p[..., 1] + 0.2) * 230], -1)


@shot.check
def check(ck):
    ref = R_P.T @ (D_W * np.linalg.solve(K_P, [UV[0], UV[1], 1.0]) - T_P)
    ck.close("step-by-step unprojection = formula", P2, ref, 1e-9)
    ck.close("matches the bilinear sample of the official unprojection (sub-pixel, approximate)", P2, bil(V["points_from_depth"][CAM], *UV), 2e-3)
    ck.close("white dot depth+cam matches the metrics file", P2, M["white_dot"]["depth_cam"], 2e-3)
    ck.close("white dot point head matches the metrics file", P1, M["white_dot"]["point_head"], 1e-4)
    ck.eq("number of valid pixels in the scene metrics", N_PIX, 363918)
    ck.close("1 normalized unit = s = 2.153 m", S, 2.1525, 1e-3)
    ck.close("illustrative points: Overall = (Acc + Comp) / 2", OVERALL, (ACC + COMP) / 2, 1e-12)
    ck.true("illustrative points: nearest-point distance <= same-pixel pairing distance", ACC <= PAIRED + 1e-12 and COMP <= PAIRED + 1e-12,
            f"acc {ACC:.3f} comp {COMP:.3f} paired {PAIRED:.3f}")
    ck.true("illustrative points: the nearest GT point of a prediction is not its own pair (so the two definitions differ)", bool((NN_PG != np.arange(6)).any()))
    popup_check(ck, [POP_M], DUR)
    ck.true("original timeline reconnects after the insert (end = original 40.1 s)", abs(tm(DUR) - 40.1) < 1e-9)
    ck.close("on-screen 0.925 = bilinear sample of the depth head at the white dot pixel (0.9252)", D_W, 0.9252, 5e-5)
    ck.eq("on-screen z-depth GT 0.918 (= white dot z-depth in 05 / 08 ÷ s)", f"{NB['zdepth'] / S:.3f}", "0.918")
    ck.record(ray=RAY_P.round(4), depth=round(D_W, 4), Xc=XC_P.round(4), P_B=P2.round(4), P_A=P1.round(4),
              gt=GT.round(4), err_A=round(float(np.linalg.norm(P1 - GT)), 4),
              err_B=round(float(np.linalg.norm(P2 - GT)), 4), points=PM,
              toy=dict(acc=round(ACC, 3), comp=round(COMP, 3), overall=round(OVERALL, 3), paired=round(PAIRED, 3)))


def outline(cv, box, a):
    if a <= 0.01:
        return
    x0, y0, x1, y1 = box
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    cv.s.polyline(pts, ORANGE, a, 2, closed=True)
    cv.g.polyline(pts, ORANGE, 0.5 * a, 4, closed=True)


@shot.frame
def frame(cv, tr):
    t = tm(tr)                                       # original timeline; the "GT" lines use real time tr
    on_m = POP_M.on(t)
    a_split = ss(t, 0.1, 0.6) * (1 - ss(t, 11.1, 11.5))
    if a_split > 0:
        left, right = split(cv, "lr", 0.38, a_split, margin=(40, 280, 1040, 860))
        lc, rc = (left[0] + left[2]) / 2, (right[0] + right[2]) / 2
        cv.rich("Method A: point head", (lc, 298), 32, alpha=a_split, fill=CYAN, anchor="ma")
        cv.rich("Regresses the point map", (lc, 344), 24, alpha=a_split, fill=style.TEXT_DIM, anchor="ma")
        cv.rich("Method B: depth head + camera head", (rc, 298), 32, alpha=a_split, fill=GOLD, anchor="ma")
        cv.rich("Then unprojects depth to 3D with the cameras", (rc, 344), 24, alpha=a_split, fill=style.TEXT_DIM, anchor="ma")
        # Method A
        a1 = ss(t, 0.4, 0.9)
        if a1 > 0:
            module_box(cv, (70, 420, 380, 510), "point head", "DPT → $P_i$ (camera 1 frame)", CYAN, a_split * a1,
                       title_kind="light")
            if not PRE_HIDE:  # pre cut: hide the white dot P1 measured on our test scene and its label
                cv.rich("White dot:", (70, 560), 28, alpha=a_split * a1)
                vector(cv, (70, 620), P1, "{:.3f}", 28, CYAN, a_split * ss(t, 1.1, 1.6), anchor="lm")   # size 28 keeps ")" off the divider line
            # explain GT as soon as Method A appears (insert T_INS to T_INS+INS, real time)
            if not PRE_HIDE:  # pre cut: hide the whole GT X*/s block measured on our test scene (title, note, coordinates, highlight box)
                a_gt = a_split * ss(tr, T_INS + 0.1, T_INS + 0.5)
                g_gt = bump(tr, T_INS + 0.6, 1.4)
                cv.rich(r"GT $X^\star/s$: reference for comparison", (70, 676), 24, alpha=a_gt, fill=style.LABEL_GOLD)
                cv.rich("= known 3D points from the scene script (÷ $s$)", (70, 712), 20, alpha=a_split * ss(tr, T_INS + 0.5, T_INS + 0.9),
                        fill=style.TEXT_DIM)
                vector(cv, (70, 772), GT, "{:.3f}", 28, style.TEXT_DIM, a_gt, anchor="lm")
                if g_gt > 0.01:
                    outline(cv, (58, 652, 412, 806), a_split * g_gt)
        # Method B: formula + three steps
        a2 = ss(t, 2.6, 3.1)
        if a2 > 0:
            st = {"a": "lit" if t > 9.2 else "dim", "D": "hl" if 7.0 < t < 9.1 else ("lit" if t > 7.0 else "dim"),
                  "K": "hl" if 4.8 < t < 6.9 else ("lit" if t > 4.8 else "dim"),
                  "t": "lit" if t > 9.2 else "dim", "b": "lit" if t > 9.2 else "dim"}
            EQ.draw(cv, EQ_XY, alpha=a_split * a2, state=st)
            cv.rich("$R_i,t_i$: extrinsics (world→cam)   $K_i$: intrinsics   $D_i(y)$: z-depth", (735, 450), 22,
                    alpha=a_split * a2, fill=style.TEXT_DIM, anchor="ma")
            rows = [(4.9, rf"K^{{-1}}[{UV[0]:.1f},\,{UV[1]:.1f},\,1]^{{\mathsf{{T}}}}=", RAY_P, "Direction ($z$=1)"),
                    (7.1, r"\times D=" + f"{D_W:.3f}" + r"\ \Rightarrow\ ", XC_P, "Camera 2 frame point"),
                    (9.3, r"R^{\mathsf{T}}(\cdot-t)\ \Rightarrow\ ", P2, "Camera 1 frame (world)")]
            for k, (t0, tex, vec, lab) in enumerate(rows):
                if PRE_HIDE:  # pre cut: hide the Method B step-by-step numbers measured on our test scene (3 rows and the small 0.925 source note)
                    break
                a = a_split * ss(t, t0, t0 + 0.5)
                if a <= 0:
                    continue
                y = (520, 625, 770)[k]                  # the source of 0.925 sits under row 2, so row 3 is lower
                cv.math(tex, (460, y), 32, style.MATH, a, "lm")
                vector(cv, (1030, y + 46), vec, "{:.3f}", 30, GOLD if k == 2 else style.MATH, a, anchor="rm")
                cv.rich(lab, (460, y + 46), 22, alpha=a, fill=style.TEXT_DIM, anchor="lm")
                if k == 1:                              # where the 0.925 depth comes from (0.9252)
                    an = a_split * ss(t, 7.5, 7.9)
                    cv.rich(f"{D_W:.3f} = depth predicted by VGGT's depth head at the white dot pixel", (460, y + 82), 19, alpha=an,
                            fill=style.LABEL_GOLD, anchor="lm")
                    # optional suggestion: say it is z-depth + GT, so it is not confused with the 0.922 (along the ray) in 08
                    cv.rich(f"(bilinear sample of depth map, normalized units; z-depth, GT {NB['zdepth'] / S:.3f})", (460, y + 106), 18,
                            alpha=an, fill=style.TEXT_DIM, anchor="lm")
            a_cmp = a_split * ss(t, 10.1, 10.6)
            if not PRE_HIDE:  # pre cut: hide the measured white-dot-to-GT error line (computed from measurements, so hidden too)
                cv.rich(f"White dot to GT: Method A {np.linalg.norm(P1 - GT):.3f}, Method B {np.linalg.norm(P2 - GT):.3f}"
                        " (this one point, units ÷ s)", (540, 900), 26, alpha=a_cmp, anchor="ma")
            if not PRE_HIDE:  # pre cut: hide the measured VGGT-predicted camera 2 intrinsics line
                cv.rich("VGGT-predicted camera 2 intrinsics: fov_w = %.2f°, $f_x$ = %.1f px (GT 60°, 448.6 px)" % (FOV_W, K_P[0, 0]),
                        (540, 950), 22, alpha=a_split * ss(t, 5.3, 5.7), fill=style.TEXT_DIM, anchor="ma")

    # full-image point cloud (VGGT measured, colored by source camera)
    a_pc = fade(t, 11.4, 13.9, 0.5, 0.4)
    if a_pc > 0:
        P = OC.at(az=-10 + 14 * ss(t, 11.4, 13.9))
        for i in range(3):
            ec.frustum(cv, P, i, 0.45, a_pc, scale=1.0 / S, label=ec.CAM_NAMES[i])
        ec.splat(cv, P, X_ALL, C_ALL, a_pc * ss(t, 11.6, 12.7), size=2)
        cv.rich("Method B (depth + camera unprojection) point cloud, one color per camera", (540, 1120), 24, alpha=a_pc,
                fill=style.TEXT_DIM, anchor="ma")

    # paper Tab. 3 and our test scene measurements
    a_t = ss(t, 13.8, 14.3) * (1 - ss(t, 37.5, 37.9))
    if a_t > 0:
        bar_chart(cv, BOX_L, [1.005, 0.826, 0.709, 0.677], ["DUSt3R", "MASt3R", "Method A", "Method B"],
                  vmax=1.1, horizontal=True, highlight=[2, 3], progress=ss(t, 14.0, 15.3), fmt="{:.3f}",
                  title="Paper Tab. 3 (ETH3D)\nChamfer distance, lower is better", alpha=a_t)
        a_r = ss(t, 16.2, 16.7) * a_t
        if not PRE_HIDE:  # pre cut: hide the measured unaligned bar chart
            bar_chart(cv, BOX_R, [PM["point_head"]["mean"], PM["depth_plus_camera"]["mean"]],
                      ["Method A", "Method B"], vmax=0.03, horizontal=True, highlight=1, progress=ss(t, 16.4, 17.4), fmt="{:.4f}",
                      title="Our test scene: no alignment\nSame-pixel error, lower = better", alpha=a_r)
        a_al = ss(t, 34.0, 34.5) * a_t
        if not PRE_HIDE:  # pre cut: hide the measured Umeyama-aligned bar chart
            bar_chart(cv, BOX_RA, [PM["point_head"]["aligned_mean"], PM["depth_plus_camera"]["aligned_mean"]],
                      ["Method A", "Method B"], vmax=0.03, horizontal=True, progress=ss(t, 34.2, 35.2), fmt="{:.4f}",
                      title="Our test scene: Umeyama-aligned\n(rotation + translation + scale)", alpha=a_al)
        a_def = a_t * ss(t, 33.9, 34.4)                      # the two definitions that stay after the popup closes
        cv.rich("Chamfer: predicted→nearest GT, GT→nearest predicted\nAverage each, then average both; Umeyama-aligned first; no unit in paper",
                (345, 640), 20, alpha=a_def, fill=style.TEXT_DIM, anchor="ma", align="m")
        if not PRE_HIDE:  # pre cut: hide the measured same-pixel distance / valid pixel count note
            cv.rich(f"Same-pixel distance between prediction and GT,\naveraged over {N_PIX:,} valid pixels;\nunits ÷ $s$ ($s$ = {S:.3f} m)",
                    (845, 730), 20, alpha=a_def, fill=style.TEXT_DIM, anchor="ma", align="m")
        if not PRE_HIDE:  # pre cut: hide the measured "different data on the two sides" note
            cv.rich("Different data: left is the paper's dataset and protocol, right is our test scene (own test)", (540, 860), 22,
                    alpha=a_t * ss(t, 34.4, 34.9), fill=style.TEXT_DIM, anchor="ma")
        # whichever block the popup is talking about gets framed in the main view
        lt = POP_M.local(t)
        outline(cv, (66, 296, 560, 615), on_m * (1 - ss(lt, 7.4, 7.8)))
        outline(cv, (580, 300, 1040, 505), on_m * ss(lt, 7.6, 8.0))
    a_w = ss(t, 38.0, 38.5)
    if a_w > 0:
        module_box(cv, (180, 520, 900, 640), "point map = depth + camera", "Authors' explanation (§4.3); not tested separately in the paper", GOLD, a_w,
                   title_kind="light")
        cv.rich("depth: each image's own z-depth   camera: 9 numbers", (540, 700), 26, alpha=a_w * ss(t, 38.5, 38.9),
                fill=style.TEXT_DIM, anchor="ma")
    POP_M.draw(cv, t, body_m)


def body_m(sub, lt, box):
    p1 = 1 - ss(lt, 7.4, 7.8)
    p2 = ss(lt, 7.6, 8.0)
    G, Pp = toy_xy(TOY_GT), toy_xy(TOY_PR)
    a0 = ss(lt, 0.2, 0.6)
    # coordinate frame (illustrative)
    sub.s.polyline([(40, 110), (40, 430), (470, 430)], style.GREY, 0.5 * a0, 1)
    # page 1: nearest points (both directions)
    for k in range(6):
        u = ss(lt, 0.8 + 0.2 * k, 1.2 + 0.2 * k)
        if u > 0:
            a, b = Pp[k], G[NN_PG[k]]
            sub.s.line(tuple(a), tuple(a + (b - a) * u), CYAN, 0.9 * p1, 2)
        u = ss(lt, 3.0 + 0.2 * k, 3.4 + 0.2 * k)
        if u > 0:
            a, b = G[k], Pp[NN_GP[k]]
            sub.s.dashed(tuple(a), tuple(a + (b - a) * u), WHITE, 0.8 * p1, 2, n=8)
    # page 2: same-pixel one-to-one pairing
    for k in range(6):
        u = ss(lt, 8.0 + 0.2 * k, 8.4 + 0.2 * k)
        if u > 0:
            a, b = Pp[k], G[k]
            sub.s.line(tuple(a), tuple(a + (b - a) * u), ORANGE, p2, 3)
    for k in range(6):
        sub.s.dot(tuple(G[k]), WHITE, 6, a0)
        sub.s.dot(tuple(Pp[k]), CYAN, 6, a0)
        if p2 > 0.01:
            sub.text((G[k][0] - 4, G[k][1] - 10), f"{k + 1}", 16, "light", p2 * a0, WHITE, "rb")
            sub.text((Pp[k][0] + 6, Pp[k][1] + 4), f"{k + 1}", 16, "light", p2 * a0, CYAN, "la")
    sub.text((60, 396), "● GT", 20, alpha=a0, fill=WHITE, anchor="la")
    sub.text((170, 396), "● Predicted", 20, alpha=a0, fill=CYAN, anchor="la")
    if p2 > 0.01:
        sub.text((285, 398), "Number = pixel", 18, alpha=p2 * a0, fill=style.TEXT_DIM, anchor="la")
    sub.text((255, 450), "Toy points: values by definition, units illustrative", 18, alpha=a0, fill=style.TEXT_DIM, anchor="ma")
    X = 500
    if p1 > 0.01:
        sub.rich("Paper Tab. 3: Chamfer (nearest point)", (X, 86), 22, "bold", p1 * ss(lt, 0.2, 0.6), anchor="la")
        sub.rich("1. Each prediction → nearest GT point", (X, 130), 22, alpha=p1 * ss(lt, 0.8, 1.2), fill=CYAN, anchor="la")
        a = p1 * ss(lt, 2.0, 2.4)
        bx = sub.rich("Accuracy = mean distance =", (X + 24, 164), 22, alpha=a, anchor="la")
        number(sub, (bx[2] + 10, (bx[1] + bx[3]) / 2), ACC, "{:.3f}", 24, CYAN, a, "lm")
        sub.rich("2. Each GT point → nearest prediction", (X, 214), 22, alpha=p1 * ss(lt, 3.0, 3.4), anchor="la")
        a = p1 * ss(lt, 4.2, 4.6)
        bx = sub.rich("Completeness = mean distance =", (X + 24, 248), 22, alpha=a, anchor="la")
        number(sub, (bx[2] + 10, (bx[1] + bx[3]) / 2), COMP, "{:.3f}", 24, style.TEXT, a, "lm")
        a = p1 * ss(lt, 4.8, 5.2)
        bx = sub.rich("3. Overall = average of both =", (X, 298), 22, alpha=a, fill=ORANGE, anchor="la")
        number(sub, (bx[2] + 10, (bx[1] + bx[3]) / 2), OVERALL, "{:.3f}", 24, ORANGE, a, "lm")
        sub.rich("Umeyama-align first (rotation, translation, scale)", (X, 352), 20, alpha=p1 * ss(lt, 5.4, 5.8),
                 fill=style.TEXT_DIM, anchor="la")
        sub.rich("No unit in paper; lower is better", (X, 384), 22, alpha=p1 * ss(lt, 5.8, 6.2), fill=style.LABEL_GOLD, anchor="la")
        sub.rich("Defined in §4.2 (DTU); Tab. 3 uses same-named metrics", (X, 418), 18, alpha=p1 * ss(lt, 6.2, 6.6),
                 fill=style.TEXT_DIM, anchor="la")
    if p2 > 0.01:
        sub.rich("Our test scene: per-pixel pairing", (X, 86), 22, "bold", p2, anchor="la")
        sub.rich("Each pixel has a predicted and a GT point", (X, 130), 22, alpha=p2 * ss(lt, 7.8, 8.2), anchor="la")
        sub.rich("(scripted test scene: GT 3D point known per pixel)", (X, 162), 18, alpha=p2 * ss(lt, 8.0, 8.4),
                 fill=style.TEXT_DIM, anchor="la")
        sub.rich("Same pixel's two points, compared directly:", (X, 204), 22, alpha=p2 * ss(lt, 8.6, 9.0), anchor="la")
        sub.math(r"\Vert P_{\mathrm{pred}}-X_{\mathrm{gt}}/s\Vert", (X + 24, 252), 30, style.MATH,
                 p2 * ss(lt, 8.8, 9.2), "lm")
        a = p2 * ss(lt, 9.4, 9.8)
        bx = sub.rich("Mean over illustrative points =", (X + 24, 282), 20, alpha=a, fill=style.TEXT_DIM, anchor="la")
        number(sub, (bx[2] + 10, (bx[1] + bx[3]) / 2), PAIRED, "{:.3f}", 22, ORANGE, a, "lm")
        sub.rich(f"Measured: mean over {N_PIX:,} valid pixels", (X, 326), 22, alpha=p2 * ss(lt, 10.3, 10.7), anchor="la")
        sub.rich(f"Unit: normalized units, 1 = $s$ = {S:.3f} m", (X, 360), 22, alpha=p2 * ss(lt, 12.9, 13.3), anchor="la")
        sub.rich("Lower is better", (X, 394), 22, alpha=p2 * ss(lt, 13.2, 13.6), fill=style.LABEL_GOLD, anchor="la")
        sub.rich("Unlike the paper: no nearest search; pixels fix pairs", (X, 428), 18, alpha=p2 * ss(lt, 13.5, 13.9),
                 fill=style.TEXT_DIM, anchor="la")
