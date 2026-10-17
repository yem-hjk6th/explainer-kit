"""E02-02 Depth, inverse depth, disparity (M, 34.5 s). Storyboard storyboard.md section E02-02.
Computed for real: the white dot's z-depth D / range r along the ray / inverse depth rho (numbers of the camera-2 cross-section, full depth and inverse-depth maps, colorbar ranges), the right image and white-dot pixel truly rendered by the imaginary right camera
(camera 2 shifted by B along its own x axis), disparity = f B / D, three points on the dD = D^2 d(rho) curve.
Illustration: the imaginary stereo baseline B = 0.12 m is a free choice (tag on #4-#5); how far the image plane sits from the center in the cross-section is only a drawing choice (marked "position illustration").
Paper numbers: Depth Pro paper Tab. 10 (Hypersim delta1, 0-1 m range) inverse depth 0.730, log depth 0.700, depth 0.657 (#8, tag "Paper Tab. 10").
Tags: #1-#7 c2.GT_TAG (GT frames); #4-#6 also "Illustration: virtual stereo pair, free baseline"; #8 Paper Tab. 10.
Notation: disparity is always written Δu (not d; in E02-07 d is DA3's ray direction).
"""
import os
import re
from functools import lru_cache

import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, ORANGE, VIOLET, Formula, ImagePanel, Plot2D, Shot, bar_chart, bump, colorbar,
                     colorize, fade, lerp, measure, ss, style)

ec, sc = c2.ec, c2.sc
NB = c2.NB()
NB1 = ec.nb()                                       # E01 scene numbers (Kinv_p etc.)
CAM = c2.CAM2
R2, _t2, C2W = sc.cameras()[CAM]                    # camera 2 in the world frame (camera 1): world->camera rotation, center
RS2, CS2 = sc._CS[CAM]                              # the same camera in the scene frame

DUR = 34.5
T = [0.3, 4.3, 8.5, 12.6, 16.4, 20.6, 25.0, 29.4, DUR]
CAPS = [(T[0], T[1], "Depth D: distance along the optical axis"),
        (T[1], T[2], "Not the straight distance to the center"),
        (T[2], T[3], "Inverse depth: one over depth"),
        (T[3], T[4], "Imagine a parallel right camera"),
        (T[4], T[5], "White dot differs by 27.3 px between images"),
        (T[5], T[6], "Disparity = inverse depth times a constant"),
        (T[6], T[7], "Inverse depth off by 0.01: only 1 cm off up close"),
        (T[7], T[8], "So inverse depth is more accurate up close")]

shot = Shot("s02_depth_disp", DUR)
shot.title("Depth vs. disparity", "E02-02 · Depth, inverse depth, disparity", 0.2, 3.0)
shot.tag(c2.GT_TAG, 0.3, T[7])
shot.tag("Illustration: virtual stereo pair, free baseline", T[3], T[6])
shot.tag("Paper Tab. 10", T[7], DUR)
for a, b, s in CAPS:
    shot.caption(a, b, s)

# ------------------------------------------------------------------ numbers
D, RNG, RHO, B_BASE = NB["D"], NB["range"], NB["rho"], NB["B"]
UV = NB1["uv"]                                      # sub-pixel position of the white dot in camera 2 (left image)
UVR = NB["uv_right"]                                # white dot in the imaginary right image
DISP = NB["disp"]
F = sc.FX
KP = NB1["Kinv_p"]
KPN = float(np.linalg.norm(KP))
TH = np.arccos(D / RNG)                             # angle between the ray and the optical axis (for the cross-section)
H_OFF = float(np.sqrt(RNG ** 2 - D ** 2))           # perpendicular distance of the white dot from the optical axis (m)

# Imaginary right camera (world frame = camera-1 frame): center = camera-2 center shifted by B along camera 2's own x axis, same orientation
C_R = C2W + B_BASE * R2[0]
CS_R = CS2 + RS2.T @ np.array([B_BASE, 0, 0])       # the same point in the scene frame (as written in e02_numbers.py L72)

# depth map / inverse-depth map (background inf / nan)
DMAP = c2.cam2_depth()
OKM = np.isfinite(DMAP)
RMAP = np.where(OKM, 1.0 / np.where(OKM, DMAP, 1.0), np.nan)
D_LO, D_HI = NB["D_min"], NB["D_max"]
R_LO, R_HI = NB["rho_min"], NB["rho_max"]


@lru_cache(maxsize=None)
def shade(Rs_key, cs_key):
    """Render a full image for any camera (scene-frame world->camera rotation Rs, center cs): copies the intersection and shading logic of scene_e01.render, leaving the E01 files unchanged.
    Returns rgb (uint8), z-depth (inf on a miss), scene-frame 3D points."""
    Rs = np.array(Rs_key).reshape(3, 3)
    cs = np.array(cs_key)
    u, v = np.meshgrid(np.arange(sc.W, dtype=float), np.arange(sc.H, dtype=float))
    dcam = np.stack([(u - sc.CX) / sc.FX, (v - sc.CY) / sc.FY, np.ones_like(u)], -1)
    d = dcam @ Rs
    o = np.broadcast_to(cs, d.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[..., 1] > 1e-9, (sc.GROUND_Y - o[..., 1]) / d[..., 1], np.inf)
        Xg = o + np.where(np.isfinite(tg), tg, 0)[..., None] * d
    tile = (np.abs(Xg[..., 0] - sc.GROUND_X[0]) <= sc.GROUND_X[1]) & (np.abs(Xg[..., 2] - sc.GROUND_Z[0]) <= sc.GROUND_Z[1])
    tg = np.where((tg > 1e-6) & tile, tg, np.inf)
    allt = np.stack([tg, sc._hit_box(o, d, sc.BOX['lo'], sc.BOX['hi'])] +
                    [sc._hit_sphere(o, d, s['c'], s['r']) for s in sc.SPHERES], -1)
    idx = np.argmin(allt, -1)
    tz = np.min(allt, -1)
    hit = np.isfinite(tz)
    Xs = o + np.where(hit, tz, np.nan)[..., None] * d
    n = np.zeros_like(Xs)
    n[idx == 0] = [0, -1, 0]
    bx = (idx == 1) & hit
    Xb = Xs[bx]
    nb = np.zeros_like(Xb)
    for ax in range(3):
        nb[np.abs(Xb[:, ax] - sc.BOX['lo'][ax]) < 1e-4, ax] = -1
        nb[np.abs(Xb[:, ax] - sc.BOX['hi'][ax]) < 1e-4, ax] = 1
    n[bx] = nb
    for k, s in enumerate(sc.SPHERES):
        m = (idx == 2 + k) & hit
        n[m] = (Xs[m] - s['c']) / s['r']
    lam = np.clip(-(n @ sc.LIGHT), 0, 1) * 0.7 + 0.3
    base = np.zeros_like(Xs)
    with np.errstate(invalid='ignore'):
        chk = (np.floor(Xs[..., 0] / 0.25) + np.floor(Xs[..., 2] / 0.25)) % 2 == 0
    g = (idx == 0) & hit
    base[g] = np.where(chk[g][:, None], [165, 165, 160], [105, 105, 102])
    base[bx] = sc.BOX['rgb']
    for k, s in enumerate(sc.SPHERES):
        base[(idx == 2 + k) & hit] = s['rgb']
    rgb = np.zeros(d.shape, float) + np.array([28, 30, 38])
    rgb[hit] = base[hit] * lam[hit][:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8), tz, Xs


def render_cam(Rs, cs):
    return shade(tuple(np.asarray(Rs, float).ravel()), tuple(np.asarray(cs, float)))


RGB_L, TZ_L, XS_L = render_cam(RS2, CS2)
RGB_R, TZ_R, XS_R = render_cam(RS2, CS_R)

# Strip crop: take only rows V0:V1 around the white dot, enlarge, stack left/right images vertically so the disparity is visible
V0, V1 = 246, 346
STRIP_W = 760
STRIP_X = 160
STRIP_Y = (345, 552)
PS_L = ImagePanel(RGB_L[V0:V1], (STRIP_X, STRIP_Y[0]), w=STRIP_W)
PS_R = ImagePanel(RGB_R[V0:V1], (STRIP_X, STRIP_Y[1]), w=STRIP_W)
PT_L = PS_L.to_screen(UV[0], UV[1] - V0)
PT_R = PS_R.to_screen(UVR[0], UVR[1] - V0)
PT_L_ON_R = PS_R.to_screen(UV[0], UV[1] - V0)       # column of the left-image white dot, projected onto the right strip (reference position)

# depth-map / inverse-depth-map panels
PAN_XY, PAN_W = (70, 330), 520
PAN_D = ImagePanel(colorize(DMAP, D_LO, D_HI), PAN_XY, w=PAN_W)
PAN_R = ImagePanel(colorize(-RMAP, -R_HI, -R_LO), PAN_XY, w=PAN_W)   # negated: near (large ρ) is still the bright color
PAN_PT = PAN_D.to_screen(UV[0], UV[1])
CB_BOX = (PAN_XY[0] + PAN_W + 22, PAN_XY[1], PAN_XY[0] + PAN_W + 46, PAN_XY[1] + PAN_W)

# small 3D view (#4)
DEP3 = 0.35
DIR_L = (sc.X_STAR - C2W) / np.linalg.norm(sc.X_STAR - C2W)
DIR_R = (sc.X_STAR - C_R) / np.linalg.norm(sc.X_STAR - C_R)
RAY_LEN = 0.5
FIT3 = np.r_[[C2W, C_R], ec.plane_corners(CAM, DEP3), (np.array(ec.plane_corners(CAM, DEP3)) + B_BASE * R2[0]),
             [C2W + RAY_LEN * DIR_L, C_R + RAY_LEN * DIR_R]]
FWD = R2[2]
AZ3 = float(np.degrees(np.arctan2(-FWD[0], FWD[2])))
OC3 = ec.viewer(az=AZ3, el=72, r=3.0, fit_pts=FIT3, box=(860, 520), center=(540, 610),
                target=np.mean([C2W, C_R], 0) + 0.25 * R2[2])

# dD curve
DRHO = 0.01
PD = Plot2D((170, 640, 990, 1090), (0, 10), (0, 1.1), xlabel="$D$ (m)", ylabel=r"$\delta D$ (m)", fmt_y="{:.1f}")
DD = np.linspace(0, 10, 201)
DDD = DD ** 2 * DRHO
PTS_D = [1.0, D, 10.0]
PTS_DD = [Z * Z * DRHO for Z in PTS_D]

# Tab. 10
TAB10 = [0.730, 0.700, 0.657]
TAB10_LABELS = ["Inverse depth $1/D$", r"Log depth $\log D$", "Depth $D$"]

# ------------------------------------------------------------------ formulas (static, built at module level)
F_R = Formula([("r", "r"), (r"=D\,", "D"), (r"\Vert K^{-1}\tilde{p}\Vert", "n")], size=62)
F_RN = Formula([("1.984=", "a"), ("1.975", "b"), (r"\times1.0044", "c")], size=54)
F_RHO = Formula([(r"\rho", "r"), (r"=\dfrac{1}{D}", "d")], size=72)
F_DU = Formula([(r"\Delta u", "du"), ("=f", "f"), (r"\cdot B", "B"), (r"/D", "D"), ("=f", "f2"), (r"\cdot B", "B2"),
                (r"\cdot\rho", "rho")], size=62, gap=0.18)
F_DUN = Formula([(r"\Delta u=", "a"), ("448.6", "f"), (r"\times0.12", "B"), ("/1.975", "D"), (r"\approx27.3", "res")],
                size=54)
F_DD = Formula([(r"\delta D", "dd"), (r"\approx D^{2}\,", "sq"), (r"\delta\rho", "dr")], size=64)
COL_F, COL_B, COL_D, COL_R = VIOLET, ORANGE, GOLD, CYAN

S_D, S_RNG, S_RHO, S_DISP = f"{D:.3f}", f"{RNG:.3f}", f"{RHO:.3f}", f"{DISP:.1f}"
S_UL, S_UR, S_DU2 = f"{UV[0]:.2f}", f"{UVR[0]:.2f}", f"{DISP:.2f}"


@shot.check
def check(ck):
    # ---- #1-#2: numbers of the cross-section and formulas
    ck.eq("white dot z-depth D on screen", S_D, "1.975")
    ck.eq("range r along the ray on screen", S_RNG, "1.984")
    ck.eq("‖K^-1 p~‖ on screen (four digits; the storyboard says 1.004, the screen says 1.0044 so the product works out)", f"{KPN:.4f}", "1.0044")
    ck.close("r = D ‖K^-1 p~‖", D * KPN, RNG, 1e-9)
    ck.eq("on-screen product is consistent: 1.975 × 1.0044 = 1.984 (three decimals)", f"{round(D, 3) * round(KPN, 4):.3f}", "1.984")
    ck.eq("three components of K^-1 p~ on screen", [f"{x:.3f}" for x in KP], ["-0.043", "0.084", "1.000"])
    ck.close("cross-section: D and r are geometrically consistent, |(D, h)| = r", np.hypot(D, H_OFF), RNG, 1e-12)
    ck.close("angle between ray and optical axis: cos = D / r", np.cos(TH), D / RNG, 1e-12)
    # ---- #3: depth map / inverse-depth map
    ck.eq("depth colorbar 1.13-3.95 m", (f"{D_LO:.2f}", f"{D_HI:.2f}"), ("1.13", "3.95"))
    ck.eq("inverse-depth colorbar 0.25-0.88 m^-1", (f"{R_LO:.2f}", f"{R_HI:.2f}"), ("0.25", "0.88"))
    ck.eq("white dot ρ = 1/D on screen", S_RHO, "0.506")
    ck.close("ρ = 1/D", RHO, 1.0 / D, 1e-12)
    ck.close("depth map at the white-dot pixel ≈ D", DMAP[int(round(UV[1])), int(round(UV[0]))], D, 5e-3)
    ck.close("inverse-depth map at the white-dot pixel ≈ ρ", RMAP[int(round(UV[1])), int(round(UV[0]))], RHO, 5e-3)
    ck.true("background pixels are inf in the depth map and background in the inverse-depth map", (~OKM).any() and np.isnan(RMAP[~OKM]).all(),
            f"background pixels {(~OKM).sum()}")
    ck.close("colorbar range = min / max of the valid pixels", (DMAP[OKM].min(), DMAP[OKM].max()), (D_LO, D_HI), 1e-12)
    ck.close("ρ range = 1/D range, swapped", (1 / D_HI, 1 / D_LO), (R_LO, R_HI), 1e-12)
    # ---- #4-#5: imaginary right camera
    ck.eq("baseline B on screen", f"{B_BASE:.2f}", "0.12")
    ck.close("right camera center = camera-2 center + B × camera-2 x axis (world frame)", np.linalg.norm(C_R - C2W), B_BASE, 1e-12)
    ck.close("world-frame form and the scene-frame form in e02_numbers give the same point", sc.to_world(CS_R), C_R, 1e-9)
    ck.close("left and right images share one rotation; camera axes are parallel (x direction)", (C_R - C2W) / B_BASE, R2[0], 1e-12)
    ck.true("copied shading logic = scene_e01.render (full camera-2 image identical pixel by pixel)", np.array_equal(RGB_L, c2.sc.full(CAM)[0]))
    ck.true("depth map likewise", np.array_equal(np.where(np.isfinite(TZ_L), TZ_L, -1), np.where(OKM, DMAP, -1)))
    ck.eq("u_L on screen (two decimals)", S_UL, "239.74")
    ck.eq("u_R on screen (two decimals)", S_UR, "212.49")
    ck.eq("Δu on screen (one decimal)", S_DISP, "27.3")
    ck.eq("Δu on screen (two decimals)", S_DU2, "27.25")
    ck.close("u_L − u_R = Δu (sub-pixel value)", UV[0] - UVR[0], DISP, 1e-9)
    ck.close("Δu = f B / D", F * B_BASE / D, DISP, 1e-9)
    ck.close("same row v (parallel stereo)", UVR[1], UV[1], 1e-9)
    ck.eq("f on screen", f"{F:.1f}", "448.6")
    ck.eq("448.6 × 0.12 / 1.975 = 27.3 (on-screen rounded values are consistent)", f"{448.6 * 0.12 / 1.975:.1f}", "27.3")
    ck.eq("fB on screen", f"{F * B_BASE:.1f}", "53.8")
    ur, vr = int(round(UVR[0])), int(round(UVR[1]))
    ul, vl = int(round(UV[0])), int(round(UV[1]))
    ck.true("right-image white-dot pixel really renders onto the red box (depth ≈ D)", abs(TZ_R[vr, ur] - D) < 0.02, f"right-image z-depth {TZ_R[vr, ur]:.4f}, D={D:.4f}")
    ck.true("the 3D point of that right-image pixel is the white dot (scene frame, error < 1 cm)",
            np.linalg.norm(XS_R[vr, ur] - sc.X_STAR_S) < 0.01, f"{np.linalg.norm(XS_R[vr, ur] - sc.X_STAR_S):.4f} m")
    ck.true("the 3D point of that left-image pixel is the white dot", np.linalg.norm(XS_L[vl, ul] - sc.X_STAR_S) < 0.01)
    ck.true("pixels on both sides have the same color (red-box front face looks the same in both; the right-image dot really lies on this face)",
            np.array_equal(RGB_L[vl, ul], RGB_R[vr, ur]), f"{RGB_L[vl, ul]} vs {RGB_R[vr, ur]}")
    ck.close("horizontal distance between the two on-screen dots = Δu × magnification", PT_L[0] - PT_R[0], DISP * PS_L.s, 1e-9)
    ck.close("the two dots have the same vertical position within their strips", PT_L[1] - STRIP_Y[0], PT_R[1] - STRIP_Y[1], 1e-9)
    ck.true("white dot lies inside the cropped rows", V0 + 3 < UV[1] < V1 - 3)
    # ---- #7: δD
    ck.eq("δD(1 m) = 1.0 cm", f"{PTS_DD[0] * 100:.1f}", "1.0")
    ck.eq("δD(white dot) = 3.9 cm", f"{PTS_DD[1] * 100:.1f}", "3.9")
    ck.eq("δD(10 m) = 1.0 m", f"{PTS_DD[2]:.1f}", "1.0")
    ck.close("δD agrees with e02_numbers", PTS_DD, [NB["dZ"][1.0], NB["dZ"][D], NB["dZ"][10.0]], 1e-12)
    ck.close("δD ≈ D² δρ agrees with the numeric difference of 1/ρ (D=1 m, δρ=0.01: first-order error < 2%)",
             abs(1.0 / (1.0 - 0.01) - 1.0), 0.01, 2e-4)
    ck.close("curve end point = (10, 1.0)", (DD[-1], DDD[-1]), (10.0, 1.0), 1e-12)
    p = PD.to_screen([0, 10], [0, 1.1])
    ck.close("curve coordinates map to the plot-box corners", p.ravel(), [170, 1090, 990, 640], 1e-9)
    # ---- #8: paper Tab. 10
    ck.eq("three Tab. 10 numbers on screen", [f"{v:.3f}" for v in TAB10], ["0.730", "0.700", "0.657"])
    mdp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refs", "DepthPro_paper.md")  # optional: paper text converted to markdown
    if os.path.exists(mdp):
        with open(mdp, encoding="utf-8") as fh:
            L = fh.read().splitlines()
        i0 = next(i for i, s in enumerate(L) if s.startswith("Table 10:"))
        seg = L[i0:i0 + 40]
        got = {}
        for name in ("Inverse-depth", "Log-depth", "Depth"):
            j = next(i for i, s in enumerate(seg) if s.strip() == name)
            got[name] = float(seg[j + 1])
        ck.eq("Tab. 10, 0-1 m column in the paper markdown", [got["Inverse-depth"], got["Log-depth"], got["Depth"]], TAB10)
    ck.true("inverse depth > log depth > depth (higher is better)", TAB10[0] > TAB10[1] > TAB10[2])
    ck.record(D=round(D, 4), r=round(RNG, 4), rho=round(RHO, 4), disp=round(DISP, 4), u_left=round(float(UV[0]), 3),
              u_right=round(float(UVR[0]), 3), f=round(F, 3), B=B_BASE, strip_rows=[V0, V1],
              dD=[round(x, 4) for x in PTS_DD], tab10=TAB10)


# ------------------------------------------------------------------ drawing
def blank_ok(a):
    return a > 0.01


# ---- cross-section (#1-#2): side view of camera 2, the plane through the optical axis and the white dot
O_PT = np.array([150.0, 470.0])
PPM = 380.0                                         # pixels per meter
PLANE_DX = 70.0                                     # image plane distance from the center (position illustration)
FOOT = O_PT + [PPM * D, 0]
P_PT = O_PT + [PPM * D, -PPM * H_OFF]
DIR_RAY = (P_PT - O_PT) / np.linalg.norm(P_PT - O_PT)
IMG_PT = O_PT + DIR_RAY * (PLANE_DX / DIR_RAY[0])   # where the ray crosses the image plane


def section(cv, t):
    a = fade(t, 0.3, T[2] - 0.2, 0.5, 0.4)
    if not blank_ok(a):
        return
    ax_end = O_PT + [PPM * 2.4, 0]
    cv.s.line(O_PT, ax_end, CYAN, 0.7 * a, 2)
    cv.text((ax_end[0], ax_end[1] - 10), "Optical axis (camera direction)", 24, alpha=a, fill=style.LABEL_CYAN, anchor="rb")
    pl = np.array([O_PT[0] + PLANE_DX, O_PT[1]])
    cv.s.line(pl + [0, -62], pl + [0, 62], style.GREY, 0.9 * a, 3)
    cv.text((pl[0] + 8, pl[1] + 70), "Image plane (position illustration)", 22, alpha=a, fill=style.TEXT_DIM, anchor="la")
    cv.text((O_PT[0] - 4, O_PT[1] - 14), "center", 26, alpha=a, fill=c2.CAM2_COLOR, anchor="rb")
    cv.s.dot(O_PT, c2.CAM2_COLOR, 6, a)
    cv.g.dot(O_PT, c2.CAM2_COLOR, 8, 0.5 * a)
    ar = ss(t, 0.8, 1.8)
    tip = O_PT + (P_PT - O_PT) * ar
    cv.s.line(O_PT, tip, (205, 205, 205), 0.75 * a, 2)
    if ar > 0.5:
        cv.s.dot(IMG_PT, (255, 255, 255), 4, a)
    # perpendicular: the white dot projected onto the axis (z coordinate = position along the axis)
    ap = ss(t, 1.4, 2.0)
    cv.s.dashed(P_PT, P_PT + (FOOT - P_PT) * ap, style.GREY, 0.85 * a, 1, 10)
    sq = 12.0
    if ap > 0.9:
        cv.s.polyline([FOOT + [-sq, 0], FOOT + [-sq, -sq], FOOT + [0, -sq]], style.GREY, 0.8 * a, 1)
    ec.star_marker(cv, P_PT, a * ss(t, 1.0, 1.5), glow=0.8, rad=7)
    cv.text((P_PT[0] + 16, P_PT[1] - 6), "dot", 28, alpha=a * ss(t, 1.2, 1.7), anchor="lm")
    # #1: D measure line (gold, along the optical axis)
    aD = ss(t, 1.7, 2.3)
    if blank_ok(aD):
        measure(cv, FOOT, O_PT, r"$D$ = " + S_D + " m", GOLD, a * aD, offset=34, th=3, label_size=32)
        cv.g.line(O_PT + [0, 34], FOOT + [0, 34], GOLD, 0.35 * a * aD * (0.6 + 0.4 * bump(t, 2.0, 1.2)), 4)
    # #2: r measure line (cyan, along the ray)
    aR = ss(t, T[1] + 0.3, T[1] + 1.0)
    if blank_ok(aR):
        measure(cv, O_PT, P_PT, r"$r$ = " + S_RNG + " m", CYAN, a * aR, offset=40, th=3, label_size=32)


def section_text(cv, t):
    a = fade(t, 0.3, T[2] - 0.2, 0.5, 0.4)
    if not blank_ok(a):
        return
    a1 = a * ss(t, 2.3, 2.9)
    cv.rich("$D$ ($z$-depth): the point's coordinate along the optical axis", (70, 650), 32, alpha=a1, fill=style.LABEL_GOLD)
    a2 = a * ss(t, T[1] + 0.9, T[1] + 1.5)
    cv.rich("$r$ (range): distance to the point along the ray", (70, 700), 32, alpha=a2, fill=style.LABEL_CYAN)
    cv.rich("MapAnything in E01-08 uses $r$", (70, 746), 26, alpha=a * ss(t, T[1] + 1.4, T[1] + 2.0),
            fill=style.TEXT_DIM)
    # formula r = D ‖K^-1 p~‖, lit term by term; D gold, r cyan on screen
    ae = a * ss(t, T[1] + 1.3, T[1] + 1.8)
    if blank_ok(ae):
        F_R.draw(cv, (540, 850), alpha=ae, part_alpha={"r": ss(t, T[1] + 1.4, T[1] + 1.8), "D": ss(t, T[1] + 2.0, T[1] + 2.4),
                                                      "n": ss(t, T[1] + 2.6, T[1] + 3.0)},
                 part_color={"r": COL_R, "D": COL_D}, part_glow={"D": bump(t, T[1] + 2.2, 0.8),
                                                                 "n": bump(t, T[1] + 2.8, 0.8)})
    an = a * ss(t, T[1] + 3.0, T[1] + 3.5)
    if blank_ok(an):
        F_RN.draw(cv, (540, 950), alpha=an, part_color={"a": COL_R, "b": COL_D},
                  part_glow={"c": bump(t, T[1] + 3.4, 0.8)})
        cv.rich(r"$K$: intrinsic matrix; $\tilde p=(u,v,1)^{\mathsf{T}}$: homogeneous coordinates of the white dot pixel", (70, 1030), 26, alpha=an,
                fill=style.TEXT_DIM)
        cv.rich(r"$K^{-1}\tilde p$ = (" + f"{KP[0]:.3f}, {KP[1]:.3f}, {KP[2]:.0f}".replace("-", "−") + r"), length " + f"{KPN:.4f}" +
                ": the ray is a bit longer than the axis direction", (70, 1076), 26, alpha=an, fill=style.TEXT_DIM)


# ---- depth map / inverse-depth map (#3)
def maps(cv, t):
    a = fade(t, T[2], T[3] - 0.05, 0.4, 0.25)
    if not blank_ok(a):
        return
    u = ss(t, T[2] + 1.5, T[2] + 2.4)               # depth map -> inverse-depth map cross-fade
    PAN_D.draw(cv, a, border=style.GREY, border_alpha=0.6)
    if u > 0.01:
        PAN_R.draw(cv, a * u, border=style.GREY, border_alpha=0.6)
    cv.rich("Cam 2 depth map $D$", (PAN_XY[0] + PAN_W / 2, PAN_XY[1] - 14), 28, alpha=a * (1 - u), anchor="mb")
    cv.rich(r"Cam 2 inverse depth map $\rho$", (PAN_XY[0] + PAN_W / 2, PAN_XY[1] - 14), 28, alpha=a * u, anchor="mb")
    colorbar(cv, CB_BOX, D_LO, D_HI, unit="m", fmt="{:.2f}", alpha=a * (1 - u), size=24)
    colorbar(cv, CB_BOX, R_HI, R_LO, unit=r"m$^{-1}$", fmt="{:.2f}", alpha=a * u, size=24)   # colorbar after negation: top = near = large ρ
    cv.text((CB_BOX[2] + 12, (CB_BOX[1] + CB_BOX[3]) / 2), "Near: bright\nFar: dark", 24, alpha=a * ss(t, T[2] + 0.4, T[2] + 0.9),
            fill=style.TEXT_DIM, anchor="lm", spacing=6)
    # background = 0 (next to the inverse-depth colorbar)
    ab = a * ss(t, T[2] + 2.8, T[2] + 3.3)
    if blank_ok(ab):
        sw = (CB_BOX[0], CB_BOX[3] + 22)
        cv.s.rect(sw[0], sw[1], sw[0] + 24, sw[1] + 24, (28, 30, 38), ab)
        cv.s.polyline([(sw[0], sw[1]), (sw[0] + 24, sw[1]), (sw[0] + 24, sw[1] + 24), (sw[0], sw[1] + 24)],
                      style.GREY, ab, 1, closed=True)
        cv.rich(r"Background (no object): $\rho$ = 0", (sw[0] + 36, sw[1] + 12), 26, alpha=ab, fill=style.TEXT, anchor="lm")
    # white dot
    ast = a * ss(t, T[2] + 0.6, T[2] + 1.1)
    ec.star_marker(cv, PAN_PT, ast, glow=0.8, rad=6)
    # right side: ρ = 1/D and the definitions
    ar = a * ss(t, T[2] + 2.0, T[2] + 2.6)
    if blank_ok(ar):
        F_RHO.draw(cv, (880, 430), alpha=ar, part_color={"r": COL_R, "d": COL_D},
                   part_glow={"r": bump(t, T[2] + 2.4, 0.9)})
        cv.rich(r"$\rho$: inverse depth", (730, 520), 30, alpha=ar, fill=style.LABEL_CYAN)
        cv.text((730, 562), "inverse depth", 26, "light", alpha=ar, fill=style.TEXT_DIM)
        cv.rich("$D$: depth (m)", (730, 606), 28, alpha=ar, fill=style.LABEL_GOLD)
    # readouts below
    a1 = a * ss(t, T[2] + 0.9, T[2] + 1.4)
    cv.rich(r"White dot: $D$ = " + S_D + " m", (70, 905), 34, alpha=a1, fill=style.LABEL_GOLD)
    a2 = a * ss(t, T[2] + 1.2, T[2] + 1.8)
    cv.rich(r"White dot: $\rho=1/D=1/" + S_D + "=" + S_RHO + r"$ m$^{-1}$", (70, 960), 34, alpha=a2, fill=style.LABEL_CYAN)
    cv.rich(r"Whole image: $D\in[" + f"{D_LO:.2f},\\,{D_HI:.2f}" + r"]$ m → $\rho\in[" + f"{R_LO:.2f},\\,{R_HI:.2f}" +
            r"]$ m$^{-1}$", (70, 1020), 28, alpha=a * ss(t, T[2] + 1.6, T[2] + 2.2), fill=style.TEXT_DIM)


# ---- small 3D view: imaginary right camera (#4)
def stereo3d(cv, t):
    a = fade(t, T[3], T[4] - 0.2, 0.4, 0.4)
    if not blank_ok(a):
        return
    P = OC3.at(az=AZ3 + 3 * ss(t, T[3], T[4]))
    cl = P(ec.disp(C2W))
    cr = P(ec.disp(C_R))
    aR = ss(t, T[3] + 0.7, T[3] + 1.5)
    ec.frustum(cv, P, CAM, DEP3, a, tex_alpha=0.65, label=None)
    if blank_ok(aR):
        ec.frustum_pose(cv, P, R2, C_R, DEP3, a * aR, color=(235, 235, 235), image=RGB_R, tex_alpha=0.65, label=None)
    # white-dot direction: each camera's ray toward the white dot (almost parallel, only the first segment is drawn)
    for c0, dd_ in ((C2W, DIR_L), (C_R, DIR_R)):
        e = P(ec.disp(c0 + RAY_LEN * dd_))
        s0 = P(ec.disp(c0))
        cv.s.line(s0, e, (255, 255, 255), 0.55 * a * (1 if c0 is C2W else aR), 1)
    e_mid = P(ec.disp(0.5 * (C2W + C_R) + RAY_LEN * 0.5 * (DIR_L + DIR_R)))
    cv.text((e_mid[0] + 10, e_mid[1] - 4), "→ white dot (still 1.975 m away)", 24, alpha=a * aR, fill=style.TEXT, anchor="lm")
    # baseline
    ab = ss(t, T[3] + 1.5, T[3] + 2.2)
    if blank_ok(ab):
        cv.g.line(cl, cr, ORANGE, 0.6 * a * ab, 4)
        measure(cv, cr, cl, r"$B$ = " + f"{B_BASE:.2f}" + " m", ORANGE, a * ab, offset=30, th=4, label_size=32)
    cv.text(cl + [-12, 8], "Cam 2 (left)", 26, alpha=a, fill=c2.CAM2_COLOR, anchor="rm")
    cv.text(cr + [12, 8], "Virtual right camera", 26, alpha=a * aR, fill=style.TEXT, anchor="lm")
    # text
    t1 = a * ss(t, T[3] + 1.8, T[3] + 2.4)
    cv.rich("$B$ (baseline): distance between the two camera centers", (70, 985), 30, alpha=t1, fill=style.LABEL_ORANGE)
    t2 = a * ss(t, T[3] + 2.3, T[3] + 2.9)
    cv.rich("Right camera = cam 2 shifted by $B$ along its own $x$ axis; orientation and intrinsics unchanged", (70, 1035), 28, alpha=t2)
    t3 = a * ss(t, T[3] + 2.8, T[3] + 3.4)
    cv.rich("$B$ = 0.12 m is an arbitrary illustrative value, not a real camera", (70, 1082), 26, alpha=t3, fill=style.TEXT_DIM)


# ---- left/right images stacked + disparity + formula (#5-#6)
def strips(cv, t):
    a = fade(t, T[4], T[6] - 0.2, 0.5, 0.4)
    if not blank_ok(a):
        return
    aL = ss(t, T[4] + 0.1, T[4] + 0.7)
    aRr = ss(t, T[4] + 0.9, T[4] + 1.6)
    PS_L.draw(cv, a * aL, border=style.GREY, border_alpha=0.7)
    PS_R.draw(cv, a * aRr, border=style.GREY, border_alpha=0.7)
    cv.badge((STRIP_X + 8, STRIP_Y[0] + 8), "Left: cam 2", 22, a * aL, anchor="la")
    cv.badge((STRIP_X + 8, STRIP_Y[1] + 8), "Right: imagined right camera, same scene re-rendered", 22, a * aRr, anchor="la")
    cv.text((STRIP_X + STRIP_W, STRIP_Y[1] + (V1 - V0) * PS_R.s + 6), f"Only the {V1 - V0} rows around the white dot", 20,
            alpha=a * aRr, fill=style.TEXT_DIM, anchor="ra")
    ad = a * ss(t, T[4] + 1.5, T[4] + 2.1)
    # column of the left-image white dot: a vertical dashed line through both images; the right-image dot is shifted left
    cv.s.dashed((PT_L[0], STRIP_Y[0]), (PT_L[0], STRIP_Y[1] + (V1 - V0) * PS_R.s), style.WHITE, 0.6 * ad, 1, 28)
    ec.star_marker(cv, PT_L, a * aL, glow=0.5, rad=6)
    ec.star_marker(cv, PT_R, a * aRr, glow=0.5, rad=6)
    # disparity line: from the right-image dot column u_R to the left-image dot column u_L (orange, horizontal)
    pu = PS_R.to_screen(UV[0], UV[1] - V0)
    seg0, seg1 = np.array([PT_R[0], PT_R[1] + 16]), np.array([pu[0], PT_R[1] + 16])
    dg = ss(t, T[4] + 2.0, T[4] + 2.7)
    if blank_ok(dg):
        glow = 0.7 * bump(t, T[5] + 1.3, 1.0) if t > T[5] else 0.0
        measure(cv, seg0, seg1, "", ORANGE, a * dg, offset=0, th=4, tick=9)
        cv.g.line(seg0, seg1, ORANGE, (0.5 + glow) * a * dg, 4)
        cv.rich(r"$\Delta u$", ((seg0[0] + seg1[0]) / 2, seg0[1] + 14), 28, alpha=a * dg, fill=style.LABEL_ORANGE,
                anchor="ma")
    # #5 readouts and definitions
    a5 = a * (1 - ss(t, T[5] - 0.2, T[5] + 0.2))
    r1 = a5 * ss(t, T[4] + 1.0, T[4] + 1.6)
    cv.rich(r"$u_L$ = " + S_UL + r" px: x coordinate of the white dot in the left image", (70, 740), 30, alpha=r1)
    cv.rich(r"$u_R$ = " + S_UR + r" px: x coordinate of the white dot in the right image", (70, 785), 30, alpha=r1)
    r2 = a5 * ss(t, T[4] + 2.2, T[4] + 2.8)
    cv.rich(r"Disparity $\Delta u=u_L-u_R$ = " + S_DU2 + " ≈ " + S_DISP + " px", (70, 840), 34, alpha=r2,
            fill=style.LABEL_ORANGE)
    cv.rich(r"Disparity $\Delta u$: x-coordinate difference of the same point between left and right images (px)", (70, 890), 28, alpha=a5 * ss(t, T[4] + 2.8, T[4] + 3.4),
            fill=style.TEXT_DIM)
    # #6 formula
    a6 = a * ss(t, T[5] + 0.2, T[5] + 0.7)
    if blank_ok(a6):
        t6 = T[5]
        pa = {"du": ss(t, t6 + 0.2, t6 + 0.6), "f": ss(t, t6 + 0.9, t6 + 1.3), "B": ss(t, t6 + 1.0, t6 + 1.4),
              "D": ss(t, t6 + 1.1, t6 + 1.5), "f2": ss(t, t6 + 2.3, t6 + 2.7), "B2": ss(t, t6 + 2.3, t6 + 2.7),
              "rho": ss(t, t6 + 2.4, t6 + 2.8)}
        F_DU.draw(cv, (540, 790), alpha=a6, part_alpha=pa,
                  part_color={"du": ORANGE, "f": COL_F, "B": COL_B, "D": COL_D, "f2": COL_F, "B2": COL_B,
                              "rho": COL_R},
                  part_glow={"du": bump(t, t6 + 0.5, 0.8), "rho": bump(t, t6 + 2.7, 0.9)})
        r_ = ss(t, t6 + 1.3, t6 + 1.8)
        cv.rich(r"$f$: focal length (px) = " + f"{F:.1f}", (70, 880), 30, alpha=a6 * r_, fill=(200, 175, 255))
        cv.rich(r"$B$: baseline (m) = " + f"{B_BASE:.2f}", (560, 880), 30, alpha=a6 * r_, fill=style.LABEL_ORANGE)
        cv.rich(r"$D$: depth (m) = " + S_D, (70, 928), 30, alpha=a6 * r_, fill=style.LABEL_GOLD)
        cv.rich(r"$\rho=1/D$ = " + S_RHO + r" m$^{-1}$: inverse depth", (560, 928), 30, alpha=a6 * ss(t, t6 + 2.4, t6 + 2.9),
                fill=style.LABEL_CYAN)
        an = a6 * ss(t, t6 + 1.7, t6 + 2.2)
        F_DUN.draw(cv, (540, 1030), alpha=an, part_color={"a": ORANGE, "f": COL_F, "B": COL_B, "D": COL_D})
        ac = a6 * ss(t, t6 + 3.0, t6 + 3.6)
        cv.rich(r"$fB$ = " + f"{F * B_BASE:.1f}" + r" px·m, same for all points: $\Delta u$ is proportional to $\rho$", (70, 1110), 30,
                alpha=ac, fill=style.TEXT)
        cv.rich(r"Near: large $\rho$, large disparity; far: small $\rho$, small disparity", (70, 1160), 28, alpha=ac * ss(t, t6 + 3.4, t6 + 3.9),
                fill=style.TEXT_DIM)


# ---- dD curve (#7)
def curve_scene(cv, t):
    a = fade(t, T[6], T[7] - 0.15, 0.4, 0.4)
    if not blank_ok(a):
        return
    af = a * ss(t, T[6] + 0.1, T[6] + 0.6)
    F_DD.draw(cv, (540, 345), alpha=af, part_color={"dd": COL_D, "dr": COL_R},
              part_glow={"dd": bump(t, T[6] + 0.6, 0.8), "dr": bump(t, T[6] + 1.0, 0.8)})
    cv.rich(r"$\delta\rho$: inverse-depth error, here 0.01 m$^{-1}$", (70, 420), 28, alpha=af, fill=style.LABEL_CYAN)
    cv.rich(r"$\delta D$: resulting depth error (m)", (70, 464), 28, alpha=af, fill=style.LABEL_GOLD)
    cv.rich(r"Differentiating $D=1/\rho$: $\delta D\approx D^{2}\,\delta\rho$", (70, 508), 26, alpha=af * ss(t, T[6] + 0.8, T[6] + 1.3),
            fill=style.TEXT_DIM)
    PD.axes(cv, a)
    PD.curve(cv, DD, DDD, CYAN, ss(t, T[6] + 0.8, T[6] + 2.4), a)
    cols = [(255, 255, 255), GOLD, ORANGE]
    names = [("1 m: 1.0 cm", (0, -56)), ("White dot 1.975 m: 3.9 cm", (90, -150)), ("10 m: 1.0 m", (-26, 0))]
    for k, (Z, dz) in enumerate(zip(PTS_D, PTS_DD)):
        ak = a * ss(t, T[6] + 1.8 + 0.8 * k, T[6] + 2.3 + 0.8 * k)
        if not blank_ok(ak):
            continue
        p = PD.to_screen(Z, dz)
        base = PD.to_screen(Z, 0)
        cv.s.dashed(p, base, cols[k], 0.6 * ak, 1, 12)
        PD.scatter(cv, [Z], [dz], cols[k], 1.0, ak, rad=8)
        off = np.array(names[k][1], float)
        lab = p + off
        if k < 2:
            cv.s.line(p + [0, -10], lab + [0, 16], cols[k], 0.5 * ak, 1)
        cv.rich(names[k][0], lab, 30, alpha=ak, fill=cols[k] if k else style.TEXT, anchor="mb" if k < 2 else "rm")


# ---- paper Tab. 10 (#8)
def tab10(cv, t):
    a = fade(t, T[7], DUR + 0.5, 0.4, 0.0)
    if not blank_ok(a):
        return
    cv.rich(r"Depth Pro paper Tab. 10: Hypersim, $\delta_1$ for the 0–1 m range (higher is better)", (540, 330), 30, alpha=a,
            anchor="mb", align="m")
    bar_chart(cv, (340, 370, 980, 650), TAB10, TAB10_LABELS, vmax=0.8, horizontal=True, highlight=0,
              progress=ss(t, T[7] + 0.5, T[7] + 2.2), fmt="{:.3f}", alpha=a, value_size=30, label_size=28,
              note="Paper Tab. 10 (Appendix B.4)")
    r1 = a * ss(t, T[7] + 1.8, T[7] + 2.4)
    cv.rich(r"$\delta_1$: fraction of pixels whose prediction/GT ratio lies between 1/1.25 and 1.25", (70, 740), 30, alpha=r1)
    r2 = a * ss(t, T[7] + 2.6, T[7] + 3.2)
    cv.rich("Same network structure; only the output representation changes; trained with MAE only", (70, 792), 30, alpha=r2)
    cv.rich("MAE (Mean Absolute Error):", (70, 842), 26, alpha=r2, fill=style.TEXT_DIM)
    cv.rich("mean of per-pixel |prediction − GT|; lower is better", (70, 880), 26, alpha=r2, fill=style.TEXT_DIM)
    r3 = a * ss(t, T[7] + 3.4, T[7] + 4.0)
    cv.rich("Inverse depth is highest in 0–1 m: most accurate up close", (70, 960), 34, alpha=r3, fill=style.LABEL_GOLD)


@shot.frame
def frame(cv, t):
    section(cv, t)
    section_text(cv, t)
    maps(cv, t)
    stereo3d(cv, t)
    strips(cv, t)
    curve_scene(cv, t)
    tab10(cv, t)
