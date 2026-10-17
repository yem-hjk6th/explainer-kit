"""E02-07 DA3 1: depth-ray -- one point = ray origin + depth × direction (M, 49 s). Storyboard: storyboard.md §E02-07.
Computed (corner tag "Scene GT"): white dot K^-1 p~ = (-0.043, 0.084, 1), d = R K^-1 p~ = (0.435, 0.111, 0.899), |d| = 1.004,
  P = t + D d = (-0.300, 0.139, 2.075) = white dot; MapAnything unit ray × along-ray depth 1.984 and d × z-depth 1.975 reach the same endpoint;
  Eq. 1 origin average = camera 2 center; Eq. 2 DLT (14×14 = 196 pixels, 392×9) + RQ recovers K (448.6, 259) and rotation 30.0°.
Measured (corner tag "Measured on our test scene"): #11 runs our own DLT + RQ on the ray map output by DA3-Large (288×288, 856 pixels):
  f_x = 426.4, f_y = 423.1 (GT 436.5, in 504-pixel units), camera 2 rotation 30.1°.
Paper numbers: #12 the camera head takes about 0.1% of the backbone compute (p.7, §7.2.1); #13 Tab. 6 Auc3 (HiRoom 48.7 / 10.8, ETH3D 25.5 / 9.9).
Illustration: the block diagram of #1; the structure diagram of #12.
"""
import numpy as np

import e02_common as c2
import e02_numbers as num
from animkit import (CYAN, GOLD, ORANGE, VIOLET, WHITE, Formula, ImagePanel, Popup, Shot, arrow, bar_chart, bump, colorize,
                     fade, input_slot, matrix, measure, module_box, number, popup_check, ss, style, table, vector)

ec, sc = c2.ec, c2.sc
NB = c2.NB()
EN = ec.nb()
ML = c2.metrics()["da3_large"]["A_nopose_camhead"]["ray"]

DUR = 55.6
SH = 2.6                                          # DLT popup extended to 6 s: content after t >= 36.6 shifts back by SH
SH2, TQ = 2.6, 45.5                               # during the quaternion popup the main view holds again at content time TQ ($q$ already on screen) for SH2
shot = Shot("s07_da3_ray", DUR)
shot.title("Rays plus depth", "E02-07 · Depth Anything 3 · part 1 · depth-ray", 0.2, 3.6)

CAPS = [(0.4, 3.8, "Any number of images in; depth and rays out"),
        (3.9, 7.3, "Ray map: 6 numbers per pixel"),
        (7.5, 10.9, "Direction d is not normalized"),
        (11.0, 15.0, "P equals t plus D times d"),
        (15.1, 18.4, "d has z = 1, so D is the z-depth"),
        (18.5, 22.4, "MapAnything splits it differently; same point"),
        (22.6, 26.6, "Whole image: shapes and element-wise operations"),
        (27.0, 30.0, "Reverse: average the camera center"),
        (30.1, 34.0, "Directions: solve a matrix with DLT"),
        (36.7, 40.7, "RQ decomposition splits out K and R"),
        (41.4, 45.4, "Real model rays solve back too"),
        (45.6, 51.1, "A lightweight camera head is added and trained jointly"),
        (51.3, 55.3, "Depth + rays only: more accurate pose")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

shot.tag(c2.GT_TAG, 0.3, 40.8)
shot.tag("Paper's order; the official code stores direction first", 3.9, 7.4)
shot.tag(c2.DERIV_TAG + ": from §3.1's $d=RK^{-1}\\tilde p$", 36.7, 40.8)
shot.tag(c2.SCENE_TAG, 41.4, 45.6)
shot.tag("Paper p.7, §7.2.1", 45.7, 51.2)
shot.tag("Paper Tab. 6", 51.3, DUR)

# ---------------------------------------------------------------- Numbers
KP, DV, DN = NB["da3_kp"], NB["da3_d"], NB["da3_dnorm"]
TV, PV, RNG = NB["da3_t"], NB["da3_P"], NB["da3_range"]
MA, DD, TC = NB["ma_unit"], NB["D"], NB["tc"]
R_W2C = EN["R"]
UV = EN["uv"]
W = sc.W
TAN = float(np.hypot(KP[0], KP[1]))                 # tan of the white dot ray's deviation from the optical axis (lateral distance on the z = 1 plane)

_PIX = np.array([[u, v, 1.0] for v in np.arange(0, W, 37) for u in np.arange(0, W, 37)])
_DIRS = _PIX @ np.linalg.inv(sc.K).T @ R_W2C                 # d = R_c2w K^-1 p~
Kh, Rh = num.rq(np.linalg.inv(num.dlt_rays(_PIX, _DIRS)))
if np.linalg.det(Rh) < 0:
    Rh, Kh = -Rh, -Kh
Kh = Kh / Kh[2, 2]
ROT_DEG = float(np.degrees(np.arccos((np.trace(Rh) - 1) / 2)))
N_PIX = 196
F504 = sc.FX * 504 / W                              # GT focal length converted to 504-pixel units
K_M = np.array(ML["K_hat_cam2_504"])
FX_M, FY_M = float(K_M[0, 0]), float(K_M[1, 1])
ROT_M = float(ML["cam2_rot_vs_cam1_deg"])
ROT_ERR_M = float(ML["rot_err_vs_truth_deg"])
REL_X, REL_Y = FX_M / F504 - 1, FY_M / F504 - 1


KNORM = float(np.linalg.norm(KP))
KN_TXT = f"$r=D\\,\\Vert K^{{-1}}\\tilde p\\Vert$ = {DD:.3f} × {KNORM:.4f} = {RNG:.3f} m (same as E02-02)"


def v3(v, nd=3):
    return ", ".join(f"{x:.{nd}f}" for x in v)


def sv(v, nd=3):                                    # vector in non-formula text (minus sign uses "−")
    return c2.fmt_vec(v, nd)


# ---------------------------------------------------------------- Images
def ray_images(n=200):
    g = (np.arange(n) + 0.5) * W / n - 0.5
    uu, vv = np.meshgrid(g, g)
    kp = np.stack([(uu - sc.CX) / sc.FX, (vv - sc.CY) / sc.FY, np.ones_like(uu)], -1)
    d = kp @ R_W2C                                   # = R_c2w K^-1 p~
    lo, hi = d.reshape(-1, 3).min(0), d.reshape(-1, 3).max(0)
    dirimg = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
    col = np.clip((TV + 1.5) / 3.0, 0, 1) * 255
    orgimg = np.broadcast_to(col.astype(np.uint8), (n, n, 3)).copy()
    return orgimg, dirimg


ORG_IMG, DIR_IMG = ray_images()
_, TZ2, _ = sc.full(sc.EX_CAM)
DEPTH_IMG = colorize(TZ2, float(np.nanmin(TZ2[np.isfinite(TZ2)])), float(np.nanmax(TZ2[np.isfinite(TZ2)])))
IN_PANELS = [ImagePanel(ec.view_rgb(i), (50, 372 + 190 * i), w=150) for i in range(3)]
P_DEPTH = ImagePanel(DEPTH_IMG, (570, 372), w=150)
P_RAY = ImagePanel(DIR_IMG, (570, 612), w=150)
P_RAY2 = ImagePanel(DIR_IMG, (60, 380), w=280)       # #2 large image
P_ORG = ImagePanel(ORG_IMG, (90, 640), w=200)
P_DIR = ImagePanel(DIR_IMG, (390, 640), w=200)
P_CAM = ImagePanel(ec.view_rgb(sc.EX_CAM), (60, 350), w=220)

# ---------------------------------------------------------------- Formulas
EQ4 = Formula([("P=", "P"), ("t", "t"), (r"+D(u,v)\cdot", "D"), ("d", "d")], size=64, gap=0.12)
EQ7 = Formula([("P=", "P"), (r"\hat{M}[\ldots,{:}3]", "t"), (r"+\hat{D}\odot", "D"), (r"\hat{M}[\ldots,3{:}]", "d")],
              size=48, gap=0.12)
EQ8 = Formula([("t_c=", "tc"), (r"\frac{1}{HW}\sum_{h,w}", "avg"), (r"M(h,w,{:}3)", "M")], size=56, gap=0.12)
EQ9 = Formula([(r"H^{*}=\arg\min_{\Vert H\Vert=1}\ \sum_{h,w}\Vert H\tilde{p}\times M(h,w,3{:})\Vert", "e")], size=38)
COL_T, COL_D, COL_DEP = ORANGE, GOLD, CYAN
TXT_T, TXT_D, TXT_DEP = style.LABEL_ORANGE, style.LABEL_GOLD, style.LABEL_CYAN

# ---------------------------------------------------------------- Popups
POP_DLT = Popup((40, 620, 1000, 620), t_in=30.1, t_out=36.3, title="DLT (direct linear transform)", corner="bl", accent=ORANGE)
POP_Q = Popup((40, 1010, 1000, 200), t_in=48.0, t_out=50.7, title="Quaternion", corner="bl", accent=VIOLET)
POP_AUC = Popup((40, 300, 1000, 340), t_in=51.1, t_out=DUR - 0.5, title="Auc3: pose accuracy", corner="tl", accent=CYAN)


def body_dlt(sub, lt, box):
    a = ss(lt, 0.3, 0.7)
    EQ9.draw(sub, (480, 90), alpha=a, anchor="mm", color=style.MATH)
    sub.rich(r"Eq. 2 · $\tilde{p}=(u,v,1)$: pixel; $M(h,w,3{:})$ = $d$: ray direction of that pixel", (480, 150), 21,
             alpha=a, fill=style.TEXT_DIM, anchor="mm", align="m")
    a2 = ss(lt, 0.9, 1.3)
    sub.rich(r"Cross product = 0 means $H\tilde{p}$ and $d$ are parallel (point the same way)", (480, 215), 28, alpha=a2, fill=style.LABEL_GOLD,
             anchor="mm", align="m")
    a3 = ss(lt, 1.5, 1.9)
    sub.rich("Each pixel gives 2 linear equations; " + f"{N_PIX} pixels stack into", (40, 290), 27, alpha=a3, fill=style.TEXT, anchor="lm")
    a4 = ss(lt, 1.9, 2.3)
    if a4 > 0.01:                                     # long 392 × 9 matrix (schematic scale)
        x0, y0 = 70, 330
        sub.s.rect(x0, y0, x0 + 90, y0 + 190, (30, 40, 56), 0.9 * a4, r=4)
        sub.s.polyline([(x0, y0), (x0 + 90, y0), (x0 + 90, y0 + 190), (x0, y0 + 190)], style.GREY, a4, 2, closed=True)
        for k in range(1, 9):
            sub.s.line((x0 + 10 * k, y0), (x0 + 10 * k, y0 + 190), style.GREY, 0.3 * a4, 1)
        sub.rich(r"$A$: $392\times9$", (x0 + 45, y0 + 212), 26, alpha=a4, fill=style.LABEL_ORANGE, anchor="mm")
        sub.rich(f"= {N_PIX} × 2 rows, 9 columns = the 9 unknowns of $H$", (200, y0 + 40), 24, alpha=a4, fill=style.TEXT_DIM,
                 anchor="lm")
    a5 = ss(lt, 2.4, 2.8)
    sub.rich(r"SVD (singular value decomposition): write $A$ ($392\times9$) as $U\Sigma V^\top$,", (200, 430), 22, alpha=a5,
             fill=style.TEXT, anchor="lm")
    sub.rich(r"the diagonal entries of $\Sigma$ are the singular values; the right singular vector of the smallest one", (200, 462), 22, alpha=a5,
             fill=style.TEXT, anchor="lm")
    sub.rich(r"= the unit vector $h$ minimizing $\Vert Ah\Vert$ → its 9 numbers form $H$ ($3\times3$)", (200, 494), 22, alpha=a5,
             fill=style.TEXT, anchor="lm")
    sub.rich(r"Goal: minimize $\sum$ subject to $\Vert H\Vert=1$ (excluding the all-zero solution)", (200, 528), 20, alpha=a5, fill=style.TEXT_DIM,
             anchor="lm")


def body_q(sub, lt, box):
    sub.rich("4 numbers (unit length) represent a 3D rotation", (40, 110), 30, alpha=ss(lt, 0.3, 0.7), fill=style.TEXT, anchor="lm")


def body_auc(sub, lt, box):
    a = ss(lt, 0.3, 0.7)
    sub.rich("Auc3: for each image pair, take the larger of the rotation error and translation-direction error (both angles)", (40, 90), 25, alpha=a, fill=style.TEXT, anchor="lm")
    a2 = ss(lt, 1.0, 1.4)
    sub.rich("Accuracy = share of pairs with error below threshold;", (40, 150), 25, alpha=a2, fill=style.LABEL_CYAN, anchor="lm")
    sub.rich("sweep the threshold over 0–3°, take the area under the curve; full score 100, higher is better", (40, 210), 25, alpha=ss(lt, 1.8, 2.2),
             fill=style.LABEL_GOLD, anchor="lm")


# ---------------------------------------------------------------- Checks
@shot.check
def check(ck):
    ck.eq("K^-1 p~ on screen", v3(KP), "-0.043, 0.084, 1.000")
    ck.eq("d on screen", v3(DV), "0.435, 0.111, 0.899")
    ck.eq("|d| on screen", f"{DN:.3f}", "1.004")
    ck.eq("t on screen", v3(TV), "-1.159, -0.080, 0.300")
    ck.eq("P on screen", v3(PV), "-0.300, 0.139, 2.075")
    ck.eq("D on screen", f"{DD:.3f}", "1.975")
    ck.eq("along-ray distance on screen", f"{RNG:.3f}", "1.984")
    ck.eq("‖K^-1 p~‖ on screen", f"{KNORM:.4f}", "1.0044")
    ck.true("r = D‖K^-1 p~‖ on-screen string", "1.975 × 1.0044 = 1.984 m" in KN_TXT, KN_TXT)
    ck.close("r = D‖K^-1 p~‖", DD * KNORM, RNG, 1e-3)
    ck.eq("MapAnything unit ray on screen", v3(MA), "-0.043, 0.083, 0.996")
    ck.eq("Eq.1 origin average on screen", v3(TC), "-1.159, -0.080, 0.300")
    ck.close("d = R_c2w K^-1 p~", DV, R_W2C.T @ KP, 1e-12)
    ck.close("P = t + D d = white dot", TV + DD * DV, EN["X_star"], 1e-9)
    ck.close("origin average = camera 2 center t", TC, TV, 1e-9)
    ck.close("unit ray × along-ray distance = K^-1 p~ × D", MA * RNG, KP * DD, 1e-9)
    ck.close("unit ray length 1", np.linalg.norm(MA), 1.0, 1e-12)
    ck.close("z component of d in the camera frame = 1", (R_W2C @ DV)[2], 1.0, 1e-12)
    ck.eq("DLT pixel count", N_PIX, NB["dlt_n"])
    ck.eq("392 = 2 × 196", 2 * N_PIX, 392)
    ck.close("DLT + RQ: K̂ = GT K", Kh, sc.K, 1e-6)
    ck.close("NB K_hat = GT K", NB["K_hat"], sc.K, 1e-6)
    ck.eq("K̂ on screen f / c", (f"{Kh[0, 0]:.1f}", f"{Kh[1, 1]:.1f}", f"{Kh[0, 2]:.0f}", f"{Kh[1, 2]:.0f}"),
          ("448.6", "448.6", "259", "259"))
    ck.eq("rotation angle on screen 30.0", f"{ROT_DEG:.1f}", "30.0")
    ck.close("R̂ = GT R", Rh, R_W2C, 1e-8)
    ck.eq("measured f_x / f_y on screen", (f"{FX_M:.1f}", f"{FY_M:.1f}"), ("426.4", "423.1"))
    ck.eq("GT f (504 px) on screen", f"{F504:.1f}", "436.5")
    ck.eq("relative error on screen", (f"{REL_X * 100:.1f}", f"{REL_Y * 100:.1f}"), ("-2.3", "-3.1"))
    ck.eq("measured rotation on screen 30.1", f"{ROT_M:.1f}", "30.1")
    ck.true("rotation error within 0.1–0.2°", 0.1 <= ROT_ERR_M <= 0.2, f"{ROT_ERR_M:.3f}")
    ck.eq("measured DLT pixel count", ML["dlt_n"], 856)
    ck.eq("ray map shape 3×288×288×6", tuple(ML["shape"]), (3, 288, 288, 6))
    ck.record(da3_d=DV.round(4), rot=round(ROT_DEG, 3), f_meas=[round(FX_M, 2), round(FY_M, 2)], rot_meas=round(ROT_M, 3))
    ck.true("quaternion popup ≥ 2.5 s", POP_Q.t_out - POP_Q.t_in >= 2.5, "")
    popup_check(ck, [POP_DLT, POP_Q, POP_AUC], DUR)


# ---------------------------------------------------------------- Drawing
S_O = np.array([90.0, 560.0])
PPU = 400.0


def zpt(z):                                          # camera-frame section: the point on the ray at z-depth z
    return np.array([S_O[0] + z * PPU, S_O[1] - z * TAN * PPU])


def axpt(z):
    return np.array([S_O[0] + z * PPU, S_O[1]])


def section(cv, a, plane_a=1.0):
    """Camera-frame section (optical axis z points right; the plane containing the ray): optical axis + z = 1 plane. Angles and lengths are true."""
    if a <= 0.01:
        return
    cv.s.line(S_O, axpt(2.3), style.GREY, 0.7 * a, 1)
    cv.rich("Optical axis $z$", (axpt(2.3)[0], S_O[1] + 14), 22, alpha=a, fill=style.TEXT_DIM, anchor="ra")
    pa = a * plane_a
    cv.s.line((axpt(1.0)[0], S_O[1] - 110), (axpt(1.0)[0], S_O[1] + 50), style.GREY, 0.8 * pa, 2)
    cv.rich("$z=1$ plane", (axpt(1.0)[0], S_O[1] - 124), 22, alpha=pa, fill=style.TEXT_DIM, anchor="mb")


def ph(t, a, b, fi=0.35, fo=0.35):
    return fade(t, a, b, fi, fo)


def row_vec(cv, y, label, vals, a, color=style.MATH, lab_color=style.TEXT, nd=3, x=60, xv=330, size=34, extra=None):
    if a <= 0.01:
        return
    cv.rich(label, (x, y), 28, alpha=a, fill=lab_color, anchor="lm")
    vector(cv, (xv, y), vals, f"{{:.{nd}f}}", size, color, a, "lm")
    if extra:
        cv.rich(extra, (xv + 400, y), 24, alpha=a, fill=style.TEXT_DIM, anchor="lm")


@shot.frame
def frame(cv, t_real):
    t = min(t_real, 33.0) if t_real < 36.6 else t_real - SH     # content time axis; the main view holds during the DLT popup
    if t > TQ:
        t = max(TQ, t - SH2)                                     # holds once more during the quaternion popup
    dm = 1.0 - 0.5 * max(POP_DLT.on(t_real), POP_AUC.on(t_real))     # POP_Q does not dim (occupies only the empty bottom area)

    # ================= #1 input / output =================
    a1 = ph(t, 0.3, 4.0, 0.3, 0.4)
    if a1 > 0.01:
        cv.rich("Input: $N$ images", (50, 336), 28, alpha=a1 * ss(t, 0.4, 0.8), fill=style.TEXT, anchor="lm")
        for i, p in enumerate(IN_PANELS):
            p.draw(cv, a1 * ss(t, 0.5 + 0.2 * i, 0.9 + 0.2 * i), border=ec.CAM_COLORS[i])
        cv.rich("$N=1$ is monocular", (50, 944), 22, alpha=a1 * ss(t, 1.1, 1.5), fill=style.TEXT_DIM, anchor="lm")
        input_slot(cv, (50, 990, 280, 1060), "Optional: camera pose", optional=True, on=0.0, alpha=a1 * ss(t, 1.3, 1.7), size=24)
        cv.rich("(covered in E02-08)", (165, 1084), 20, alpha=a1 * ss(t, 1.3, 1.7), fill=style.TEXT_DIM, anchor="mm")
        mb = module_box(cv, (310, 560, 470, 760), "DA3", "ViT + Dual-DPT", CYAN, a1 * ss(t, 1.4, 1.8), title_size=34,
                        sub_size=18)
        for y in (450, 690):
            arrow(cv, (mb["r"][0] + 4, 660), (545, y), WHITE, a1 * ss(t, 1.8, 2.2), 2, 10)
        cv.rich("Output (one set per image)", (560, 336), 26, alpha=a1 * ss(t, 1.8, 2.2), fill=style.TEXT, anchor="lm")
        P_DEPTH.draw(cv, a1 * ss(t, 2.0, 2.4), border=CYAN)
        cv.rich("$\\hat D$: depth, $H\\times W$\n$H,W$: image height, width", (740, 450), 25, alpha=a1 * ss(t, 2.2, 2.6),
                fill=style.LABEL_CYAN, anchor="lm")
        P_RAY.draw(cv, a1 * ss(t, 2.4, 2.8), border=GOLD)
        cv.rich(r"$\hat M$: ray map, $H\times W\times6$", (740, 690), 25, alpha=a1 * ss(t, 2.6, 3.0),
                fill=style.LABEL_GOLD, anchor="lm")
        input_slot(cv, (560, 880, 1030, 960), "Camera: translation, rotation, FoV", optional=True, on=0.0,
                   alpha=a1 * ss(t, 3.0, 3.4), size=24)
        cv.rich("(optional output, gray)", (795, 990), 20, alpha=a1 * ss(t, 3.0, 3.4), fill=style.TEXT_DIM, anchor="mm")

    # ================= #2 ray map: 6 numbers per pixel =================
    a2 = ph(t, 3.9, 7.6, 0.4, 0.4)
    if a2 > 0.01:
        P_RAY2.draw(cv, a2, border=GOLD)
        cv.rich(r"Cam 2 ray map $M$ ($H\times W\times6$)", (60, 350), 22, alpha=a2, fill=style.LABEL_GOLD,
                anchor="lm")
        wp = P_RAY2.to_screen(UV[0] * 200 / W, UV[1] * 200 / W)
        ec.star_marker(cv, wp, a2 * ss(t, 4.2, 4.6), glow=0.7, rad=6)
        cv.rich(f"The 6 numbers of the white dot pixel $(u,v)=({UV[0]:.1f},\\ {UV[1]:.1f})$ in this map", (700, 380), 25, alpha=a2 * ss(t, 4.3, 4.7),
                fill=style.TEXT, anchor="mm")
        vals = list(TV) + list(DV)
        cx0, cw, cy = 70, 150, 760
        for k in range(6):
            ak = a2 * ss(t, 4.8 + 0.12 * k, 5.2 + 0.12 * k)
            col = COL_T if k < 3 else COL_D
            x0 = cx0 + cw * k
            cv.s.rect(x0 + 4, cy - 40, x0 + cw - 4, cy + 40, tuple(c * 0.25 for c in col), 0.9 * ak, r=8)
            cv.s.polyline([(x0 + 4, cy - 40), (x0 + cw - 4, cy - 40), (x0 + cw - 4, cy + 40), (x0 + 4, cy + 40)], col,
                          0.9 * ak, 2, closed=True)
            number(cv, (x0 + cw / 2, cy), vals[k], "{:.3f}", 34, style.MATH, ak, "mm")
        if a2 * ss(t, 4.6, 5.0) > 0.01:
            cv.s.line(wp, (cx0 + 3 * cw, cy - 44), GOLD, 0.5 * a2 * ss(t, 4.6, 5.0), 1)
        a_b = a2 * ss(t, 5.7, 6.1)
        if a_b > 0.01:
            for lo, hi, col, lab in [(0, 3, COL_T, "First 3 slots: origin $t$ (camera center)"),
                                     (3, 6, COL_D, "Last 3 slots: direction $d$")]:
                xa, xb = cx0 + cw * lo + 8, cx0 + cw * hi - 8
                cv.s.polyline([(xa, cy + 52), (xa, cy + 62), (xb, cy + 62), (xb, cy + 52)], col, a_b, 2)
                cv.rich(lab, ((xa + xb) / 2, cy + 92), 25, alpha=a_b,
                        fill=style.LABEL_ORANGE if lo == 0 else style.LABEL_GOLD, anchor="mm")
        a_c = a2 * ss(t, 6.2, 6.6)
        cv.rich("$3+3=6$: two parts concatenated, not multiplied", (540, 920), 28, alpha=a_c, fill=style.TEXT, anchor="mm")
        cv.rich("$t$: where the ray starts; $d$: where it points (cam 2 → world frame)", (540, 975), 23, alpha=a_c, fill=style.TEXT_DIM,
                anchor="mm")
        cv.rich("World frame = cam 1's camera frame", (540, 1025), 23, alpha=a_c, fill=style.TEXT_DIM, anchor="mm")

    # ================= #3–#5 section =================
    a35 = ph(t, 7.4, 18.6, 0.4, 0.5)
    if a35 > 0.01:
        section(cv, a35 * dm)
        a_ray = ss(t, 8.0, 9.0)
        ptP = zpt(1.0)
        # unit vector K^-1 p~ (gold): optical center → z = 1 plane
        arrow(cv, S_O, ptP, GOLD, a35 * a_ray, 3, 12, progress=a_ray)
        cv.s.dot(S_O, ORANGE, 6, a35)
        cv.rich("Center (cam 2 center $t$)", (S_O[0] - 6, S_O[1] + 36), 22, alpha=a35 * ss(t, 7.6, 8.0),
                fill=style.LABEL_ORANGE, anchor="la")
        if a_ray > 0.5:
            cv.rich(r"$K^{-1}\tilde p$", (ptP[0] - 12, ptP[1] - 26), 24, alpha=a35 * ss(t, 8.8, 9.2),
                    fill=style.LABEL_GOLD, anchor="rb")
        # ray extension: #4 reaches the white dot, #5 then labels z
        u4 = ss(t, 12.6, 14.0)
        zend = 1.0 + (DD - 1.0) * u4
        if t > 11.0:
            pe = zpt(zend)
            cv.s.line(ptP, pe, GOLD, 0.9 * a35, 3)
            cv.g.line(ptP, pe, GOLD, 0.5 * a35, 3)
            if t > 12.6:
                ec.star_marker(cv, pe, a35 * dm, glow=0.8, rad=6)
        a_z = ss(t, 15.2, 15.8) * (1 - ss(t, 18.0, 18.6))
        if a_z > 0.01:
            e1, e2 = axpt(1.0), axpt(DD)
            cv.s.line(zpt(DD), e2, style.GREY, 0.8 * a_z * a35, 1)
            cv.s.line(S_O + (0, 90), e1 + (0, 90), CYAN, a_z * a35, 3)
            cv.rich("$d$ has $z=1$", ((S_O[0] + e1[0]) / 2, S_O[1] + 118), 24, alpha=a_z * a35, fill=style.LABEL_CYAN,
                    anchor="mm")
            u5 = ss(t, 16.2, 17.2)
            cv.s.line(S_O + (0, 150), axpt(1 + (DD - 1) * u5) + (0, 150), CYAN, a_z * a35, 3)
            if u5 > 0.9:
                cv.rich("$D$ = 1.975 = $z$ coordinate of the white dot", ((S_O[0] + e2[0]) / 2, S_O[1] + 184), 26,
                        alpha=a_z * a35 * ss(u5, 0.9, 1.0), fill=style.LABEL_CYAN, anchor="mm")
            cv.s.dot(axpt(DD), CYAN, 4, a_z * a35)

    # ---- #3 readout
    a3 = ph(t, 7.4, 11.2, 0.3, 0.4)
    if a3 > 0.01:
        cv.rich(r"Camera frame: $K^{-1}\tilde p$ = " + sv(KP) + r", $\tilde p=(u,v,1)$ white dot pixel, $K$ intrinsics", (60, 790), 25,
                alpha=a3 * ss(t, 8.8, 9.3), fill=style.TEXT, anchor="lm")
        cv.rich(r"World frame: $d=R\,K^{-1}\tilde p$ = " + sv(DV) + f", $\\Vert d\\Vert$ = {DN:.3f}", (60, 860), 27,
                alpha=a3 * ss(t, 9.6, 10.1), fill=style.LABEL_GOLD, anchor="lm")
        cv.rich("$R$: rotation from cam 2 to world (DA3 convention, opposite of VGGT's world → camera)", (60, 925), 23,
                alpha=a3 * ss(t, 10.0, 10.5), fill=style.TEXT_DIM, anchor="lm")
        cv.rich("Direction not divided by its own length: camera-frame $z$ component stays 1", (60, 985), 25, alpha=a3 * ss(t, 10.3, 10.8),
                fill=style.TEXT, anchor="lm")

    # ---- #4 formula + readout
    a4 = ph(t, 11.0, 15.2, 0.4, 0.4)
    if a4 > 0.01:
        EQ4.draw(cv, (540, 230), alpha=a4, part_alpha={"t": 0.3 + 0.7 * ss(t, 11.5, 11.9), "D": 0.3 + 0.7 * ss(t, 12.0, 12.4),
                                                        "d": 0.3 + 0.7 * ss(t, 12.4, 12.8), "P": 1.0},
                 part_color={"t": COL_T, "D": COL_DEP, "d": COL_D},
                 part_glow={"t": bump(t, 11.6, 0.8), "D": bump(t, 12.1, 0.8), "d": bump(t, 12.5, 0.8),
                            "P": bump(t, 14.0, 1.0)})
        row_vec(cv, 800, "Origin $t$ (m)", TV, a4 * ss(t, 11.5, 11.9), (255, 190, 120), TXT_T, extra="= cam 2 center")
        row_vec(cv, 860, "Depth $D$ (m)", [DD], a4 * ss(t, 12.0, 12.4), (130, 230, 255), TXT_DEP, x=60)
        row_vec(cv, 920, "Direction $d$", DV, a4 * ss(t, 12.4, 12.8), (255, 220, 120), TXT_D)
        row_vec(cv, 1000, r"$P=t+D\,d$ (m)", PV, a4 * ss(t, 13.8, 14.2), (255, 255, 255), style.TEXT,
                extra=r"= white dot $X^\star$")
        cv.rich("$P$: 3D point of the pixel (world frame); $D(u,v)$: $z$-depth of that pixel", (540, 305), 22, alpha=a4 * ss(t, 11.3, 11.7),
                fill=style.TEXT_DIM, anchor="mm")

    # ---- #5 readout
    a5 = ph(t, 15.0, 18.6, 0.4, 0.5)
    if a5 > 0.01:
        EQ4.draw(cv, (540, 230), alpha=a5 * 0.6, part_color={"t": COL_T, "D": COL_DEP, "d": COL_D})
        cv.rich("The camera-frame $z$ component of $d$ = 1; moving $D$ times along $d$ gives $z$ coordinate $D$", (60, 850), 26, alpha=a5 * ss(t, 15.6, 16.0),
                fill=style.TEXT, anchor="lm")
        cv.rich("$D$ = $z$-depth = the point's coordinate along the optical axis (the same $D$ as in E02-02)", (60, 910), 25,
                alpha=a5 * ss(t, 17.0, 17.5), fill=style.LABEL_CYAN, anchor="lm")

    # ================= #6 side-by-side comparison =================
    a6 = ph(t, 18.4, 22.6, 0.4, 0.4)
    if a6 > 0.01:
        cv.s.line((540, 350), (540, 840), style.GREY, 0.5 * a6, 1)
        cv.rich("MapAnything", (290, 330), 32, "bold", alpha=a6, fill=style.TEXT, anchor="mm")
        cv.rich("DA3", (810, 330), 32, "bold", alpha=a6, fill=c2.DA3_COLOR, anchor="mm")
        s = 190.0
        for side, (ox, col) in enumerate([(50.0, (235, 235, 235)), (570.0, c2.DA3_COLOR)]):
            o = np.array([ox, 600.0])
            zp = lambda z: np.array([o[0] + z * s, o[1] - z * TAN * s])
            cv.s.line(o, np.array([o[0] + 2.1 * s, o[1]]), style.GREY, 0.6 * a6, 1)
            if side == 0:                                      # unit arc: length 1
                th = np.arctan(TAN)
                arc = [o + s * np.array([np.cos(x), -np.sin(x)]) for x in np.linspace(-0.25, 0.4, 20)]
                cv.s.polyline(arc, style.GREY, 0.6 * a6, 1)
                tip1 = o + s * np.array([np.cos(th), -np.sin(th)])
            else:
                cv.s.line((o[0] + s, o[1] - 90), (o[0] + s, o[1] + 40), style.GREY, 0.8 * a6, 2)
                tip1 = zp(1.0)
            u1 = ss(t, 19.0 + 0.1 * side, 19.8 + 0.1 * side)
            arrow(cv, o, o + (tip1 - o) * u1, GOLD, a6, 3, 12)
            u2 = ss(t, 20.4, 21.4)
            zend = 1.984 * np.cos(np.arctan(TAN)) if side == 0 else DD
            endp = zp(zend)
            cv.s.dashed(tip1, tip1 + (endp - tip1) * u2, GOLD, 0.9 * a6, 2, 14)
            cv.s.dot(o, ORANGE, 5, a6)
            if u2 > 0.95:
                ec.star_marker(cv, endp, a6 * ss(u2, 0.95, 1.0), glow=0.7, rad=6)
        L = a6 * ss(t, 19.6, 20.1)
        cv.rich("Unit ray (length 1)\n" + sv(MA), (290, 700), 24, alpha=L, fill=style.LABEL_GOLD, anchor="mm", align="m")
        cv.rich("× ray depth $r$ = 1.984", (290, 790), 26, alpha=a6 * ss(t, 20.4, 20.9), fill=style.TEXT, anchor="mm")
        R_ = a6 * ss(t, 19.6, 20.1)
        cv.rich("Unnormalized $d$\n$z$ component = 1", (810, 700), 24, alpha=R_, fill=style.LABEL_GOLD, anchor="mm", align="m")
        cv.rich("× $z$-depth $D$ = 1.975", (810, 790), 26, alpha=a6 * ss(t, 20.4, 20.9), fill=style.TEXT, anchor="mm")
        cv.rich("Both ends reach the same white dot (cross-section drawn in the camera frame for comparison)", (540, 900), 27, alpha=a6 * ss(t, 21.2, 21.7),
                fill=style.LABEL_CYAN, anchor="mm")
        cv.rich("MapAnything's direction is in the camera frame, DA3's $d$ in the world frame", (540, 960), 24, alpha=a6 * ss(t, 21.4, 21.9),
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("$r$: distance along the ray; $D$: $z$-depth; " + KN_TXT, (540, 1010), 22, alpha=a6 * ss(t, 21.4, 21.9),
                fill=style.TEXT_DIM, anchor="mm")

    # ================= #7 shape =================
    a7 = ph(t, 22.4, 27.0, 0.4, 0.4)
    if a7 > 0.01:
        EQ7.draw(cv, (540, 230), alpha=a7, part_color={"t": COL_T, "D": COL_DEP, "d": COL_D},
                 part_alpha={"t": 0.4 + 0.6 * ss(t, 24.0, 24.4), "D": 0.4 + 0.6 * ss(t, 24.4, 24.8),
                             "d": 0.4 + 0.6 * ss(t, 24.8, 25.2)})
        cv.rich(r"$[\ldots,{:}3]$ = origin is the paper's order; the official code is reversed", (540, 292), 22, alpha=a7 * ss(t, 23.0, 23.4),
                fill=style.TEXT_DIM, anchor="mm")
        mb1 = module_box(cv, (40, 350, 330, 440), r"$\hat D$", r"$(N,H,W)$", CYAN, a7 * ss(t, 22.8, 23.2), title_size=32,
                         sub_size=26)
        mb2 = module_box(cv, (380, 350, 700, 440), r"$\hat M$", r"$(N,H,W,6)$", GOLD, a7 * ss(t, 23.0, 23.4),
                         title_size=32, sub_size=26)
        mb3 = module_box(cv, (750, 350, 1040, 440), r"$P$", r"$(N,H,W,3)$", WHITE, a7 * ss(t, 23.6, 24.0),
                         title_size=32, sub_size=26)
        arrow(cv, (mb2["r"][0] + 4, 395), (mb3["l"][0] - 4, 395), WHITE, a7 * ss(t, 23.6, 24.0), 2, 10)
        cv.rich(r"$N$: number of images; $H,W$: height, width of the ray map; $\odot$: element-wise product (each pixel's $\hat D$ times its 3 direction components)", (540, 485), 21,
                alpha=a7 * ss(t, 23.6, 24.0), fill=style.TEXT_DIM, anchor="mm")
        P_ORG.draw(cv, a7 * ss(t, 24.0, 24.4), border=COL_T)
        P_DIR.draw(cv, a7 * ss(t, 24.4, 24.8), border=COL_D)
        cv.rich("Origin $\\hat M[\\ldots,{:}3]$\nsame at every pixel (constant color block)", (190, 880), 22, alpha=a7 * ss(t, 24.2, 24.6),
                fill=style.LABEL_ORANGE, anchor="mm", align="m")
        cv.rich("Direction $\\hat M[\\ldots,3{:}]$\n3 channels → RGB", (490, 880), 22, alpha=a7 * ss(t, 24.6, 25.0),
                fill=style.LABEL_GOLD, anchor="mm", align="m")
        cv.rich("Cam 2 ray map (GT)", (390, 600), 22, alpha=a7 * ss(t, 24.0, 24.4), fill=style.TEXT_DIM, anchor="mm")
        a_n = a7 * ss(t, 25.4, 25.9)
        cv.s.rect(640, 560, 1040, 940, (20, 24, 32), 0.85 * a_n, r=12)
        cv.rich("Actual model output\n\nFor a 504 input, DA3-Large's ray map\nis 288×288 (the ray branch is not\ninterpolated back to input resolution).\nThe official API does not return ray maps by default,\nso we tapped it from an intermediate stage.", (660, 580), 21, alpha=a_n, fill=style.TEXT, anchor="la")

    # ================= #8 Eq. 1 =================
    a8 = ph(t, 27.0, 30.3, 0.4, 0.3)
    if a8 > 0.01:
        EQ8.draw(cv, (540, 230), alpha=a8, part_color={"tc": COL_T, "M": COL_T},
                 part_alpha={"avg": 0.4 + 0.6 * ss(t, 28.3, 28.7), "M": 0.4 + 0.6 * ss(t, 27.6, 28.0)})
        cv.rich("$t_c$: camera center estimated from the ray map; $M(h,w,{:}3)$ = paper's order, first 3 numbers = origin", (540, 345), 22,
                alpha=a8 * ss(t, 27.4, 27.8), fill=style.TEXT_DIM, anchor="mm")
        for k, lab in enumerate([r"$(h,w)=(0,0)$", r"$(h,w)=(0,1)$", "……", r"$(h,w)=(517,517)$"]):
            ak = a8 * ss(t, 27.6 + 0.2 * k, 28.0 + 0.2 * k)
            y = 440 + 70 * k
            cv.rich(lab, (60, y), 24, alpha=ak, fill=style.TEXT_DIM, anchor="lm")
            if k != 2:
                vector(cv, (400, y), TV, "{:.3f}", 34, (255, 190, 120), ak, "lm")
        cv.s.line((60, 730), (900, 730), style.GREY, a8 * ss(t, 28.4, 28.8), 1)
        cv.rich("Take the average $t_c$", (60, 790), 28, alpha=a8 * ss(t, 28.6, 29.0), fill=style.LABEL_ORANGE, anchor="lm")
        vector(cv, (400, 790), TC, "{:.3f}", 38, (255, 190, 120), a8 * ss(t, 28.8, 29.2), "lm")
        cv.rich("= cam 2 center (origin $t$ of #2)", (60, 860), 26, alpha=a8 * ss(t, 29.2, 29.6), fill=style.TEXT, anchor="lm")
        cv.rich(r"$H\times W$ = 518×518 (our test scene's GT ray map)", (60, 920), 22, alpha=a8 * ss(t, 29.2, 29.6),
                fill=style.TEXT_DIM, anchor="lm")

    # ================= #9 DLT: main view =================
    a9 = ph(t, 30.0, 34.3, 0.4, 0.4)
    if a9 > 0.01:
        P_CAM.draw(cv, a9 * dm, border=ec.CAM_COLORS[sc.EX_CAM])
        cv.rich("Cam 2", (170, 336), 22, alpha=a9 * dm, fill=style.TEXT_DIM, anchor="mm")
        ks = np.arange(0, W, 37)
        for v in ks:
            for u in ks:
                p = P_CAM.to_screen(u, v)
                cv.s.dot(p, GOLD, 2.5, a9 * ss(t, 30.4, 31.0) * dm)
        cv.rich(f"14×14 = {N_PIX}  pixels:\nevery pixel's position $(u,v)$ and\nray direction $d$ are known", (320, 350), 25,
                alpha=a9 * ss(t, 30.6, 31.0) * dm, fill=style.TEXT, anchor="la")
        cv.rich("Goal: find a $3\\times3$ matrix $H$\nso that $H\\tilde p$ points along $d$", (320, 480), 25,
                alpha=a9 * ss(t, 31.0, 31.5) * dm, fill=style.LABEL_ORANGE, anchor="la")

    # ================= #10 RQ =================
    a10 = ph(t, 34.0, 38.4, 0.4, 0.4)
    if a10 > 0.01:
        cv.rich(r"$H$: DLT solution, determined only up to a scale factor ($\propto$)", (60, 340), 24, alpha=a10 * ss(t, 34.3, 34.7),
                fill=style.TEXT_DIM, anchor="lm")
        cv.rich(r"$H\propto R\,K^{-1}$", (60, 410), 40, alpha=a10 * ss(t, 34.4, 34.8), fill=style.TEXT, anchor="lm")
        cv.rich(r"$H^{-1}\propto K\,R^{\mathsf{T}}$ (upper triangular × orthogonal)", (60, 490), 40, alpha=a10 * ss(t, 35.0, 35.4),
                fill=style.TEXT, anchor="lm")
        cv.rich(r"RQ decomposition → $\hat K$ (upper triangular), $R^{\mathsf{T}}$ (orthogonal)", (60, 570), 30, alpha=a10 * ss(t, 35.5, 35.9),
                fill=style.LABEL_ORANGE, anchor="lm")
        cv.rich(r"$\hat K=$", (90, 760), 44, alpha=a10 * ss(t, 35.8, 36.1), fill=style.TEXT, anchor="lm")
        fxs, cxs = f"{Kh[0, 0]:.1f}", f"{Kh[0, 2]:.0f}"
        Ms = [[fxs, "0", cxs], ["0", fxs, cxs], ["0", "0", "1"]]
        pa = {(0, 0): ss(t, 36.0, 36.3), (0, 2): ss(t, 36.4, 36.7), (1, 1): ss(t, 36.8, 37.1), (1, 2): ss(t, 37.1, 37.4)}
        pa.update({k: ss(t, 35.9, 36.2) * 0.5 for k in [(0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]})
        matrix(cv, (480, 760), Ms, "{}", 44, style.MATH, a10 * ss(t, 35.8, 36.1), "mm", part_alpha=pa,
               box={(0, 0): bump(t, 36.1, 0.8), (0, 2): bump(t, 36.5, 0.8), (1, 1): bump(t, 36.9, 0.8),
                    (1, 2): bump(t, 37.2, 0.8)})
        cv.rich("$f_x,f_y$: focal lengths (px); $c_x,c_y$: principal point (px)", (60, 900), 24, alpha=a10 * ss(t, 36.2, 36.6),
                fill=style.TEXT_DIM, anchor="lm")
        cv.rich(f"Rotation angle = {ROT_DEG:.1f}° (GT 30°)", (60, 980), 34, alpha=a10 * ss(t, 37.2, 37.6),
                fill=style.LABEL_GOLD, anchor="lm")
        cv.rich(r"$\hat K$ equals GT $K$ (448.6, 259)", (60, 1040), 24, alpha=a10 * ss(t, 37.2, 37.6),
                fill=style.TEXT_DIM, anchor="lm")

    # ================= #11 real model =================
    a11 = ph(t, 38.6, 43.2, 0.4, 0.4)
    if a11 > 0.01:
        cv.rich("We run the same DLT + RQ on the cam 2 ray map\noutput by DA3-Large (288×288, 856 pixels used)", (540, 350), 28,
                alpha=a11 * ss(t, 38.8, 39.2), fill=style.TEXT, anchor="mm", align="m")
        rows = [[r"$\hat f_x$ (focal length)", f"{FX_M:.1f}", f"{F504:.1f}", f"−{abs(REL_X) * 100:.1f} %"],
                [r"$\hat f_y$ (focal length)", f"{FY_M:.1f}", f"{F504:.1f}", f"−{abs(REL_Y) * 100:.1f} %"],
                ["Cam 2 rotation angle", f"{ROT_M:.1f}°", "30°", "Rotation error about 0.1–0.2°\n(varies with solver)"]]
        table(cv, (50, 440), [270, 230, 190, 280], ["Quantity", "Our DLT result", "GT", "Difference from GT"], rows, row_h=100, size=28,
              reveal=3 * ss(t, 39.4, 41.4), alpha=a11)
        cv.rich("Unit of $f$: pixels on the 504-pixel image. Difference from GT = (result − GT) ÷ GT; closer to 0 is better", (540, 930), 22,
                alpha=a11 * ss(t, 40.0, 40.4), fill=style.TEXT_DIM, anchor="mm")
        cv.rich("Our DLT: Eq. 2 form, coordinates not normalized, 856 pixels", (540, 985), 22, alpha=a11 * ss(t, 40.4, 40.8),
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("Rotation angle: how far cam 2 is turned relative to cam 1; rotation error: angle between predicted $\\hat R$ and GT $R$\n(not 30.1 − 30)", (540, 1055),
                22, alpha=a11 * ss(t, 40.8, 41.2), fill=style.TEXT_DIM, anchor="mm", align="m")

    # ================= #12 camera head =================
    a12 = ph(t, 43.0, 46.0, 0.4, 0.4)
    if a12 > 0.01:
        cv.rich("Solving the camera from the whole ray map touches every pixel, which is costly", (540, 340), 25, alpha=a12 * ss(t, 43.2, 43.6),
                fill=style.TEXT_DIM, anchor="mm")
        P_CAM2 = IN_PANELS[sc.EX_CAM]
        cv.rich("Camera token: 1 per image,\nsummarizing the image's tokens", (190, 520), 24, alpha=a12 * ss(t, 43.4, 43.8),
                fill=style.LABEL_GOLD, anchor="mm", align="m")
        cv.s.rect(150, 580, 230, 640, tuple(c * 0.4 for c in GOLD), 0.9 * a12 * ss(t, 43.4, 43.8), r=6)
        cv.s.polyline([(150, 580), (230, 580), (230, 640), (150, 640)], GOLD, a12 * ss(t, 43.4, 43.8), 2, closed=True)
        mb = module_box(cv, (330, 570, 640, 650), "camera head", "Lightweight, reads only the camera token", CYAN,
                        a12 * ss(t, 43.8, 44.2), title_size=30, sub_size=22)
        arrow(cv, (236, 610), (326, 610), WHITE, a12 * ss(t, 43.8, 44.2), 2, 10)
        for k, (lab, col) in enumerate([("Translation $t$", COL_T), ("Rotation $q$", VIOLET), (r"FoV $\theta_v,\theta_h$", CYAN)]):
            ak = a12 * ss(t, 44.4 + 0.25 * k, 44.8 + 0.25 * k)
            y = 540 + 70 * k
            arrow(cv, (644, 610), (740, y + 10), WHITE, ak, 2, 8)
            cv.s.rect(750, y - 24, 1000, y + 44, tuple(c * 0.25 for c in col), 0.9 * ak, r=8)
            cv.s.polyline([(750, y - 24), (1000, y - 24), (1000, y + 44), (750, y + 44)], col, ak, 2, closed=True)
            cv.rich(lab, (875, y + 10), 28, alpha=ak, fill=style.TEXT, anchor="mm")
        cv.rich("About 0.1% of backbone compute", (540, 860), 34, alpha=a12 * ss(t, 45.0, 45.4), fill=style.LABEL_GOLD, anchor="mm")
        cv.rich("$t$: camera translation; $q$: rotation (quaternion);\n$\\theta_v,\\theta_h$: vertical, horizontal field of view", (540, 945), 23,
                alpha=a12 * ss(t, 45.0, 45.4), fill=style.TEXT_DIM, anchor="mm", align="m")

    # ================= #13 Tab. 6 =================
    a13 = ph(t, 46.0, 99.0, 0.4, 0.0)
    if a13 > 0.01:
        cv.rich("depth + ray: the network learns depth and ray maps; depth + cam: learns depth and camera parameters", (540, 1180), 23,
                alpha=a13 * ss(t, 46.2, 46.6), fill=style.TEXT_DIM, anchor="mm")
        bar_chart(cv, (430, 720, 950, 1100), [48.7, 10.8, 25.5, 9.9],
                  ["HiRoom\ndepth + ray", "HiRoom\ndepth + cam", "ETH3D\ndepth + ray", "ETH3D\ndepth + cam"], vmax=60,
                  horizontal=True, highlight=[0, 2], progress=ss(t, 46.3, 47.2), fmt="{:.1f}",
                  title="Tab. 6 · Auc3", better="higher", alpha=a13, label_size=22)

    POP_DLT.draw(cv, t_real, body_dlt)
    POP_Q.draw(cv, t_real, body_q)
    POP_AUC.draw(cv, t_real, body_auc)
