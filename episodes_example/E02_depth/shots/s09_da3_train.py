"""E02-09 DA3 3: training -- normalization, five losses, teacher-student (M, 36.5 s). Storyboard: storyboard.md §E02-09.
Computed: mean distance 2.153 m (363,918 valid pixels), white dot depth 1.975 → 0.918, RANSAC over 6 sparse points (the pair of points 1 and 2: s=1.420, b=1.211,
      threshold 0.258, 5 inliers), inlier refit (1.502, 1.201), DA3-Large white dot output without pose 0.918 (measured model output).
Illustration: the sparse depth, the bad point (the 5th), and the teacher's relative depth map are all built from GT (FAKE_TAG); the RANSAC inlier criterion is our implementation.
Paper numbers: five losses (§3.3), Eq. 8 (§4.1–4.2), training schedule (§3.4), Tab. 7 e (HiRoom Auc3 39.2 → 11.2; DTU CD 3.82 → 3.29).
"""
import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, GREEN, GREY, ORANGE, RED, VIOLET, WHITE, Formula, ImagePanel, Plot2D, Popup, Shot,
                     arrow, bar_chart, colorize, counter, fade, lerp, number, popup_check, ss, style)

ec, sc = c2.ec, c2.sc
NB = c2.NB()
MET = c2.metrics()["da3_large"]["A_nopose_camhead"]

DUR = 38.0
shot = Shot("s09_da3_train", DUR)
shot.title("DA3 3. Training", "Training · normalization · losses · teacher-student", 0.2, 3.0)

CAPS = [(0.3, 4.3, "GT divided by mean distance first"),
        (4.3, 7.0, "So the main output has no meters"),
        (7.0, 11.0, "The five losses are summed"),
        (11.0, 14.0, "Real-data depth is sparse and dirty"),
        (14.0, 17.0, "Teacher: trained on synthetic data only"),
        (17.0, 21.0, "Align the teacher to the sparse depth"),
        (21.0, 25.0, "RANSAC: guess from two points, count inliers"),
        (25.0, 28.5, "Refit on the inliers to get dense labels"),
        (28.5, 32.0, "GT for the first 120k steps, then labels"),
        (32.0, DUR, "Without teacher: most get worse")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

# ---- Corner tags
shot.tag(c2.GT_TAG, 0.3, 7.0)
shot.tag(c2.SCENE_TAG, 3.6, 7.0)
shot.tag("Paper + our inference", 4.3, 7.0)
shot.tag("Paper §3.3", 7.0, 11.0)
shot.tag(c2.FAKE_TAG, 11.0, 28.5)
shot.tag("Paper Eq. 8", 17.0, 21.0)
shot.tag("Inlier criterion: our implementation", 21.0, 25.0)
shot.tag("Paper §3.4", 28.5, 32.0)
shot.tag("Paper Tab. 7", 32.0, DUR)

# ---------------------------------------------------------------- Numbers
S_NORM = NB["s_norm"]
N_VALID = NB["n_valid"]
Z = NB["D"]
Z_N = Z / S_NORM
DS, DSP, DT = np.asarray(NB["ransac_D"]), np.asarray(NB["ransac_sparse"]), np.asarray(NB["ransac_teacher"])
BEST = NB["ransac_best"]                      # (inlier count, (i, j), s, b, threshold)
H_S, H_B, H_THR = float(BEST[2]), float(BEST[3]), float(BEST[4])
HI, HJ = BEST[1]
RS_HAT, RB_HAT = float(NB["ransac_s"]), float(NB["ransac_b"])
UV_W = ec.nb()["uv"]
SP_PX = np.array([[80, 420], [200, 480], [UV_W[0], UV_W[1]], [330, 200], [420, 380], [150, 300]], float)
BAD = 4


def _ransac():
    best = None
    for i in range(6):
        for j in range(i + 1, 6):
            s_, b_ = np.linalg.solve(np.array([[DT[i], 1], [DT[j], 1]]), DSP[[i, j]])
            if s_ <= 0:
                continue
            r = s_ * DT + b_ - DSP
            med = np.median(r)
            thr = np.mean(np.abs(r - med))
            inl = np.abs(r - med) < thr
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), (i, j), s_, b_, thr, inl, float(med))
    return best


RB = _ransac()
INL, MED = RB[5], RB[6]
FIT_S, FIT_B = Plot2D.fit_line(DT[INL], DSP[INL])
SCALE_M = MET["scale_to_m"]
DIFF_PCT = abs(MET["scale_to_m"] / S_NORM - 1) * 100

# ---------------------------------------------------------------- Point cloud (drawn as in E01-07)
X_GT, RGB_GT, _, _ = ec.cloud(4)
OC = ec.viewer(az=-12, el=30, r=7.0, fit_pts=np.r_[X_GT, [c for _, _, c in sc.cameras()]], box=(820, 400),
               center=(540, 600))
LINE_IDX = np.random.default_rng(1).choice(len(X_GT), 40, replace=False)

# ---------------------------------------------------------------- Images
IMG = ec.view_rgb(c2.CAM2)
TZ = c2.cam2_depth()
DT_IMG = (TZ - 1.2) / 1.5                                   # illustration: teacher's relative depth (built from GT)
LAB_IMG = FIT_S * DT_IMG + FIT_B                            # dense label
TEACH_RGB = colorize(DT_IMG, 0.0, 1.9)
LABEL_RGB = colorize(LAB_IMG, 1.1, 4.0)

# ---------------------------------------------------------------- Formulas
TERM_COL = {"d": CYAN, "m": ORANGE, "p": GOLD, "c": VIOLET, "g": GREEN}
EQ_L = Formula([(r"\mathcal{L}=", "L"), (r"\mathcal{L}_{D}", "d"), (r"+\mathcal{L}_{M}", "m"), (r"+\mathcal{L}_{P}", "p"),
                (r"+\beta\,\mathcal{L}_{C}", "c"), (r"+\alpha\,\mathcal{L}_{\mathrm{grad}}", "g")], size=56)
LEGEND = [("d", r"$\mathcal{L}_D$", "Depth: error times confidence $D_c$,|minus $\\lambda_c\\log D_c$ (keeps confidence from collapsing to 0)"),
          ("m", r"$\mathcal{L}_M$", "Ray map: error between predicted and GT ray maps"),
          ("p", r"$\mathcal{L}_P$", "Points: error between the 3D points built from depth × rays and GT"),
          ("c", r"$\mathcal{L}_C$", "Camera: error between predicted and GT cameras"),
          ("g", r"$\mathcal{L}_{\mathrm{grad}}$", "Depth gradient: penalizes only wrong local shape")]
EQ8 = Formula([(r"(\hat s,\hat b)=\arg\min_{s>0,\,b}", "arg"), (r"\sum_p m_p", "m"),
               (r"\left(s\tilde D_p+b-D_p\right)^2", "res")], size=48)

# ---------------------------------------------------------------- regression plot
PLOT = Plot2D((170, 700, 930, 1130), (-0.1, 1.0), (0, 3.8), xlabel=r"Teacher relative depth $\tilde D_p$",
              ylabel="Sparse depth $D_p$ (m)", fmt_x="{:.1f}", fmt_y="{:.1f}")
PT_LABELS = [f"$p_{k + 1}$" for k in range(6)]
PT_LABELS[2] = "$p_3$ white dot"
PT_LABELS[BAD] = "$p_5$ bad point"

POP = Popup(box=(560, 305, 480, 330), t_in=21.0, t_out=25.0, title="RANSAC", corner="tr", accent=ORANGE)

DEF_AUC1 = "Auc3: for each image pair, take the larger of the rotation error and translation-direction error (both angles)"
DEF_AUC2 = "Accuracy = fraction of image pairs with error below the threshold; sweep the threshold over 0–3°, take the area under the curve; full score 100, higher is better"
DEF_CD = "CD: mean distance between nearest points of the two clouds; the paper gives no unit"


@shot.check
def check(ck):
    ck.eq("mean distance on screen", f"{S_NORM:.3f}", "2.153")
    ck.eq("valid pixel count", f"{N_VALID:,}", "363,918")
    ck.eq("white dot depth on screen", f"{Z:.3f}", "1.975")
    ck.eq("white dot depth ÷ mean distance on screen", f"{Z_N:.3f}", "0.918")
    ck.eq("R-6: DA3-Large white dot output without pose", f"{MET['D_white_raw']:.3f}", "0.918")
    ck.eq("R-6: whole-image scale difference %", f"{DIFF_PCT:.2f}", "0.16")
    ck.eq("R-7: multiplier to meters on screen", f"{SCALE_M:.3f}", "2.149")
    ck.eq("bad point GT on screen", f"{DS[BAD]:.3f}", "1.774")
    ck.eq("bad point sparse value on screen", f"{DSP[BAD]:.3f}", "3.193")
    ck.eq("RANSAC best hypothesis = points (0, 1)", (int(HI), int(HJ)), (0, 1))
    ck.eq("RANSAC rerun gives the same pair", RB[1], tuple(BEST[1]))
    ck.eq("s on screen", f"{H_S:.3f}", "1.420")
    ck.eq("b on screen", f"{H_B:.3f}", "1.211")
    ck.eq("threshold on screen", f"{H_THR:.3f}", "0.258")
    ck.eq("inlier count", int(INL.sum()), 5)
    ck.true("bad point (the 5th) is not an inlier", not bool(INL[BAD]), INL.tolist())
    ck.close("rerun s = NB", RB[2], H_S, 1e-9)
    ck.close("rerun b = NB", RB[3], H_B, 1e-9)
    ck.close("rerun threshold = NB", RB[4], H_THR, 1e-9)
    ck.eq("refit s", f"{FIT_S:.3f}", "1.502")
    ck.eq("refit b", f"{FIT_B:.3f}", "1.201")
    ck.close("fit_line = NB ransac_s", FIT_S, RS_HAT, 1e-9)
    ck.close("fit_line = NB ransac_b", FIT_B, RB_HAT, 1e-9)
    ck.close("fit_line = np.polyfit", FIT_S, float(np.polyfit(DT[INL], DSP[INL], 1)[0]), 1e-9)
    ck.close("only the 5th sparse point was altered", float(np.abs(DSP - DS).sum() - abs(DSP[BAD] - DS[BAD])), 0.0, 1e-12)
    ck.eq("Tab. 7 e Auc3", ("39.2", "11.2"), ("39.2", "11.2"))
    popup_check(ck, [POP], DUR)
    ck.record(s_norm=round(S_NORM, 4), ransac=[H_S, H_B, H_THR], refit=[FIT_S, FIT_B], diff_pct=round(DIFF_PCT, 3))


# ---------------------------------------------------------------- A: normalization (0.3–7.0)
def sec_norm(cv, t):
    a = fade(t, 0.3, 7.0, 0.5, 0.4)
    if a <= 0.01:
        return
    P = OC.at(az=-12 + 8 * ss(t, 0.3, 7.0))
    a_c = ss(t, 0.4, 1.2) * a
    for i in range(3):
        r = ec.frustum(cv, P, i, 0.45, a_c)
        if r is not None:
            C = r[0]
            cv.text((float(np.clip(C[0], 70, 1010)), C[1] + 14), ec.CAM_NAMES[i], 24, alpha=a_c,
                    fill=ec.CAM_COLORS[i], anchor="ma")
    ec.splat(cv, P, X_GT, RGB_GT, 0.8 * a_c, size=2)
    o = P(ec.disp(np.zeros(3)))
    a_l = ss(t, 1.2, 2.0) * a
    for j in LINE_IDX:
        cv.s.line(o, P(ec.disp(X_GT[j])), GOLD, 0.35 * a_l, 1)
    cv.g.dot(o, GOLD, 8, 0.6 * a_l)
    cv.s.dot(o, GOLD, 5, a_l)
    cv.rich("Origin = cam 1's position", (o[0] - 16, o[1] - 2), 22, alpha=a_l, fill=GOLD, anchor="rm")
    cv.rich(f"{N_VALID:,} valid points from the three images, each connected to the origin", (540, 305), 26, alpha=a_l * (1 - ss(t, 4.3, 4.7)), anchor="ma")
    cv.rich("Metric output comes from a separate DA3-metric (single-image input)", (540, 305), 28, alpha=a * ss(t, 5.4, 5.9), fill=GOLD, anchor="ma")
    pw = P(ec.disp(sc.X_STAR))
    a_w = ss(t, 2.0, 2.6) * a
    cv.s.line(o, pw, WHITE, 0.7 * a_w, 2)
    ec.star_marker(cv, pw, a_w)

    # number row
    u_m = 1 - ss(t, 4.4, 5.2)                                   # "m" fades out
    cv.rich("Mean distance", (110, 880), 34, alpha=a_l, anchor="lm")
    counter(cv, (330, 880), ss(t, 1.4, 3.0), 0.0, S_NORM, fmt="{:.3f}", size=40, alpha=a_l, color=style.LABEL_GOLD,
            anchor="lm")
    cv.rich("m", (470, 880), 34, alpha=a_l, fill=style.LABEL_GOLD, anchor="lm")
    a_z = ss(t, 3.0, 3.5) * a
    k = ss(t, 3.4, 4.2)
    cv.rich("White dot depth", (110, 945), 34, alpha=a_z, anchor="lm")
    number(cv, (330, 945), lerp(Z, Z_N, k), "{:.3f}", 40, style.LABEL_GOLD, a_z, anchor="lm")
    cv.rich("m", (470, 945), 34, alpha=a_z * (1 - k) * u_m, fill=style.LABEL_GOLD, anchor="lm")
    cv.rich("$D$ ÷ mean distance", (560, 945), 28, alpha=a_z * ss(t, 3.6, 4.0), fill=style.TEXT_DIM, anchor="lm")
    a_r = ss(t, 3.8, 4.4) * a
    cv.rich(f"With no pose given, DA3-Large outputs {MET['D_white_raw']:.3f} for the white dot,", (110, 1006), 26, alpha=a_r, anchor="lm")
    cv.rich(f"consistent with dividing by the mean distance: turning the three depth maps into meters needs a factor of {SCALE_M:.3f},", (110, 1040), 26, alpha=a_r, anchor="lm")
    cv.rich(f"{DIFF_PCT:.2f} % below the mean distance of {S_NORM:.3f} m", (110, 1074), 26, alpha=a_r, anchor="lm")
    cv.rich(r"DA3: mean $\ell_2$ length from valid points to the origin.", (110, 1110), 22, alpha=a_l, fill=style.TEXT_DIM, anchor="lm")
    cv.rich("The paper does not say which frame the origin is in; here we use cam 1.", (110, 1142), 22, alpha=a_l, fill=style.TEXT_DIM,
            anchor="lm")


# ---------------------------------------------------------------- B: five losses (7.0–11.0)
def sec_loss(cv, t):
    a = fade(t, 7.0, 11.0, 0.4, 0.4)
    if a <= 0.01:
        return
    lit = {"L": "lit"}
    for k, key in enumerate(["d", "m", "p", "c", "g"]):
        lit[key] = "lit" if t > 7.5 + 0.5 * k else "dim"
    EQ_L.draw(cv, (540, 380), alpha=a, state=lit, part_color=TERM_COL)
    for k, (key, sym, txt) in enumerate(LEGEND):
        ak = ss(t, 7.6 + 0.5 * k, 8.1 + 0.5 * k) * a
        y = 520 + 98 * k
        cv.s.rect(70, y - 30, 78, y + 30, TERM_COL[key], ak, r=2)
        cv.rich(sym, (105, y), 38, alpha=ak, fill=TERM_COL[key], anchor="lm")
        parts = txt.split("|")
        for q, part in enumerate(parts):
            cv.rich(part, (260, y + (q - 0.5 * (len(parts) - 1)) * 34), 27, alpha=ak, anchor="lm")
    cv.rich(r"$\alpha=\beta=1$; every term is an L1 error", (540, 1040), 30, alpha=a * ss(t, 9.9, 10.4), fill=GOLD, anchor="mm")


# ---------------------------------------------------------------- C: sparse depth + teacher (11.0–17.0)
def sec_sparse(cv, t):
    a = fade(t, 11.0, 17.0, 0.4, 0.4)
    if a <= 0.01:
        return
    k = ss(t, 14.0, 14.8)
    w = lerp(620, 440, k)
    pl = ImagePanel(IMG, (lerp(230, 70, k), lerp(380, 400, k)), w=w)
    pl.draw(cv, a, border=c2.CAM2_COLOR, title="Cam 2 image" if t > 11.5 else None, title_size=26)
    n = ss(t, 11.6, 13.4)
    for i in range(6):
        u = ss(n * 7 - i, 0, 1) * a
        if u <= 0.01:
            continue
        p = pl.to_screen(*SP_PX[i])
        col = RED if i == BAD else CYAN
        if i == 2:
            ec.star_marker(cv, p, u, glow=0.6, rad=5)
        else:
            cv.g.dot(p, col, 9, 0.5 * u)
            cv.s.dot(p, col, 5, u)
        cv.rich(PT_LABELS[i] if i != 2 else "$p_3$", (p[0] + 10, p[1] - 12), 20, alpha=u, fill=col if i == BAD else WHITE,
                anchor="lb")
    a_b = ss(t, 13.0, 13.6) * a * (1 - k)
    cv.rich(f"Bad point $p_5$: {DS[BAD]:.3f} → {DSP[BAD]:.3f} m", (540, 1030), 30, alpha=a_b, fill=RED, anchor="mm")
    cv.rich("Sparse depth $D_p$: only 6 pixels have metric values", (540, 1085), 26, alpha=ss(t, 12.2, 12.8) * a * (1 - k), anchor="mm")
    # teacher
    a_t = ss(t, 14.6, 15.4) * a
    if a_t > 0.01:
        pt = ImagePanel(TEACH_RGB, (570, 400), w=440)
        pt.draw(cv, a_t, border=c2.DA3_COLOR, title=r"Teacher's relative depth $\tilde D$", title_size=26)
        arrow(cv, (518, 620), (562, 620), c2.DA3_COLOR, a_t, th=3, head=14)
        cv.rich("Teacher = a single-image relative-depth model", (540, 905), 28, alpha=ss(t, 14.8, 15.3) * a, anchor="mm")
        cv.rich("(DINOv2 + DPT), trained on synthetic data only", (540, 945), 28, alpha=ss(t, 15.0, 15.5) * a, anchor="mm")
        cv.rich(r"$\tilde D$: near-far order only, no meters", (540, 1000), 26, alpha=ss(t, 15.8, 16.3) * a, fill=style.TEXT_DIM,
                anchor="mm")


# ---------------------------------------------------------------- D: alignment + RANSAC (17.0–28.5)
def sec_align(cv, t):
    a = fade(t, 17.0, 28.5, 0.4, 0.4)
    if a <= 0.01:
        return
    # Eq. 8 (above during 17–21, collapsed afterwards)
    a_eq = ss(t, 17.2, 17.8) * (1 - ss(t, 20.8, 21.2)) * a
    if a_eq > 0.01:
        st = {"arg": "lit", "m": "hl" if 18.8 < t < 20.0 else "lit", "res": "lit"}
        EQ8.draw(cv, (540, 365), alpha=a_eq, state=st, notes={
            "arg": r"$\hat s$: scale; $\hat b$: offset (m)",
            "m": "$m_p$: whether this pixel has depth",
            "res": r"$\tilde D_p$: teacher's relative depth; $D_p$: sparse depth (m)"})
        cv.rich("Same formula as E02-03, wrapped in RANSAC", (540, 590), 28, alpha=a_eq * ss(t, 19.6, 20.1), fill=GOLD,
                anchor="mm")
    # regression plot
    a_p = ss(t, 17.6, 18.2) * a
    PLOT.axes(cv, a_p)
    cols = [CYAN] * 6
    cols[BAD] = RED
    if t > 23.8:
        cols = [GREEN if INL[i] else RED for i in range(6)]
    lab = list(PT_LABELS)
    lab[3] = lab[5] = None
    PLOT.scatter(cv, DT, DSP, cols, ss(t, 18.0, 19.6), a_p, labels=lab)
    a_x = ss(t, 19.2, 19.8) * a_p
    q4, q6 = PLOT.to_screen(DT[3], DSP[3]), PLOT.to_screen(DT[5], DSP[5])
    cv.rich(PT_LABELS[3], (q4[0] + 6, q4[1] - 14), 24, alpha=a_x, anchor="lb")
    cv.rich(PT_LABELS[5], (q6[0] - 6, q6[1] + 14), 24, alpha=a_x, anchor="rt")
    # 7: hypothesis line, residuals, threshold band
    a_h = (1 - ss(t, 25.0, 25.6)) * a
    if t > 21.0 and a_h > 0.01:
        PLOT.line(cv, H_S, H_B, ORANGE, ss(t, 21.2, 22.0), a_h)
        for i in (HI, HJ):
            p = PLOT.to_screen(DT[i], DSP[i])
            cv.g.dot(p, GOLD, 14, 0.5 * a_h * ss(t, 21.2, 21.8))
        PLOT.residuals(cv, DT, DSP, H_S, H_B, RED, ss(t, 22.0, 23.0), 0.8 * a_h)
        a_band = ss(t, 23.0, 23.6) * a_h
        for sgn in (-1, 1):
            PLOT.line(cv, H_S, H_B - MED + sgn * H_THR, ORANGE, 1.0, 0.55 * a_band, th=2, glow=0.0, dashed=True)
        x0 = 80
        cv.rich(f"Solved from points $p_{HI + 1}$, $p_{HJ + 1}$:", (x0, 330), 26, alpha=ss(t, 21.2, 21.7) * a_h, anchor="lm")
        cv.rich(f"$s$ = {H_S:.3f}, $b$ = {H_B:.3f} m", (x0, 372), 28, alpha=ss(t, 21.8, 22.3) * a_h,
                fill=ORANGE, anchor="lm")
        cv.rich(f"Threshold = {H_THR:.3f} m (half-width of the dashed band)", (x0, 414), 26, alpha=ss(t, 23.2, 23.7) * a_h, anchor="lm")
        cv.rich(f"{int(INL.sum())} inliers out of 6", (x0, 456), 30, alpha=ss(t, 23.8, 24.3) * a_h, fill=GREEN,
                anchor="lm")
        cv.rich("Bad point $p_5$ is outside", (x0, 498), 26, alpha=ss(t, 24.2, 24.7) * a_h, fill=RED, anchor="lm")
    # 8: inlier refit
    u = ss(t, 25.4, 26.6)
    if t > 25.2:
        s, b = Plot2D.morph((H_S, H_B), (FIT_S, FIT_B), u)
        PLOT.line(cv, s, b, GREEN, 1.0, a * ss(t, 25.2, 25.6), th=4)
        x0 = 80
        cv.rich("Least squares on the 5 inliers only:", (x0, 330), 26, alpha=ss(t, 25.4, 25.9) * a, anchor="lm")
        cv.rich(r"$\hat s$ =", (x0, 380), 30, alpha=ss(t, 25.6, 26.0) * a, anchor="lm")
        number(cv, (x0 + 70, 380), s, "{:.3f}", 34, style.LABEL_GOLD, ss(t, 25.6, 26.0) * a, anchor="lm")
        cv.rich(r"$\hat b$ =", (x0 + 230, 380), 30, alpha=ss(t, 25.6, 26.0) * a, anchor="lm")
        number(cv, (x0 + 300, 380), b, "{:.3f}", 34, style.LABEL_GOLD, ss(t, 25.6, 26.0) * a, anchor="lm",
               unit="m")
        cv.rich("(the values used to build the data were 1.5 and 1.2)", (x0, 430), 22, alpha=ss(t, 26.8, 27.3) * a, fill=style.TEXT_DIM,
                anchor="lm")
        a_l = ss(t, 26.9, 27.6) * a
        if a_l > 0.01:
            pl = ImagePanel(LABEL_RGB, (740, 345), w=280)
            pl.draw(cv, a_l, border=GREEN)
            cv.rich(r"Dense label $=\hat s\tilde D+\hat b$", (880, 330), 24, alpha=a_l, anchor="mb")
            cv.rich("One metric depth per pixel", (880, 648), 22, alpha=a_l, fill=style.TEXT_DIM, anchor="mt")


# ---------------------------------------------------------------- E: training schedule (28.5–32.0)
def sec_sched(cv, t):
    a = fade(t, 28.5, 32.0, 0.4, 0.4)
    if a <= 0.01:
        return
    x0, x1, y0, y1 = 100.0, 980.0, 660.0, 760.0
    k = 880.0 / 200_000
    xs = x0 + 120_000 * k
    cv.rich("The depth target at each step comes from:", (100, 520), 30, alpha=a, anchor="lm")
    g1 = ss(t, 28.8, 29.8)
    g2 = ss(t, 29.8, 30.6)
    if g1 > 0.01:
        cv.s.rect(x0, y0, x0 + (xs - x0) * g1, y1, tuple(0.8 * c for c in CYAN), a, r=4)
    if g2 > 0.01:
        cv.s.rect(xs, y0, xs + (x1 - xs) * g2, y1, tuple(0.8 * c for c in ORANGE), a, r=4)
        cv.g.rect(xs, y0, xs + (x1 - xs) * g2, y1, ORANGE, 0.35 * a, r=4)
    cv.rich("GT depth", ((x0 + xs) / 2, (y0 + y1) / 2), 30, alpha=a * g1, anchor="mm")
    cv.rich("Teacher labels", ((xs + x1) / 2, (y0 + y1) / 2), 28, alpha=a * g2, anchor="mm")
    for v, lab in [(0, "0"), (120_000, "120k"), (200_000, "200k")]:
        x = x0 + v * k
        cv.s.line((x, y1), (x, y1 + 14), GREY, a, 2)
        cv.rich(lab, (x, y1 + 22), 26, alpha=a, anchor="ma")
    cv.rich("Training steps", (540, y1 + 70), 24, alpha=a, fill=style.TEXT_DIM, anchor="ma")
    a_s = ss(t, 29.8, 30.4) * a
    cv.s.dashed((xs, y0 - 40), (xs, y1 + 10), GOLD, 0.8 * a_s, 2, n=10)
    cv.rich("Step 120,000: switch to teacher labels", (xs, y0 - 52), 26, alpha=a_s, fill=GOLD, anchor="mb")
    cv.rich("128 H100 GPUs", (540, 960), 34, alpha=a * ss(t, 30.8, 31.4), anchor="mm")


# ---------------------------------------------------------------- F: ablation (32.0–36.5)
def sec_abl(cv, t):
    a = ss(t, 32.0, 32.5)
    if a <= 0.01:
        return
    p = ss(t, 32.4, 34.2)
    bar_chart(cv, (380, 430, 980, 560), [39.2, 11.2], ["With teacher", "Without teacher"], vmax=46, horizontal=True,
              highlight=1, progress=p, fmt="{:.1f}", title="HiRoom Auc3", better="higher", alpha=a)
    p2 = ss(t, 34.0, 35.6)
    bar_chart(cv, (380, 720, 980, 850), [3.82, 3.29], ["With teacher", "Without teacher"], vmax=4.5, horizontal=True,
              highlight=1, progress=p2, fmt="{:.2f}", title="DTU CD", better="lower", alpha=a * ss(t, 33.8, 34.3),
              note="Paper Tab. 7 e")
    cv.rich(DEF_AUC1, (540, 950), 23, alpha=a * ss(t, 33.0, 33.6), fill=style.TEXT_DIM, anchor="mm")
    cv.rich(DEF_AUC2, (540, 985), 22, alpha=a * ss(t, 33.0, 33.6), fill=style.TEXT_DIM, anchor="mm")
    cv.rich(DEF_CD, (540, 1050), 23, alpha=a * ss(t, 35.0, 35.6), fill=style.TEXT_DIM, anchor="mm")
    cv.rich("Without the teacher, Auc3 drops sharply; CD instead improves", (540, 1120), 26, alpha=a * ss(t, 35.6, 36.1),
            fill=GOLD, anchor="mm")


# ---------------------------------------------------------------- Popup body
RANSAC_LINES = ["1) Pick any two points, solve one $(s,b)$", "2) Compute each point's residual $r_p$ = value on the line − $D_p$",
                "3) Threshold = mean absolute deviation of the residuals from their median", r"4) Points with $|r_p-\mathrm{med}|$ < threshold are inliers",
                "5) 6 points give 15 two-point pairs; the one with the most inliers wins"]


def pop_body(sub, lt, box):
    x0, y0 = box[0], box[1]
    for i, s in enumerate(RANSAC_LINES):
        sub.rich(s, (x0 + 4, y0 + 22 + 56 * i), 22, alpha=ss(lt, 0.5 + 0.5 * i, 0.9 + 0.5 * i), anchor="lm")


@shot.frame
def frame(cv, t):
    sec_norm(cv, t)
    sec_loss(cv, t)
    sec_sparse(cv, t)
    sec_align(cv, t)
    sec_sched(cv, t)
    sec_abl(cv, t)
    POP.draw(cv, t, pop_body)
