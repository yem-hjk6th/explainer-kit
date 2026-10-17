"""E02-03 Relative vs metric depth: scale ambiguity and affine alignment (M, 36 s). Storyboard storyboard.md section E02-03.
Computed for real: after scaling the scene + cameras ×2 together the white-dot depth goes 1.975 -> 3.951 m while camera 2's pixels stay put (same assertion as in E01); 5 pixel positions and GT depths;
      normal equations N·(s, b) = rhs (lsq_N / lsq_rhs), (ŝ, b̂) = (1.545, 1.175 m), RMS residual 0.008 m; Plot2D.fit_line = (ŝ, b̂).
Illustration: the "relative depth" D~ is built from GT (GT ÷ 1.5 − 0.8 + small noise, eps in e02_numbers.py); #5-#7 carry a tag.
Paper numbers: none (#8 only has labels pointing to E02-06 / E02-09). Sources: Depth Pro section 2 p.3, DA3 Eq. 8.
Notation: affine alignment is written (ŝ, b̂), b = offset, not t (t in E02 is the ray origin / camera translation).
"""
import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, GREY, ORANGE, RED, VIOLET, Formula, ImagePanel, Plot2D, Popup, Shot, bump, fade,
                     lerp, matrix, number, popup_check, ss, style)

ec, sc = c2.ec, c2.sc

DUR = 36.0
shot = Shot("s03_scale_align", DUR)
shot.title("Scale and affine fit", "relative depth vs metric depth", 0.2, 3.8)
shot.tag(c2.GT_TAG, 0.3, 7.8)
shot.tag("Illustration: relative depth built from GT (GT ÷ 1.5 − 0.8 + small noise)", 16.0, 30.0)

CAPS = [(0.3, 4.0, "Scene and cameras scaled up 2x together"),
        (4.0, 7.8, "Same photo, double the depth"),
        (7.8, 12.0, "Relative depth: off by a scale and an offset"),
        (12.0, 16.0, "Offset free too: each dataset has its own zero"),
        (16.0, 20.5, "Alignment: solve s and b by least squares"),
        (20.5, 25.5, "Two unknowns, two equations"),
        (25.5, 30.0, "After alignment only 0.8 cm off"),
        (30.0, 36.0, "Evaluation and labels both need this step")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

# ------------------------------------------------------------------ data (all from c2.NB())
NB = c2.NB()
CAM = sc.EX_CAM
R2, t2, C2 = sc.cameras()[CAM]
XS = sc.X_STAR
UV = sc.numbers()["uv"]                               # white-dot pixel in camera 2 (sub-pixel)
D_STAR = float(NB["D"])
LS_D, LS_DT, LS_N, LS_RHS = NB["lsq_D"], NB["lsq_Dt"], NB["lsq_N"], NB["lsq_rhs"]
S_HAT, B_HAT, RMS = float(NB["s_hat"]), float(NB["b_hat"]), float(NB["lsq_rms"])
NAMES = NB["lsq_names"]
RES = S_HAT * LS_DT + B_HAT - LS_D                   # residual r_k = fitted value − GT (m)

# 5 pixels: same as lines 89-100 of e02_numbers.py
_PICK = [(float(UV[0]), float(UV[1]))]
for _sp in sc.SPHERES:
    _Xc = _sp["c"] - np.array([0, 0, _sp["r"]])
    _p, _ = sc.project(R2, t2, sc.to_world(_Xc))
    _PICK.append((float(_p[0]), float(_p[1])))
_PICK += [(259.0, 500.0), (60.0, 330.0)]
PICK = np.array(_PICK)
PCOL = [(255, 255, 255), (95, 145, 245), (245, 205, 70), (235, 135, 95), (140, 215, 200)]
PNAME = ["White dot", "Blue ball", "Yellow ball", "Near ground", "Far ground"]

# ------------------------------------------------------------------ 3D scene (#1-#2)
CLOUD, CRGB, _, _ = ec.cloud(5)
_cen = [sc.cameras()[i][2] for i in range(3)]
FIT = 2.0 * np.r_[np.zeros((1, 3)), np.array(_cen), [XS], ec.plane_corners(0, 0.45), ec.plane_corners(1, 0.45),
                  ec.plane_corners(2, 0.45), CLOUD[::7]]
OC = ec.viewer(az=-4.0, el=52.0, r=7.0, fit_pts=FIT, box=(900, 400), center=(540, 495),
               target=2.0 * np.mean([np.zeros(3), C2, XS], 0) * 0.75)
IMG = ec.view_rgb(CAM)
PA = ImagePanel(IMG, (100, 905), w=330)
PB = ImagePanel(IMG, (650, 905), w=330)
PH = ImagePanel(IMG, (60, 490), w=340)

# ------------------------------------------------------------------ formulas
F_REL = Formula([(r"D\approx", "D"), (r"s\,\cdot", "s"), (r"\tilde D+", "dt"), ("b", "b")], size=66, gap=0.2)
REL_COL = {"D": style.LABEL_GOLD, "s": style.LABEL_ORANGE, "dt": style.LABEL_CYAN, "b": style.LABEL_ORANGE}
F_ARG = Formula([(r"(\hat s,\hat b)=", "hat"), (r"\arg\min_{s,b}\ \sum_k(", "op"), ("s", "s"), (r"\tilde D_k", "dt"),
                 ("+b", "b"), ("-D_k", "dk"), (r")^{2}", "sq")], size=46, gap=0.12)
ARG_COL = {"hat": style.LABEL_ORANGE, "s": style.LABEL_ORANGE, "b": style.LABEL_ORANGE, "dt": style.LABEL_CYAN,
           "dk": style.LABEL_GOLD}
ARG_XY = (540, 330)
F_RES = Formula([r"r_k=\hat s\,\tilde D_k+\hat b-D_k"], size=40)
F_RMS = Formula([r"\mathrm{RMS}=\sqrt{\frac{1}{n}\sum_k r_k^{2}}"], size=40)

# normal equations (symbolic / numeric)
M_SYM = [[r"\sum\tilde D_k^{2}", r"\sum\tilde D_k"], [r"\sum\tilde D_k", "n"]]
V_SYM = [[r"\hat s"], [r"\hat b"]]
R_SYM = [[r"\sum\tilde D_k D_k"], [r"\sum D_k"]]
M_NUM = [[float(LS_N[0, 0]), float(LS_N[0, 1])], [float(LS_N[1, 0]), "5"]]
R_NUM = [[float(LS_RHS[0])], [float(LS_RHS[1])]]

# ------------------------------------------------------------------ plots
PL = Plot2D((480, 490, 1010, 830), (-0.1, 1.1), (1.0, 3.0), xlabel=None, ylabel=None,
            xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[1.0, 1.5, 2.0, 2.5, 3.0], fmt_x="{:g}", fmt_y="{:g}",
            tick_size=22)
PR = Plot2D((100, 1040, 420, 1200), (0.5, 5.5), (-1.5, 1.5), xticks=[(k, str(k)) for k in range(1, 6)],
            yticks=[-1, 0, 1], fmt_y="{:g}", tick_size=22, label_size=22,
            ylabel="Residual $r_k$ (cm); x axis $k$ = pixel index")
LBL_OFF = [((12, -10), "lb"), ((12, -10), "lb"), ((-12, 12), "ra"), ((12, -10), "lb"), ((12, -10), "lb")]

# popup: #3 definition + #4 background (two parts of the same window)
POP = Popup((50, 300, 980, 640), t_in=7.5, t_out=16.0, title="Relative depth vs. metric depth",
            corner="tr", accent=GOLD)


@shot.check
def check(ck):
    ck.eq("white dot z-depth D on screen", f"{D_STAR:.3f}", "1.975")
    ck.eq("white-dot depth on screen after scene + cameras ×2", f"{float(NB['scale2_D']):.3f}", "3.951")
    ck.close("×2 = double the depth", float(NB["scale2_D"]), 2 * D_STAR, 1e-9)
    for k in (2.0,):                                   # pixels unchanged (same as E01 A-18): white-dot projection in cameras 1-3
        for i in range(3):
            Ri, ti, ci = sc.cameras()[i]
            uv0, _ = sc.project(Ri, ti, XS)
            uvk, _ = sc.project(Ri, -Ri @ (k * ci), k * XS)
            ck.close(f"×{k:g}: camera {i + 1} white-dot pixel unchanged", uvk, uv0, 1e-9)
    # 5 pixels: recomputed z-depth = lsq_D from the script; camera 2 pixels also unchanged after ×2
    _, dz, _ = sc.render(CAM, PICK)
    ck.close("GT depth of the 5 pixels = lsq_D", dz, LS_D, 1e-9)
    ck.close("white-dot pixel = the white dot of e02_numbers", PICK[0], UV, 1e-9)
    for j in range(5):
        pj, _ = sc.project(R2, -R2 @ (2.0 * C2), 2.0 * sc.unproject(R2, t2, PICK[j], dz[j]))
        ck.close(f"pixel {j + 1} position unchanged after ×2", pj, PICK[j], 1e-8)
    # illustrative relative depth = GT ÷ 1.5 − 0.8 + small noise
    ck.true("D~ = D/1.5 − 0.8 + ε (|ε| ≤ 0.015)", float(np.abs(LS_DT - (LS_D / 1.5 - 0.8)).max()) <= 0.0151,
            f"{np.abs(LS_DT - (LS_D / 1.5 - 0.8)).max():.4f}")
    # normal equations / solution
    A = np.stack([LS_DT, np.ones_like(LS_DT)], 1)
    ck.close("N = AᵀA", A.T @ A, LS_N, 1e-12)
    ck.close("rhs = AᵀD", A.T @ LS_D, LS_RHS, 1e-12)
    ck.close("N·(ŝ, b̂) = rhs", LS_N @ np.array([S_HAT, B_HAT]), LS_RHS, 1e-10)
    s_f, b_f = Plot2D.fit_line(LS_DT, LS_D)
    ck.close("Plot2D.fit_line s = ŝ", s_f, S_HAT, 1e-10)
    ck.close("Plot2D.fit_line b = b̂", b_f, B_HAT, 1e-10)
    ck.close("np.polyfit = (ŝ, b̂)", np.polyfit(LS_DT, LS_D, 1), [S_HAT, B_HAT], 1e-10)
    ck.close("RMS recomputed", float(np.sqrt(np.mean(RES ** 2))), RMS, 1e-12)
    ck.eq("left side of the normal equations on screen: 1.827 / 2.624 / 5", (f"{LS_N[0, 0]:.3f}", f"{LS_N[0, 1]:.3f}", f"{LS_N[1, 0]:.3f}",
                                         f"{LS_N[1, 1]:.0f}"), ("1.827", "2.624", "2.624", "5"))
    ck.eq("right side of the normal equations on screen: 5.906 / 9.930", (f"{LS_RHS[0]:.3f}", f"{LS_RHS[1]:.3f}"), ("5.906", "9.930"))
    ck.eq("ŝ on screen", f"{S_HAT:.3f}", "1.545")
    ck.eq("b̂ on screen (m)", f"{B_HAT:.3f}", "1.175")
    ck.eq("RMS residual on screen (m)", f"{RMS:.3f}", "0.008")
    ck.eq("RMS residual ≈ 0.8 cm", f"{RMS * 100:.1f}", "0.8")
    ck.true("residuals all lie within the small plot's y range (cm)", float(np.abs(RES * 100).max()) < 1.5, f"{np.abs(RES * 100).max():.2f}")
    ck.true("all 5 points lie within the large plot", bool(np.all((LS_DT > -0.1) & (LS_DT < 1.1) & (LS_D > 1.0) & (LS_D < 3.0))))
    ck.eq("caption count = 8 storyboard sentences", len(CAPS), 8)
    popup_check(ck, [POP], DUR)
    ck.record(D=round(D_STAR, 4), D2=round(float(NB["scale2_D"]), 4), s_hat=round(S_HAT, 4), b_hat=round(B_HAT, 4),
              rms=round(RMS, 5), lsq_D=LS_D.round(4), lsq_Dt=LS_DT.round(4), picks=PICK.round(1),
              residual_cm=(RES * 100).round(3))


# ------------------------------------------------------------------ frame
def text_at(cv, s, xy, size, col=style.TEXT, a=1.0, anchor="lm"):
    if a > 0.01:
        cv.rich(s, xy, size, alpha=a, fill=col, anchor=anchor)


def eq_row(cv, cy, MA, MV, MR, size, a, fmtA="{:.3f}", pa_A=None, pa_R=None, colV=style.LABEL_ORANGE, box_R=None):
    """One row: matrix · vector = matrix (total width centered at x = 540)."""
    rg = 0.75
    wA = matrix(cv, (0, 0), MA, fmtA, size, alpha=0, row_gap=rg)[1]
    wV = matrix(cv, (0, 0), MV, "{:.3f}", size, alpha=0, row_gap=rg)[1]
    wR = matrix(cv, (0, 0), MR, "{:.3f}", size, alpha=0, row_gap=rg)[1]
    wa, wv, wr = wA[2] - wA[0], wV[2] - wV[0], wR[2] - wR[0]
    gap = 0.55 * size
    weq = 0.9 * size
    x = 540 - (wa + wv + wr + weq + 3 * gap) / 2
    matrix(cv, (x + wa / 2, cy), MA, fmtA, size, alpha=a, part_alpha=pa_A, row_gap=rg)
    x += wa + gap
    matrix(cv, (x + wv / 2, cy), MV, "{:.3f}", size, color=colV, alpha=a, row_gap=rg)
    x += wv + gap
    cv.math("=", (x + weq / 2, cy), size, style.MATH, a)
    x += weq + gap
    _, bb = matrix(cv, (x + wr / 2, cy), MR, "{:.3f}", size, alpha=a, part_alpha=pa_R, box=box_R, row_gap=rg)
    return bb


def frame(cv, t):
    dm = 1.0 - 0.6 * POP.on(t)
    # ================================================================= #1-#2: 3D + two photos
    a1 = ss(t, 0.2, 0.8) * (1.0 - ss(t, 15.6, 16.3))
    if a1 > 0.01:
        kk = lerp(1.0, 2.0, ss(t, 1.0, 3.2))
        P = OC.at(az=-4.0 + 8.0 * ss(t, 0.0, 8.0))
        ec.splat(cv, P, kk * CLOUD, CRGB, 0.5 * a1 * dm, size=2)
        for i in range(3):
            ec.frustum(cv, P, i, 0.45, a1 * dm, tex_alpha=0.5 if i == CAM else 0.0, label=ec.CAM_NAMES[i], scale=kk)
        xs = P(ec.disp(kk * XS))
        ec.star_marker(cv, xs, a1 * dm, glow=0.7 + 0.5 * bump(t, 3.2, 1.0), rad=6, label="dot")
        # magnification factor
        text_at(cv, "$\\lambda$ =", (70, 322), 34, style.LABEL_ORANGE, a1 * dm)
        number(cv, (140, 322), kk, "{:.1f}", 38, style.LABEL_ORANGE, a1 * dm, "lm")
        text_at(cv, r"$\lambda$: factor by which the whole scene (cameras included) is scaled about cam 1", (70, 362), 22, style.TEXT_DIM, a1 * dm)
        # two camera-2 photos
        for P_, x0, title, aa, val_fn in [(PA, 100, "Before: cam 2 photo", ss(t, 0.4, 0.9), lambda: D_STAR),
                                         (PB, 650, "After: cam 2 photo", ss(t, 1.0, 1.5), lambda: D_STAR * kk)]:
            al = a1 * aa * dm
            if al > 0.01:
                text_at(cv, title, (x0, 832), 26, style.TEXT_DIM, al)
                text_at(cv, "White dot z-depth $D$ =", (x0, 872), 28, style.TEXT, al)
                number(cv, (x0 + 232, 872), val_fn(), "{:.3f}", 40, style.LABEL_GOLD, al, "lm", unit="m")
                P_.draw(cv, al, border=style.GREY, border_alpha=0.7)
                P_.mark(cv, UV[0], UV[1], (255, 255, 255), rad=5, alpha=al)
        a_eq = ss(t, 4.2, 4.8) * a1 * dm
        if a_eq > 0.01:
            cv.math("=", (540, 1060), 78, style.MATH, a_eq)
            text_at(cv, "Pixel-identical", (540, 1118), 26, style.TEXT, a_eq, "mm")
            text_at(cv, "$D$: distance of the white dot along cam 2's optical axis (m)", (540, 262), 26, style.TEXT_DIM, a_eq, "mm")
        a_n = ss(t, 5.4, 6.0) * a1 * dm
        text_at(cv, "One image holds no geometry that fixes scale", (540, 788), 30, style.LABEL_GOLD, a_n, "mm")

    # ================================================================= #5-#8: scatter plot
    a2 = ss(t, 16.0, 16.6)
    if a2 > 0.01:
        a_f = a2 * ss(t, 16.1, 16.7)
        pl_a = a2 * ss(t, 17.0, 17.5)
        # formula + symbol definitions
        F_ARG.draw(cv, ARG_XY, alpha=a_f, part_color=ARG_COL,
                   part_glow={"dt": bump(t, 18.0, 1.0), "dk": bump(t, 18.6, 1.0), "hat": bump(t, 19.7, 1.0),
                              "sq": 0.9 * bump(t, 27.2, 1.4)})
        a_d = a2 * ss(t, 16.7, 17.2)
        text_at(cv, r"$k$: pixel index ($n$ = 5 pixels in all); $\tilde D_k$: illustrative relative depth; $D_k$: GT (m)", (540, 392), 22,
                style.TEXT_DIM, a_d, "mm")
        text_at(cv, r"$s,\ b$: scale and offset to find; $\hat s,\ \hat b$: solved estimates; argmin: the $s,\ b$ that minimize the right side", (540, 428),
                22, style.TEXT_DIM, a_d, "mm")
        # left: camera-2 photo + 5 pixels
        a_h = a2 * ss(t, 16.9, 17.4)
        if a_h > 0.01:
            PH.draw(cv, a_h, border=style.GREY, border_alpha=0.7)
            text_at(cv, "Cam 2 photo", (60, 474), 24, style.TEXT_DIM, a_h)
            for k in range(5):
                ak = a_h * ss(t, 17.2 + 0.3 * k, 17.6 + 0.3 * k)
                PH.mark(cv, PICK[k, 0], PICK[k, 1], PCOL[k], rad=5, alpha=ak, label=str(k + 1),
                        label_offset=(10, -8), size=26, label_color=PCOL[k])
                text_at(cv, f"{k + 1} {PNAME[k]}", (60 + (0, 110, 220, 0, 170)[k], 856 + (0, 0, 0, 34, 34)[k]), 24,
                        PCOL[k], ak)
        # right: scatter plot
        if pl_a > 0.01:
            PL.axes(cv, pl_a)
            text_at(cv, r"$\tilde D$ (relative depth, illustration)", ((480 + 1010) / 2, 885), 26, style.LABEL_CYAN, pl_a, "mm")
            text_at(cv, "$D$ (m, GT)", (480, 474), 26, style.LABEL_GOLD, pl_a, "lb")
            prog = ss(t, 17.6, 19.6)
            Pp = PL.scatter(cv, LS_DT, LS_D, PCOL, prog, pl_a, rad=6)
            for k in range(5):
                ak = pl_a * ss(prog * 6 - k, 0, 1)
                off, anc = LBL_OFF[k]
                text_at(cv, str(k + 1), (Pp[k][0] + off[0], Pp[k][1] + off[1]), 24, PCOL[k], ak, anc)
            # end of #6: fitted line; #7: residual vertical lines
            lp = ss(t, 23.9, 25.3)
            if lp > 0.01:
                PL.line(cv, S_HAT, B_HAT, ORANGE, lp, pl_a, th=3)
            rp = ss(t, 26.0, 28.0)
            if rp > 0.01:
                PL.residuals(cv, LS_DT, LS_D, S_HAT, B_HAT, RED, rp, pl_a, th=2)
            # in-plot result (from the end of #6)
            ar = ss(t, 25.4, 25.9) * pl_a
            if ar > 0.01:
                text_at(cv, "$\\hat s$ =", (505, 520), 30, style.LABEL_ORANGE, ar)
                number(cv, (565, 520), S_HAT, "{:.3f}", 34, style.LABEL_ORANGE, ar, "lm")
                text_at(cv, "$\\hat b$ =", (505, 560), 30, style.LABEL_ORANGE, ar)
                number(cv, (565, 560), B_HAT, "{:.3f}", 34, style.LABEL_ORANGE, ar, "lm", unit="m")
        # ---- #6: normal equations (symbolic -> numeric -> solution)
        a6 = a2 * ss(t, 20.6, 21.1) * (1.0 - ss(t, 25.4, 25.9))
        if a6 > 0.01:
            pa_A = {(0, 0): 0.25 + 0.75 * ss(t, 21.2, 21.6), (0, 1): 0.25 + 0.75 * ss(t, 21.6, 22.0),
                    (1, 0): 0.25 + 0.75 * ss(t, 21.6, 22.0), (1, 1): 0.25 + 0.75 * ss(t, 22.0, 22.4)}
            pa_R = {(0, 0): 0.25 + 0.75 * ss(t, 22.4, 22.8), (1, 0): 0.25 + 0.75 * ss(t, 22.8, 23.2)}
            eq_row(cv, 992, M_SYM, V_SYM, R_SYM, 28, a6, pa_A=pa_A, pa_R=pa_R)
            text_at(cv, r"$\sum$: sum over $k$ = 1…5; $n$ = 5 is the pixel count; $\hat s,\ \hat b$ are the two unknowns to solve", (540, 1078), 22,
                    style.TEXT_DIM, a6 * ss(t, 21.0, 21.5), "mm")
            a6n = a6 * ss(t, 22.9, 23.4)
            if a6n > 0.01:
                eq_row(cv, 1150, M_NUM, V_SYM, R_NUM, 30, a6n, box_R={(0, 0): bump(t, 23.4, 1.0), (1, 0): bump(t, 23.8, 1.0)})
            a6s = a6 * ss(t, 24.2, 24.7)
            if a6s > 0.01:
                text_at(cv, "Solution: $\\hat s$ = " + f"{S_HAT:.3f}" + ", $\\hat b$ = " + f"{B_HAT:.3f}" + " m", (540, 1224), 28,
                        style.LABEL_ORANGE, a6s, "mm")
        # ---- #7: residuals
        a7 = a2 * ss(t, 25.9, 26.4) * (1.0 - ss(t, 29.8, 30.3))
        if a7 > 0.01:
            F_RES.draw(cv, (745, 996), alpha=a7)
            text_at(cv, "$r_k$: fitted value − GT (red vertical lines in the large plot)", (745, 1030), 22, style.TEXT_DIM, a7, "mm")
            text_at(cv, "Too short to see; magnified (cm) at lower left", (745, 1060), 22, style.TEXT_DIM, a7, "mm")
            PR.axes(cv, a7)
            PR.stem(cv, np.arange(1, 6), RES * 100, RED, ss(t, 26.4, 28.2), a7)
            a7r = a7 * ss(t, 27.8, 28.4)
            if a7r > 0.01:
                F_RMS.draw(cv, (640, 1136), alpha=a7r, part_glow={0: bump(t, 28.2, 1.0)})
                number(cv, (815, 1139), RMS, "{:.3f}", 38, style.LABEL_ORANGE, a7r, "lm", unit="m")
                text_at(cv, "= 0.8 cm (RMS residual, $n$ = 5, on these 5 illustrative points only)", (745, 1194), 24, style.LABEL_ORANGE, a7r,
                        "mm")
        # ---- #8: two label cards
        for k, (t0, name, col, line) in enumerate([(30.3, "Depth Pro", c2.DP_COLOR, "datasets without meters are normalized before the loss (E02-06)"),
                                                   (32.8, "DA3", c2.DA3_COLOR, "teacher relative depth aligned to sparse depth (E02-09)")]):
            a8 = a2 * ss(t, t0, t0 + 0.5)
            if a8 > 0.01:
                y0 = 980 + 110 * k
                cv.s.rect(60, y0, 1020, y0 + 84, (24, 28, 36), 0.92 * a8, r=14)
                cv.s.line((60, y0 + 6), (60, y0 + 78), col, a8, 6)
                text_at(cv, name + ": " + line, (90, y0 + 42), 30, style.TEXT, a8)

    # ================================================================= #3-#4: popup
    def body(sub, lt, box):
        a1_ = ss(lt, 0.3, 0.8)
        F_REL.draw(sub, (490, 118), alpha=a1_, part_color=REL_COL,
                   part_glow={"D": bump(lt, 0.9, 0.8), "dt": bump(lt, 1.9, 0.8), "s": bump(lt, 2.7, 0.8),
                              "b": bump(lt, 3.1, 0.8)})
        rows = [(0.9, 196, "$D$: metric depth, every pixel directly in meters", style.LABEL_GOLD),
                (1.9, 236, r"$\tilde D$: relative depth output by the model", style.LABEL_CYAN),
                (2.7, 276, "$s$ scale, $b$ offset: one pair per image, unknown beforehand", style.LABEL_ORANGE),
                (3.5, 316, "Such output is also called affine-invariant", style.TEXT)]
        for t0, y, s, col in rows:
            a = ss(lt, t0, t0 + 0.4)
            if a > 0.01:
                sub.rich(s, (60, y), 28, alpha=a, fill=col, anchor="lm")
        a2_ = ss(lt, 4.4, 4.8)
        if a2_ > 0.01:
            sub.s.line((40, 366), (940, 366), GREY, 0.5 * a2_, 1)
            sub.badge((940, 396), "Background note", 22, a2_, anchor="ra")
            sub.rich("The offset $b$ must be free too:", (60, 396), 28, alpha=a2_, fill=style.LABEL_ORANGE, anchor="lm")
            rows2 = [(4.9, 448, "Mixed datasets: disparity has unknown baseline and zero point", style.TEXT),
                     (5.8, 492, "→ neither scale nor offset matches", style.LABEL_ORANGE),
                     (6.6, 540, "Disparity = x-coordinate difference of the same point in a stereo pair (px)", style.TEXT_DIM)]
            for t0, y, s, col in rows2:
                a = ss(lt, t0, t0 + 0.4)
                if a > 0.01:
                    sub.rich(s, (60, y), 26 if col != style.TEXT_DIM else 22, alpha=a, fill=col, anchor="lm")

    POP.draw(cv, t, body)


shot.frame(frame)
