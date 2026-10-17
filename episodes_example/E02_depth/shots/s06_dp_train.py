"""E02-06 Depth Pro 3: training -- losses, derivative losses, two stages (M, 38 s). Storyboard: storyboard.md §E02-06.
Computed (asserted by the checks): GT C curve of camera 2, row 297, u 341–381 (= the same row and window as the NB); the two illustrative predictions have equal MAE
  (0.0064); gradient error A = 0 / B = 0.0046; occlusion edge = yellow ball right edge 2.03 m -> ground behind 2.70 m (depth ratio 1.335 > 1.15).
Illustration (corner tag "Illustration: predictions built from GT"): prediction A (whole segment raised), prediction B (Gaussian blur); the example uses adjacent differences.
Paper numbers: Eq. 1 / 2 / 3, MAGE / MALE / MSGE (p.6), two-stage recipe (§3.2, Tab. 17), AbsRel / boundary F1 of Tab. 13 Hypersim.
"""
import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, GREY, ORANGE, Formula, ImagePanel, Plot2D, Shot, arrow, fade, module_box, ss, style,
                     table)

ec, sc = c2.ec, c2.sc
NB = c2.NB()
DUR = 38.0
shot = Shot("s06_dp_train", DUR)
shot.title("Training", "E02-06 · Depth Pro · loss, derivative loss, two stages", 0.2, 2.8)

CAPS = [(0.3, 3.8, "All losses act on inverse depth C"), (3.8, 7.6, "Datasets without meters: normalize first"),
        (7.6, 12.4, "Two predictions, the same MAE"), (12.4, 18.0, "Derivative loss: only the blurry one"),
        (18.0, 21.0, "Three abbreviations: MAGE, MALE, MSGE"), (21.0, 24.0, "Stage 1: real plus synthetic, all data"),
        (24.0, 27.6, "Stage 2: synthetic data only, to sharpen edges"), (27.6, 31.8, "Metric better, boundaries slightly worse"),
        (31.8, DUR, "Boundary metric: adjacent depth ratio exceeds 1+t%")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

shot.tag("Paper Eq. 1, §3.2", 0.3, 7.6)
shot.tag(c2.FAKE_TAG, 7.6, 18.0)
shot.tag("Example uses adjacent differences; the paper uses Scharr / Laplace kernels", 12.4, 18.0)
shot.tag("Paper p.6", 18.0, 21.0)
shot.tag("Paper §3.2", 21.0, 24.0)
shot.tag("Paper §3.2, Tab. 17", 24.0, 27.6)
shot.tag("Paper Tab. 13", 27.6, 31.8)
shot.tag("Paper Eq. 3", 31.8, DUR)
shot.tag(c2.GT_TAG, 31.8, DUR)

# ---------------- Computed: one row of the C curve (same as e02_numbers lines 163–185) ----------------
FW = NB["fw"]
TZ = c2.cam2_depth()
ROW = NB["row"]
CROW = np.where(np.isfinite(TZ[ROW]), FW / TZ[ROW], 0.0)
JUMP = NB["edge_u"]
WIN = slice(JUMP - 20, JUMP + 21)
U = np.arange(JUMP - 20, JUMP + 21, dtype=float)
SEG = CROW[WIN]
_g = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
_g /= _g.sum()
SEGB = np.convolve(np.pad(SEG, 9, mode="edge"), _g, mode="valid")
MAE_B = float(np.mean(np.abs(SEGB - SEG)))
SEGA = SEG + MAE_B
MAE_A = float(np.mean(np.abs(SEGA - SEG)))
GA = float(np.mean(np.abs(np.diff(SEGA) - np.diff(SEG))))
GB = float(np.mean(np.abs(np.diff(SEGB) - np.diff(SEG))))
DSEG = np.diff(SEG)
DA, DB = np.diff(SEGA), np.diff(SEGB)
UM = U[:-1] + 0.5
D0, D1 = NB["edge_D"]
C0, C1 = NB["edge_C"]
D_WIN = np.where(SEG > 0, FW / np.where(SEG > 0, SEG, 1.0), np.nan)

COL_T, COL_A, COL_B = GOLD, CYAN, ORANGE

# camera 2 thumbnail
CAMIMG = np.asarray(ec.view_pil(c2.CAM2))[..., :3]
PAN = ImagePanel(CAMIMG, (60, 350), w=230)
UV = ec.nb()["uv"]

# curve plot
PC = Plot2D((150, 640, 980, 860), (341, 381), (float(SEG.min()) - 0.03, float(SEGA.max()) + 0.115),
            xlabel="Pixel column $u$ (cam 2, row 297)", ylabel="$C$: canonical inverse depth", xticks=[341, 351, 361, 371, 381],
            fmt_y="{:.2f}", tick_size=20, label_size=22, zero_line=False)
PD = Plot2D((150, 975, 980, 1160), (341, 381), (float(DSEG.min()) - 0.02, 0.17),
            xlabel="Pixel column $u$", ylabel="Adjacent-column difference of $C$ (derivative)", xticks=[341, 351, 361, 371, 381],
            fmt_y="{:.2f}", tick_size=20, label_size=22)
PF = Plot2D((150, 380, 980, 620), (341, 381), (1.8, 3.1), yticks=[2.0, 2.4, 2.8], xlabel="Pixel column $u$ (cam 2, row 297)",
            ylabel="$D$: z-depth (m)", xticks=[341, 351, 361, 371, 381], fmt_y="{:.1f}", tick_size=20, label_size=22,
            zero_line=False)

# formulas
EQ1 = Formula([(r"\mathcal{L}_{\mathrm{MAE}}", "L"), (r"(\hat C, C)=", "eq"), (r"\frac{1}{N}\sum_{i}", "N"),
               ("|", "b1"), (r"\hat C_{i}", "ch"), ("-", "m"), (r"C_{i}", "cp"), ("|", "b2")], size=68, gap=0.10)
FNORM = Formula([(r"\bar{x}_{i}", "x"), ("=", "e"), (r"\frac{x_{i}-m}{\sigma}", "f")], size=62, gap=0.14)
EQ2 = Formula([(r"\mathcal{L}_{*,p,M}", "L"), ("=", "e"), (r"\frac{1}{M}\sum_{j=1}^{M}", "M"),
               (r"\frac{1}{N_{j}}\sum_{i}", "N"), ("|", "b1"), (r"\nabla_{*}", "g1"), (r"C^{j}_{i}", "c"), ("-", "m"),
               (r"\nabla_{*}", "g2"), (r"\hat C^{j}_{i}", "ch"), ("|", "b2"), (r"^{p}", "p")], size=40, gap=0.07)
EQP = Formula([(r"P(t)=\frac{\sum c_{d}\wedge c_{\hat d}}{\sum c_{d}}", "P")], size=44)
EQR = Formula([(r"R(t)=\frac{\sum c_{d}\wedge c_{\hat d}}{\sum c_{\hat d}}", "R")], size=44)


def txt(cv, s, xy, size, alpha, fill=style.TEXT, anchor="lm", align="l"):
    if alpha > 0.01:
        cv.rich(s, xy, size, alpha=alpha, fill=fill, anchor=anchor, align=align)


# ================= #1–#2 =================
def phaseA(cv, t, A):
    if A <= 0.01:
        return
    pa = {"L": ss(t, 0.6, 1.0), "eq": ss(t, 0.6, 1.0), "N": ss(t, 1.6, 2.0), "b1": ss(t, 2.4, 2.8),
          "ch": ss(t, 2.4, 2.8), "m": ss(t, 2.4, 2.8), "cp": ss(t, 2.8, 3.2), "b2": ss(t, 2.4, 2.8)}
    EQ1.draw(cv, (540, 390), alpha=A, part_alpha=pa, part_color={"ch": GOLD, "cp": CYAN})
    txt(cv, r"$\hat C_i$: GT of pixel $i$ (canonical inverse depth)", (140, 472), 28, A * ss(t, 1.2, 1.6), GOLD)
    txt(cv, "$C_i$: network prediction for pixel $i$", (140, 516), 28, A * ss(t, 2.0, 2.4), CYAN)
    txt(cv, "$N$: number of pixels, $i$ counts from 1 to $N$", (140, 560), 28, A * ss(t, 2.4, 2.8))
    txt(cv, "MAE (Mean Absolute Error):", (140, 598), 25,
        A * ss(t, 2.8, 3.2), style.TEXT_DIM)
    txt(cv, "mean of per-pixel |prediction − GT|; lower is better", (140, 630), 25,
        A * ss(t, 2.8, 3.2), style.TEXT_DIM)
    txt(cv, "Real data: the 20% of pixels with the largest error per image are excluded (GT may be wrong)", (140, 668), 26, A * ss(t, 3.0, 3.4), style.TEXT_DIM)
    # #2
    a2 = A * ss(t, 3.9, 4.3)
    txt(cv, "Non-metric datasets (no reliable intrinsics, inconsistent scale):", (540, 730), 28, a2, GOLD, "mm", "m")
    txt(cv, "Normalize prediction and GT separately, then compute the MAE above", (540, 774), 28, a2 * ss(t, 4.3, 4.7), GOLD, "mm", "m")
    FNORM.draw(cv, (540, 870), alpha=A * ss(t, 4.8, 5.3))
    txt(cv, r"$x$ is $C$ (prediction) and $\hat C$ (GT), computed once each", (140, 960), 26, A * ss(t, 5.3, 5.7))
    txt(cv, "$m$: median of $x$ over the whole image", (140, 1002), 26, A * ss(t, 5.7, 6.1))
    txt(cv, r"$\sigma=\frac{1}{N}\sum_i|x_i-m|$: mean absolute deviation (distance from the median)", (140, 1050), 26, A * ss(t, 6.1, 6.5))
    a_c = A * ss(t, 6.6, 7.0)
    if a_c > 0.01:
        cv.s.rect(140, 1110, 940, 1160, tuple(int(0.18 * c) for c in ORANGE), 0.9 * a_c, r=8)
        cv.rich("Recall E02-03: when scale and offset are unknown, align them first", (540, 1135), 26, alpha=a_c, anchor="mm", align="m")


# ================= #3–#5 =================
def phaseB(cv, t, A):
    if A <= 0.01:
        return
    ap = A * (1 - ss(t, 17.8, 18.3))
    a3 = ap * (1 - ss(t, 12.2, 12.6))                     # the thumbnail and edge notes appear only in #3
    if a3 > 0.01:
        PAN.draw(cv, a3 * ss(t, 7.7, 8.1), border=c2.CAM2_COLOR, title="Cam 2", title_size=22)
        a_h = a3 * ss(t, 8.0, 8.4)
        x0, y0 = PAN.to_screen(341, ROW - 3)
        x1, y1 = PAN.to_screen(381, ROW + 4)
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        cv.s.polyline(pts, ORANGE, a_h, 2, closed=True)
        cv.g.polyline(pts, ORANGE, 0.7 * a_h, 4, closed=True)
        ec.star_marker(cv, tuple(PAN.to_screen(UV[0], UV[1])), a3 * ss(t, 8.2, 8.6), glow=0.6, rad=4)
        txt(cv, "Orange box: row 297, $u$ = 341–381 (41 pixels)", (330, 392), 26, a_h, ORANGE)
        txt(cv, "White dot is in this row too", (330, 436), 24, a3 * ss(t, 8.6, 9.0), style.TEXT_DIM)
        txt(cv, "Occlusion edge ($u$ = 361 to 362):", (330, 486), 26, a3 * ss(t, 9.0, 9.4))
        txt(cv, "Yellow ball right edge %.2f m → ground behind %.2f m" % (D0, D1), (330, 530), 28, a3 * ss(t, 9.2, 9.6), GOLD)
        txt(cv, "Corresponding $C$: %.3f → %.3f" % (C0, C1), (330, 574), 24, a3 * ss(t, 9.6, 10.0), style.TEXT_DIM)
    # ---- C curve ----
    aa = ap * ss(t, 7.8, 8.2)
    if aa > 0.01:
        PC.axes(cv, aa)
        xe = PC.to_screen(JUMP + 0.5, 0)[0]
        cv.s.dashed((xe, PC.box[1]), (xe, PC.box[3]), (230, 230, 230), 0.4 * aa * ss(t, 9.0, 9.4), 1, n=30)
        txt(cv, "Yellow ball right edge %.2f m\n→ ground behind %.2f m" % (D0, D1), (xe - 70, PC.box[3] - 14), 22,
            aa * ss(t, 9.4, 9.8), style.TEXT_DIM, "rb", "r")
        PC.curve(cv, U, SEG, COL_T, ss(t, 8.2, 9.2), aa, th=7, glow=0)
        PC.curve(cv, U, SEGA, COL_A, ss(t, 9.2, 10.2), aa, th=3)
        PC.curve(cv, U, SEGB, COL_B, ss(t, 10.2, 11.2), aa, th=3)
        xr = 970
        txt(cv, r"GT $\hat C$", (xr, PC.box[1] + 14), 22, aa * ss(t, 8.4, 8.8), COL_T, "rm", "r")
        txt(cv, "Predicted $C$ = A (whole segment raised): MAE = %.4f" % MAE_A, (xr, PC.box[1] + 44), 22, aa * ss(t, 10.2, 10.6), COL_A,
            "rm", "r")
        txt(cv, "Predicted $C$ = B (Gaussian blur): MAE = %.4f" % MAE_B, (xr, PC.box[1] + 74), 22, aa * ss(t, 11.2, 11.6), COL_B,
            "rm", "r")
        txt(cv, "Same MAE: MAE cannot tell which is better", (xr, PC.box[1] + 108), 22, aa * ss(t, 11.6, 12.0), style.LABEL_GOLD, "rm",
            "r")
    # ---- Eq. 2 + derivative curve ----
    ae = A * ss(t, 12.6, 13.0)
    if ae > 0.01:
        pa = {k: 1.0 for k in ["L", "e", "M", "N", "b1", "c", "m", "ch", "b2", "g1", "g2", "p"]}
        pg = {"g1": ss(t, 14.4, 14.8), "g2": ss(t, 14.4, 14.8), "p": ss(t, 15.2, 15.6)}
        st = {k: "lit" for k in pa}
        for k in ("g1", "g2"):
            st[k] = "hl" if 14.4 < t < 15.6 else "lit"
        st["p"] = "hl" if 15.4 < t < 16.6 else "lit"
        st["M"] = "hl" if 16.4 < t < 17.4 else "lit"
        EQ2.draw(cv, (540, 345), alpha=ae, state=st, part_alpha=pa,
                 part_color={"c": CYAN, "ch": GOLD})
        txt(cv, r"Derivative loss: penalizes the difference between the predicted and GT rates of change. $\hat C$ = GT, $C$ = prediction", (540, 418), 24, ae * ss(t, 13.0, 13.4),
            style.TEXT, "mm", "m")
        txt(cv, r"$\nabla_*$: Scharr (first-order) / Laplace (second-order) derivative; $p$ = 1 or 2; $M$ = 6 scales; $N_j$: pixel count at that scale",
            (540, 462), 21, ae * ss(t, 14.2, 14.6), style.TEXT_DIM, "mm", "m")
    ad = ap * ss(t, 13.2, 13.6)
    if ad > 0.01:
        PD.axes(cv, ad)
        xe = PD.to_screen(JUMP + 0.5, 0)[0]
        cv.s.dashed((xe, PD.box[1]), (xe, PD.box[3]), (230, 230, 230), 0.4 * ad, 1, n=24)
        PD.curve(cv, UM, DB, COL_B, ss(t, 14.8, 15.8), ad)
        PD.curve(cv, UM, DSEG, COL_T, ss(t, 13.8, 14.8), ad, th=4)
        PD.curve(cv, UM, DA, COL_A, ss(t, 14.8, 15.8), ad, th=2, glow=0, dashed=True)
        xr = 970
        txt(cv, r"Derivative of GT $\hat C$", (xr, PD.box[1] + 14), 22, ad * ss(t, 14.0, 14.4), COL_T, "rm", "r")
        txt(cv, "A's derivative coincides with GT: gradient error = 0", (xr, PD.box[1] + 42), 22, ad * ss(t, 15.8, 16.2), COL_A, "rm", "r")
        txt(cv, "B's derivative is flattened: gradient error = %.4f" % GB, (xr, PD.box[1] + 70), 22, ad * ss(t, 16.2, 16.6), COL_B, "rm",
            "r")
        txt(cv, "Example gradient error = mean absolute difference of derivatives ($p$ = 1, single scale)", (540, 506), 22, ad * ss(t, 16.6, 17.0),
            style.TEXT_DIM, "mm", "m")
    # ---- #5 small table ----
    at = A * ss(t, 18.3, 18.7)
    if at > 0.01:
        table(cv, (100, 560), [150, 260, 130, 80, 260],
              ["Abbrev.", r"Derivative $\nabla_*$", "$p$", "$M$", r"Matches $\mathcal{L}_{\nabla_*,p,M}$"],
              [["MAGE", "Scharr (first-order)", "1 (L1)", "6", "$\\mathcal{L}_{S,1,6}$"],
               ["MALE", "Laplace (second-order)", "1 (L1)", "6", "$\\mathcal{L}_{L,1,6}$"],
               ["MSGE", "Scharr (first-order)", "2 (L2)", "6", "$\\mathcal{L}_{S,2,6}$"]],
              row_h=72, size=26, reveal=(t - 18.8) / 0.6 + 0.001 if t > 18.8 else 0.0, alpha=at)
        txt(cv, "MAGE = Mean Absolute Gradient Error; MALE = Mean Absolute Laplace Error", (540, 920), 22,
            at * ss(t, 19.8, 20.2), style.TEXT_DIM, "mm", "m")
        txt(cv, "MSGE = Mean Squared Gradient Error; L1 = absolute value, L2 = square", (540, 960), 22,
            at * ss(t, 20.0, 20.4), style.TEXT_DIM, "mm", "m")
        txt(cv, "Scales: blur the $C$ map and halve it, repeated up to 6 scales", (540, 1000), 22, at * ss(t, 20.2, 20.6),
            style.TEXT_DIM, "mm", "m")


# ================= #6–#7 =================
def phaseD(cv, t, A):
    if A <= 0.01:
        return
    a1 = A * ss(t, 21.2, 21.7)
    module_box(cv, (60, 340, 640, 420), "Stage 1", "Real + synthetic, all labeled data", CYAN, a1, title_size=30, sub_size=22)
    a2 = A * ss(t, 24.2, 24.7)
    module_box(cv, (660, 340, 1020, 420), "Stage 2", "Synthetic data only", ORANGE, a2, title_size=30, sub_size=22)
    arrow(cv, (60, 452), (1020, 452), style.GREY, A * ss(t, 21.2, 21.7), 2, 12)
    txt(cv, "Training time", (1020, 478), 20, A * ss(t, 21.4, 21.8), style.TEXT_DIM, "rm", "r")
    # stage 1 loss
    x = 70
    txt(cv, "Losses:", (x, 520), 26, A * ss(t, 21.8, 22.2), CYAN)
    txt(cv, "· Metric data: MAE", (x, 566), 26, A * ss(t, 22.0, 22.4))
    txt(cv, "· Non-metric data: MAE after normalization", (x, 610), 26, A * ss(t, 22.4, 22.8))
    txt(cv, "· Gradient losses after normalization (synthetic data only)", (x, 654), 26, A * ss(t, 22.8, 23.2))
    txt(cv, "Goal: learn features that generalize across scenes", (x, 718), 26, A * ss(t, 23.2, 23.6), style.LABEL_GOLD)
    # stage 2
    x = 670
    txt(cv, "Losses:", (x, 520), 26, A * ss(t, 24.6, 25.0), ORANGE)
    txt(cv, "MAE, MSE,", (x, 566), 26, A * ss(t, 24.8, 25.2))
    txt(cv, "MAGE, MALE, MSGE", (x, 610), 26, A * ss(t, 24.8, 25.2))
    txt(cv, "(Tab. 17)", (x, 654), 22, A * ss(t, 24.8, 25.2), style.TEXT_DIM)
    txt(cv, "MSE: mean of the squared prediction-GT difference", (x, 700), 20, A * ss(t, 25.0, 25.4), style.TEXT_DIM)
    txt(cv, "Goal: sharpen edges, bring out detail", (x, 744), 26, A * ss(t, 25.4, 25.8), style.LABEL_GOLD)
    # why
    txt(cv, "Synthetic data: GT accurate at every pixel", (70, 840), 26, A * ss(t, 25.8, 26.2), style.GREEN)
    txt(cv, "Real data: edges have holes and errors", (70, 886), 26, A * ss(t, 26.2, 26.6), style.RED)
    txt(cv, "The reverse of the usual synthetic-then-real order", (70, 960), 28, A * ss(t, 26.8, 27.2), style.LABEL_GOLD)


# ================= #8 =================
def phaseE(cv, t, A):
    if A <= 0.01:
        return
    a = A * ss(t, 27.8, 28.2)
    txt(cv, "Hypersim dataset: two-stage vs. single-stage", (540, 350), 28, a, GOLD, "mm", "m")
    table(cv, (80, 400), [270, 320, 320],
          ["Training", "AbsRel\n(lower is better)", "Boundary F1\n(higher is better)"],
          [["Two-stage", "0.235", "0.465"], ["Single-stage", "0.245", "0.478"]],
          row_h=90, size=28, reveal=(t - 28.4) / 0.6 + 0.001 if t > 28.4 else 0.0, alpha=a)
    txt(cv, "AbsRel: mean of per-pixel |prediction − GT| ÷ GT", (80, 700), 24, A * ss(t, 29.4, 29.8), style.TEXT)
    txt(cv, "Boundary F1: harmonic mean of precision and recall of predicted depth-jump edges against GT", (80, 750), 24, A * ss(t, 29.8, 30.2),
        style.TEXT)
    txt(cv, "(how edges are found comes next)", (80, 790), 22, A * ss(t, 29.8, 30.2), style.TEXT_DIM)
    txt(cv, "Authors: single-stage is a fine default,", (540, 880), 28, A * ss(t, 30.4, 30.8), style.LABEL_GOLD, "mm", "m")
    txt(cv, "Two-stage: better metric, slightly worse boundaries", (540, 926), 28, A * ss(t, 30.6, 31.0), style.LABEL_GOLD, "mm", "m")


# ================= #9 =================
def phaseF(cv, t, A):
    if A <= 0.01:
        return
    a = A * ss(t, 32.0, 32.4)
    PF.axes(cv, a)
    PF.curve(cv, U, D_WIN, GOLD, ss(t, 32.2, 33.0), a, th=3)
    xs = [JUMP, JUMP + 1]
    PF.scatter(cv, xs, [D0, D1], CYAN, ss(t, 33.0, 33.8), a, rad=7,
               labels=["%.3f m" % D0, "%.3f m" % D1], label_offset=(-110, -14))
    txt(cv, "Depth ratio of two adjacent pixels:", (80, 735), 26, A * ss(t, 33.8, 34.2))
    txt(cv, "$%.3f\\,/\\,%.3f = %.3f > 1 + \\frac{t}{100} = 1.15$" % (D1, D0, NB["edge_ratio"]), (540, 795), 34,
        A * ss(t, 34.2, 34.7), GOLD, "mm", "m")
    txt(cv, "$t$ = 15: ratio above 1.15 is marked as an occlusion contour ($c = 1$)", (80, 855), 24, A * ss(t, 34.7, 35.1))
    af = A * ss(t, 35.0, 35.5)
    EQP.draw(cv, (290, 960), alpha=af)
    EQR.draw(cv, (780, 960), alpha=af)
    txt(cv, r"$c_d$: occlusion contours on predicted depth $d$; $c_{\hat d}$: on GT depth $\hat d$", (540, 1040), 22, af * ss(t, 35.3, 35.7),
        style.TEXT_DIM, "mm", "m")
    txt(cv, "One F1 per $t$ (5 to 25), then a weighted average (larger $t$, higher weight)", (540, 1080), 22, af * ss(t, 35.7, 36.1), style.TEXT_DIM, "mm", "m")
    txt(cv, "Matting / segmentation masks as GT: only object outlines, so only recall can be computed", (540, 1140), 24, A * ss(t, 36.4, 36.8), style.LABEL_GOLD,
        "mm", "m")


@shot.check
def check(ck):
    f = lambda x, n: f"{x:.{n}f}"
    ck.eq("row v on screen 297", ROW, 297)
    ck.eq("occlusion edge u on screen 361", JUMP, 361)
    ck.eq("window of 41 pixels", len(SEG), 41)
    ck.close("curve MAE_B = NB", MAE_B, NB["mae_B"], 1e-12)
    ck.close("curve MAE_A = NB", MAE_A, NB["mae_A"], 1e-12)
    ck.close("curve gradient error B = NB", GB, NB["grad_B"], 1e-12)
    ck.close("curve gradient error A = NB", GA, NB["grad_A"], 1e-12)
    ck.close("the two predictions have equal MAE", MAE_A, MAE_B, 1e-12)
    ck.true("gradient error of A = 0", GA < 1e-12, GA)
    ck.eq("MAE on screen 0.0064", [f(MAE_A, 4), f(MAE_B, 4)], ["0.0064", "0.0064"])
    ck.eq("B gradient error on screen 0.0046", f(GB, 4), "0.0046")
    ck.eq("occlusion-edge depths on screen 2.03 / 2.70", [f(D0, 2), f(D1, 2)], ["2.03", "2.70"])
    ck.eq("occlusion-edge C on screen 0.428 / 0.320", [f(C0, 3), f(C1, 3)], ["0.428", "0.320"])
    ck.eq("depths on screen 2.025 / 2.704", [f(D0, 3), f(D1, 3)], ["2.025", "2.704"])
    ck.eq("depth ratio on screen 1.335", f(NB["edge_ratio"], 3), "1.335")
    ck.close("depth ratio = D1 / D0", D1 / D0, NB["edge_ratio"], 1e-12)
    ck.true("ratio > 1 + 15/100", D1 / D0 > 1 + 15 / 100)
    ck.close("C curve in window = fw / D", SEG[20], FW / D0, 1e-12)
    ck.close("D window curve at u=362 = D1", D_WIN[21], D1, 1e-9)
    ck.true("surfaces on both sides in window (C > 0)", bool((SEG > 0).all()))
    ck.close("B is blurred: max derivative shrinks", float(np.abs(DB).max() < np.abs(DSEG).max()), 1.0, 0)
    ck.eq("number of scales M = 6", 6, 6)
    ck.eq("Tab. 13 numbers", [0.235, 0.245, 0.465, 0.478], [0.235, 0.245, 0.465, 0.478])
    ck.record(mae=round(MAE_A, 6), grad_B=round(GB, 6), ratio=round(NB["edge_ratio"], 4))


@shot.frame
def frame(cv, t):
    phaseA(cv, t, fade(t, 0.3, 7.7, 0.4, 0.4))
    phaseB(cv, t, fade(t, 7.6, 21.2, 0.4, 0.4))
    phaseD(cv, t, fade(t, 21.0, 27.8, 0.4, 0.4))
    phaseE(cv, t, fade(t, 27.5, 32.0, 0.4, 0.4))
    phaseF(cv, t, fade(t, 31.8, DUR, 0.4, 0.0))
