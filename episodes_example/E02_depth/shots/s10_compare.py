"""E02-10 explanatory comparison + wrap-up (S, 38 s). Storyboard: storyboard.md §E02-10.
Same white dot, same scene, Depth Pro on the left and DA3 on the right: what input each needs, what it outputs, how it represents depth internally, where meters come from, and what each is suited for.
Corner tag "Our take" for the whole shot; the table and the bottom of each row state whether it is the paper text, the official code, or our own understanding.
Computed: all numbers of the #7 small table (`c2.metrics()`, measured model output); the checks assert the strings shown on screen.
Illustration: none (the thumbnails in the split screen are the real images from E01; the white dot position is converted from the GT pixel).
Paper numbers: #6 DA3 paper Tab. 11 (ETH3D δ1: DA3-metric 0.917, Depth Pro 0.386; reported by the DA3 authors).
Code reading: #4 right column (README L235, api.py Umeyama).
"""
import numpy as np

import e02_common as c2
from animkit import GOLD, ORANGE, Shot, bar_chart, ss, style, table
from animkit.layout import input_slot

ec = c2.ec
DUR = 38.0
T_SPLIT_END = 13.6          # split screen ends, #5 table comes in
T_TAB5_END = 19.6
T_BAR_END = 24.2

shot = Shot("s10_compare", DUR)
shot.title("Comparison", "Depth Pro vs DA3", 0.2, 2.6)
shot.tag(c2.VIEW_TAG, 0.2, DUR)
shot.tag(c2.CODE_TAG, 10.4, T_SPLIT_END)
shot.tag("DA3 paper Tab. 11", T_TAB5_END + 0.2, T_BAR_END)
shot.tag(c2.SCENE_TAG, T_BAR_END, DUR)

CAPS = [(0.3, 3.2, "The same white dot: what each method needs"), (3.2, 6.0, "What it outputs"), (6.0, 9.0, "How depth is represented inside"),
        (9.0, 13.5, "Where meters come from"), (13.6, 19.4, "What each suits"), (19.6, 24.0, "Metric comparisons from the paper"),
        (24.2, 31.1, "Our scene: shapes all correct"), (31.1, DUR, "Meters: from priors or external focal length / pose")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

M = c2.metrics()
DP, D3L, D3M, TR = M["depth_pro"], M["da3_large"], M["da3_metric"], M["truth"]
D_TRUE = TR["D_white"]


def _pct(x, nd):
    s = f"{100 * (x / D_TRUE - 1):+.{nd}f}".replace("-", "−")
    return s


ROWS7 = [
    ("1. Depth Pro\nown focal estimate", f"{DP['D_white_est']:.3f} m ({_pct(DP['D_white_est'], 0)} %)", f"{DP['stats_est_raw']['absrel']:.3f}",
     f"{DP['stats_est_scale']['absrel']:.3f}"),
    ("2. Depth Pro\nGT focal length", f"{DP['D_white_gtf']:.3f} m ({_pct(DP['D_white_gtf'], 0)} %)",
     f"{DP['stats_gtf_raw']['absrel']:.3f}", "—"),
    ("3. DA3-metric\nGT focal length", f"{D3M['D_white_m']:.3f} m ({_pct(D3M['D_white_m'], 1)} %)",
     f"{D3M['stats_raw']['absrel']:.3f}", "—"),
    ("4. DA3-Large\nno pose", f"{D3L['A_nopose_camhead']['D_white_raw']:.3f}\n(normalized units)", "—",
     f"{D3L['A_nopose_camhead']['stats_scale']['absrel']:.4f}"),
    ("5. DA3-Large\nGT pose", f"{D3L['C_gtpose']['D_white_scaled']:.3f} m ({_pct(D3L['C_gtpose']['D_white_scaled'], 1)} %)",
     f"{D3L['C_gtpose']['stats_raw']['absrel']:.4f}", "—"),
]
EXPECT7 = [("1.259 m (−36 %)","0.357", "0.030"), ("1.555 m (−21 %)", "0.205", "—"), ("2.136 m (+8.1 %)", "0.082", "—"),
           ("0.918\n(normalized units)", "—", "0.0063"), ("1.978 m (+0.1 %)", "0.0035", "—")]


@shot.check
def check(ck):
    for i, (r, e) in enumerate(zip(ROWS7, EXPECT7)):
        ck.eq(f"#7 row {i + 1} white dot depth on screen", r[1], e[0])
        ck.eq(f"#7 row {i + 1} raw AbsRel on screen", r[2], e[1])
        ck.eq(f"#7 row {i + 1} scale-multiplied AbsRel on screen", r[3], e[2])
    ck.eq("GT white dot depth 1.975", f"{D_TRUE:.3f}", "1.975")
    ck.eq("Tab. 11 DA3-metric", f"{0.917:.3f}", "0.917")
    ck.true("DA3-metric converted with GT focal length (436.5)", abs(D3M["f_504_used"] - 436.477) < 0.01, D3M["f_504_used"])
    ck.record(tab11={"DA3-metric": 0.917, "Depth Pro": 0.386},
              d_white={"dp_est": DP["D_white_est"], "dp_gtf": DP["D_white_gtf"], "da3m": D3M["D_white_m"],
                       "da3L_A_raw": D3L["A_nopose_camhead"]["D_white_raw"],
                       "da3L_C": D3L["C_gtpose"]["D_white_scaled"]})


LX, RX = 290, 790
TW = 104
THUMBS = [ec.view_pil(i) for i in range(3)]
UV = np.array(M["white_uv_518"])
ROW_TOP = [365, 640, 815, 960]
ROW_NAME = ["Input", "Output", "Depth representation", "Where meters come from"]
ROW_T = [0.4, 3.4, 6.2, 9.2]
DIM = style.TEXT_DIM


def chip(cv, box, text, color, alpha, size=22):
    x0, y0, x1, y1 = box
    if alpha <= 0.01:
        return
    cv.s.rect(x0, y0, x1, y1, tuple(int(0.18 * c) for c in color), 0.9 * alpha, r=8)
    cv.s.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, 0.9 * alpha, 2, closed=True)
    cv.rich(text, ((x0 + x1) / 2, (y0 + y1) / 2), size, alpha=alpha, anchor="mm", align="m")


def thumbs(cv, cx, alpha, only_cam2):
    x0 = cx - 1.5 * TW - 12
    for i in range(3):
        x = x0 + i * (TW + 12)
        y = 402
        a = alpha * (1.0 if (not only_cam2 or i == c2.CAM2) else 0.28)
        ec.paste_img(cv, THUMBS[i], (x, y), TW, a)
        cv.s.polyline([(x, y), (x + TW, y), (x + TW, y + TW), (x, y + TW)], ec.CAM_COLORS[i], 0.8 * a, 2, closed=True)
        if i == c2.CAM2:
            ec.star_marker(cv, (x + UV[0] / 518 * TW, y + UV[1] / 518 * TW), alpha=a, glow=0.6, rad=4)


def src(cv, cx, y, text, alpha):
    cv.rich(text, (cx, y), 19, alpha=alpha, fill=(150, 160, 175), anchor="mm", align="m")


def split_view(cv, t, A):
    cv.s.line((540, 300), (540, 1235), style.GREY, 0.5 * A, 1)
    cv.text((LX, 330), "Depth Pro", 36, "bold", ss(t, 0.2, 0.7) * A, c2.DP_COLOR, "mm")
    cv.text((RX, 330), "Depth Anything 3 (DA3)", 32, "bold", ss(t, 0.2, 0.7) * A, c2.DA3_COLOR, "mm")
    for k in range(4):
        a = ss(t, ROW_T[k], ROW_T[k] + 0.4) * A
        if a > 0.01:
            cv.s.line((40, ROW_TOP[k]), (1040, ROW_TOP[k]), style.GREY, 0.35 * a, 1)
            cv.text((50, ROW_TOP[k] + 6), ROW_NAME[k], 20, "bold", a, style.LABEL_GOLD, "la")

    # 1 input
    a1 = ss(t, 0.5, 1.2) * A
    thumbs(cv, LX, a1, True)
    thumbs(cv, RX, a1, False)
    cv.text((LX, 522), "Only cam 2 is used (the other two are not)", 20, alpha=ss(t, 1.2, 1.7) * A, fill=DIM, anchor="mm")
    cv.text((RX, 522), "All three images together", 20, alpha=ss(t, 1.5, 2.0) * A, fill=DIM, anchor="mm")
    a_s = ss(t, 1.8, 2.4) * A
    if a_s > 0.01:
        input_slot(cv, (LX - 100, 540, LX + 100, 580), "Pose: none", optional=False, on=0.0, alpha=a_s, size=20,
                   color=c2.DP_COLOR)
        input_slot(cv, (RX - 110, 540, RX + 110, 580), "Pose (optional)", optional=True, on=1.0, alpha=a_s, size=20,
                   color=c2.DA3_COLOR)
    src(cv, LX, 612, "Paper's own words", ss(t, 2.5, 3.0) * A)
    src(cv, RX, 612, "Paper's own words (dashed = optional)", ss(t, 2.5, 3.0) * A)

    # 2 output
    items = [(LX, 0, "$D_m$: metric depth, in meters", c2.DP_COLOR, 3.7), (LX, 1, r"$\theta$: field of view, FoV (degrees)", c2.DP_COLOR, 4.2),
             (RX, 0, "$D$: depth, not in meters", c2.DA3_COLOR, 3.9), (RX, 1, "Ray map → camera, point cloud", c2.DA3_COLOR, 4.4)]
    for cx, k, txt, col, t0 in items:
        chip(cv, (cx - 230, 674 + 48 * k, cx + 230, 716 + 48 * k), txt, col, ss(t, t0, t0 + 0.5) * A)
    src(cv, LX, 782, "Paper's own words", ss(t, 5.0, 5.5) * A)
    src(cv, RX, 782, "Paper + inference ($D$ is in normalized-scale units)", ss(t, 5.0, 5.5) * A)

    # 3 representation
    a3 = ss(t, 6.4, 6.9) * A
    cv.rich("Canonical inverse depth $C=(f/w)/D_m$", (LX, 874), 25, alpha=a3, anchor="mm", align="m")
    cv.rich("z-depth $D$ × ray direction $d$ (unnormalized)", (RX, 874), 24, alpha=ss(t, 6.8, 7.3) * A, anchor="mm", align="m")
    cv.rich("$f$ focal length, $w$ image width (px): $f/w$ converts back to meters", (LX, 908), 19, alpha=ss(t, 7.4, 7.9) * A, fill=DIM,
            anchor="mm", align="m")
    cv.rich(r"Point = origin $t$ + $D\,d$", (RX, 908), 19, alpha=ss(t, 7.6, 8.1) * A, fill=DIM, anchor="mm", align="m")
    src(cv, LX, 940, "Paper's own words", ss(t, 8.2, 8.7) * A)
    src(cv, RX, 940, "Paper's own words", ss(t, 8.2, 8.7) * A)

    # 4 where meters come from
    L = ["Prior on 'how big objects usually are'", "+ the estimated focal length $f$", "→ converted to meters"]
    for k, s in enumerate(L):
        cv.rich(s, (LX, 1000 + 30 * k), 22, alpha=ss(t, 9.3 + 0.4 * k, 9.8 + 0.4 * k) * A, anchor="mm", align="m")
    R = ["Main model gives no meters; two ways:", "1. DA3-metric, which needs the focal length as input:", "$D$ = focal length × output ÷ 300",
         "2. Give a metric pose: the API uses Umeyama", "to scale the output to the pose's scale,", "the network itself gives no meters"]
    ar = ss(t, 10.2, 10.7) * A
    cv.badge((RX, 972), c2.CODE_TAG, 19, ar, anchor="ma", fill=style.LABEL_GOLD, border=(140, 120, 80))
    for k, s in enumerate(R):
        cv.rich(s, (RX, 1034 + 27 * k), 21, alpha=ss(t, 10.5 + 0.35 * k, 11.0 + 0.35 * k) * A, anchor="mm", align="m")
    src(cv, LX, 1196, "Paper's own words + our reading", ss(t, 11.0, 11.5) * A)
    src(cv, RX, 1196, "Official code (README L235, api.py) + our reading", ss(t, 12.4, 12.9) * A)


def tab5(cv, t, A):
    rows = [("One photo needing real distances and sharp edges\n(e.g. single-image novel view synthesis)", "Depth Pro"),
            ("Several images needing consistent geometry and cameras\n(reconstruction, video, pose estimation)", "DA3"),
            ("Only one image, no meters needed", "Either works\nDA3 on one image is monocular")]
    rev = float(np.clip((t - 14.0) / 1.4, 0, 3))
    table(cv, (90, 380), [560, 340], ["Your situation", "Which to pick"], rows, row_h=150, size=26, reveal=rev, alpha=A)
    cv.text((540, 1060), "The three rows above are our reading of the two papers, not their wording", 24, alpha=A * ss(t, 18.3, 18.8),
            fill=style.TEXT, anchor="mm")


def bar6(cv, t, A):
    bar_chart(cv, (180, 430, 900, 680), [0.917, 0.386], ["DA3-metric", "Depth Pro"], vmax=1.0, horizontal=True,
              highlight=0, progress=ss(t, 20.2, 21.6), fmt="{:.3f}", title="Metric depth $\\delta_1$ (ETH3D dataset)",
              better="higher", alpha=A, note="DA3 paper Tab. 11")
    cv.rich(r"$\delta_1$: fraction of pixels whose prediction/GT ratio lies between 1/1.25 and 1.25", (540, 780), 22,
            alpha=A * ss(t, 21.6, 22.1), fill=DIM, anchor="mm", align="m")
    notes = ["Reported by the DA3 authors; DA3-metric is a separate monocular model, not the main DA3 model",
             "The paper does not say whether Depth Pro used estimated focal length or GT intrinsics"]
    for k, s in enumerate(notes):
        cv.rich(s, (540, 840 + 36 * k), 22, alpha=A * ss(t, 22.2 + 0.6 * k, 22.7 + 0.6 * k), fill=style.TEXT,
                anchor="mm", align="m")


def tab7(cv, t, A):
    X0, Y0, RH = 60, 330, 104
    rev = float(np.clip((t - 25.0) / 1.6, 0, 5))
    rows = [list(r) for r in ROWS7]
    table(cv, (X0, Y0), [330, 250, 150, 170], ["Method\n(where focal length / pose come from)", "White dot depth\n(GT 1.975 m)",
                                                 "Whole-image AbsRel\nlower is better", "AbsRel after\nscaling"],
          rows, row_h=RH, size=23, reveal=rev, alpha=A)
    # bracket between 2. and 3.: "same GT focal length"
    ab = ss(t, 28.4, 28.9) * A
    if ab > 0.01:
        ya, yb = Y0 + RH * 2 + 10, Y0 + RH * 4 - 10
        xb = X0 + 900 + 14
        cv.s.polyline([(xb - 8, ya), (xb, ya), (xb, yb), (xb - 8, yb)], style.LABEL_GOLD, ab, 2)
        cv.text_rot((xb + 26, (ya + yb) / 2), "Same GT focal length", 20, alpha=ab, fill=style.LABEL_GOLD, angle=270)
    n = ["AbsRel: mean of per-pixel |prediction − GT| ÷ GT",
         "Scale = median(GT / prediction), one number per row (row 4: shared by the three images); lower AbsRel is better",
         "Pixel sets: rows 1-2 cam 2 @518, row 3 cam 2 @504, rows 4-5 all three @504",
         "The meters in row 5 come from the API scaling depth to the input pose's scale with Umeyama; the network does not output meters itself"]
    for k, s in enumerate(n):
        cv.rich(s, (X0, 980 + 36 * k), 21, alpha=A * ss(t, 33.0 + 0.8 * k, 33.5 + 0.8 * k), fill=DIM, anchor="lm",
                align="l")


@shot.frame
def frame(cv, t):
    A1 = 1 - ss(t, T_SPLIT_END - 0.4, T_SPLIT_END + 0.1)
    if A1 > 0.01:
        split_view(cv, t, A1)
    A5 = ss(t, T_SPLIT_END, T_SPLIT_END + 0.5) * (1 - ss(t, T_TAB5_END - 0.4, T_TAB5_END + 0.1))
    if A5 > 0.01:
        tab5(cv, t, A5)
    A6 = ss(t, T_TAB5_END, T_TAB5_END + 0.5) * (1 - ss(t, T_BAR_END - 0.5, T_BAR_END))
    if A6 > 0.01:
        bar6(cv, t, A6)
    A7 = ss(t, T_BAR_END, T_BAR_END + 0.5)
    if A7 > 0.01:
        tab7(cv, t, A7)
