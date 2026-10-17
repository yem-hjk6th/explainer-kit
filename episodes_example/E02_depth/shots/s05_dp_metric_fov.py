"""E02-05 Depth Pro 2: decoding -> canonical inverse depth -> meters; FOV head (M, 41 s). Storyboard storyboard.md section E02-05.
Computed for real: camera 2's C map (= fw / GT z-depth, background inf not drawn), white dot C = 0.438, D_m = f_px / (w C) = 1.975 m,
  a 1 m wide bar covers 227.1 px = 0.438 × 518 (drawn with similar triangles in the popup), f_px = 259 / tan 30° = 448.6,
  decoder stream sizes 96 / 48 / 24 -> 768 / 384 / 192 / 96 / 48 -> multiples up to 1536.
Read from code (tag "Official code"): #1 the decoder's 5 feature streams, upsampling sizes, level-by-level fusion, output head ×2.
Derivation (tag "Derivation"): #4 the 1 m wide bar, #5 "estimate C from the image".
Paper: structure of the FOV head (section 3.3, C.2), 256 channels (Tab. 16), delta 25% of Tab. 3 (Depth Pro 64.6, second best SPEC 34.6; all on-screen numbers are reproduced).
Measured (tag "Measured on our test scene"): #8 Depth Pro on camera 2: θ = 71.0°, f_px = 363.1, white dot C = 0.557,
  white-dot depth 1.555 m (with the GT focal length) / 1.259 m (with its own estimated focal length).
Illustration: #1 the small animation in the popup (DPT upsample -> add -> conv).
"""
import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, ORANGE, VIOLET, WHITE, Formula, ImagePanel, Popup, Shot, arrow, arrow_path, bar_chart,
                     bump, colorbar, colorize, fade, grid_square, module_box, popup_check, pulse_dot, ss, style)

ec, sc = c2.ec, c2.sc
NB = c2.NB()
EN = ec.nb()                                           # scene numbers (the white-dot pixel is in EN["uv"])
MT = c2.metrics()["depth_pro"]

DUR = 42.0
W = sc.W                                               # 518
UV = EN["uv"]
D_W, C_W, FW = NB["D"], NB["C"], NB["fw"]
F_PX = FW * W                                          # 448.6
PX1M = NB["px_1m"]                                     # 227.1
TH_GT = float(sc.FOV_DEG)                              # 60°

shot = Shot("s05_dp_metric_fov", DUR)
shot.title("Decoding, metric scale", "E02-05 · Depth Pro · canonical inverse depth + FOV head", 0.2, 2.6)

# ---------------- timeline ----------------
T1_0, T1_1 = 0.3, 8.6            # #1 decoder
T2, T3, T4, T5 = 8.6, 12.4, 16.4, 22.2
T6, T7, T8, T9 = 25.2, 29.4, 32.8, 38.0
T_B0, T_B1 = 8.4, 25.0           # left: C map + right: formula
T_F0, T_F1 = 25.0, 32.7          # FOV head
T_8_0, T_8_1 = 32.6, 38.2
T_9_0 = 37.9

CAPS = [(0.4, 4.4, "Decoder fuses coarse to fine"),
        (8.8, 12.2, "Output C: canonical inverse depth"),
        (12.4, 16.2, "To meters: D equals f divided by w times C"),
        (16.4, 21.8, "C: fraction of image width ÷ actual width"),
        (22.2, 25.0, "This ratio is readable from the image"),
        (25.2, 29.2, "Focal length goes to the FoV head"),
        (29.4, 32.6, "FoV 60°, focal length 448.6 px"),
        (32.8, 37.6, "Our test scene: estimated 71°, depth 19% shorter still"),
        (38.0, 42.0, "Paper: more images with focal error within 25%")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

shot.tag(c2.CODE_TAG, T1_0, T1_1)
shot.tag(c2.GT_TAG, T_B0 + 0.2, T_B1)
shot.tag(c2.DERIV_TAG, T4, T6 - 0.4)
shot.tag("Paper §3.3, C.2", T_F0 + 0.2, T7)
shot.tag(c2.GT_TAG, T7, T_F1)
shot.tag(c2.SCENE_TAG, T_8_0 + 0.2, T_8_1)
shot.tag("Paper Tab. 3", T_9_0 + 0.2, DUR)

# ================= #1 decoder =================
SIZES = [768, 384, 192, 96, 48]                        # side lengths of the 5 streams after upsampling
SRC = [96, 96, 96, 48, 24]                             # side lengths of each stream before upsampling
OPS = [s // r for s, r in zip(SIZES, SRC)]             # 8 4 2 2 2
ROW_Y = [420 + 118 * k for k in range(5)]
DISP = [100, 84, 68, 52, 38]                           # displayed square side (not to scale)
NODE_X, SQ_X = 760, 530
FEATS = [("Layer 5 intermediate features", "$96\\times96$"), ("Layer 11 intermediate features", "$96\\times96$"),
         ("Scale 1 output", "$96\\times96$"), ("Scale 2 output", "$48\\times48$"),
         ("Scale 3 output $24^2$", "and image encoder output $24^2$")]
NOTE5 = "Each of the two streams ×2 to $48^2$, concatenated, then 1×1 conv"          # encoder.py L305-324: upsample2 / upsample_lowres → cat → fuse_lowres
T_FUSE = [7.0, 6.4, 5.8, 5.2, 4.6]                     # time when the node of row k lights up (bottom to top)
FUSE_PATH = np.array([[NODE_X, ROW_Y[4]], [NODE_X, ROW_Y[0]]], float)

POP1 = Popup((60, 960, 900, 280), t_in=4.2, t_out=8.2, title="DPT (covered in E01-05)", corner="bl", accent=CYAN,
             point_to=(NODE_X, ROW_Y[4] + 18), tag="Illustration")


def body_dpt(sub, lt, box):
    sub.rich("Feature maps are upsampled, added, and convolved step by step,\nending in a per-pixel map", (24, 70), 24, alpha=ss(lt, 0.2, 0.6), anchor="la")
    y = 190
    steps = [(110, 34, "Feature map (small)", CYAN, 0.8), (320, 56, "Upsample", CYAN, 1.5), (480, 56, "Add", ORANGE, 2.2),
             (680, 56, "Conv", CYAN, 3.0)]
    for x, side, lab, col, t0 in steps:
        a = ss(lt, t0, t0 + 0.4)
        grid_square(sub, (x, y), side, 5, None, col, a)
        sub.rich(lab, (x, y + 52), 20, alpha=a, fill=style.TEXT_DIM, anchor="ma")
    for (x0, x1, t0, lab) in [(140, 285, 1.1, "×2"), (350, 450, 1.9, ""), (515, 645, 2.7, "")]:
        u = ss(lt, t0, t0 + 0.4)
        arrow(sub, (x0, y), (x1, y), WHITE, u, 2, 10)
    sub.text((212, y - 18), "×2", 20, "light", ss(lt, 1.3, 1.7), style.TEXT_DIM, "mb")
    # the second input of the addition (features of the next level) comes in from above
    a = ss(lt, 2.0, 2.4)
    grid_square(sub, (400, y - 62), 36, 5, None, VIOLET, a)
    arrow(sub, (410, y - 42), (462, y - 30), VIOLET, a, 2, 9)
    sub.rich("Other feature stream", (440, y - 60), 18, alpha=a, fill=style.TEXT_DIM, anchor="lm")
    sub.rich("→ per-pixel map", (775, y), 22, alpha=ss(lt, 3.4, 3.8), fill=style.TEXT_DIM, anchor="lm")


def phase1(cv, t, A):
    if A <= 0.01:
        return
    heads = [(195, "5 feature streams", 0.6), (408, "Upsample", 1.4), (SQ_X, "After upsampling", 1.6), (NODE_X, "Fuse", 3.6),
             (945, "Output", 7.0)]
    for x, s, t0 in heads:
        cv.rich(s, (x, 338), 22, alpha=A * ss(t, t0, t0 + 0.4), fill=style.TEXT_DIM, anchor="mm")
    for k in range(5):
        y = ROW_Y[k]
        a1 = A * ss(t, 0.6 + 0.25 * k, 1.1 + 0.25 * k)
        title, sub = FEATS[k]
        module_box(cv, (30, y - 40, 360, y + 40), title, sub, CYAN, a1, title_size=26, sub_size=20)
        a2 = A * ss(t, 1.6 + 0.2 * k, 2.1 + 0.2 * k)
        arrow(cv, (368, y), (446, y), WHITE, a2, 2, 10)
        if k == 4:
            cv.rich(f"each ×{OPS[k]}", (407, y - 14), 22, alpha=a2, fill=style.TEXT_DIM, anchor="mb")
            cv.rich(NOTE5, (380, y + 46), 20, alpha=a2, fill=style.LABEL_CYAN, anchor="lm")
        else:
            cv.text((407, y - 14), f"×{OPS[k]}", 22, "light", a2, style.TEXT_DIM, "mb")
        grid_square(cv, (SQ_X, y), DISP[k], 6, None, CYAN, a2)
        cv.rich(f"${SIZES[k]}^2$", (SQ_X + DISP[k] / 2 + 12, y), 24, alpha=a2, fill=style.TEXT, anchor="lm")
        # fusion node
        an = A * ss(t, T_FUSE[k] - 0.4, T_FUSE[k] + 0.1)
        if an > 0.01:
            arrow(cv, (668, y), (NODE_X - 20, y), WHITE, an, 2, 9)
            cv.s.dot((NODE_X, y), ORANGE, 15, 0.45 * an)
            cv.g.dot((NODE_X, y), ORANGE, 15, 0.5 * an * bump(t, T_FUSE[k], 1.0))
            if k == 4:
                cv.rich("Start", (NODE_X + 28, y), 20, alpha=an, fill=style.LABEL_ORANGE, anchor="lm")
            else:
                cv.text((NODE_X, y), "+", 28, "light", an, (20, 20, 28), "mm")
        if k < 4:                                      # fuse upward from the next (coarser) level
            u = ss(t, T_FUSE[k] - 0.5, T_FUSE[k])
            arrow(cv, (NODE_X, ROW_Y[k + 1] - 18), (NODE_X, y + 18), ORANGE, A, 3, 12, progress=u)
            if u > 0.5:
                cv.text((NODE_X + 22, (y + ROW_Y[k + 1]) / 2), "×2", 20, "light", A * ss(u, 0.5, 1), style.LABEL_ORANGE, "lm")
    # light dot travels bottom to top
    pu = ss(t, 4.6, 7.0)
    if pu > 0.02:
        pulse_dot(cv, FUSE_PATH, pu, GOLD, A * (1 - ss(t, 7.0, 7.3)))
    # output head
    ao = A * ss(t, 7.0, 7.5)
    if ao > 0.01:
        arrow(cv, (NODE_X + 20, ROW_Y[0]), (850, ROW_Y[0]), ORANGE, ao, 3, 12)
        module_box(cv, (852, ROW_Y[0] - 40, 1040, ROW_Y[0] + 40), "Head", "Upsample ×2", CYAN, ao, title_size=26,
                   sub_size=20, glow=0.6 * bump(t, 7.2, 1.0))
        a3 = A * ss(t, 7.5, 8.0)
        arrow(cv, (946, ROW_Y[0] + 44), (946, 560), WHITE, a3, 2, 10)
        grid_square(cv, (946, 640), 130, 8, None, CYAN, a3, glow=0.5 * bump(t, 7.7, 1.0))
        cv.rich("$1536\\times1536$", (946, 722), 24, alpha=a3, fill=style.TEXT, anchor="ma")
        cv.rich("$C$: canonical inverse depth", (946, 758), 22, alpha=a3, fill=style.LABEL_CYAN, anchor="ma")
        cv.rich("Channels 256 (Tab. 16)", (946, 822), 22, alpha=a3, fill=style.TEXT_DIM, anchor="ma")


# ================= #2-#5 C map + formula =================
CAM_D = c2.cam2_depth()
VALID = np.isfinite(CAM_D)
C_IMG = np.where(VALID, FW / np.where(VALID, CAM_D, 1.0), np.nan)
C_LO, C_HI = float(np.nanmin(C_IMG)), float(np.nanmax(C_IMG))
STOPS_REV = [(1 - u, c) for u, c in reversed(style.CMAP_DEPTH)]     # large C (near) = bright
C_COL = colorize(C_IMG, C_LO, C_HI, stops=STOPS_REV)
PAN = ImagePanel(C_COL, (60, 400), w=400)
MARK_XY = np.array(PAN.to_screen(UV[0], UV[1]))

POP2 = Popup((572, 380, 468, 400), t_in=8.9, t_out=12.3, title="$C$: canonical inverse depth", corner="tr",
             accent=CYAN, title_size=20, title_dy=3)

FML = Formula([("D_m", "D"), ("=", "eq"), ("f_{px}", "f"), (r"\,/\,(", "sl"), ("w", "w"), (r"\,C", "C"), (")", "rp")],
              size=54, gap=0.08)
FML_XY = (790, 450)
COL = {"D": GOLD, "f": ORANGE, "w": WHITE, "C": CYAN}

# popup 3: 1 m wide bar (similar triangles, drawn for real)
POP3 = Popup((60, 850, 960, 350), t_in=T4, t_out=22.0, title="How wide a 1 m wide object appears in the image", corner="bl", accent=GOLD)
K_PL, S_M = 0.3, 150.0                                 # image-plane display scale; displayed pixels per 1 m
O_X, O_Y = 50.0, 205.0
PL_X = O_X + F_PX * K_PL                               # image plane distance from the center
BAR_X = O_X + D_W * S_M                                # bar distance from the center
HALF_BAR = 0.5 * S_M
PL_HALF = 0.5 * W * K_PL
INT_HALF = HALF_BAR * (PL_X - O_X) / (BAR_X - O_X)     # intercept of the ray on the image plane (half width)
INT_PX = 2 * INT_HALF / K_PL                           # converted back to image pixels


def body_bar(sub, lt, box):
    a0 = ss(lt, 0.2, 0.6)
    sub.s.dot((O_X, O_Y), WHITE, 6, a0)
    sub.rich("center", (O_X, O_Y + 20), 20, alpha=a0, fill=style.TEXT_DIM, anchor="ma")
    sub.s.line((PL_X, O_Y - PL_HALF), (PL_X, O_Y + PL_HALF), WHITE, 0.9 * a0, 3)
    sub.rich(f"Image plane, width $w={W}$ px", (PL_X, O_Y - PL_HALF - 12), 20, alpha=a0, fill=style.TEXT_DIM, anchor="mb")
    a1 = ss(lt, 0.9, 1.4)
    sub.s.line((BAR_X, O_Y - HALF_BAR), (BAR_X, O_Y + HALF_BAR), GOLD, a1, 6)
    sub.g.line((BAR_X, O_Y - HALF_BAR), (BAR_X, O_Y + HALF_BAR), GOLD, 0.6 * a1, 10)
    sub.rich(f"1 m wide\n(depth $D={D_W:.3f}$ m)", (BAR_X + 14, O_Y - HALF_BAR - 4), 20, alpha=a1, fill=style.LABEL_GOLD,
             anchor="la")
    a2 = ss(lt, 1.6, 2.4)
    for sgn in (-1, 1):                                # from the center through both ends of the bar, extended to the image plane
        sub.s.line((O_X, O_Y), (O_X + (BAR_X - O_X) * a2, O_Y + sgn * HALF_BAR * a2), GOLD, 0.7 * a1, 1)
        sub.s.dot((PL_X, O_Y + sgn * INT_HALF), GOLD, 5, ss(lt, 2.0, 2.4))
    a3 = ss(lt, 2.4, 3.0)
    sub.s.line((PL_X, O_Y - INT_HALF), (PL_X, O_Y + INT_HALF), GOLD, a3, 6)
    sub.s.line((PL_X - 24, O_Y + 56), (PL_X - 2, O_Y + INT_HALF), GOLD, 0.8 * a3, 1)
    sub.rich(f"{PX1M:.1f} px", (PL_X - 28, O_Y + 62), 22, alpha=a3, fill=style.LABEL_GOLD, anchor="ra")
    # text on the right: f_px / D = 227.1; 227.1 / 518 = C
    X = 450
    sub.rich(f"Similar triangles: pixels covered $=f_{{px}}\\times1/D$", (X, 92), 22, alpha=ss(lt, 2.6, 3.0), anchor="lm")
    sub.rich(f"$={F_PX:.1f}\\times1/{D_W:.3f}={PX1M:.1f}$ px", (X, 138), 26, alpha=ss(lt, 3.2, 3.6), fill=style.MATH,
             anchor="lm")
    sub.rich(f"Divide by image width $w={W}$ px: $\\dfrac{{{PX1M:.1f}}}{{{W}}}={C_W:.3f}$", (X, 206), 26, alpha=ss(lt, 3.9, 4.3),
             fill=style.LABEL_CYAN, anchor="lm")
    sub.rich(f"= the white dot's $C$", (X, 270), 22, alpha=ss(lt, 4.3, 4.7), fill=style.TEXT_DIM,
             anchor="lm")


def body_c(sub, lt, box):
    sub.rich("Output: one number $C$ per pixel, unit 1/m", (16, 74), 22, alpha=ss(lt, 0.2, 0.6), anchor="lm")
    sub.rich("$C=\\dfrac{f_{px}}{w}\\cdot\\dfrac{1}{D}$", (16, 150), 38, alpha=ss(lt, 0.7, 1.1), fill=style.MATH,
             anchor="lm")
    rows = [(1.4, "$D$: z-depth of the pixel (m)"), (1.8, "$f_{px}$: focal length (px)"), (2.2, f"$w$: image width (px) = {W}"),
            (2.6, "Closer means larger $C$")]
    for k, (t0, s) in enumerate(rows):
        sub.rich(s, (16, 214 + 36 * k), 21, alpha=ss(lt, t0, t0 + 0.35), fill=style.TEXT_DIM, anchor="lm")
    sub.rich(f"Our test scene, cam 2: $C$ ranges from {C_LO:.2f} to {C_HI:.2f}", (16, 368), 21, alpha=ss(lt, 3.0, 3.4), fill=style.LABEL_CYAN,
             anchor="lm")


NUM_X = 556


def phaseB(cv, t, A):
    if A <= 0.01:
        return
    a_p = A * ss(t, T_B0 + 0.1, T_B0 + 0.6)
    PAN.draw(cv, a_p, border=c2.CAM2_COLOR, title="Cam 2: $C$ map (computed from GT)", title_size=24)
    colorbar(cv, (474, 400, 494, 800), C_LO, C_HI, "", "{:.2f}", stops=STOPS_REV, alpha=a_p, size=22)
    a_m = A * ss(t, T2 + 0.4, T2 + 0.9)
    if a_m > 0.01:
        ec.star_marker(cv, tuple(MARK_XY), a_m, glow=0.8, rad=5)
        cv.rich(f"White dot $C={C_W:.3f}$", tuple(MARK_XY + np.array([0, 24])), 26, alpha=a_m, fill=style.LABEL_CYAN, anchor="ma")
    # ---- formula (#3) ----
    a_f = A * ss(t, T3 - 0.1, T3 + 0.4)
    if a_f > 0.01:
        st = {"D": "dim", "eq": "lit", "f": "dim", "sl": "lit", "w": "dim", "C": "dim", "rp": "lit"}
        notes = {}
        t_f, t_w, t_c, t_d = T3 + 0.5, T3 + 1.3, T3 + 2.1, T3 + 3.0
        if t > t_f:
            st["f"] = "hl" if t < t_w else "lit"
            notes["f"] = "Focal length (px)"
        if t > t_w:
            st["w"] = "hl" if t < t_c else "lit"
            notes["w"] = "Image width (px)"
        if t > t_c:
            st["C"] = "hl" if t < t_d else "lit"
            notes["C"] = "Canonical inverse depth"
        if t > t_d:
            st["D"] = "hl"
            notes["D"] = "Metric depth (m)"
        dim = 1 - 0.55 * ss(t, T4, T4 + 0.4) * (1 - ss(t, 22.0, 22.4))
        FML.draw(cv, FML_XY, alpha=a_f * dim, state=st, part_color=COL, notes=notes)
        # numbers
        rows = [(t_f, f"$f_{{px}}$ = {F_PX:.1f} px", ORANGE, 620), (t_w, f"$w$ = {W} px", WHITE, 664),
                (t_c, f"$C$ = {C_W:.3f}", CYAN, 708)]
        for t0, s, col, y in rows:
            cv.rich(s, (NUM_X, y), 30, alpha=a_f * dim * ss(t, t0, t0 + 0.4), fill=col, anchor="lm")
        a_r = a_f * dim * ss(t, t_d, t_d + 0.5)
        cv.rich(f"$D_m$ = {F_PX:.1f} / ({W} × {C_W:.3f})", (NUM_X, 764), 30, alpha=a_r, fill=GOLD, anchor="lm")
        cv.rich(f"= {D_W:.3f} m", (NUM_X + 34, 810), 36, alpha=a_r * ss(t, t_d + 0.5, t_d + 1.0), fill=GOLD, anchor="lm")
    # ---- #5 text (after the popup closes) ----
    a5 = A * ss(t, T5, T5 + 0.4)
    if a5 > 0.01:
        cv.rich("$C$: canonical inverse depth, unit 1/m = fraction of image width the object covers ÷ object's actual width (m)", (540, 850), 28, alpha=a5, anchor="ma", align="m")
        cv.rich(f"Example: 1 m wide, covers {PX1M / W:.3f} of image width → $C={C_W:.3f}$", (540, 906), 28,
                alpha=a5 * ss(t, T5 + 0.7, T5 + 1.1), fill=style.LABEL_CYAN, anchor="ma", align="m")
        cv.rich("Covered width: measured from the image; usual size: from priors; no focal length needed", (540, 962), 26,
                alpha=a5 * ss(t, T5 + 1.4, T5 + 1.8), fill=style.TEXT_DIM, anchor="ma", align="m")


# ================= #6-#7 FOV head =================
CW, CGAP, CX0 = 262, 56, 90
CONV = [(CX0 + k * (CW + CGAP), 520) for k in range(3)]
CONV_TXT = [("Conv 1", "Kernel 3 · stride 2", "Channels 128 → 64"), ("Conv 2", "Kernel 3 · stride 2", "Channels 64 → 32"),
            ("Conv 3", "Kernel 6 · stride 1", "Channels 32 → 1")]
FOV_FML = Formula([("f_{px}", "f"), ("=", "e1"), (r"\frac{w/2}{\tan(\theta/2)}", "def"), ("=", "e2"),
                   (r"\frac{259}{\tan 30^{\circ}}", "num"), (r"\ =448.6", "res")], size=46, gap=0.18)
FOV_XY = (540, 1000)


def phaseF(cv, t, A):
    if A <= 0.01:
        return
    a_in = A * ss(t, T_F0 + 0.4, T_F0 + 0.9)
    module_box(cv, (60, 340, 500, 416), "Frozen depth network features", None, CYAN, a_in, title_size=26)
    module_box(cv, (540, 340, 980, 416), "Features of the third ViT (FoV only)", None, VIOLET, a_in, title_size=26)
    a_ar = A * ss(t, T_F0 + 1.0, T_F0 + 1.6)
    arrow(cv, (280, 418), (CONV[0][0] + 100, CONV[0][1] - 2), WHITE, a_ar, 2, 12)
    arrow_path(cv, np.array([(760, 418), (760, 466), (CONV[0][0] + 170, 466), (CONV[0][0] + 170, CONV[0][1] - 2)], float),
                    WHITE, a_ar, 2, 12)
    cv.rich("Two input streams", (540, 450), 22, alpha=a_ar, fill=style.TEXT_DIM, anchor="mm")
    for k, (x, y) in enumerate(CONV):
        ak = A * ss(t, T_F0 + 1.6 + 0.35 * k, T_F0 + 2.1 + 0.35 * k)
        tt, l1, l2 = CONV_TXT[k]
        module_box(cv, (x, y, x + CW, y + 80), tt, None, CYAN, ak, title_size=28)
        cv.rich(l1, (x + CW / 2, y + 92), 22, alpha=ak, fill=style.TEXT_DIM, anchor="ma")
        cv.rich(l2, (x + CW / 2, y + 124), 22, alpha=ak, fill=style.LABEL_CYAN, anchor="ma")
        if k < 2:
            arrow(cv, (x + CW + 4, y + 40), (x + CW + CGAP - 4, y + 40), WHITE, ak, 2, 10)
    # small feature-map size sketches (illustration): after the sum 24² -> 12² -> 6² -> 1
    for gx, half, n, lab, t0 in [(52, 22, 6, "$24^2$", 0.4), (380, 17, 4, "$12^2$", 1.6),
                                 (698, 13, 3, "$6^2$", 1.95), (1016, 6, 0, "1", 2.3)]:
        ag = A * ss(t, T_F0 + t0, T_F0 + t0 + 0.5)
        if n:
            grid_square(cv, (gx, 640), 2 * half, n, None, CYAN, ag)
        else:
            cv.s.dot((gx, 640), CYAN, half, ag)
        cv.rich(lab, (gx, 640 + half + 8), 20, alpha=ag, fill=style.TEXT, anchor="ma")
    cv.rich("Small maps: illustrative", (540, 704), 20, alpha=A * ss(t, T_F0 + 2.4, T_F0 + 2.9), fill=style.TEXT_DIM, anchor="mm")
    # θ output
    a_o = A * ss(t, T_F0 + 2.9, T_F0 + 3.4)
    TX = CONV[2][0] + CW / 2
    arrow(cv, (TX, CONV[2][1] + 150), (TX, 718), WHITE, a_o, 2, 12)
    module_box(cv, (TX - 150, 722, TX + 150, 790), r"$\theta$: horizontal field of view", None, GOLD, a_o, title_size=26,
               glow=0.6 * bump(t, T_F0 + 3.2, 1.0))
    cv.rich("1 number", (TX, 800), 22, alpha=a_o, fill=style.TEXT_DIM, anchor="ma")
    cv.rich("Trained separately after depth training,\nwith L2 loss (squared prediction-label difference)", (60, 730), 22, alpha=A * ss(t, T_F0 + 3.4, T_F0 + 3.9),
            fill=style.TEXT_DIM, anchor="la")
    # light dot
    up = ss(t, T_F0 + 1.4, T_F0 + 4.2)
    allp = np.array([(CONV[0][0] + 130, 470), (CONV[0][0] + 130, 560), (CONV[2][0] + CW / 2, 560), (TX, 722)], float)
    if up > 0.02:
        pulse_dot(cv, allp, up, GOLD, A * (1 - ss(t, T_F0 + 4.2, T_F0 + 4.5)))
    # #7: θ -> f_px
    a7 = A * ss(t, T7 - 0.1, T7 + 0.4)
    if a7 > 0.01:
        cv.rich(f"Our test scene GT $\\theta={TH_GT:.0f}^{{\\circ}}$", (TX, 828), 24, alpha=a7, fill=style.LABEL_GOLD,
                anchor="ma")
        st = {"f": "lit", "e1": "lit", "def": "lit", "e2": "dim", "num": "dim", "res": "dim"}
        if t > T7 + 1.0:
            st["e2"], st["num"] = "lit", "lit"
        if t > T7 + 1.8:
            st["e2"], st["res"] = "lit", "hl"
        FOV_FML.draw(cv, FOV_XY, alpha=a7, state=st, part_color={"f": ORANGE, "def": WHITE, "num": WHITE, "res": ORANGE},
                     notes={"def": f"$w$ = image width {W} px; $\\theta$ = horizontal field of view"} if t > T7 + 0.3 else {})
        cv.rich("$f_{px}$: focal length (px); $w/2=259$", (540, 1100), 24, alpha=a7 * ss(t, T7 + 1.4, T7 + 1.9),
                fill=style.TEXT_DIM, anchor="ma", align="m")


# ================= #8 Depth Pro measured =================
BX0, BY0, BX1, BY1 = 110, 590, 660, 960
VMAX8 = 2.2
RATIO = MT["f_px_est"] / F_PX
SC8 = {"fov": MT["fov_deg"], "f": MT["f_px_est"], "d_est": MT["D_white_est"], "d_gtf": MT["D_white_gtf"],
       "c": MT["C_white"]}


def phase8(cv, t, A):
    if A <= 0.01:
        return
    t0 = T_8_0 + 0.3
    cv.rich("Focal length Depth Pro estimates itself on cam 2:", (70, 346), 26, alpha=A * ss(t, t0, t0 + 0.4), fill=style.TEXT_DIM,
            anchor="la")
    cv.rich(f"$\\theta$ = {SC8['fov']:.1f}° (GT {TH_GT:.0f}°)", (70, 392), 32, alpha=A * ss(t, t0 + 0.4, t0 + 0.9),
            fill=style.TEXT, anchor="la")
    cv.rich(f"$f_{{px}}$ = {SC8['f']:.1f} (GT {F_PX:.1f})", (70, 442), 32, alpha=A * ss(t, t0 + 0.9, t0 + 1.4),
            fill=ORANGE, anchor="la")
    cv.rich("White dot depth $D_m$ (same $C$ map, only the focal length changed):", (70, 512), 24, alpha=A * ss(t, t0 + 1.6, t0 + 2.0),
            fill=style.TEXT_DIM, anchor="la")
    pg = ss(t, t0 + 1.8, t0 + 3.0)
    if pg > 0.01:
        bar_chart(cv, (BX0, BY0, BX1, BY1), [SC8["d_gtf"], SC8["d_est"]], ["With GT focal length\n$f_{px}$=448.6", "With its own estimated focal length\n$f_{px}$=363.1"],
                  vmax=VMAX8, color=CYAN, highlight=1, hl_color=ORANGE, progress=pg, fmt="{:.3f}", alpha=A, unit="m",
                  value_size=28, label_size=22)
        yref = BY1 - (BY1 - BY0) * D_W / VMAX8
        ar = A * ss(t, t0 + 2.6, t0 + 3.2)
        cv.s.dashed((BX0, yref), (BX1, yref), WHITE, 0.8 * ar, 2, 40)
        cv.rich(f"GT {D_W:.3f} m", (BX1 + 8, yref), 24, alpha=ar, fill=style.TEXT, anchor="lm")
    # formula on the right
    xr = 730
    a1 = A * ss(t, t0 + 3.2, t0 + 3.7)
    cv.rich("A smaller focal estimate shrinks $D_m$ in proportion:", (xr, 680), 22, alpha=a1, fill=style.TEXT_DIM, anchor="la")
    cv.rich(f"{SC8['d_gtf']:.3f} × $\\dfrac{{{SC8['f']:.1f}}}{{{F_PX:.1f}}}$", (xr, 730), 28,
            alpha=A * ss(t, t0 + 3.6, t0 + 4.0), fill=style.MATH, anchor="la")
    cv.rich(f"= {SC8['d_gtf']:.3f} × {RATIO:.3f}", (xr, 824), 28, alpha=A * ss(t, t0 + 4.0, t0 + 4.4), fill=style.MATH,
            anchor="la")
    cv.rich(f"= {SC8['d_est']:.3f} m", (xr, 870), 30, alpha=A * ss(t, t0 + 4.3, t0 + 4.7), fill=ORANGE, anchor="la")
    cv.rich(f"{(1 - RATIO) * 100:.0f}% shorter than {SC8['d_gtf']:.3f} m", (xr, 926), 24, alpha=A * ss(t, t0 + 4.5, t0 + 4.9),
            fill=style.LABEL_ORANGE, anchor="la")
    cv.rich("The whole depth map shrinks by the same ratio", (xr, 966), 22, alpha=A * ss(t, t0 + 4.7, t0 + 5.1), fill=style.TEXT_DIM,
            anchor="la")
    a2 = A * ss(t, t0 + 4.6, t0 + 5.2)
    cv.rich(f"Even with the GT focal length only {SC8['d_gtf']:.3f} m (GT {D_W:.3f} m):\n$C$ itself is also too large; Depth Pro's white dot $C$ = {SC8['c']:.3f}, "
            f"GT-derived {C_W:.3f}", (540, 1062), 24, alpha=a2, fill=style.TEXT, anchor="ma", align="m")


# ================= #9 paper Tab. 3 =================
def phase9(cv, t, A):
    if A <= 0.01:
        return
    bar_chart(cv, (260, 500, 920, 800), [64.6, 34.6], ["Depth Pro", "SPEC (second best)"], vmax=100.0, color=CYAN, highlight=0,
              hl_color=style.GREEN, progress=ss(t, T_9_0 + 0.5, T_9_0 + 2.0), fmt="{:.1f}", title="Tab. 3 (PPR10K): $\\delta_{25\\%}$",
              better="higher", note="Source: Depth Pro paper Tab. 3", alpha=A, unit="%", label_size=26)
    cv.rich(r"$\delta_{25\%}$ = fraction of images with focal-length relative error below 25%", (540, 900), 26,
            alpha=A * ss(t, T_9_0 + 1.2, T_9_0 + 1.7), anchor="ma", align="m")
    cv.rich("Denominator = all PPR10K test images; PPR10K is the paper's dataset, not our scene", (540, 950), 22,
            alpha=A * ss(t, T_9_0 + 1.8, T_9_0 + 2.3), fill=style.TEXT_DIM, anchor="ma", align="m")


# ================= check =================
@shot.check
def check(ck):
    f = lambda x, n: f"{x:.{n}f}"
    # decoder (read from code)
    ck.eq("side lengths of the 5 streams after upsampling", SIZES, [768, 384, 192, 96, 48])
    ck.eq("upsampling factors 96->768 ×8, 96->384 ×4, 96->192 ×2, 48->96 ×2, 24->48 ×2", OPS, [8, 4, 2, 2, 2])
    ck.eq("each level of the fusion chain ×2 (48->96->192->384->768)", [SIZES[k] // SIZES[k + 1] for k in range(4)], [2, 2, 2, 2])
    ck.eq("output head ×2: 768 -> 1536", SIZES[0] * 2, 1536)
    ck.eq("scale 3 and the image encoder are each ×2 before concatenation: 24->48", [24 * 2, 24 * 2], [48, 48])
    ck.eq("FOV head feature maps 24->12->6->1 (illustration)", [24, 24 // 2, 24 // 4, 1], [24, 12, 6, 1])
    ck.eq("image width w", W, 518)
    # C and D_m (computed for real)
    ck.eq("white-dot C on screen", f(C_W, 3), "0.438")
    ck.eq("white-dot D on screen", f(D_W, 3), "1.975")
    ck.eq("f_px on screen", f(F_PX, 1), "448.6")
    ck.eq("pixels covered by 1 m width on screen", f(PX1M, 1), "227.1")
    ck.close("C = f_px / (w D)", C_W, F_PX / (W * D_W), 1e-9)
    ck.close("D_m = f_px / (w C) = 1.975", F_PX / (W * C_W), D_W, 1e-9)
    ck.close("227.1 / 518 = C", PX1M / W, C_W, 1e-9)
    ck.close("similar triangles in the popup: intercept converted to pixels = 227.1", INT_PX, PX1M, 1e-6)
    ck.true("the bar in the popup is farther than the image plane (no drawing glitch)", BAR_X > PL_X)
    ck.eq("C range on screen", [f(C_LO, 2), f(C_HI, 2)], ["0.22", "0.77"])
    ck.close("C min / max = NB", [C_LO, C_HI], [NB["C_min"], NB["C_max"]], 1e-9)
    px = int(round(UV[0])), int(round(UV[1]))
    ck.close("C map at the white-dot pixel ≈ white-dot C (sub-pixel difference)", C_IMG[px[1], px[0]], C_W, 0.01)
    ck.true("C-map background = nan (not drawn)", bool(np.isnan(C_IMG[~VALID]).all()) and int((~VALID).sum()) > 0)
    # focal length
    ck.close("f_px = (w/2) / tan(θ/2) = 259 / tan 30°", (W / 2) / np.tan(np.radians(TH_GT / 2)), F_PX, 1e-6)
    ck.eq("w/2 on screen: 259", int(W / 2), 259)
    ck.eq("GT θ on screen: 60", int(TH_GT), 60)
    # measured
    ck.eq("θ on screen", f(MT["fov_deg"], 1) + "°", "71.0°")
    ck.eq("f_px (estimated) on screen", f(MT["f_px_est"], 1), "363.1")
    ck.eq("white-dot depth with the GT focal length on screen", f(MT["D_white_gtf"], 3), "1.555")
    ck.eq("white-dot depth with its own estimated focal length on screen", f(MT["D_white_est"], 3), "1.259")
    ck.eq("ratio on screen", f(RATIO, 3), "0.809")
    ck.eq("white-dot C (Depth Pro) on screen", f(MT["C_white"], 3), "0.557")
    ck.close("1.555 × 363.1 / 448.6 ≈ 1.259", 1.555 * 363.1 / 448.6, 1.259, 0.002)
    ck.close("D_gtf × f_est / f_gt = D_est (exact)", MT["D_white_gtf"] * RATIO, MT["D_white_est"], 1e-3)
    ck.eq("19% shorter on screen", f"{(1 - RATIO) * 100:.0f}%", "19%")
    ck.close("GT C in the metrics = C of this shot", MT["C_white_truth"], C_W, 1e-9)
    ck.true("1.555 with the GT focal length is still below the GT 1.975", MT["D_white_gtf"] < D_W)
    # paper numbers
    ck.eq("Tab. 3 numbers", [64.6, 34.6], [64.6, 34.6])
    popup_check(ck, [POP1, POP2, POP3], DUR)
    ck.record(C_range=[round(C_LO, 3), round(C_HI, 3)], white_uv=np.round(UV, 1).tolist(), ratio=round(RATIO, 4))


@shot.frame
def frame(cv, t):
    phase1(cv, t, fade(t, T1_0, T1_1, 0.4, 0.4))
    phaseB(cv, t, fade(t, T_B0, T_B1, 0.4, 0.5))
    phaseF(cv, t, fade(t, T_F0, T_F1, 0.5, 0.4))
    phase8(cv, t, fade(t, T_8_0, T_8_1, 0.4, 0.4))
    phase9(cv, t, fade(t, T_9_0, DUR, 0.4, 0.0))
    POP1.draw(cv, t, body_dpt)
    POP2.draw(cv, t, body_c)
    POP3.draw(cv, t, body_bar)

