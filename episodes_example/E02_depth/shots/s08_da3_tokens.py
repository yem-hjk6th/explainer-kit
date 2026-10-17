"""E02-08 DA3 shot 2: a plain ViT (token rearrangement) + camera token + Dual-DPT (M, 47 s). Storyboard: storyboard.md §E02-08.
Computed: each image has 1 + 1296 = 1297 tokens; the white dot token is no. 737 within its frame (slot 0 + patch 736); cross-view layout 1×1297+737 = 2034
      (reshape index mapping); within-frame / cross-view attention size (3x); 24-layer bar (8 cross-view layers, 16 within-frame layers, first cross-view layer = layer 9);
      9 = 3 + 4 + 2; measured model output (DA3-Large without pose: camera 2 rotation 30.07°; with GT pose: white dot 1.978 m).
Illustration: only 5 tokens per row are drawn (illustrative scale); boxes of the camera token / pose input slot / Dual-DPT structure diagram; the flicker toggle.
Paper numbers: Tab. 7 f / g (HiRoom F1, DTU CD, 7Scenes F1), Tab. 7 a / b / d (HiRoom Auc3), p = 0.2 (§3.4).
Code reading (official code): slot 0 is CLS in layers 0–7 and becomes the camera token from layer 8; QK-norm and 2D RoPE are added from layer 8; layer bar; E_c structure.
"""
import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, ORANGE, VIOLET, Formula, Popup, Shot, TokenStrip, arrow, arrow_path, attn_blocks, bump,
                     fade, flicker, fold, input_slot, module_box, popup_check, pulse_dot, reshape_check, reshape_rows,
                     ss, style, token)
from animkit.tensor import reshape_ids

DUR = 47.0
shot = Shot("s08_da3_tokens", DUR)
shot.title("DA3 token layout", "Depth Anything 3 · §3.2", 0.2, 3.6)
shot.tag("Paper p.17", 0.3, 3.9)
shot.tag("Not to scale: 5 tokens drawn per row", 7.6, 12.0)
shot.tag(c2.CODE_TAG, 12.0, 15.5)
shot.tag("Not to scale: 4 tokens drawn per frame", 15.5, 18.5)
shot.tag(c2.CODE_TAG, 21.0, 31.5)
shot.tag("Illustration", 25.5, 31.5)
shot.tag("Paper §3.4", 31.5, 34.0)
shot.tag("Paper Tab. 7", 34.0, 38.5)
shot.tag(c2.SCENE_TAG, 36.6, 38.5)
shot.tag("Paper Fig. 3", 38.5, 43.0)
shot.tag("Paper Tab. 7", 43.0, DUR)
shot.tag("The authors' explanation: two thirds of VGGT-style layers are not pretrained", 43.0, DUR)

CAPS = [(0.3, 3.8, "The backbone is just DINOv2; no extra layers"), (3.8, 7.6, "First, attention within each image"),
        (7.6, 12.0, "Then the three images go in one row"), (12.0, 15.5, "Same weights; only token order changes"),
        (15.5, 18.5, "Cross-view layer: 3x the compute"), (18.5, 21.0, "With only one image, it is monocular"),
        (21.0, 25.5, "First 8 layers: own frame only, then alternate"), (25.5, 31.5, "From layer 8, one camera token per image"),
        (31.5, 34.0, "In training, 20% of samples get a pose"), (34.0, 38.5, "Given pose: better reconstruction"),
        (38.5, 43.0, "Dual-DPT: first half shared, second half separate"), (43.0, DUR, "Old approach: nearly all metrics drop")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

NB = c2.NB()
MT = c2.metrics()["da3_large"]
T_TOK = NB["T"]                                # 1297
WHITE_PATCH = NB["patch504_idx"]               # 736
WHITE_VIEW = NB["tok_view"]                    # 737
WHITE_CROSS = NB["tok_cross"]                  # 2034
ROW2 = c2.CAM2                                 # camera holding the white dot (0-based)
GREY_CLS = (120, 126, 138)
CYC = style.TOKEN["patch"]
GOLDC = style.TOKEN["camera"]
CAMC = c2.ec.CAM_COLORS
LAYER_KIND = ["frame" if (i < 8 or i % 2 == 0) else "cross" for i in range(24)]
N_CROSS, N_FRAME = LAYER_KIND.count("cross"), LAYER_KIND.count("frame")
FIRST_CROSS = LAYER_KIND.index("cross")
V_NOPOSE_ROT = MT["A_nopose_camhead"]["pose"]["cam2_rot_deg"]
V_GT_D = MT["C_gtpose"]["D_white_raw"]
S_ROT, S_GT = f"{V_NOPOSE_ROT:.2f}", f"{V_GT_D:.3f}"
S_WIN, S_CROSS_A = f"{NB['attn_within'] / 1e6:.2f}", f"{NB['attn_cross'] / 1e7:.2f}"


# ---- Dual-DPT diagram thumbnails: measured on this scene (real output of da3_B_nopose_raypose, camera 2)----
def _thumbs():
    import cv2
    d = c2.npz("da3_B_nopose_raypose")
    dep = d["depth"][ROW2].astype(np.float64)
    ok = np.isfinite(dep)
    lo, hi = np.percentile(dep[ok], [2, 98])
    g = np.clip((dep - lo) / (hi - lo + 1e-9), 0, 1)
    g[~ok] = 1.0
    dimg = cv2.cvtColor(cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    ray = d["ray"][0, ROW2, :, :, :3].astype(np.float64)
    ray = ray / (np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-9)
    rimg = np.clip((ray + 1) * 127.5, 0, 255).astype(np.uint8)
    return dimg, rimg, ray


TH_DEPTH, TH_RAY, TH_RAYV = _thumbs()

# ---- Sentence 2: real token strip ----
STRIP = TokenStrip(T_TOK, show=fold(T_TOK, head=7, tail=2, force=[WHITE_VIEW]), rows=3, center=(580, 700), size=38,
                   row_pitch=160)
# ---- Sentence 3: reshape at illustrative scale (5 per row)----
KD, NF = 5, 3
DPITCH, DROWP, DSIZE = 56, 100, 42
DCEN = (540, 680)
W_ROW, W_COL = ROW2, 3                         # the cell representing the white dot token in the illustration

SH_A = Formula([r"(3,\ 1297,\ 1024)"], size=48)
SH_B = Formula([r"(1,\ 3891,\ 1024)"], size=48)
SH_1A = Formula([r"(1,\ 1297,\ 1024)"], size=40)
SH_1B = Formula([r"(1,\ 1\times1297,\ 1024)"], size=40)


@shot.check
def check(ck):
    ids = reshape_ids((NF, T_TOK), lambda a: a.reshape(1, NF * T_TOK))
    r = reshape_check((NF, T_TOK), ids)
    ck.true("(3,1297)↔(1,3891) one-to-one", r["bijective"])
    ck.eq("token (i,k) lands at i·1297+k", r["dst_index"][:, 1], r["src_index"][:, 0] * T_TOK + r["src_index"][:, 1])
    flat = ROW2 * T_TOK + WHITE_VIEW
    pos = int(np.where((r["src_index"][:, 0] == ROW2) & (r["src_index"][:, 1] == WHITE_VIEW))[0][0])
    ck.eq("white dot: lands at 2034 after reshape", int(r["dst_index"][pos, 1]), flat)
    ck.eq("white dot patch index 736", WHITE_PATCH, 736)
    ck.eq("white dot token within frame 737 = 1 + 736", WHITE_VIEW, 1 + WHITE_PATCH)
    ck.eq("white dot cross-view 2034 = 1×1297+737", WHITE_CROSS, 1 * T_TOK + WHITE_VIEW)
    ck.eq("white dot is in camera 2 (row 1, 0-based)", ROW2, 1)
    ck.eq("on screen: white dot patch (row, col) = (20, 16)", f"{NB['patch504'][0]}, {NB['patch504'][1]}", "20, 16")
    ck.eq("tokens per image 1 + 1296 = 1297", T_TOK, 1 + 36 * 36)
    ck.eq("3 images total 3891", NF * T_TOK, 3891)
    ck.eq("within-frame size 3×1297²", NB["attn_within"], 3 * T_TOK ** 2)
    ck.eq("cross-view size 3891²", NB["attn_cross"], 3891 ** 2)
    ck.eq("cross-view = 3 × within-frame", NB["attn_cross"], 3 * NB["attn_within"])
    ck.eq("on screen within-frame 5.05e6", S_WIN, "5.05")
    ck.eq("on screen cross-view 1.51e7", S_CROSS_A, "1.51")
    ck.eq("layer bar: cross-view layer count 8", N_CROSS, 8)
    ck.eq("layer bar: within-frame layer count 16", N_FRAME, 16)
    ck.eq("layer bar: first cross-view layer = layer 9", FIRST_CROSS, 9)
    ck.true("layer bar: layers 0–7 are all within-frame", all(k == "frame" for k in LAYER_KIND[:8]))
    ck.eq("concatenation 9 = 3 + 4 + 2", 3 + 4 + 2, 9)
    ck.eq("VGGT layers without pretraining are 48/72 = 2/3", 48 * 3, 72 * 2)
    ck.eq("on screen: camera 2 rotation without pose", S_ROT, "30.07")
    ck.eq("on screen: white dot depth with GT pose", S_GT, "1.978")
    ck.record(white=[WHITE_PATCH, WHITE_VIEW, WHITE_CROSS], layers=LAYER_KIND, rot=V_NOPOSE_ROT, d_gt=V_GT_D)
    ck.true("thumbnails: depth / rays come from measured npz (shape)", TH_DEPTH.shape[2] == 3 and TH_RAY.shape[2] == 3 and TH_RAYV.shape[-1] == 3)
    ck.true("thumbnails: ray directions are normalized", bool(np.allclose(np.linalg.norm(TH_RAYV, axis=-1), 1.0, atol=1e-6)))
    ck.eq("on screen: 6 + 1 = 7 channels", 6 + 1, 7)
    popup_check(ck, [POP, POP_Q], DUR)


# ---------------------------------------------------------------- Helpers
def fade_(t, a, b, fi=0.4, fo=0.3):
    return fade(t, a, b, fi, fo)


def rbox(cv, x0, y0, x1, y1, color, alpha, glow=0.0, th=2):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    cv.s.polyline(pts, color, alpha, th, closed=True)
    if glow > 0.01:
        cv.g.polyline(pts, color, alpha * glow, 4, closed=True)


def hbars(cv, x0, y0, labels, values, unit_w, colors, alpha, grow, fmt="{:g}", h=44, gap=34, size=26, hl=None):
    for j, (lab, v) in enumerate(zip(labels, values)):
        y = y0 + j * (h + gap)
        w = max(unit_w * v * grow, 2)
        c = colors[j]
        cv.s.rect(x0, y, x0 + w, y + h, c, (0.9 if hl == j else 0.55) * alpha, r=4)
        if hl == j:
            cv.g.rect(x0, y, x0 + w, y + h, c, 0.45 * alpha, r=4)
        cv.rich(lab, (x0 - 14, y + h / 2), size, alpha=alpha, anchor="rm", align="r")
        cv.text((x0 + w + 12, y + h / 2), fmt.format(v), size, "light", alpha * ss(grow, 0.6, 1.0), style.TEXT, "lm")


def small_row(cv, xy, n, pitch, size, alpha, color=CYC, first=None):
    for k in range(n):
        token(cv, (xy[0] + k * pitch, xy[1]), size, first if (k == 0 and first) else color, alpha)


# ---------------------------------------------------------------- Popup (short DPT talk inside the Dual-DPT shot)
POP = Popup((60, 995, 960, 245), 38.9, 42.4, "DPT (E01-05): tokens rearranged to 2D, fused step by step", corner="bl", accent=CYAN)
POP_Q = Popup((60, 1010, 960, 225), 28.9, 31.5, "Quaternion", corner="tl", accent=VIOLET, point_to=(440, 990))


def pop_q_body(sub, lt, box):
    x0, y0, x1, y1 = box
    for j in range(4):
        a = ss(lt, 0.2 + 0.1 * j, 0.5 + 0.1 * j)
        xa = x0 + 330 + j * 56
        sub.s.rect(xa + 2, y0 + 8, xa + 52, y0 + 58, VIOLET, 0.7 * a, r=5)
    sub.rich("4 numbers (unit length) represent a 3D rotation", (x0 + 450, y0 + 105), 28, alpha=ss(lt, 0.8, 1.2), anchor="mm")


def pop_body(sub, lt, box):
    x0, y0, x1, y1 = box
    xs = [x0 + 90, x0 + 330, x0 + 570, x0 + 810]
    labs = ["Token sequence", "Reshaped to 2D maps", "Upsample and fuse step by step", "Pixel-level output"]
    for i, (x, lab) in enumerate(zip(xs, labs)):
        a = ss(lt, 0.2 + 0.35 * i, 0.55 + 0.35 * i)
        sub.s.rect(x - 90, y0 + 30, x + 90, y0 + 100, CYAN, 0.25 * a, r=10)
        sub.rich(lab, (x, y0 + 65), 24, alpha=a, anchor="mm")
        if i:
            arrow(sub, (xs[i - 1] + 92, y0 + 65), (x - 94, y0 + 65), style.WHITE, a, 2, 10)
    sub.rich("DPT = a decoding head that turns ViT tokens into per-pixel predictions", (x0 + 450, y0 + 150), 24, alpha=ss(lt, 1.7, 2.1),
             fill=style.TEXT_DIM, anchor="mm")


# ---------------------------------------------------------------- Frame
@shot.frame
def frame(cv, t):
    # ---------- 1: DA3 one bar vs VGGT one bar ----------
    a1 = fade_(t, 0.4, 3.9)
    if a1 > 0.01:
        y0, y1 = 420, 1040
        for i in range(24):
            ai = a1 * ss(t, 0.6 + 0.02 * i, 0.9 + 0.02 * i)
            ya = y0 + (y1 - y0) * i / 24
            cv.s.rect(170, ya + 1, 250, ya + (y1 - y0) / 24 - 1, CYAN, 0.55 * ai, r=3)
        for i in range(72):
            ai = a1 * ss(t, 0.9 + 0.012 * i, 1.2 + 0.012 * i)
            ya = y0 + (y1 - y0) * i / 72
            col, al = (CYAN, 0.55) if i < 24 else (GREY_CLS, 0.5)
            cv.s.rect(640, ya + 0.6, 720, ya + (y1 - y0) / 72 - 0.6, col, al * ai, r=2)
        cv.rich("DA3 (ViT-L)", (210, 380), 30, alpha=a1, anchor="mb")
        cv.rich("VGGT (E01)", (680, 380), 30, alpha=a1, anchor="mb")
        cv.rich("24 layers\nall with DINOv2\npretrained weights", (270, 700), 26, alpha=a1 * ss(t, 1.4, 1.9), anchor="lm")
        cv.rich("DINOv2\n24 layers", (740, y0 + (y1 - y0) / 6), 26, alpha=a1 * ss(t, 2.0, 2.5), anchor="lm")
        cv.rich("48 new layers\n(trained from scratch)", (740, y0 + (y1 - y0) * 2 / 3), 26, alpha=a1 * ss(t, 2.2, 2.7), fill=style.TEXT_DIM,
                anchor="lm")
        cv.rich("DA3's claim: two thirds of VGGT's layers are not pretrained", (540, 1110), 28, alpha=a1 * ss(t, 2.6, 3.1),
                fill=style.LABEL_GOLD, anchor="mm")

    # ---------- 2: real token strip (3, 1297, 1024) ----------
    a2 = fade_(t, 3.9, 7.7)
    if a2 > 0.01:
        for r in range(3):
            ar = a2 * ss(t, 4.0 + 0.2 * r, 4.5 + 0.2 * r)
            for i in STRIP.shown:
                p = STRIP.xy(i, r)
                if i == 0:
                    token(cv, p, STRIP.size, GREY_CLS, ar)
                else:
                    token(cv, p, STRIP.size, CYC, ar)
            for gx in STRIP.gap_x:
                cv.text((gx, STRIP.y[r]), "…", 26, "light", ar, style.TEXT_DIM, "mm", shadow=False)
            cv.text((min(STRIP.x.values()) - 40, STRIP.y[r]), f"Cam {r + 1}", 28, alpha=ar, fill=CAMC[r], anchor="rm")
        pw = STRIP.xy(WHITE_VIEW, ROW2)
        aw = a2 * ss(t, 5.6, 6.1)
        c2.ec.star_marker(cv, pw, aw, glow=0.5, rad=4)
        # one box per row: within-frame attention
        af = a2 * fade(t, 4.8, 7.5, 0.4, 0.3)
        for r in range(3):
            rbox(cv, min(STRIP.x.values()) - 30, STRIP.y[r] - 36, max(STRIP.x.values()) + 30, STRIP.y[r] + 36, ORANGE,
                 af, glow=bump(t, 5.0 + 0.3 * r, 1.0))
        a_sh = a2 * ss(t, 4.1, 4.6)
        SH_A.draw(cv, (540, 340), alpha=a_sh)
        cv.rich(r"3 images, each with $1+1296=1297$ tokens ($504\times504$, patch $14\times14$), 1024-dim", (540, 388), 24,
                alpha=a_sh, fill=style.TEXT_DIM, anchor="mm")
        p0 = STRIP.xy(0, 0)
        a_c = a2 * ss(t, 4.8, 5.3)
        cv.s.line(p0 + [0, -22], p0 + [0, -48], GREY_CLS, a_c, 2)
        cv.rich("Slot 0: in layers 0–7 it is DINOv2's CLS\n(whole-image summary token, gray cell)", p0 + [-14, -54], 24, alpha=a_c,
                fill=(190, 195, 205), anchor="lb")
        cv.rich("White dot token: number $737$ in the frame (counted from 0)\n$=$ slot 0 $+$ patch $736$ (row 20, column 16)", pw + [0, 56], 26,
                alpha=aw, fill=style.LABEL_GOLD, anchor="ma")

    # ---------- 3: reshape at illustrative scale ----------
    a3 = fade_(t, 7.7, 12.1)
    if a3 > 0.01:
        u = ss(t, 9.0, 10.8)
        pos, uk, _ = reshape_rows(NF, KD, u, DCEN, DCEN, DPITCH, DROWP, stagger=0.55, arc=26, group_gap=18)
        a_big = a3 * ss(t, 7.8, 8.3)
        for idx in range(NF * KD):
            i, k = divmod(idx, KD)
            hl = (i == W_ROW and k == W_COL)
            token(cv, pos[idx], DSIZE, GREY_CLS if k == 0 else CYC, a_big, glow=0.0)
            if hl:
                c2.ec.star_marker(cv, pos[idx], a_big, glow=0.5, rad=4)
        a_lab = a_big * (1 - ss(t, 8.9, 9.3))
        for i in range(NF):
            cv.text((DCEN[0] - 2 * DPITCH - 44, DCEN[1] + (i - 1) * DROWP), f"Cam {i + 1}", 28, alpha=a_lab,
                    fill=CAMC[i], anchor="rm")
            y = DCEN[1] + (i - 1) * DROWP
            rbox(cv, DCEN[0] - 2 * DPITCH - 30, y - 34, DCEN[0] + 2 * DPITCH + 30, y + 34, ORANGE, a_lab * 0.7)
        a_gl = a_big * fade(t, 10.6, 12.0, 0.4, 0.3)
        if a_gl > 0.01:
            xs = pos[:, 0]
            rbox(cv, xs.min() - 32, DCEN[1] - 36, xs.max() + 32, DCEN[1] + 36, CYAN, a_gl, glow=bump(t, 10.8, 1.0))
        a_sa = a3 * (1 - ss(t, 8.9, 9.3)) * ss(t, 7.9, 8.4)
        a_sb = a3 * ss(t, 9.8, 10.4)
        SH_A.draw(cv, (540, 385), alpha=a_sa)
        SH_B.draw(cv, (540, 385), alpha=a_sb)
        cv.rich(r"Cameras $3$, each image $1297$ tokens, $3\times1297=3891$", (540, 438), 24, alpha=a3 * ss(t, 8.0, 8.5),
                fill=style.TEXT_DIM, anchor="mm")
        a_w = a3 * ss(t, 10.9, 11.4)
        if a_w > 0.01:
            p = pos[W_ROW * KD + W_COL]
            cv.rich(f"White dot token: $1\\times1297+737={WHITE_CROSS}$", p + [0, -50], 28, alpha=a_w, fill=style.LABEL_GOLD,
                    anchor="mb")
        cv.rich("In input order; cam 1 is the reference view", (540, 900), 26, alpha=a3 * ss(t, 11.0, 11.5), fill=style.TEXT_DIM,
                anchor="mm")
        cv.rich("Reference view: the image placed first; the output world frame = its camera frame", (540, 945), 22, alpha=a3 * ss(t, 11.4, 11.9),
                fill=style.TEXT_DIM, anchor="mm")

    # ---------- 4: same weights, two layouts ----------
    a4 = fade_(t, 12.1, 15.6)
    if a4 > 0.01:
        for cx, nrow, nk, pitch, lab in ((270, 3, 5, 30, "Per-frame layout\n$(3,\\ 1297,\\ 1024)$"),
                                          (790, 1, 15, 22, "Cross-view layout\n$(1,\\ 3891,\\ 1024)$")):
            ar = a4 * ss(t, 12.2, 12.7)
            for r in range(nrow):
                small_row(cv, (cx - (nk - 1) * pitch / 2, 720 + (r - (nrow - 1) / 2) * 44), nk, pitch, 24, ar)
            cv.rich(lab, (cx, 830), 26, alpha=ar, anchor="mm")
            g = bump(t, 13.0 if cx < 500 else 13.4, 1.0)
            module_box(cv, (cx - 130, 460, cx + 130, 560), "$W_Q,\\ W_K,\\ W_V$", "Same weights", ORANGE,
                       a4 * ss(t, 12.4, 12.9), glow=g)
            arrow(cv, (cx, 566), (cx, 640), ORANGE, a4 * ss(t, 12.8, 13.2), 2, 12)
        cv.rich("Paper: no structural change; cross-view comes from rearranging tokens", (540, 930), 32, alpha=a4 * ss(t, 13.6, 14.1),
                fill=style.LABEL_GOLD, anchor="mm")
        cv.rich("Code: from layer 8 on, QK-norm and 2D RoPE are added; no layers added", (540, 985), 26, alpha=a4 * ss(t, 14.2, 14.7),
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("$W_Q,W_K,W_V$: three linear layers turning a token into query, key, value", (540, 1040), 22,
                alpha=a4 * ss(t, 13.0, 13.5), fill=style.TEXT_DIM, anchor="mm")
        cv.rich("QK-norm: normalize query and key separately before computing attention", (540, 1085), 22, alpha=a4 * ss(t, 14.4, 14.9),
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("2D RoPE: rotate query and key by the token's row and column position in the image, so attention knows relative position", (540, 1130), 22,
                alpha=a4 * ss(t, 14.6, 15.1), fill=style.TEXT_DIM, anchor="mm")

    # ---------- 5: three times the compute ----------
    a5 = fade_(t, 15.6, 18.6)
    if a5 > 0.01:
        N, K, cell = 3, 4, 24
        attn_blocks(cv, (110, 580), N, K, cell, "frame", alpha=a5 * ss(t, 15.7, 16.2))
        attn_blocks(cv, (630, 580), N, K, cell, "global", alpha=a5 * ss(t, 16.6, 17.1))
        al, ar_ = a5 * ss(t, 15.9, 16.4), a5 * ss(t, 16.8, 17.3)
        cv.rich("Per frame: 3 blocks of $1297^2$", (254, 475), 28, alpha=al, anchor="mm")
        cv.rich("Cross-view: 1 block of $3891^2$", (774, 475), 28, alpha=ar_, anchor="mm")
        cv.rich(f"$3\\times1297^2={S_WIN}\\times10^6$", (254, 940), 30, alpha=al * ss(t, 16.2, 16.7), fill=CYAN, anchor="mm")
        cv.rich(f"$3891^2={S_CROSS_A}\\times10^7$", (774, 940), 30, alpha=ar_ * ss(t, 17.2, 17.7), fill=ORANGE, anchor="mm")
        cv.rich("Each cell = the attention score of one token pair, per head per layer", (540, 990), 24, alpha=a5 * ss(t, 16.3, 16.8),
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("Cross-view is 3x per-frame", (540, 1060), 36, alpha=a5 * ss(t, 17.6, 18.1), fill=style.LABEL_GOLD, anchor="mm")

    # ---------- 6: N = 1, the two layouts coincide ----------
    a6 = fade_(t, 18.6, 21.1)
    if a6 > 0.01:
        m = ss(t, 19.7, 20.4)
        yA, yB = 600, 600 + 130 * (1 - m)
        aA, aB = a6 * ss(t, 18.7, 19.1), a6 * ss(t, 18.9, 19.3)
        small_row(cv, (540 - 2 * DPITCH, yA), KD, DPITCH, DSIZE, aA, first=GREY_CLS)
        small_row(cv, (540 - 2 * DPITCH, yB), KD, DPITCH, DSIZE, aB * (1 - 0.0), first=GREY_CLS)
        cv.rich("Per-frame layout", (540 - 2 * DPITCH - 50, yA), 26, alpha=aA * (1 - m), anchor="rm")
        cv.rich("Cross-view layout", (540 - 2 * DPITCH - 50, yB), 26, alpha=aB * (1 - m), anchor="rm")
        SH_1A.draw(cv, (540, 410), alpha=aA)
        SH_1B.draw(cv, (540, yB + 62), alpha=aB * (1 - m))
        cv.rich("$N=1$: only one image; $N$ = number of input images", (540, 480), 24, alpha=aA, fill=style.TEXT_DIM, anchor="mm")
        cv.rich("Both layouts are the same sequence", (540, 730), 30, alpha=a6 * ss(t, 20.2, 20.7), fill=style.LABEL_GOLD, anchor="mm")
        cv.rich("One image reduces to monocular depth at no extra cost", (540, 820), 30, alpha=a6 * ss(t, 20.3, 20.8), anchor="mm")

    # ---------- 7: 24-layer bar ----------
    a7 = fade_(t, 21.1, 25.6)
    if a7 > 0.01:
        x0, cw, yy, hh = 90, 37.5, 580, 50
        for i in range(24):
            ai = a7 * ss(t, 21.3 + 0.03 * i, 21.6 + 0.03 * i)
            cross = LAYER_KIND[i] == "cross"
            col = ORANGE if cross else CYAN
            xa = x0 + cw * i
            cv.s.rect(xa + 1.5, yy, xa + cw - 1.5, yy + hh, col, (0.8 if cross else 0.45) * ai, r=4)
            cv.text((xa + cw / 2, yy - 14), str(i), 18, "light", ai, style.TEXT_DIM, "mm", shadow=False)
        cv.rich("Layer index (from 0; ViT-L has 24 layers)", (990, 528), 24, alpha=a7 * ss(t, 21.3, 21.8), fill=style.TEXT_DIM,
                anchor="rm")
        # legend
        al = a7 * ss(t, 22.4, 22.9)
        cv.s.rect(150, 392, 186, 424, CYAN, 0.5 * al, r=4)
        cv.rich(f"Per-frame attention ({N_FRAME} layers)", (200, 408), 26, alpha=al, anchor="lm")
        cv.s.rect(560, 392, 596, 424, ORANGE, 0.85 * al, r=4)
        cv.rich(f"Cross-view attention ({N_CROSS} layers)", (610, 408), 26, alpha=al, anchor="lm")
        # bracket
        ab = a7 * ss(t, 21.9, 22.4)
        xa, xb = x0 + 2, x0 + cw * 8 - 2
        cv.s.line((xa, yy + hh + 14), (xb, yy + hh + 14), CYAN, ab, 2)
        cv.rich("Layers 0–7\nper-frame only", ((xa + xb) / 2, yy + hh + 24), 24, alpha=ab, anchor="ma")
        ab2 = a7 * ss(t, 22.6, 23.1)
        xc, xd = x0 + cw * 8 + 2, x0 + cw * 24 - 2
        cv.s.line((xc, yy + hh + 14), (xd, yy + hh + 14), ORANGE, ab2, 2)
        cv.rich("Layers 8–23: even layers per-frame, odd layers cross-view", ((xc + xd) / 2, yy + hh + 24), 24, alpha=ab2, anchor="ma")
        # layer 9
        a9 = a7 * ss(t, 23.3, 23.8)
        x9 = x0 + cw * 9 + cw / 2
        cv.s.line((x9, 475), (x9, yy - 30), GOLD, a9, 2)
        cv.g.dot((x9, yy + hh / 2), GOLD, 14, 0.5 * a9 * (0.5 + 0.5 * bump(t, 23.4, 1.2)))
        cv.rich("Layer 9: the first cross-view layer", (x9, 465), 28, alpha=a9, fill=style.LABEL_GOLD, anchor="mb")
        cv.rich("Paper $L_s:L_g=2:1$; the code can only be read as\nper-frame : cross-view = 16 : 8", (540, 880), 28, alpha=a7 * ss(t, 24.0, 24.6),
                anchor="mm")
        cv.rich("$L_s$ = paper's per-frame part, $L_g$ = cross-view part", (540, 960), 22, alpha=a7 * ss(t, 24.2, 24.8),
                fill=style.TEXT_DIM, anchor="mm")

    # ---------- 8: camera token + pose input ----------
    a8 = fade_(t, 25.6, 31.6)
    if a8 > 0.01:
        # slot 0: gray → gold
        sw = ss(t, 26.5, 26.9)
        for k in range(7):
            p = (100 + 62 * k, 395)
            ak = a8 * ss(t, 25.7 + 0.05 * k, 26.1 + 0.05 * k)
            if k == 0:
                col = tuple(GREY_CLS[j] + (GOLDC[j] - GREY_CLS[j]) * sw for j in range(3))
                token(cv, p, 44, col, ak, glow=0.7 * bump(t, 26.7, 1.2))
            else:
                token(cv, p, 44, CYC, ak)
        cv.rich("Tokens of one image (illustration)", (100, 340), 24, alpha=a8, fill=style.TEXT_DIM, anchor="lm")
        cv.rich("Layers 0–7: slot 0 = CLS (gray)", (520, 372), 24, alpha=a8 * ss(t, 25.9, 26.3), fill=(190, 195, 205),
                anchor="lm")
        cv.rich("From layer 8: replaced by the camera token (gold)", (520, 416), 24, alpha=a8 * ss(t, 26.5, 26.9), fill=style.LABEL_GOLD,
                anchor="lm")
        # two input slots
        asl = a8 * ss(t, 27.0, 27.5)
        input_slot(cv, (60, 500, 500, 568), "With pose", optional=False, on=1.0, color=GOLD, alpha=asl)
        input_slot(cv, (580, 500, 1020, 568), "Without pose", optional=True, on=1.0, color=VIOLET, alpha=asl)
        al_ = a8 * ss(t, 27.5, 28.0)
        cv.rich("$c_i=E_c(t_i,q_i,\\theta_{v,i},\\theta_{h,i})$", (280, 612), 28, alpha=al_, anchor="mm")
        cv.rich("$E_c$: MLP (9 → 512 → 1024)\n+ 4-layer transformer", (280, 690), 26, alpha=al_, anchor="mm")
        cv.rich("$c_i$ = camera token of image $i$", (280, 760), 22, alpha=al_, fill=style.TEXT_DIM, anchor="mm")
        ar_ = a8 * ss(t, 28.0, 28.5)
        cv.rich("2 learned tokens", (800, 612), 30, alpha=ar_, anchor="mm")
        cv.rich("one for the reference view\none shared by the other views", (800, 690), 26, alpha=ar_, anchor="mm")
        # concatenation: t(3) + q(4) + f(2) = 9
        ac = a8 * ss(t, 28.0, 28.4)
        cv.rich("The 9 numbers fed to $E_c$ (concatenated, not multiplied)", (60, 820), 24, alpha=ac, fill=style.TEXT_DIM, anchor="lm")
        cs = 44
        groups = [("t", 3, (100, 170, 255)), ("q", 4, VIOLET), ("θ", 2, ORANGE)]
        xs0, yc = 60, 880
        xcur = xs0
        centers = []
        for gi, (nm, n, col) in enumerate(groups):
            ag = ac * ss(t, 28.2 + 0.4 * gi, 28.6 + 0.4 * gi)
            for j in range(n):
                xa = xcur + j * cs
                cv.s.rect(xa + 2, yc - cs / 2 + 2, xa + cs - 2, yc + cs / 2 - 2, col, 0.7 * ag, r=5)
            centers.append((xcur + n * cs / 2, col, nm, ag))
            if gi < 2:
                cv.text((xcur + n * cs + 18, yc), "+", 32, "light", ag * ss(t, 28.6 + 0.4 * gi, 28.9 + 0.4 * gi),
                        style.TEXT, "mm")
            xcur += n * cs + 36
        aeq = ac * ss(t, 29.5, 29.9)
        cv.text((xcur - 8, yc), "= 9", 32, "light", aeq, style.LABEL_GOLD, "lm")
        (ct, colt, _, agt), (cq, colq, _, agq), (cf, colf, _, agf) = centers
        cv.rich("$t$ translation (3)", (ct, yc + 40), 24, alpha=agt, fill=colt, anchor="ma")
        cv.rich("$q$ rotation quaternion (4)", (cq, yc + 76), 24, alpha=agq, fill=(205, 185, 255), anchor="ma")
        cv.s.line((cq, yc + cs / 2 + 2), (cq, yc + 72), colq, 0.6 * agq, 1)
        cv.rich("$\\theta_v,\\theta_h$: vertical, horizontal field of view\n(radians, 2 numbers)", (cf - 20, yc + 40), 24, alpha=agf, fill=colf,
                anchor="la")

    # ---------- 9: flicker ----------
    a9 = fade_(t, 31.6, 34.1)
    if a9 > 0.01:
        cv.rich("Each training sample gets a pose with probability $p=0.2$", (540, 420), 34, alpha=a9 * ss(t, 31.7, 32.2),
                fill=style.LABEL_GOLD, anchor="mm")
        fl = flicker(t, 10, p=0.2, period=0.45, seed=15, t0=31.6)
        for j in range(10):
            r, c = divmod(j, 5)
            bx = 70 + c * 188
            by = 540 + r * 120
            input_slot(cv, (bx, by, bx + 160, by + 66), f"Sample {j + 1}", optional=True, on=float(fl[j]), color=GOLD,
                       alpha=a9 * ss(t, 31.9 + 0.04 * j, 32.3 + 0.04 * j), size=24)
        cv.rich("Bright = this sample gets a pose; gray = it does not", (540, 830), 24, alpha=a9 * ss(t, 32.4, 32.9), fill=style.TEXT_DIM,
                anchor="mm")
        cv.rich("$p$ = probability of giving a pose, paper §3.4: 0.2", (540, 880), 24, alpha=a9 * ss(t, 32.6, 33.1), fill=style.TEXT_DIM,
                anchor="mm")

    # ---------- 10: Tab. 7 f / g + definitions + measured on this scene ----------
    a10 = fade_(t, 34.1, 38.6)
    if a10 > 0.01:
        g = ss(t, 34.6, 36.0)
        cv.rich("Both rows fuse the point cloud with GT pose (Tab. 7 f / g)", (540, 335), 26, alpha=a10, fill=style.TEXT_DIM, anchor="mm")
        specs = [("HiRoom F1", [65.8, 73.8], "higher", "{:.1f}", 410), ("DTU CD", [3.65, 2.14], "lower", "{:.2f}", 560),
                 ("7Scenes F1", [58.4, 46.0], "higher", "{:.1f}", 710)]
        from animkit import bar_chart
        for k, (ttl, vals, bet, fm, yb) in enumerate(specs):
            ak = a10 * ss(t, 34.3 + 0.4 * k, 34.8 + 0.4 * k)
            bar_chart(cv, (400, yb + 20, 900, yb + 110), vals, ["Input without pose (f)", "Input with pose (g)"], vmax=max(vals) * 1.15, horizontal=True,
                      highlight=1, progress=g, fmt=fm, title=ttl, better=bet, alpha=ak, value_size=24, label_size=24)
        cv.rich("drops instead", (900, 800), 28, alpha=a10 * ss(t, 36.0, 36.5), fill=ORANGE, anchor="mm")
        ad = a10 * ss(t, 35.4, 35.9)
        cv.rich("F1: harmonic mean of precision and recall of the reconstructed vs. GT point cloud within a distance threshold", (540, 862), 23, alpha=ad,
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("CD (Chamfer distance): mean distance between nearest points of the two clouds; the paper gives no unit", (540, 904), 23, alpha=ad,
                fill=style.TEXT_DIM, anchor="mm")
        am = a10 * ss(t, 36.7, 37.2)
        cv.rich(f"No pose given: cam 2 rotation angle ${S_ROT}^\\circ$ (GT $30^\\circ$)", (540, 970), 28, alpha=am, anchor="mm")
        cv.rich(f"GT pose given: white dot ${S_GT}$ m (GT $1.975$ m)", (540, 1020), 28, alpha=am, anchor="mm")
        cv.rich("When a pose is given, the API uses Umeyama to scale the output to the input pose's scale", (540, 1070), 22, alpha=am, fill=style.TEXT_DIM,
                anchor="mm")

    # ---------- 11: Dual-DPT structure ----------
    a11 = fade_(t, 38.6, 43.1)
    if a11 > 0.01:
        k = lambda a, b: a11 * ss(t, a, b)
        bA = module_box(cv, (250, 305, 830, 405), "4 feature streams", "Tokens output by 4 ViT layers", CYAN, k(38.7, 39.1))
        bB = module_box(cv, (250, 450, 830, 550), "Shared reassemble", "Tokens rearranged to 2D maps (shared by depth and rays)", style.WHITE,
                        k(39.3, 39.7), glow=0.5 * bump(t, 39.6, 1.0))
        bC1 = module_box(cv, (60, 640, 500, 740), "fusion (depth)", "Fuse", CYAN, k(40.0, 40.4))
        bC2 = module_box(cv, (580, 640, 1020, 740), "fusion (rays)", "Fuse", GOLD, k(40.1, 40.5))
        bD1 = module_box(cv, (60, 800, 400, 920), "Output layer", "Depth + confidence (2 channels)", CYAN, k(40.8, 41.2), sub_size=22)
        bD2 = module_box(cv, (580, 800, 920, 940), "Output layer", "Rays + confidence\n(6 + 1 = 7 channels)", GOLD, k(40.9, 41.3),
                         sub_size=22)
        arrow(cv, bA["b"], bB["t"], style.WHITE, k(39.3, 39.6), 2, 12)
        for tgt, col in ((bC1["t"], CYAN), (bC2["t"], GOLD)):
            arrow_path(cv, np.array([bB["b"], (bB["b"][0], 595), (tgt[0], 595), tgt]), col, k(40.0, 40.4), 2, 12)
        arrow(cv, (bC1["b"][0] - 40, bC1["b"][1]), (bD1["t"][0], bD1["t"][1]), CYAN, k(40.8, 41.1), 2, 12)
        arrow(cv, (bC2["b"][0] + 0, bC2["b"][1]), (bD2["t"][0], bD2["t"][1]), GOLD, k(40.9, 41.2), 2, 12)
        cv.rich("First half shared", (130, 500), 26, alpha=k(39.6, 40.0), fill=style.LABEL_GOLD, anchor="mm")
        cv.rich("Second half separate", (130, 595), 26, alpha=k(40.4, 40.8), fill=style.LABEL_GOLD, anchor="mm")
        # ---- thumbnails of each node ----
        ath = k(39.0, 39.5)                                  # token sequence: a row of small squares
        small_row(cv, (860, 355), 6, 24, 18, ath)
        ath = k(39.6, 40.0)                                  # 2D feature map: grid of squares
        rng = np.random.RandomState(8)
        for r_ in range(4):
            for c_ in range(4):
                v = 0.25 + 0.6 * rng.rand()
                xg, yg = 872 + c_ * 26, 460 + r_ * 26
                cv.s.rect(xg, yg, xg + 23, yg + 23, tuple(int(CYC[j] * v) for j in range(3)), 0.9 * ath, r=3)
        ad_ = k(41.0, 41.5)                                  # depth map / ray RGB map (measured on this scene)
        x1_, y1_ = c2.ec.paste_img(cv, TH_DEPTH, (420, 810), 110, alpha=ad_)[:2]
        rbox(cv, 420, 810, 530, 920, CYAN, ad_, th=1)
        c2.ec.paste_img(cv, TH_RAY, (935, 815), 110, alpha=ad_)
        rbox(cv, 935, 815, 1045, 925, GOLD, ad_, th=1)
        cv.rich("Depth maps", (475, 937), 22, alpha=ad_, fill=CYAN, anchor="mm")
        cv.rich("Ray maps", (990, 937), 22, alpha=ad_, fill=GOLD, anchor="mm")
        cv.rich("Thumbnails: measured on our test scene (ray map: RGB = direction)", (540, 972), 22, alpha=ad_, fill=style.TEXT_DIM, anchor="mm")

    # ---------- 12: Tab. 7 structure ablation ----------
    a12 = fade_(t, 43.1, DUR + 0.5)
    if a12 > 0.01:
        cv.rich("Tab. 7, HiRoom Auc3 (higher is better)", (540, 360), 32, alpha=a12 * ss(t, 43.2, 43.7), anchor="mm")
        labs = ["DA3's design", "Two separate DPTs", "VGGT style: two stacked transformers\n(same parameter count)"]
        vals = [39.2, 5.59, 3.72]
        hbars(cv, 480, 470, labs, vals, 11.0, [GOLD, style.GREY, style.GREY], a12 * ss(t, 43.4, 43.9),
              ss(t, 43.8, 45.2), fmt="{:g}", h=56, gap=60, hl=0)
        aa = a12 * ss(t, 44.4, 44.9)
        cv.rich("Auc3: for each image pair, take the larger of the rotation error and translation-direction error (both angles)", (540, 850), 24, alpha=aa,
                fill=style.TEXT_DIM, anchor="mm")
        cv.rich("Accuracy = share of pairs with error below threshold;", (540, 892), 24, alpha=aa, fill=style.TEXT_DIM, anchor="mm")
        cv.rich("sweep the threshold over 0–3°, take the area under the curve; full score 100, higher is better", (540, 934), 24, alpha=aa, fill=style.TEXT_DIM,
                anchor="mm")

    POP.draw(cv, t, pop_body)
    POP_Q.draw(cv, t, pop_q_body)
