"""E01-02 Input: the 518-pixel image is cut into 37x37 patches, and why each vector is 1024-dimensional (S, 35.6 s). Storyboard storyboard.md section E01-02.
Computed for real: the grid (14 px per cell), K = 37x37 = 1369, the cell holding the white dot and its index 794, the real pixels in that cell;
      why 1024 dims: one cell is 14x14x3 = 588 numbers (flattened by channel, same order as the conv weight [1024, 3, 14, 14]);
      patch embed parameters 1024x588 + 1024 = 603,136; 16 heads x 64 = 1024; shapes are read from the local weight-file header and the official code.
      Popup: the same 588 numbers go into ViT-S / B / L / g, widths 384 / 768 / 1024 / 1536 (official code); 1024 is set by the model size, not computed.
Illustration: the numbers inside the DINOv2 output vector.
"""
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import e01_common as ec
import scene_e01 as sc
from animkit import CYAN, GOLD, Formula, Shot, arrow, bump, fade, lerp, module_box, ss, style
from animkit.readout import number
from e01_common import Popup, popup_check

DUR = 35.6
T_WHY = 15.0
shot = Shot("s02_patchify", DUR)
shot.title("Input: cut into patches", "Patchify · VGGT §3.2", 0.2, 2.8)
shot.tag("Illustration: numbers in a vector", 12.6, DUR)
CAPS = [(0.3, 3.0, "Each image is 518×518 pixels"), (3.0, 6.0, "Cut into small 14×14 patches"),
        (6.0, 9.0, "37 patches per side, 1369 in total"), (9.0, 12.0, "The white dot is in row 21, column 17"),
        (12.0, T_WHY, "DINOv2 (24 layers) encodes the whole image into 1369 vectors"),
        # why 1024 dims
        (15.0, 17.4, "Why is each vector 1024-dimensional?"), (17.4, 19.8, "One cell: 14×14×3 = 588 numbers"),
        (19.8, 22.0, "Flatten by channel into one column"), (22.0, 24.8, "Multiply by W, add b: 588 dims become 1024"),
        (24.8, 27.2, "All 1369 cells share the same W"), (27.2, 29.6, "All 24 layers after that stay at 1024 dims"),
        # Popup "where 1024 comes from": wording avoids "can only" and "lift dims to hold whole-image information"
        (29.6, 32.6, "Width fixed by choosing the ViT-L size"),
        (32.6, DUR, "VGGT's backbone keeps 1024, with no projection")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

NB = ec.nb()
CAM = sc.EX_CAM
IMG = ec.view_pil(CAM)
IMG_RGB = ec.view_rgb(CAM)
PATCH, G = sc.PATCH, sc.W // sc.PATCH
ROW, COL = NB["patch_rc"]
UV = NB["uv"]
crop = IMG_RGB[ROW * PATCH:(ROW + 1) * PATCH, COL * PATCH:(COL + 1) * PATCH]

# Operators are merged into the adjacent term (a standalone = / \cdot drops to the baseline, a library v0 issue)
EQ = Formula(["K=", r"\frac{518}{14}\cdot\frac{518}{14}", r"=37\times 37", "=1369"], size=58)
EQ_XY = (540, 250)
IDX = Formula([r"21\times 37+17", "=794"], size=40)
VEC = np.random.default_rng(2).uniform(0.25, 1.0, 16)                       # illustration: numbers in the vector

# ---------------- One cell -> 588 -> W -> 1024 ----------------
N_PIX = PATCH * PATCH * 3                    # 588
C_DIM = 1024
N_PARAM = C_DIM * N_PIX + C_DIM              # 603,136
CH_TINT = [(255, 95, 95), (110, 235, 130), (100, 160, 255)]
CH_NAME = ["R", "G", "B"]

CROP_BOX = (70, 300, 150)                    # position of the enlarged white-dot cell (x, y, side)
SL_S = 104                                   # side of each of the three channel slices
SL_XY = [(262 + 26 * c, 300 + 20 * c) for c in range(3)]
KS = 0.38                                    # column height: 0.38 px per component (only to compare the lengths of 588 and 1024)
P_X, P_W, P_H = 128, 26, N_PIX * KS
X_X, X_W, X_H = 650, 26, C_DIM * KS
ROWB_Y = 790                                 # center line of the second row (p -> W -> x)
P_TOP = ROWB_Y - P_H / 2
X_TOP = ROWB_Y - X_H / 2
W_BOX = (262, ROWB_Y - 85, 540, ROWB_Y + 85)
PANEL_X = 740                                # right-hand explanation column

F588 = Formula([(r"14\times14", "hw"), (r"\times3", "c"), ("=588", "n")], size=46)
F588_XY = (600, 382)
FXW = Formula([("x", "x"), ("=W", "W"), ("p", "p"), ("+b", "b")], size=54, gap=0.18)
FXW_XY = (620, 382)
CHAIN = Formula([(r"(3,\,518,\,518)", "img"), (r"\to(1369,\,588)", "p"), (r"\to(1369,\,1024)", "x"),
                 (r"\to\cdots\to(1369,\,1024)", "vit")], size=34, gap=0.15)
CHAIN_XY = (540, 1095)
SEQ = [794, 0, 1, 2, 38, 520, 1000, 1368, 794]       # cell indices swept in turn while "sharing the same W"
SEQ_T0, SEQ_DT = 25.0, 0.26


def patch_of(k):
    r, c = divmod(k, G)
    return IMG_RGB[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]


def p_column(k):
    """The 588 numbers of cell k flattened by (channel, row, column) (= the flatten order of the conv weight [1024, 3, 14, 14]), drawn as one column colored by channel."""
    v = patch_of(k).transpose(2, 0, 1).reshape(3, -1).astype(np.float32) / 255.0      # (3, 196)
    col = np.concatenate([v[c][:, None] * np.asarray(CH_TINT[c], np.float32)[None] for c in range(3)])
    return np.clip(col[:, None, :], 0, 255).astype(np.uint8)                           # (588, 1, 3)


def x_column(k):
    """Illustration: the 1024 numbers of the output vector (not real values), drawn in cyan shades by magnitude."""
    v = np.random.default_rng(1000 + k).uniform(0.15, 1.0, C_DIM).astype(np.float32)
    col = v[:, None] * np.asarray(CYAN, np.float32)[None]
    return np.clip(col[:, None, :], 0, 255).astype(np.uint8)


P_COLS = {k: p_column(k) for k in set(SEQ)}
X_COLS = {k: x_column(k) for k in set(SEQ)}
SLICES = []
for c in range(3):
    s_ = np.zeros_like(crop, np.float32)
    s_[:] = crop[..., c:c + 1].astype(np.float32) / 255.0 * np.asarray(CH_TINT[c], np.float32)
    SLICES.append(np.clip(s_, 0, 255).astype(np.uint8))


def paste_box(cv, arr, box, alpha=1.0, nearest=False):
    """Stretch an RGB array to box = (x, y, w, h) (aspect ratio not kept) and paste it into the sharp layer."""
    x, y, w, h = box
    w, h = int(round(w)), int(round(h))
    x, y = int(round(x)), int(round(y))
    if alpha <= 0.01 or w < 1 or h < 1:
        return
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    small = cv2.resize(np.ascontiguousarray(arr), (w, h), interpolation=interp).astype(np.float32)
    img = cv.s.img
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xb <= xa or yb <= ya:
        return
    roi = img[ya:yb, xa:xb].astype(np.float32)
    img[ya:yb, xa:xb] = np.clip(roi * (1 - alpha) + small[ya - y:yb - y, xa - x:xb - x] * alpha + 0.5, 0,
                                255).astype(np.uint8)


def outline(cv, box, color, alpha, th=2):
    x, y, w, h = box
    cv.s.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], color, alpha, th, closed=True)


# ---------------- Source of the shapes: local weight-file header + official code ----------------
_ROOT = Path(__file__).resolve().parents[3]          # repo root; clone official code into third_party/
VIT_PY = _ROOT / "third_party" / "vggt" / "vggt" / "layers" / "vision_transformer.py"


weight_shapes = ec.s01_s02_weight_shapes          # local weight-file header (shapes only)


def vit_cfg(name):
    """embed_dim, depth and num_heads of vit_small / vit_base / vit_large / vit_giant2 in the official code."""
    src = VIT_PY.read_text(encoding="utf-8")
    body = src[src.index(f"def {name}("):]
    body = body[:body.index("return model")]
    return {k: int(re.search(rf"{k}=(\d+)", body).group(1)) for k in ("embed_dim", "depth", "num_heads")}


# ---------------- Popup: where 1024 comes from ----------------
_cx0, _cy0, _cx1, _cy1 = CHAIN.part_box("x", CHAIN_XY)
# popup height 660: its bottom edge fully covers the two "Parameters: ..." lines of the main frame instead of cutting through them
POP = Popup((340, 290, 720, 660), t_in=29.6, t_out=35.0, title="Where $1024$ comes from", corner="tr",
            point_to=((_cx0 + _cx1) / 2 + 30, _cy0 - 6))
HEADS, HEAD_DIM = 16, 64
TIERS = [("ViT-S/14", "vit_small", 384), ("ViT-B/14", "vit_base", 768), ("ViT-L/14", "vit_large", 1024),
         ("ViT-g/14", "vit_giant2", 1536)]   # widths are checked one by one against the official code in check
TIER_L = 2


def body_pop(sub, lt, box):
    """Three segments (popup-local time): 0-1.7 the same 588 numbers enter four model sizes with different widths; 1.7-3.0 ViT-L = 16 heads x 64;
    3.0- VGGT uses exactly DINOv2 ViT-L, the code adds no projection in between, so the backbone is also 1024."""
    x0, y0, x1, y1 = box
    lx = x0 + 120                                     # start of the bars
    kx = (x1 - lx - 70) / 1536.0                      # px per dimension (the widest, 1536, just fits)
    a1 = ss(lt, 0.2, 0.6)
    sub.rich("The same $588$ numbers into ViTs of different sizes (all patch 14):", (x0 + 4, y0 + 4), 22, alpha=a1)
    for j, (nm, _, wd) in enumerate(TIERS):
        yj = y0 + 44 + j * 44
        u = ss(lt, 0.4 + 0.25 * j, 0.9 + 0.25 * j)
        hl = j == TIER_L
        sub.text((x0 + 4, yj + 13), nm, 22, "light", a1, style.LABEL_GOLD if hl else style.TEXT_DIM, "lm")
        L = wd * kx * u
        if hl and lt > 1.7:                           # segment 2: the ViT-L bar splits into 16 heads
            u2 = ss(lt, 1.8, 2.6)
            seg = wd * kx / HEADS
            for h in range(HEADS):
                g2 = 3.0 * u2
                c = CYAN if h % 2 == 0 else (60, 150, 170)
                sub.s.rect(lx + h * seg + g2 / 2, yj, lx + (h + 1) * seg - g2 / 2, yj + 26, c, 0.9 * a1, r=3)
        else:
            sub.s.rect(lx, yj, lx + max(L, 1), yj + 26, CYAN if hl else (60, 120, 135), 0.85 * a1, r=4)
        if u > 0.05:
            sub.text((lx + L + 8, yj + 13), str(wd) + (" ← VGGT" if hl else ""), 22, "light",
                     a1 * u, style.LABEL_GOLD if hl else style.TEXT_DIM, "lm")
    sub.rich("Source: DINOv2 paper Tab. 17; official code vit_small / base / large / giant2", (x0 + 4, y0 + 224), 18,
             alpha=a1 * ss(lt, 1.2, 1.6), fill=style.TEXT_DIM)
    # Segment 2: ViT-L's 1024 = 16 heads x 64
    a2 = ss(lt, 1.8, 2.3)
    sub.rich(r"ViT-L: $16$ attention heads $\times$ $64$ dims per head $=1024$, 24 layers", (x0 + 4, y0 + 262), 22, alpha=a2)
    sub.rich("Source: original ViT paper Tab. 1 (Large: 24 layers, width 1024, 16 heads)", (x0 + 4, y0 + 292), 18,
             alpha=a2 * ss(lt, 2.4, 2.8), fill=style.TEXT_DIM)
    # Segment 3: VGGT uses exactly DINOv2 ViT-L, with no projection in between
    a3 = ss(lt, 3.0, 3.4)
    if a3 > 0.01:
        cy = y0 + 328
        sub.rich("VGGT loads DINOv2 ViT-L pretrained weights (shapes in the weight file):", (x0 + 4, cy), 20, alpha=a3)
        chips = [r"patch embed: $[1024,\,3,\,14,\,14]$", "DINOv2 24 layers: width $1024$",
                 "camera / register token: $1024$ dims", "24 rounds (frame + global): $1024$"]
        cw, ch = (x1 - x0 - 18) / 2, 38
        for k, s_ in enumerate(chips):
            u = ss(lt, 3.2 + 0.2 * k, 3.6 + 0.2 * k)
            cxk = x0 + 4 + (k % 2) * (cw + 12) + 30 * (1 - u)
            cyk = cy + 30 + (k // 2) * (ch + 8)
            sub.s.rect(cxk, cyk, cxk + cw, cyk + ch, (36, 50, 62), a3 * u, r=8)
            sub.s.polyline([(cxk, cyk), (cxk + cw, cyk), (cxk + cw, cyk + ch), (cxk, cyk + ch)], CYAN,
                           0.6 * a3 * u, 1, closed=True)
            sub.rich(s_, (cxk + 10, cyk + ch / 2), 19, alpha=a3 * u, anchor="lm")
        sub.rich("Code has no projection after DINOv2 → backbone width is $1024$ too", (x0 + 4, cy + 128),
                 22, alpha=a3 * ss(lt, 4.1, 4.5), fill=style.LABEL_CYAN)


@shot.check
def check(ck):
    ck.eq("518 / 14 is an integer", sc.W % PATCH, 0)
    ck.eq("patches per side", G, 37)
    ck.eq("K = 37×37", NB["K_patch"], 1369)
    ck.eq("white-dot row and column (0-based)", [ROW, COL], [21, 17])
    ck.eq("patch index", NB["patch_idx"], ROW * G + COL)
    ck.true("white-dot pixel lies inside this cell", COL * PATCH <= UV[0] < (COL + 1) * PATCH and ROW * PATCH <= UV[1] < (ROW + 1) * PATCH)
    # 588 -> 1024
    ck.eq("numbers in one cell: 14×14×3", N_PIX, 588)
    ck.eq("patch embed parameters 1024×588 + 1024", N_PARAM, 603136)
    ck.eq("16 heads × 64 dims", HEADS * HEAD_DIM, C_DIM)
    ck.true("flatten order = flatten order of weight [1024,3,14,14]: component c·196 + r·14 + col",
            np.array_equal(crop.transpose(2, 0, 1).reshape(-1)[[0, 196, 392 + 14 * 3 + 5]],
                           [crop[0, 0, 0], crop[0, 0, 1], crop[3, 5, 2]]))
    if not VIT_PY.exists():
        ck.record(skipped_official_code=f"{VIT_PY.name} not found: clone facebookresearch/vggt into third_party/vggt to run this check")
    else:
     ck.eq("official vit_large: embed_dim / depth / num_heads", list(vit_cfg("vit_large").values()), [1024, 24, 16])
     ck.eq("widths of the four sizes in the official code (S / B / L / g)", [vit_cfg(fn)["embed_dim"] for _, fn, _ in TIERS],
          [wd for _, _, wd in TIERS])
    ws = weight_shapes()
    if ws is None:
        ck.record(weight_header="no VGGT-1B weight file on this machine; shape assertions skipped")
    else:
        ck.eq("weight header patch embed proj.weight", ws["aggregator.patch_embed.patch_embed.proj.weight"], [1024, 3, 14, 14])
        ck.eq("weight header patch embed proj.bias", ws["aggregator.patch_embed.patch_embed.proj.bias"], [1024])
        ck.eq("weight header camera_token", ws["aggregator.camera_token"], [1, 2, 1, 1024])
        ck.eq("weight header register_token", ws["aggregator.register_token"], [1, 2, 4, 1024])
        nb_ = len({int(m.group(1)) for k in ws for m in [re.search(r"aggregator\.patch_embed\.blocks\.(\d+)\.", k)]
                   if m})
        ck.eq("DINOv2 layer count (weight header)", nb_, 24)
        qkv = [ws[k] for k in ws if re.fullmatch(r"aggregator\.(frame|global)_blocks\.\d+\.attn\.qkv\.weight", k)]
        ck.true("VGGT backbone has 48 layers, every qkv is [3072, 1024]", len(qkv) == 48 and all(s == [3072, 1024] for s in qkv),
                f"{len(qkv)} layers")
        ck.eq("concatenated to 2048 before the heads (depth_head.projects.0 input channels)", ws["depth_head.projects.0.weight"][1], 2048)
    ck.true("all swept cell indices are within 0-1368", all(0 <= k < NB["K_patch"] for k in SEQ))
    popup_check(ck, [POP], DUR)
    ck.record(uv=UV.round(2), patch_rc=[ROW, COL], patch_idx=NB["patch_idx"], n_pix=N_PIX, n_param=N_PARAM)


def layout(t):
    """Image top-left corner and width: shrinks to the left at sentence 5."""
    u = ss(t, 12.0, 12.8)
    return lerp(180, 40, u), lerp(360, 520, u), lerp(720, 330, u)


def overview(cv, t, ov):
    """First 15 s; ov = overall fade-out factor."""
    x0, y0, w = layout(t)
    s = w / sc.W
    a_img = ss(t, 0.3, 1.0) * ov
    ec.paste_img(cv, IMG, (x0, y0), w, a_img)
    cv.s.polyline([(x0, y0), (x0 + w, y0), (x0 + w, y0 + w), (x0, y0 + w)], ec.CAM_COLORS[CAM], 0.8 * a_img, 2,
                  closed=True)
    cv.text((x0, y0 + w + 10), "Camera 2", 26, alpha=a_img, fill=ec.CAM_COLORS[CAM])

    # ruler
    a_r = fade(t, 0.8, 6.2, 0.5, 0.5)
    if a_r > 0:
        cv.s.line((x0, y0 - 26), (x0 + w, y0 - 26), style.GREY, a_r, 2)
        for x in (x0, x0 + w):
            cv.s.line((x, y0 - 34), (x, y0 - 18), style.GREY, a_r, 2)
        cv.text((x0 + w / 2, y0 - 34), "518 px", 30, "light", a_r, anchor="mb")
        cv.s.line((x0 - 26, y0), (x0 - 26, y0 + w), style.GREY, a_r, 2)
        cv.text((x0 - 34, y0 + w / 2), "518\npx", 26, "light", a_r, anchor="rm", align="right")

    # grid: vertical lines sweep first, then horizontal lines
    a_g = ss(t, 3.1, 3.4) * ov
    if a_g > 0:
        pv, ph = ss(t, 3.2, 4.6), ss(t, 4.4, 5.8)
        for k in range(G + 1):
            if k / G <= pv + 1e-9:
                x = x0 + k * PATCH * s
                cv.s.line((x, y0), (x, y0 + w), CYAN, 0.45 * a_g, 1)
            if k / G <= ph + 1e-9:
                y = y0 + k * PATCH * s
                cv.s.line((x0, y), (x0 + w, y), CYAN, 0.45 * a_g, 1)
        a14 = fade(t, 3.6, 6.0, 0.4, 0.4)
        if a14 > 0:
            q = PATCH * s
            cv.s.rect(x0, y0, x0 + q, y0 + q, GOLD, 0.6 * a14)
            cv.text((x0 + q + 8, y0 + q / 2), "14×14 px", 26, "text", a14, style.LABEL_GOLD, "lm")

    # formula K = 518/14 · 518/14 = 37×37 = 1369
    a_eq = fade(t, 6.0, 12.0, 0.4, 0.5)
    if a_eq > 0:
        pa = {1: ss(t, 6.2, 6.7), 2: ss(t, 7.2, 7.5), 3: ss(t, 7.9, 8.2)}
        EQ.draw(cv, EQ_XY, alpha=a_eq, part_alpha=pa, part_glow={3: bump(t, 8.2, 0.8)})
        cv.math("K", (300, 320), 30, style.LABEL_GOLD, a_eq, "rm")
        cv.text((310, 320), "= patches per image", 24, alpha=a_eq, fill=style.TEXT_DIM, anchor="lm")
        a37 = ss(t, 7.2, 7.6) * (1 - ss(t, 8.8, 9.2))
        cv.text((x0 + w + 16, y0 + w / 2), "37 rows", 30, alpha=a37, fill=style.LABEL_CYAN, anchor="lm")
        cv.text((x0 + w / 2, y0 + w + 46), "37 columns", 30, alpha=a37, fill=style.LABEL_CYAN, anchor="ma")

    # white dot: row, column, cell
    a_w = ss(t, 9.0, 9.5) * ov
    if a_w > 0:
        q = PATCH * s
        a_band = a_w * (1 - ss(t, 12.0, 12.5))
        cv.s.add_poly([(x0, y0 + ROW * q), (x0 + w, y0 + ROW * q), (x0 + w, y0 + (ROW + 1) * q),
                       (x0, y0 + (ROW + 1) * q)], GOLD, 0.22 * a_band)
        cv.s.add_poly([(x0 + COL * q, y0), (x0 + (COL + 1) * q, y0), (x0 + (COL + 1) * q, y0 + w),
                       (x0 + COL * q, y0 + w)], GOLD, 0.22 * a_band)
        cx0, cy0 = x0 + COL * q, y0 + ROW * q
        cv.s.polyline([(cx0, cy0), (cx0 + q, cy0), (cx0 + q, cy0 + q), (cx0, cy0 + q)], GOLD, a_w, 2, closed=True)
        ec.star_marker(cv, (x0 + UV[0] * s, y0 + UV[1] * s), a_w, glow=0.6 + 0.4 * bump(t, 9.4, 1.0), rad=4)
        a_lab = ss(t, 9.4, 9.9) * (1 - ss(t, 12.0, 12.4))
        cv.text((540, 1120), "Row 21, column 17 (counting from 0)", 32, alpha=a_lab, anchor="mm")
        a_idx = ss(t, 10.2, 10.6) * a_lab
        cv.text((420, 1175), "Index", 30, alpha=a_idx, anchor="rm", fill=style.TEXT_DIM)
        IDX.draw(cv, (440, 1175), alpha=a_idx, anchor="lm", part_glow={1: bump(t, 10.8, 0.8)})

    # Sentence 5: the whole image goes into DINOv2 (24-layer ViT-L, all patches in the image attend to each other), producing 37x37 vectors
    a5 = ss(t, 12.3, 12.8) * ov
    if a5 > 0:
        b = module_box(cv, (405, 610, 655, 750), "DINOv2", "24-layer ViT-L", CYAN, a5, glow=0.5 * bump(t, 13.3, 0.8))
        arrow(cv, (x0 + w + 6, y0 + w / 2), b["l"], style.WHITE, a5, progress=ss(t, 12.6, 13.0))
        cv.text((530, 762), "Whole image computed together", 24, alpha=a5, fill=style.TEXT_DIM, anchor="ma")
        a_o = ss(t, 13.3, 13.8) * ov
        if a_o > 0:
            arrow(cv, b["r"], (650, 680), style.WHITE, a_o, progress=ss(t, 13.3, 13.6))
            pitch, gx0, gy0 = 9.0, 662.0, 515.0
            rows_on = G * ss(t, 13.5, 14.3)
            for r in range(G):
                if r >= rows_on:
                    break
                ys = gy0 + r * pitch
                cv.s.rect(gx0, ys, gx0 + G * pitch - 2, ys + pitch - 2, CYAN, 0.55 * a_o)
            for k in range(1, G):
                cv.s.line((gx0 + k * pitch - 1, gy0), (gx0 + k * pitch - 1, gy0 + min(rows_on, G) * pitch),
                          (16, 20, 30), a_o, 1)
            cx, cy = gx0 + COL * pitch, gy0 + ROW * pitch
            a_h = ss(t, 14.2, 14.5) * ov
            cv.s.rect(cx - 1, cy - 1, cx + pitch, cy + pitch, GOLD, a_h)
            cv.text((gx0 + G * pitch / 2, gy0 - 10), "37 × 37 vectors", 26, alpha=a_o, fill=style.LABEL_CYAN,
                    anchor="mb")
            # vector of the white-dot cell (values illustrative)
            if a_h > 0:
                bx, by = 1062.0, 520.0
                cv.s.line((cx + pitch, cy + pitch / 2), (bx - 12, by + 160), GOLD, 0.7 * a_h, 1)
                for k in range(len(VEC)):
                    yk = by + k * 20
                    cv.s.rect(bx - 10 - 22 * VEC[k], yk, bx, yk + 15, CYAN, (0.35 + 0.5 * VEC[k]) * a_h, r=3)
                cv.text((bx, by + len(VEC) * 20 + 8), "1024 dims", 26, alpha=a_h, fill=style.LABEL_CYAN, anchor="ra")
            a_n = ss(t, 14.3, 14.7) * ov
            cv.math(r"C=1024", (662, 900), 30, style.LABEL_CYAN, a_n, "lm")
            cv.text((800, 900), "(token dim)", 24, alpha=a_n, fill=style.TEXT_DIM, anchor="lm")
            cv.text((540, 960), "DINOv2's own CLS and register outputs are dropped; only 1369 are kept", 24, alpha=a_n,
                    fill=style.TEXT_DIM, anchor="ma")
            cv.text((540, 996), "DINOv2 has ~304M parameters, 24% of the full model's 1.26B", 24, alpha=a_n,
                    fill=style.TEXT_DIM, anchor="ma")


def seq_k(t):
    """In the "same W shared" sentence, the cell currently being swept."""
    if t < SEQ_T0:
        return 794
    return SEQ[min(int((t - SEQ_T0) / SEQ_DT), len(SEQ) - 1)]


def why_1024(cv, t, dim):
    """One cell -> 588 numbers -> multiply by W, add b -> 1024 dims; all 1369 cells share W; 24 layers stay at 1024."""
    # The white-dot cell flies out of the small image and enlarges (15.2-16.0)
    x0, y0, w = layout(t)
    q = PATCH * w / sc.W
    u = ss(t, 15.2, 16.0)
    cx, cy, cs = CROP_BOX
    fx, fy, fs = lerp(x0 + COL * q, cx, u), lerp(y0 + ROW * q, cy, u), lerp(q, cs, u)
    a_c = ss(t, 15.1, 15.3) * dim
    k = seq_k(t)
    to_mini = ss(t, 24.8, 25.0)                       # sentence 10: switch to the whole small image + scan box
    if a_c > 0.01:
        paste_box(cv, crop, (fx, fy, fs, fs), a_c * (1 - to_mini), nearest=True)
        outline(cv, (fx, fy, fs, fs), GOLD, a_c * (1 - to_mini))
        du, dv = (UV[0] - COL * PATCH) / PATCH, (UV[1] - ROW * PATCH) / PATCH      # position of the white dot inside the cell
        ec.star_marker(cv, (fx + du * fs, fy + dv * fs), a_c * (1 - to_mini) * ss(t, 15.6, 16.0), glow=0.5, rad=5)
        if to_mini > 0.01:
            paste_box(cv, IMG_RGB, (cx, cy, cs, cs), a_c * to_mini)
            outline(cv, (cx, cy, cs, cs), ec.CAM_COLORS[CAM], 0.8 * a_c * to_mini)
            r_, c_ = divmod(k, G)
            qq = cs / G
            outline(cv, (cx + c_ * qq - 1, cy + r_ * qq - 1, qq + 2, qq + 2), GOLD, a_c * to_mini, 2)
        lab = "White-dot cell: 14×14 px" if to_mini < 0.5 else "Whole image of camera 2"
        cv.text((cx + cs / 2, cy + cs + 10), lab, 22, alpha=a_c * ss(t, 15.8, 16.2), fill=style.TEXT_DIM, anchor="ma")

    # Three channel slices: slide out of the cell at 17.4, flatten into the p column from 19.8
    a_sl = ss(t, 17.4, 17.7) * dim
    if a_sl > 0.01:
        for c in (2, 1, 0):
            uo = ss(t, 17.5 + 0.2 * c, 18.3 + 0.2 * c)
            sx, sy = lerp(cx, SL_XY[c][0], uo), lerp(cy, SL_XY[c][1], uo)
            ss_ = lerp(cs, SL_S, uo)
            uf = ss(t, 19.9 + 0.35 * c, 20.9 + 0.35 * c)            # flatten
            seg_box = (P_X, P_TOP + c * P_H / 3, P_W, P_H / 3)
            bx = (lerp(sx, seg_box[0], uf), lerp(sy, seg_box[1], uf), lerp(ss_, seg_box[2], uf),
                  lerp(ss_, seg_box[3], uf))
            if uf < 0.999:
                paste_box(cv, SLICES[c], bx, a_sl * (1 - uf), nearest=True)
                outline(cv, bx, CH_TINT[c], a_sl * (1 - uf))
                cv.text((bx[0] + bx[2] - 6, bx[1] + bx[3] - 4), CH_NAME[c], 20, "light", a_sl * (1 - uf) * uo,
                        CH_TINT[c], "rb")
            if uf > 0.001:
                seg = P_COLS[794][c * 196:(c + 1) * 196]
                paste_box(cv, seg, bx, a_sl * uf)
    # 588 = 14×14×3
    a_588 = fade(t, 17.9, 22.2, 0.4, 0.4) * dim
    if a_588 > 0.01:
        F588.draw(cv, F588_XY, alpha=a_588, anchor="lm",
                  part_alpha={"c": ss(t, 18.2, 18.6), "n": ss(t, 18.8, 19.2)}, part_glow={"n": bump(t, 19.0, 0.8)},
                  notes={"hw": "width × height (pixels)", "c": "R, G, B", "n": "count"})

    # p column (after flattening)
    a_p = ss(t, 20.0, 20.4) * dim
    if a_p > 0.01:
        if t > 21.6:
            paste_box(cv, P_COLS[k], (P_X, P_TOP, P_W, P_H), a_p)
        outline(cv, (P_X, P_TOP, P_W, P_H), style.WHITE, 0.6 * a_p, 1)
        for c in range(3):
            cv.text((P_X - 8, P_TOP + (c + 0.5) * P_H / 3), f"{CH_NAME[c]} 196", 18, "light", a_p, CH_TINT[c], "rm")
        cv.rich("$p$: this cell's 588 numbers", (P_X + P_W / 2, P_TOP + P_H + 14), 22, alpha=a_p * ss(t, 21.0, 21.4),
                anchor="ma", align="m")
        cv.text((P_X + P_W / 2, P_TOP + P_H + 46), "(pixels normalized by mean, std)", 18, alpha=a_p * ss(t, 21.0, 21.4),
                fill=style.TEXT_DIM, anchor="ma")

    # x = W p + b
    a_f = ss(t, 22.0, 22.5) * dim
    if a_f > 0.01:
        FXW.draw(cv, FXW_XY, alpha=a_f, anchor="lm", part_glow={"x": bump(t, 24.0, 0.8)},
                 notes={"x": "$1024$ dims", "W": "$1024\\times588$", "p": "$588$ numbers", "b": "$1024$ biases"})
        b = module_box(cv, W_BOX, "$W$", r"$1024\times588$ parameters", CYAN, a_f, glow=0.5 * bump(t, 22.6, 0.8))
        arrow(cv, (P_X + P_W + 8, ROWB_Y), b["l"], style.WHITE, a_f, 2, 14, progress=ss(t, 22.3, 22.8))
        a_o = ss(t, 23.1, 23.5) * dim
        arrow(cv, b["r"], (X_X - 8, ROWB_Y), style.WHITE, a_o, 2, 14, progress=ss(t, 23.1, 23.5))
        cv.rich("$+\\,b$", ((W_BOX[2] + X_X) / 2, ROWB_Y - 12), 26, alpha=a_o, anchor="mb")
        ug = ss(t, 23.3, 24.2)
        if ug > 0.01:
            hh = X_H * ug
            paste_box(cv, X_COLS[k], (X_X, ROWB_Y - hh / 2, X_W, hh), a_o)
            outline(cv, (X_X, ROWB_Y - hh / 2, X_W, hh), CYAN, 0.8 * a_o, 1)
        a_xl = ss(t, 24.0, 24.4) * dim
        cv.rich("$x$: $1024$-dim vector", (X_X + X_W / 2, X_TOP + X_H + 12), 22, alpha=a_xl, fill=style.LABEL_CYAN,
                anchor="ma", align="m")
        # length comparison: extend the top and bottom edges of p to the side of the x column
        for yy in (P_TOP, P_TOP + P_H):
            cv.s.dashed((P_X + P_W, yy), (X_X - 4, yy), style.GREY, 0.35 * a_xl, 1, n=40)

    # Sentence 10: all 1369 cells share the same W (right-hand explanation column)
    a_s = ss(t, 24.9, 25.3) * dim
    if a_s > 0.01:
        idx = SEQ.index(k) if t >= SEQ_T0 else 0
        cv.text((PANEL_X, 610), "This cell now:", 24, alpha=a_s, fill=style.TEXT_DIM, anchor="lm")
        number(cv, (PANEL_X + 150, 610), k, "{:d}", 30, style.LABEL_GOLD, a_s, "lm")
        cv.text((PANEL_X, 650), "(1369 cells in total, indexed 0–1368)", 20, alpha=a_s, fill=style.TEXT_DIM, anchor="lm")
        cv.rich("Every cell: the same $W$,\nthe same $b$", (PANEL_X, 700), 24, alpha=a_s * ss(t, 25.4, 25.8))
        cv.rich("= a 14×14 conv, stride 14\n(DINOv2's patch embed)", (PANEL_X, 790), 24, alpha=a_s * ss(t, 25.9, 26.3),
                fill=style.LABEL_CYAN)
        cv.rich("Params: $1024\\times588+1024$\n$=603{,}136$", (PANEL_X, 880), 24, alpha=a_s * ss(t, 26.4, 26.8),
                fill=style.TEXT_DIM)
        del idx

    # Sentence 11: shape chain, 24 layers stay at 1024
    a_ch = ss(t, 27.2, 27.6) * dim
    if a_ch > 0.01:
        pa = {"img": 1.0, "p": ss(t, 27.5, 27.9), "x": ss(t, 28.0, 28.4), "vit": ss(t, 28.5, 28.9)}
        CHAIN.draw(cv, CHAIN_XY, alpha=a_ch, part_alpha=pa,
                   box={"x": bump(t, 28.4, 1.2), "vit": bump(t, 28.9, 1.2)}, hl_color=CYAN,
                   notes={"img": "One image: 3 channels", "p": "Cut, flatten", "x": "Multiply by $W$, add $b$", "vit": "After the 24-layer ViT-L"})
        cv.text((540, 1190), "Backbone: 24 rounds at 1024 dims; before the heads, a round's frame + global outputs are concatenated → 2048", 20,
                alpha=a_ch * ss(t, 29.0, 29.4), fill=style.TEXT_DIM, anchor="mm")


@shot.frame
def frame(cv, t):
    ov = 1 - ss(t, T_WHY, T_WHY + 0.6)
    if ov > 0.01:
        overview(cv, t, ov)
    if t > T_WHY:
        why_1024(cv, t, 1 - 0.55 * POP.on(t))
    POP.draw(cv, t, body_pop)
