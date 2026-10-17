"""E02-04 Depth Pro 1: multi-scale tiling + shared encoder (M, 44 s). Storyboard storyboard.md section E02-04.
Computed for real: 518 -> 1536 resize (×2.965, pixel-center mapping), tile starts / stride / overlap / tile count (25 + 9 + 1 = 35), (35, 3, 384, 384),
      384 / 16 = 24 -> (35, 1024, 24, 24), the white dot's coordinate on the 1536 image (711.9, 880.5) and which two tiles cover it, Voronoi boundary y = 912 and the owning tile,
      token in tile (19, 8) -> merged map (55, 44), tokens per row 21+18+18+18+21 = 96, attention entries 9216² vs 36×576² (about 1/7.1),
      tokens through the linear layers 20,736 = 2.25 × 9,216.
Illustration: token values in the output feature map are not drawn (only the 24×24 grid); everything else is the real camera-2 tiling, nothing fabricated.
Paper numbers: Tab. 16 overlap 25%, Tab. 9 latency 341 vs 392 ms, DIS-5k boundary recall 0.080 vs 0.065 (tagged "Paper Tab. 9").
Official code: middle scale 768, the finest level also takes layers 5 and 11 (tag "Official code").
Tags: the picture is the scene image -> c2.GT_TAG; #3, #6 also c2.CODE_TAG; #6-#7 also "Illustration"; #10 paper numbers tagged "Paper Tab. 9 + our computation".
"""
import cv2
import numpy as np

import e02_common as c2
from animkit import (CYAN, GOLD, GREEN, ORANGE, VIOLET, Formula, ImagePanel, Popup, Shot, T, arrow, bar_chart, bump,
                     fade, lerp, module_box, popup_check, ss, style, tile_starts)

ec, sc = c2.ec, c2.sc
NB = c2.NB()

DUR = 44.0
CAPS = [(0.3, 4.0, "Input one image; output inverse depth and field of view"),
        (4.2, 8.4, "Whole image into a ViT: too costly"),
        (9.2, 13.2, "Cut into 384 tiles at three scales"),
        (13.4, 17.4, "Finest level: 5×5 tiles, stride 288"),
        (17.6, 21.6, "Middle 9 tiles, coarsest 1 tile, 35 in all"),
        (21.8, 26.2, "35 tiles share one ViT"),
        (26.4, 29.6, "Another ViT sees the whole downscaled image"),
        (29.8, 33.0, "White dot sits in a two-tile overlap"),
        (33.2, 37.4, "Merge: overlaps cut along the midline"),
        (37.6, 41.6, "Tiling makes attention about 7x cheaper")]
shot = Shot("s04_dp_tiles", DUR)
shot.title("Depth Pro 1. Tiling", "Multi-scale patches · shared ViT encoder", 0.2, 2.8)
shot.tag(c2.GT_TAG, 0.3, 37.5)
shot.tag(c2.CODE_TAG, 9.2, 13.4)
shot.tag(c2.CODE_TAG, 21.8, 26.2)
shot.tag("Illustration: token values not drawn", 22.0, 29.6)
shot.tag("Paper Tab. 9 + our computation", 37.6, DUR)
for a_, b_, s_ in CAPS:
    shot.caption(a_, b_, s_)

# ------------------------------------------------------------------ image and tiles
IMG518 = np.ascontiguousarray(ec.view_rgb(c2.CAM2)).astype(np.uint8)
IMG1536 = cv2.resize(IMG518, (1536, 1536), interpolation=cv2.INTER_LINEAR)       # half-pixel centers = align_corners=False
IMG768 = cv2.resize(IMG518, (768, 768), interpolation=cv2.INTER_LINEAR)
IMG384 = cv2.resize(IMG518, (384, 384), interpolation=cv2.INTER_AREA)
STR1, STR2 = int(round(NB["stride1"])), int(round(NB["stride2"]))
XS1 = tile_starts(1536, 384, STR1)
XS2 = tile_starts(768, 384, STR2)
XS3 = tile_starts(384, 384, STR1)
P1 = np.asarray(NB["p1536"], float)                                              # white dot on the 1536 image (x, y)
KEEP = tuple(int(v) for v in NB["keep1"])                                        # (row, col)
COVER_R, COVER_C = [list(map(int, v)) for v in NB["cover1"]]
VL = [float(v) for v in NB["voronoi_lines1"]]
TOK = tuple(int(v) for v in NB["tok_in_patch"])                                  # token within the tile (row, col)
MERGED = tuple(int(v) for v in NB["tok_merged"])
UV518 = np.asarray(ec.nb()["uv"], float)
BASE_TOK = (XS1[KEEP[0]] // 16, XS1[KEEP[1]] // 16)                              # tile start converted to a token count

# position of the large 1536 image (x, y, width): keyframes of each stage
K1, K2, K3, K4, K9 = (500, 385, 480), (280, 710, 520), (70, 420, 480), (120, 420, 800), (270, 330, 540)


def mix(a, b, u):
    return tuple(lerp(p, q, u) for p, q in zip(a, b))


def geom(t):
    g = mix(K1, K2, ss(t, 4.2, 5.0))
    g = mix(g, K3, ss(t, 9.2, 10.0))
    g = mix(g, K4, ss(t, 13.4, 14.2))
    return mix(g, K9, ss(t, 33.2, 34.0))


_PANELS = {}


def panel_at(x, y, w):
    key = (int(round(x)), int(round(y)), int(round(w)))
    if key not in _PANELS:
        _PANELS[key] = ImagePanel(IMG1536, (key[0], key[1]), w=key[2])
    return _PANELS[key]


def main_alpha(t):
    return float(np.clip(ss(t, 1.2, 1.8) * (1 - ss(t, 17.4, 18.0)) + ss(t, 29.8, 30.4) * (1 - ss(t, 37.2, 37.6)), 0, 1))


STAGE = panel_at(*K4)
_O = STAGE.tiles(None, XS1, XS1, 384, alpha=0.0)
ROW_TOK = [int(round(v)) for v in np.diff(_O["vor_x"]) / 16.0]                  # tokens kept per tile in one row

# the other two scales, side-by-side thumbnails for sentence 3 (uniform on-screen scale: each tile is 120 px)
P768 = ImagePanel(IMG768, (610, 660), w=240)
P384 = ImagePanel(IMG384, (890, 780), w=120)

# ------------------------------------------------------------------ 35 tile thumbnails (real crops)
THUMB, BORD = [], []
for r in XS1:
    for c in XS1:
        THUMB.append(cv2.resize(IMG1536[r:r + 384, c:c + 384], (190, 190), interpolation=cv2.INTER_AREA))
        BORD.append(CYAN)
for r in XS2:
    for c in XS2:
        THUMB.append(cv2.resize(IMG768[r:r + 384, c:c + 384], (190, 190), interpolation=cv2.INTER_AREA))
        BORD.append(VIOLET)
THUMB.append(cv2.resize(IMG384, (190, 190), interpolation=cv2.INTER_AREA))
BORD.append(GREEN)
N_BLK = len(THUMB)

_tg = np.full((96, 96, 3), (20, 34, 42), np.uint8)                               # 24×24 token grid (illustration, values not drawn)
for _i in range(24):
    for _j in range(24):
        _tg[_i * 4 + 1:_i * 4 + 4, _j * 4 + 1:_j * 4 + 4] = (60, 150, 170)
TOKGRID = _tg


def card(cv, img, border, x, y, size, alpha):
    if alpha <= 0.01:
        return
    ec.paste_img(cv, img, (x, y), size, alpha)
    cv.s.polyline([(x, y), (x + size, y), (x + size, y + size), (x, y + size)], border, 0.9 * alpha, 1, closed=True)


def say(cv, s, xy, size=26, alpha=1.0, fill=style.TEXT, anchor="mm"):
    cv.rich(s, xy, size, alpha=alpha, fill=fill, anchor=anchor, align="m" if anchor[0] == "m" else ("r" if anchor[0] == "r" else "l"))


# ------------------------------------------------------------------ popups
POP1 = Popup((60, 300, 960, 395), t_in=4.2, t_out=9.0, title="ViT recap: what a token is, why attention is costly", corner="tl")
POP2 = Popup((40, 880, 1000, 350), t_in=33.2, t_out=37.6, title="Voronoi merge: overlaps cut along the midline", corner="bl")
CROP16 = cv2.resize(IMG1536[55 * 16:56 * 16, 44 * 16:45 * 16], (64, 64), interpolation=cv2.INTER_NEAREST)
S_ATTN = "$1536/16=%d$, $%d^2=%d$ tokens; per layer per head $%d^2=%.2f\\times10^{7}$ entries" % (
    1536 // 16, 1536 // 16, int(NB["n_tok_plain"]), int(NB["n_tok_plain"]), NB["attn_plain"] / 1e7)


def body1(sub, lt, box):
    x0, y0, x1, y1 = box
    a = ss(lt, 0.3, 0.7)
    sub.rich("ViT: cut every 16×16 pixels into a cell; one cell = 1 token (1024-dim)", (x0 + 4, y0 + 2), 24, alpha=a, anchor="la")
    ap = ss(lt, 0.8, 1.2)
    if ap > 0.01:
        sub.s.blit_rgb(CROP16, int(x0 + 14), int(y0 + 56), ap)
        sub.s.polyline([(x0 + 14, y0 + 56), (x0 + 78, y0 + 56), (x0 + 78, y0 + 120), (x0 + 14, y0 + 120)],
                       ORANGE, ap, 2, closed=True)
        sub.rich("16×16 px", (x0 + 46, y0 + 128), 20, alpha=ap, fill=style.TEXT_DIM, anchor="ma", align="m")
    at = ss(lt, 1.3, 1.7)
    if at > 0.01:
        arrow(sub, (x0 + 92, y0 + 88), (x0 + 142, y0 + 88), style.WHITE, at, 2, 12, progress=at)
        sub.s.rect(x0 + 158, y0 + 56, x0 + 184, y0 + 120, CYAN, 0.85 * at, r=4)
        sub.rich("1 token", (x0 + 171, y0 + 128), 20, alpha=at, fill=style.LABEL_CYAN, anchor="ma", align="m")
    am = ss(lt, 2.0, 2.4)
    if am > 0.01:
        gx, gy, cs = x0 + 340, y0 + 54, 14
        for i in range(8):
            for j in range(8):
                sub.s.rect(gx + j * cs + 1, gy + i * cs + 1, gx + (j + 1) * cs - 1, gy + (i + 1) * cs - 1, CYAN,
                           (0.18 + 0.30 * ((i * 3 + j * 5) % 4) / 3) * am, r=1)
        sub.rich("Each layer: every token attends to all tokens", (gx + 8 * cs + 20, gy + 18), 24, alpha=am, anchor="lm")
        sub.rich("N tokens → $N^2$ entries (per head)", (gx + 8 * cs + 20, gy + 60), 22, alpha=am, fill=style.TEXT_DIM,
                 anchor="lm")
    ac = ss(lt, 2.9, 3.3)
    sub.rich(S_ATTN, (x0 + 4, y0 + 194), 26, alpha=ac, fill=style.LABEL_GOLD, anchor="la")
    ad = ss(lt, 3.7, 4.1)
    sub.rich("DINOv2 originally uses 14×14 cells; the authors interpolated the weights to 16×16 (paper B.3)", (x0 + 4, y0 + 244), 20, alpha=ad,
             fill=style.TEXT_DIM, anchor="la")


def body2(sub, lt, box):
    x0, y0, x1, y1 = box
    r, c = KEEP
    cy = [XS1[r] + 192, XS1[r + 1] + 192]
    sub.rich("Each tile keeps only the part closer to its own center", (x0 + 4, y0 + 0), 26, alpha=ss(lt, 0.3, 0.7), fill=style.LABEL_GOLD)
    sub.rich("Tiles in rows %d and %d have centers $y=%d$ and $%d$ (tile start + 192); the midpoint $y=%d$ is the boundary" % (r, r + 1, cy[0], cy[1], VL[2]),
             (x0 + 4, y0 + 40), 22, alpha=ss(lt, 0.9, 1.3))
    sub.rich("White dot $y=%.1f<%d$ → tile %d (row %d, col %d, counted from 0)" % (P1[1], VL[2], KEEP[0] * 5 + KEEP[1], r, c),
             (x0 + 4, y0 + 80), 24, alpha=ss(lt, 1.6, 2.0))
    sub.rich("Token in tile $(%d,%d)$ → merged map $(%d,%d)$" % (TOK[0], TOK[1], MERGED[0], MERGED[1]), (x0 + 4, y0 + 118), 26,
             alpha=ss(lt, 2.3, 2.7), fill=style.LABEL_CYAN)
    sub.rich("$%d=%d+%d$, $%d=%d+%d$; tile start $%d$ px ÷ 16 = token $%d$" % (
        MERGED[0], BASE_TOK[0], TOK[0], MERGED[1], BASE_TOK[1], TOK[1], XS1[r], BASE_TOK[0]),
        (x0 + 4, y0 + 154), 20, alpha=ss(lt, 2.7, 3.1), fill=style.TEXT_DIM)
    ab = ss(lt, 3.2, 3.6)
    if ab > 0.01:
        sub.rich("Tokens per row $%s=%d=1536/16$" % ("+".join(str(v) for v in ROW_TOK), sum(ROW_TOK)),
                 (x0 + 4, y0 + 190), 22, alpha=ab, anchor="la")
        bx, bw = x0 + 4, 900.0
        xx = bx
        for k, n in enumerate(ROW_TOK):
            wseg = bw * n / sum(ROW_TOK)
            sub.s.rect(xx + 1, y0 + 222, xx + wseg - 1, y0 + 254, CYAN if k % 2 == 0 else (60, 130, 150), 0.85 * ab, r=3)
            sub.rich(str(n), (xx + wseg / 2, y0 + 238), 22, alpha=ab, fill=(10, 20, 28), anchor="mm", align="m")
            xx += wseg


# ------------------------------------------------------------------ formulas (static, built at module level)
FQ = Formula([("25", "a"), ("+9", "b"), ("+1", "c"), ("=35", "n")], size=56, gap=0.2)
FS = Formula([("(35", "b"), (r",\,3", "c"), (r",\,384", "h"), (r",\,384)", "w")], size=46, gap=0.16)

# ------------------------------------------------------------------ layout constants for sentences 5-7
STACK5 = (360.0, 640.0, 190.0, 5.0)          # top-left x, y of the bottom-left card; side length; per-card offset
STACK6 = (205.0, 480.0, 92.0, 2.2)
CX1, CX2 = 280, 800                           # center lines of the left and right columns
BOX1 = (60, 660, 500, 780)
BOX2 = (580, 660, 1020, 780)
OUT1 = (205.0, 940.0)                         # top-left of the bottom-left card of the output feature-map stack
OUT2 = (754.0, 940.0)
T_FLY = 22.6


def stack_geom(t):
    u = ss(t, 21.8, 22.4)
    return tuple(lerp(a, b, u) for a, b in zip(STACK5, STACK6))


def wglyph(cv, x, y, color, alpha, kind):
    for i in range(3):
        for j in range(3):
            if kind == "sq":
                cv.s.rect(x + j * 13, y + i * 13, x + j * 13 + 10, y + i * 13 + 10, color, alpha, r=2)
            else:
                cv.s.dot((x + j * 13 + 5, y + i * 13 + 5), color, 5.5, alpha)


@shot.check
def check(ck):
    ck.eq("image resized to 1536", IMG1536.shape, (1536, 1536, 3))
    ck.eq("scale factor on screen", f"{NB['k1536']:.3f}", "2.965")
    ck.eq("finest-level starts", tile_starts(1536, 384, 288), [0, 288, 576, 864, 1152])
    ck.eq("middle-level starts", XS2, [0, 192, 384])
    ck.eq("coarsest-level starts", XS3, [0])
    ck.eq("stride on screen", f"{NB['stride1']:.0f}", "288")
    ck.eq("overlap on screen", f"{NB['overlap1']:.0f}", "96")
    ck.eq("overlap / tile width = 25%", f"{NB['overlap1'] / 384:.0%}", "25%")
    ck.eq("tile count 25 + 9 + 1", [len(XS1) ** 2, len(XS2) ** 2, len(XS3) ** 2], [25, 9, 1])
    ck.eq("35 tiles in total", N_BLK, 35)
    ck.eq("5 tiles in a row cover the whole image", XS1[-1] + 384, 1536)
    ck.eq("white-dot 1536 coordinates on screen", [f"{P1[0]:.1f}", f"{P1[1]:.1f}"], ["711.9", "880.5"])
    ck.close("white-dot 1536 coordinates (pixel-center mapping)", P1, [711.87, 880.48], 0.01)
    ck.close("pixel-center mapping (u+0.5)·1536/518−0.5", (UV518 + 0.5) * 1536 / 518 - 0.5, P1, 1e-9)
    ck.eq("white-dot pixel in camera 2 on screen", [f"{UV518[0]:.1f}", f"{UV518[1]:.1f}"], ["239.7", "296.6"])
    ck.eq("owner(white dot) = (2, 2)", _O["owner"](711.87, 880.48), (2, 2))
    ck.eq("keep1 = (2, 2)", KEEP, (2, 2))
    ck.true("tiles covering the white dot: rows [2, 3], column [2]", COVER_R == [2, 3] and COVER_C == [2], f"{COVER_R} {COVER_C}")
    ck.eq("tile index 2×5+2 = 12", KEEP[0] * 5 + KEEP[1], 12)
    ck.eq("patch_idx1 = 12", int(NB["patch_idx1"]), 12)
    ck.true("white dot lies in both covering tiles", all(XS1[rr] <= P1[1] < XS1[rr] + 384 for rr in COVER_R) and
            all(XS1[cc] <= P1[0] < XS1[cc] + 384 for cc in COVER_C), "")
    ck.eq("Voronoi boundary y on screen", f"{VL[2]:.0f}", "912")
    ck.eq("two tile centers", [XS1[2] + 192, XS1[3] + 192], [768, 1056])
    ck.true("white dot is above the boundary", P1[1] < VL[2], f"{P1[1]:.1f} < {VL[2]}")
    ck.eq("token within tile on screen", f"({TOK[0]}, {TOK[1]})", "(19, 8)")
    ck.eq("token index within tile 464", int(NB["tok_in_patch_idx"]), 464)
    ck.eq("merged-map token on screen", f"({MERGED[0]}, {MERGED[1]})", "(55, 44)")
    ck.eq("merged-map token index 5324", int(NB["tok_merged_idx"]), 5324)
    ck.eq("merged = tile-start token + token within tile", (BASE_TOK[0] + TOK[0], BASE_TOK[1] + TOK[1]), MERGED)
    ck.eq("tokens per row 21+18+18+18+21", ROW_TOK, [21, 18, 18, 18, 21])
    ck.eq("96 per row in total", sum(ROW_TOK), 96)
    ck.eq("96 = 1536/16", 1536 // 16, 96)
    ck.eq("24×24 tokens per tile", 384 // 16, 24)
    ck.eq("24² = 576", 24 ** 2, 576)
    ck.eq("scale 2: stride 192, overlap 192", [STR2, int(NB["overlap2"])], [192, 192])
    ck.close("scale 2: white-dot 768 coordinates", NB["p768"], [355.685, 439.991], 0.01)
    ck.eq("scale 2: covered by 2×2 tiles", [len(v) for v in NB["cover2"]], [2, 2])
    ck.eq("scale 2: owned by (1, 1)", list(map(int, NB["keep2"])), [1, 1])
    ck.eq("96² = 9216 tokens", int(NB["n_tok_plain"]), 9216)
    ck.eq("scaled up directly: 8.49×10^7 per layer per head", f"{NB['attn_plain'] / 1e7:.2f}", "8.49")
    ck.eq("Depth Pro 36×576² on screen: 1.19×10^7", f"{NB['attn_dp'] / 1e7:.2f}", "1.19")
    ck.eq("36×576²", int(NB["attn_dp"]), 36 * 576 ** 2)
    ck.eq("36 = 35 tiles + 1 downscaled whole image", 35 + 1, 36)
    ck.eq("ratio on screen 1/7.1", f"1/{NB['attn_ratio']:.1f}", "1/7.1")
    ck.eq("tokens through the linear layers", int(NB["tok_dp"]), 20736)
    ck.eq("token ratio 2.25", f"{NB['tok_ratio']:.2f}", "2.25")
    ck.eq("paper Tab. 9 numbers", list(LAT) + list(REC), [341, 392, 0.080, 0.065])
    ck.true("4 Voronoi boundary lines", len(VL) == 4, "")
    popup_check(ck, [POP1, POP2], DUR)
    ck.record(p1536=P1.round(2), starts=XS1, row_tokens=ROW_TOK, tiles=N_BLK)


LAT, REC = (341, 392), (0.080, 0.065)          # Tab. 9: Depth Pro tiling vs the same ViT-L scaled up directly to 1536 (C-11, cited secondhand)


# ------------------------------------------------------------------ visuals of each sentence
def ph1(cv, t):
    a = fade(t, 0.3, 4.6, 0.4, 0.5)
    if a < 0.01:
        return
    say(cv, "Input: any image is first resized to 1536×1536", (540, 318), 30, a)
    PS = ImagePanel(IMG518, (80, 475), w=300)
    PS.draw(cv, a * ss(t, 0.4, 0.9), border=c2.CAM2_COLOR, title="Cam 2 image 518×518")
    ar = ss(t, 1.0, 1.5)
    arrow(cv, (392, 625), (490, 625), style.WHITE, a, 2, 14, progress=ar)
    say(cv, "×%.3f" % NB["k1536"], (441, 598), 24, a * ar, style.TEXT_DIM)
    say(cv, "1536×1536", (740, 366), 28, a * ss(t, 1.4, 1.9))
    a2 = ss(t, 2.0, 2.5) * a
    arrow(cv, (740, 872), (740, 904), style.WHITE, a2, 2, 12, progress=a2)
    module_box(cv, (520, 905, 960, 985), "Depth Pro", None, style.WHITE, a2, title_size=30)
    say(cv, "$C$: canonical inverse depth, unit 1/m = fraction of image width the object covers ÷ object's actual width (m)", (60, 1205), 20, ss(t, 3.3, 3.8) * a,
        style.TEXT_DIM, "lm")
    for (box, ttl, sub, t0, xe) in [((60, 1055, 560, 1190), "$C$: canonical inverse depth", r"$1536\times1536$ (covered in E02-05)", 2.8, 330),
                                    ((580, 1055, 1020, 1190), r"$\theta$: horizontal field of view", "1 number", 3.3, 800)]:
        u = ss(t, t0, t0 + 0.5) * a
        arrow(cv, (740, 990), (xe, 1050), style.WHITE, u, 2, 12, progress=ss(t, t0, t0 + 0.5))
        module_box(cv, box, ttl, sub, CYAN, u, title_size=28, sub_size=22)


def ph2(cv, t, P, am):
    a = ss(t, 4.9, 5.2) * (1 - ss(t, 9.2, 9.8)) * am
    if a < 0.01:
        return
    P.grid(cv, cell=16, color=CYAN, alpha=0.5 * a, progress=ss(t, 5.0, 7.0))
    P.cell(cv, 55, 44, cell=16, color=ORANGE, alpha=ss(t, 6.0, 6.4) * a, fill=0.7, glow=0.8)
    say(cv, "96 × 96\n= 9216 tokens", (815, 960), 26, a * ss(t, 6.4, 6.9), anchor="lm")


def ph34(cv, t, P, am):
    # sentence 3: three scales side by side
    a3 = (1 - ss(t, 13.4, 13.8))
    if 9.2 <= t < 13.9:
        a768, a384 = ss(t, 10.0, 10.5) * a3, ss(t, 10.6, 11.1) * a3
        P768.draw(cv, a768, border=style.GREY)
        P384.draw(cv, a384, border=style.GREY)
        say(cv, "×0.5", (730, 644), 24, a768, style.TEXT_DIM)
        say(cv, "×0.25", (950, 764), 24, a384, style.TEXT_DIM)
        P768.tiles(cv, XS2, XS2, 384, progress=ss(t, 11.2, 12.4), alpha=a768)
        P384.tiles(cv, XS3, XS3, 384, progress=ss(t, 11.9, 12.5), alpha=a384)
        al = ss(t, 12.2, 12.7) * a3
        say(cv, "1536×1536\n5×5 = 25 tiles", (310, 930), 26, al, anchor="mt")
        say(cv, "768×768\n3×3 = 9 tiles", (730, 930), 26, al, anchor="mt")
        say(cv, "384×384\n1 tile", (950, 930), 26, al, anchor="mt")
        say(cv, "The paper omits the middle scale;\nofficial code: half size", (730, 1010), 22, ss(t, 12.7, 13.2) * a3, style.TEXT_DIM, "mt")
    # sentences 3-4: the 25 tiles on the main image
    if 10.8 <= t < 18.0 and am > 0.01:
        k = int((t - 13.9) / 0.4) if 13.9 <= t < 15.9 else -1
        hl = {(0, k): GOLD} if 0 <= k <= 4 else None
        P.tiles(cv, XS1, XS1, 384, progress=ss(t, 10.8, 12.2), alpha=am, highlight=hl,
                overlap_color=ORANGE, overlap_alpha=0.4 * ss(t, 15.8, 16.6))
    # sentence 4: starts, stride, overlap
    if 13.4 <= t < 18.0:
        a4 = (1 - ss(t, 17.4, 18.0))
        say(cv, "Tile start", (66, 358), 22, a4 * ss(t, 13.8, 14.2), style.TEXT_DIM)
        k = int((t - 13.9) / 0.4)
        for c, x in enumerate(XS1):
            ac = ss(t, 13.9 + 0.4 * c, 14.2 + 0.4 * c) * a4
            sx = float(P.to_screen(x, 0)[0])
            lit = bump(t, 13.9 + 0.4 * c, 0.8)
            say(cv, str(x), (sx, 358), 24, ac, style.LABEL_GOLD if lit > 0.2 else style.TEXT_DIM)
            cv.s.line((sx, 372), (sx, 414), GOLD, 0.7 * ac, 1)
        say(cv, "Stride %d px" % STR1, (778, 358), 24, a4 * ss(t, 15.3, 15.8), style.TEXT_DIM, "lm")
        ab = ss(t, 15.9, 16.4) * a4
        if ab > 0.01:
            xa, xb = float(P.to_screen(STR1, 0)[0]), float(P.to_screen(384, 0)[0])
            cv.s.line((xa, 400), (xb, 400), ORANGE, ab, 2)
            for xx in (xa, xb):
                cv.s.line((xx, 394), (xx, 406), ORANGE, ab, 2)
            cv.s.line((xb - 6, 398), (xb - 6, 334), ORANGE, 0.7 * ab, 1)
            say(cv, "Overlap %d px = %.0f%% of tile width (Tab. 16)" % (NB["overlap1"], 100 * NB["overlap1"] / 384), (120, 318), 26,
                ab, style.ORANGE, "lm")


def ph5(cv, t):
    a = (1 - ss(t, 21.0, 21.6))
    if 17.6 <= t < 21.8:
        pa = {"a": ss(t, 17.9, 18.3), "b": ss(t, 18.6, 19.0), "c": ss(t, 19.2, 19.6), "n": ss(t, 19.8, 20.2)}
        FQ.draw(cv, (540, 345), alpha=a, part_alpha=pa, part_color={"a": CYAN, "b": VIOLET, "c": GREEN},
                part_glow={"n": bump(t, 20.0, 0.8)},
                notes={"a": "Finest 5×5", "b": "Middle 3×3", "c": "Coarsest 1 tile", "n": "Tiles"})
        a2 = a * ss(t, 20.2, 20.6)
        FS.draw(cv, (540, 905), alpha=a2, notes={"b": "Tiles (batch)", "c": "RGB", "h": "H", "w": "W"})
        say(cv, "Stacked along the batch dim, fed in at once", (540, 1040), 30, a * ss(t, 20.8, 21.2))


def ph567(cv, t):
    """Left column: stack of 35 tiles -> patch encoder -> output stack; right column: whole image -> image encoder -> one feature map."""
    lc = ss(t, 17.8, 18.2) * (1 - ss(t, 29.4, 30.0))
    if lc < 0.01:
        return
    x0, y0, sz, off = stack_geom(t)
    done = 0
    # draw the output stack + arrow + box first (cards fly over the box)
    a_box = ss(t, 22.0, 22.5) * lc
    if a_box > 0.01:
        module_box(cv, BOX1, "patch encoder", "ViT-L, DINOv2 weights", CYAN, a_box, title_size=28, sub_size=22)
        wglyph(cv, BOX1[0] + 14, BOX1[1] + 40, GOLD, a_box, "sq")
        say(cv, "35 tiles use the same weights", (BOX1[0] + 6, 815), 22, a_box, style.TEXT_DIM, "lm")
        arrow(cv, (CX1, 786), (CX1, 858), style.WHITE, a_box * ss(t, 22.5, 23.0), 2, 12, progress=ss(t, 22.5, 23.0))
    for i in range(N_BLK):
        ta = T_FLY + 0.05 * i
        ua, ub = ss(t, ta, ta + 0.45), ss(t, ta + 0.45, ta + 0.9)
        px, py = x0 + off * i, y0 - off * i
        if ua <= 0:
            drop = ss(t, 18.2 + 0.045 * i, 18.45 + 0.045 * i)
            card(cv, THUMB[i], BORD[i], px, py - (1 - drop) * 30, sz, drop * lc)
        else:                                          # already fed in: leave a faint shadow in the stack
            card(cv, THUMB[i], BORD[i], px, py, sz, 0.2 * lc)
        if 0 < ua < 1:
            s2 = lerp(sz, 56, ua)
            tx, ty = CX1 - 28, 720 - 28
            card(cv, THUMB[i], BORD[i], lerp(px, tx, ua), lerp(py, ty, ua), s2, lc * (1 - ss(ua, 0.55, 1.0)))
        if ub > 0:
            ox, oy = OUT1[0] + 2.2 * i, OUT1[1] - 2.2 * i
            card(cv, TOKGRID, CYAN, ox, lerp(790, oy, ub), 92, ub * lc)
        if ua >= 1:
            done += 1
    a6 = ss(t, 22.2, 22.7) * lc
    cv.math(r"(35,\,3,\,384,\,384)", (CX1, 372), 26, style.LABEL_CYAN, a6)
    if t >= T_FLY:
        say(cv, T("{n} / 35 tiles done", n=done), (CX1 + 220, 815), 20, a_box, style.TEXT_DIM, "rm")
    ao = ss(t, 23.6, 24.0) * lc
    cv.math(r"(35,\,1024,\,24,\,24)", (CX1, 1058), 28, style.LABEL_CYAN, ao)
    say(cv, "Each tile: 24×24 tokens", (CX1, 1092), 24, ao)
    say(cv, "The finest level also takes outputs of layers 5 and 11\n(official code): 50 more tiles", (CX1, 1124), 22, ss(t, 24.6, 25.2) * lc,
        style.TEXT_DIM, "mt")
    # ---- sentence 7: right column
    a7 = ss(t, 26.4, 26.9) * lc
    if a7 > 0.01:
        PI = ImagePanel(IMG384, (730, 425), w=140)
        PI.draw(cv, a7, border=style.GREY, title="Whole image resized to 384")
        a8 = ss(t, 26.8, 27.3) * lc
        module_box(cv, BOX2, "image encoder", "Separate weights", VIOLET, a8, title_size=28, sub_size=22)
        wglyph(cv, BOX2[0] + 14, BOX2[1] + 40, VIOLET, a8, "dot")
        arrow(cv, (CX2, 574), (CX2, 654), style.WHITE, a8, 2, 12, progress=ss(t, 26.9, 27.4))
        a9 = ss(t, 27.5, 28.0) * lc
        arrow(cv, (CX2, 786), (CX2, 934), style.WHITE, a9, 2, 12, progress=a9)
        card(cv, TOKGRID, CYAN, OUT2[0], OUT2[1] - 6 * (1 - a9), 92, a9)
        cv.math(r"(1024,\,24,\,24)", (CX2, 1058), 28, style.LABEL_CYAN, ss(t, 28.0, 28.4) * lc)
        say(cv, "Gives whole-image context", (CX2, 1092), 24, ss(t, 28.2, 28.7) * lc)


def ph8(cv, t, P, am):
    if not (29.8 <= t < 33.4) or am < 0.01:
        return
    P.tiles(cv, XS1, XS1, 384, progress=1.0, alpha=0.33 * am, stagger=0)
    xs, ys = [XS1[c] for c in COVER_C], [XS1[r] for r in COVER_R]
    hlk = ss(t, 30.6, 31.0)
    hl = {(i, j): (ORANGE, CYAN)[i] for i in range(len(ys)) for j in range(len(xs))}
    o = P.tiles(cv, xs, ys, 384, progress=1.0, alpha=am * hlk, highlight=hl, stagger=5)
    # overlap band flash
    ya, yb = ys[1], ys[0] + 384
    q = [P.to_screen(xs[0], ya), P.to_screen(xs[0] + 384, ya), P.to_screen(xs[0] + 384, yb), P.to_screen(xs[0], yb)]
    fl = ss(t, 31.2, 31.6) * (0.5 + 0.4 * np.sin((t - 31.2) * 7.0)) * am
    cv.s.add_poly([tuple(p) for p in q], GOLD, 0.45 * fl)
    for (r, c), s in [((COVER_R[0], COVER_C[0]), "Tile (%d, %d)" % (COVER_R[0], COVER_C[0])),
                      ((COVER_R[1], COVER_C[0]), "Tile (%d, %d)" % (COVER_R[1], COVER_C[0]))]:
        if r == COVER_R[1]:   # the next tile: move the label to the tile's lower left, away from the white dot
            pt = P.to_screen(XS1[c], XS1[r] + 384)
            cv.badge((pt[0] + 10, pt[1] - 44), s, 22, hlk * am, anchor="la")
        else:
            pt = P.to_screen(XS1[c], XS1[r])
            cv.badge((pt[0] + 10, pt[1] + 14), s, 22, hlk * am, anchor="la")
    ap = ss(t, 30.8, 31.3) * am
    sp = P.to_screen(P1[0], P1[1])
    ec.star_marker(cv, sp, ap, glow=0.6 + 0.4 * bump(t, 31.0, 1.0), rad=5)
    arrow(cv, (sp[0] + 8, sp[1] + 8), (sp[0] + 40, sp[1] + 60), style.WHITE, ap, 2, 8)
    cv.badge((sp[0] + 40, sp[1] + 64),"White dot (%.1f, %.1f)" % (P1[0], P1[1]), 24, ap, anchor="la")
    say(cv, "Cam 2 pixel (%.1f, %.1f) → 1536 image (%.1f, %.1f)" % (UV518[0], UV518[1], P1[0], P1[1]), (540, 318), 26,
        ap)
    say(cv, "518 → 1536 scaled by pixel centers; tile (row, col) counted from 0", (540, 362), 24, ap, style.TEXT_DIM)


def ph9(cv, t, P, am):
    if not (33.2 <= t < 37.7) or am < 0.01:
        return
    fill = 0.28 * ss(t, 34.4, 35.0) * (1 - ss(t, 35.2, 35.7))
    P.tiles(cv, XS1, XS1, 384, progress=1.0, alpha=0.5 * am, stagger=0, voronoi=ss(t, 34.0, 34.6),
            voronoi_fill=True, voronoi_fill_alpha=fill,
            keep=KEEP if t > 35.0 else None, keep_alpha=ss(t, 35.0, 35.6))
    sp = P.to_screen(P1[0], P1[1])
    ec.star_marker(cv, sp, am, glow=0.6, rad=4)
    ly = float(P.to_screen(0, VL[2])[1])
    x_r = float(P.to_screen(1536, 0)[0])
    say(cv, "White dot $y=%.1f$" % P1[1], (P.x0 - 12, sp[1] - 4), 24, am * ss(t, 33.8, 34.3), anchor="rm")
    say(cv, "Boundary $y=%d$" % VL[2], (x_r + 12, ly + 8), 24, am * ss(t, 34.2, 34.7), anchor="lm")
    say(cv, "Goes to tile %d\n(row %d, col %d)" % (KEEP[0] * 5 + KEEP[1], KEEP[0], KEEP[1]),
        (x_r + 12, float(P.to_screen(0, 560)[1])), 22, am * ss(t, 35.2, 35.7), style.LABEL_GOLD, "lm")


def ph10(cv, t):
    a = ss(t, 37.6, 38.0)
    if a < 0.01:
        return
    # A: attention entries
    ua = ss(t, 38.0, 39.4)
    bar_chart(cv, (480, 410, 980, 570), [NB["attn_plain"] / 1e7, NB["attn_dp"] / 1e7],
              ["Same ViT-L scaled up directly\n$9216^2$", "Depth Pro tiling\n$36\\times576^2$"], vmax=9.5, horizontal=True,
              progress=ua, fmt="{:.2f}", title="Attention entries per layer per head ($10^7$ entries, lower is better)", highlight=1, hl_color=CYAN,
              color=(120, 130, 145), alpha=a, label_size=22)
    say(cv, "about 1/%.1f" % NB["attn_ratio"], (760, 530), 38, ss(t, 39.2, 39.7), style.LABEL_GOLD, "lm")
    say(cv, "36 = 35 tiles + 1 downscaled whole image", (540, 604), 22, ss(t, 39.4, 39.9), style.TEXT_DIM)
    say(cv, "But multiscale + overlap + whole image make %.2fx as many tokens pass the linear layers (%s ÷ %s)" % (NB["tok_ratio"], f"{int(NB['tok_dp']):,}",
                                                         f"{int(NB['n_tok_plain']):,}"),
        (540, 644), 24, ss(t, 39.8, 40.3), style.LABEL_GOLD)
    # B: Tab. 9
    ub = ss(t, 40.3, 41.6)
    bar_chart(cv, (80, 760, 480, 925), list(LAT), ["Depth Pro\ntiling", "ViT-L\nscaled up"], vmax=450, progress=ub, fmt="{:.0f}",
              title="Latency, ms", better="lower", highlight=0, hl_color=CYAN, color=(120, 130, 145),
              alpha=a * ss(t, 40.2, 40.7), label_size=22)
    bar_chart(cv, (600, 760, 1000, 925), list(REC), ["Depth Pro\ntiling", "ViT-L\nscaled up"], vmax=0.09, progress=ub,
              fmt="{:.3f}", title="DIS-5k boundary recall", better="higher", highlight=0, hl_color=CYAN,
              color=(120, 130, 145), alpha=a * ss(t, 40.2, 40.7), label_size=22)
    say(cv, "Paper Tab. 9; baseline = the same ViT-L scaled up directly to 1536", (540, 1022), 20, ss(t, 41.0, 41.5), style.TEXT_DIM)
    say(cv, "Boundary recall: fraction of GT depth-jump edges the prediction also finds (details in E02-06)", (540, 1068), 24, ss(t, 41.4, 41.9))
    say(cv, "Shared weights: the authors say they may learn a scale-invariant representation (\"may\" in the paper; not tested separately)", (540, 1118), 24, ss(t, 42.3, 42.8),
        style.LABEL_CYAN)
    say(cv, "ViT structure unchanged; pretrained weights swapped in directly (paper p.4)", (540, 1166), 24, ss(t, 43.0, 43.5), style.TEXT_DIM)


@shot.frame
def frame(cv, t):
    am = main_alpha(t)
    P = panel_at(*geom(t))
    if am > 0.01:
        P.draw(cv, am, border=c2.CAM2_COLOR)
    ph1(cv, t)
    ph2(cv, t, P, am)
    ph34(cv, t, P, am)
    ph5(cv, t)
    ph567(cv, t)
    ph8(cv, t, P, am)
    ph9(cv, t, P, am)
    ph10(cv, t)
    POP1.draw(cv, t, body1)
    POP2.draw(cv, t, body2)
