"""E01-09 MapAnything inputs and tokens: optional inputs are added + the scale token (M, 79.6 s). Storyboard storyboard.md section E01-09.
Shapes: paper §3.1 + code (one 518x518 image). Measured (our test scene, 3 images): metric scale error under three input settings. The definition sentence for m is the same as in 08 (end of P7, held for 2.4 s); the "our test scene" phrase and the corner tag are shared.
Illustration: random switching of the input slots (the probabilities are the paper's values, the flicker sequence is random numbers from a fixed seed); the 4 numbers in the LN pop-up.
Each encoder component gets a small pop-up with a small animation: 7 pop-ups following the paper's components:
  P1 image encoder (DINOv2 ViT-G), P2 ray encoder (pixel unshuffle + shallow conv), P3 depth split first (normalized depth + log scale),
  P4 global quantities (one 4-layer MLP per quantity, broadcast to every patch), P5 LN, P6 additive fusion, P7 scale token (global layers only -> MLP -> exp -> m).
  The 16-layer G/F strip and the "-> 2-layer MLP -> exp -> m" live in P7; the scale-token label sits to the right of the grid.
Translation is the translation normalized by the mean length (not a unit vector); scale is encoded only for metric inputs, otherwise set to 0; each global quantity has its own MLP,
  broadcast to 37x37 and then added (as in the code); the ray encoder layers come from the MA weight file.
"""
from pathlib import Path

import numpy as np

import e01_common as ec
import scene_e01 as sc
from animkit import (CYAN, GOLD, GREEN, GREY, ORANGE, RED, WHITE, Formula, ImagePanel, Shot, TokenStrip, arrow,
                     bar_chart, bump, colorbar, colorize, fade, flicker, fold, input_slot, lerp, link, module_box,
                     number, pulse_dot, ss, style, tensor_block, token)
from e01_common import Popup, popup_check

# m is explained as in 08 (the same definition sentence, e01_common.M_DEF). Before P7 retracts, INS seconds are inserted and held,
# so the definition sentence stays lit long enough to read (in frame, t -> tm(t); P7 content stays on its last frame, the highlight of the definition sentence uses real time)
T_INS, INS = 67.8, 2.4
DUR = 77.2 + INS


def tm(t):
    """Real time -> original timeline (stops at T_INS during the insert)."""
    return t if t < T_INS else (T_INS if t < T_INS + INS else t - INS)


def sh(x):
    """Original timeline -> real time."""
    return x if x < T_INS - 1e-9 else x + INS


REAL = [0.0]                                      # current real time (used for the highlight of the definition sentence in P7)
shot = Shot("s09_ma_inputs", DUR)
shot.title("MapAnything inputs", "Encoders · fusion · scale token · §3.1", 0.2, 2.6)
shot.tag("Fusion formula: as written in the paper", 39.2, 52.2)
shot.tag("Illustration: switches flicker; probabilities are the paper's", sh(67.8), sh(73.2))
ec.scene_tag(shot, sh(73.2), DUR, "MapAnything weights: CC-BY-NC 4.0")
CAPS = [(0.3, 2.8, "Anything we already know can be fed in"), (2.8, 5.4, "Intrinsics as rays; depth and pose work too"),
        (5.4, 8.0, "Each input is encoded into 1536 dims"),
        # P1 image encoder
        (8.0, 10.6, "Image: through DINOv2 ViT-G"), (10.6, 13.2, "Split into 37×37 patches, 14 px each"),
        (13.2, 15.8, "One 1536-dim feature per patch"),
        # P2 ray encoder
        (15.8, 18.4, "ray map: 3 numbers per pixel"), (18.4, 21.0, "Each 14×14 patch becomes 588 numbers"),
        (21.0, 23.6, "Then a shallow conv makes it 1536 dims"),
        # P3 depth split first
        (23.6, 26.2, "Split depth first: mean depth + normalized depth"), (26.2, 28.8, "Take the log of the scale; 0 if not metric"),
        (28.8, 31.4, "Normalized depth goes through a conv of the same structure"),       # the weights are a different set from the ray conv (depth conv_in [588,196,3,3])
        # P4 global quantities
        (31.4, 34.0, "Translation is split too: divide by mean length"), (34.0, 36.6, "Each quantity has its own 4-layer MLP"),
        (36.6, 39.2, "The 1536-dim vector is added to every patch"),
        # P5 LN
        (39.2, 41.8, "Before merging, each branch gets its own LN"), (41.8, 44.4, "LN: subtract the mean, then divide by the std"),
        (44.4, 47.0, "Match the scales, then add; neither dominates"),
        # P6 addition
        (47.0, 49.6, "Add it if given, skip it if not"), (49.6, 52.2, "Shape unchanged after adding: 1536×37×37"),
        # token rows
        (52.2, 54.8, "Tokens per view stay the same"), (54.8, 57.4, "View 1 gets one fixed vector"),
        (57.4, 60.0, "Then append a scale token"),
        # P7 scale token
        (60.0, 62.6, "The scale token enters global layers only"), (62.6, 65.2, "In global layers it sees all views"),
        (65.2, 67.8, "MLP, then exp, gives the scale factor m"),        (67.8, 70.6, "Inputs are randomly hidden in training"), (70.6, 73.2, "One model handles every combination"),
        # the 9.5% group is "intrinsics + metric pose"; it does not fit in one sentence, so it is split into two
        (73.2, 75.2, "Our test scene: images only, scale error 72%"), (75.2, 77.2, "With intrinsics and metric pose: 9.5%")]
CAPS = ([c for c in CAPS if c[1] <= T_INS + 1e-9]
        + [(T_INS, T_INS + INS, "m: metric scale factor, scales the scene into meters")]            # same sentence as sentence 1 of 08
        + [(sh(a), sh(b), s) for a, b, s in CAPS if a >= T_INS - 1e-9])
for a, b, s in CAPS:
    shot.caption(a, b, s)

MA = ec.ma_metrics()
NB = ec.nb()
CAM = sc.EX_CAM
S = NB["s"]
SLOTS = [("Image (always given)", False, 390, CYAN), ("ray (= intrinsics)", True, 480, GOLD), ("Depth", True, 570, GOLD),
         ("Pose Q, T", True, 660, ORANGE), ("Scale (if metric)", True, 750, GREEN)]
ENC = [("ViT-G, first 24 layers", "$1536\\times37\\times37$"), ("Conv (unshuffle 14)", r"$588\to1536$, $37\times37$"),
       ("Conv (unshuffle 14)", r"$196\to1536$, $37\times37$"), ("4-layer MLP each", r"$1536$, broadcast to $37\times37$"),
       ("4-layer MLP (log scale)", "$1536$; 0 if not metric")]
FUSE = Formula([(r"F_{E,i}=\mathrm{LN}(\mathrm{LN}(F_{I,i})", "img"), (r"+\sum_k a_{k,i}\,\mathrm{LN}(F_{k,i}))", "sum")],
               size=40)
STRIP = TokenStrip(1369, show=fold(1369, head=6, tail=2), rows=3, center=(500, 960), size=30, row_pitch=52)
OPEN_PLAN = {2: 48.4, 4: 48.6}                   # P6: the depth and scale branches are disconnected (illustration, the same set as the example in the pop-up), reconnected at 52 s

PB = (60, 795, 960, 450)                         # P1-P6 sit below the encoder diagram (the encoders reach down to y = 784)
P1 = Popup(PB, 8.1, 15.4, "Image encoder: DINOv2 ViT-G", corner="tl", accent=CYAN)
P2 = Popup(PB, 15.8, 23.4, "ray encoder: pixel unshuffle + shallow conv", corner="tl", accent=GOLD)
P3 = Popup(PB, 23.8, 31.2, "Depth: first split into “normalized depth + scale”", corner="tl", accent=GOLD)
P4 = Popup(PB, 31.6, 39.0, "Pose, scale: one 4-layer MLP per quantity", corner="tl", accent=ORANGE)
P5 = Popup(PB, 39.4, 46.8, "LN (LayerNorm): normalize each branch", corner="tr", accent=WHITE, tag="Toy values")
P6 = Popup(PB, 47.2, 52.0, "Additive fusion: add if given, skip if not", corner="tr", accent=WHITE)
P7 = Popup((60, 320, 960, 560), 60.1, 67.8, "scale token: ends up as $m$", corner="tl", accent=GREEN)
POPS = [P1, P2, P3, P4, P5, P6, P7]
ROW_POP = {0: [P1], 1: [P2], 2: [P3], 3: [P4], 4: [P3, P4]}          # which pop-up explains which row

# ---- real data used by the pop-ups ----
_uu, _vv = np.meshgrid(np.arange(sc.W, dtype=float), np.arange(sc.H, dtype=float))
_DIR = np.stack([(_uu - sc.CX) / sc.FX, (_vv - sc.CY) / sc.FY, np.ones_like(_uu)], -1)
RAYMAP = _DIR / np.linalg.norm(_DIR, axis=-1, keepdims=True)              # 3x518x518 unit rays (camera frame)
_RGB = np.stack([128 + 220 * RAYMAP[..., 0], 128 + 220 * RAYMAP[..., 1], 60 + 180 * (RAYMAP[..., 2] - 0.8) / 0.2], -1)
RAY_IMG = np.clip(_RGB, 0, 255).astype(np.uint8)
_, _ZMAP, _ = sc.full(CAM)
RD2 = _ZMAP * np.linalg.norm(_DIR, axis=-1)                               # camera-2 depth along the ray (ground truth, meters; background inf)
RD_OK = np.isfinite(RD2)
ZD2 = float(RD2[RD_OK].mean())                                           # mean depth z_d-hat (valid pixels)
DR = NB["ray_depth"]
UV = NB["uv"]
PR, PC = NB["patch_rc"]
IMG_P1 = ImagePanel(ec.view_rgb(0), (40, 112), w=190)
RAY_P = ImagePanel(RAY_IMG, (40, 112), w=170)
DEP_P = ImagePanel(colorize(RD2, 1.3, 4.1), (40, 112), w=165)
NRM_P = ImagePanel(colorize(RD2 / ZD2, 1.3 / ZD2, 4.1 / ZD2), (330, 134), w=102)   # shrunk and placed low, label above the image
LN_A = np.array([8.0, 2.0, -4.0, 10.0])                                  # illustration: 4 numbers of an image token
LN_B = np.array([0.2, -0.1, 0.4, 0.3])                                   # illustration: 4 numbers of a ray token


def ln(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()                                      # gamma = 1, beta = 0


LN_A2, LN_B2 = ln(LN_A), ln(LN_B)
LN_SUM = LN_A2 + LN_B2
F_LN = Formula([r"\mathrm{LN}(x)=", r"\gamma\odot\frac{x-\mu}{\sigma}", r"+\beta"], size=40, gap=0.1)
MLP_Q = [4, 128, 256, 512, 1536]                                         # MA weight file: the MLP for the quaternion
NRM_LOG = float(np.log1p(DR / ZD2))                                      # code apply_log_to_norm: log(1+x) before the depth encoder
# The code sends 1 extra CLS per view into the global layers (use_register_tokens_from_encoder: True; dinov2.py L200-205)
MA_YAML = Path(__file__).resolve().parents[3] / "third_party" / "map-anything" / "configs" / "model" / "mapanything.yaml"


@shot.check
def check(ck):
    ck.eq("ray encoding channels: 3×14×14", 3 * 14 * 14, 588)
    ck.eq("depth encoding channels: 1×14×14", 1 * 14 * 14, 196)
    ck.eq("518 / 14 = 37", 518 // 14 if 518 % 14 == 0 else None, 37)
    ck.eq("tokens per view 37×37", 37 * 37, 1369)
    ck.eq("global sequence length 3×1369+1 (paper)", 3 * 1369 + 1, 4108)
    if MA_YAML.exists():
        ck.true("code: released config has use_register_tokens_from_encoder: True (1 extra CLS per view)",
                "use_register_tokens_from_encoder: True" in MA_YAML.read_text(encoding="utf-8"), str(MA_YAML))
    else:
        ck.record(skipped_official_code="clone facebookresearch/map-anything into third_party/map-anything to run this check")
    ck.eq("global sequence length (code) 3×(1369+1)+1", 3 * (1369 + 1) + 1, 4111)
    ck.eq("on-screen log(1+normalized depth): white dot 0.981 → 0.683", (f"{DR / ZD2:.3f}", f"{NRM_LOG:.3f}"), ("0.981", "0.683"))
    ck.close("probability of no geometric input 0.1 + 0.9×0.5³", 0.1 + 0.9 * 0.5 ** 3, 0.2125, 1e-12)
    ck.close("every pixel of the ray map is a unit vector", np.linalg.norm(RAYMAP, axis=-1), np.ones(RAYMAP.shape[:2]), 1e-12)
    ck.eq("ray map shape 3×518×518", (3,) + RAYMAP.shape[:2], (3, 518, 518))
    ck.close("camera-2 depth map at the white dot = 1.984 m (along the ray)", RD2[int(round(UV[1])), int(round(UV[0]))], DR, 5e-3)
    ck.eq("on-screen ẑ_d = 2.023 m, white dot normalized 0.981, log = 0.705",
          (f"{ZD2:.3f}", f"{DR / ZD2:.3f}", f"{np.log(ZD2):.3f}"), ("2.023", "0.981", "0.705"))
    ck.true("depth colorbar range covers all valid pixels", RD2[RD_OK].min() >= 1.3 and RD2[RD_OK].max() <= 4.1,
            f"{RD2[RD_OK].min():.3f}–{RD2[RD_OK].max():.3f}")
    ck.close("LN example: image token", LN_A2, [0.730, -0.365, -1.461, 1.095], 6e-4)
    ck.close("LN example: ray token", LN_B2, [0.0, -1.604, 1.069, 0.535], 6e-4)
    ck.close("after LN mean 0, std 1", [LN_A2.mean(), LN_A2.std(), LN_B2.mean(), LN_B2.std()], [0, 1, 0, 1], 1e-12)
    ck.eq("on-screen μ, σ", (f"{LN_A.mean():g}", f"{LN_A.std():.3f}", f"{LN_B.mean():.1f}", f"{LN_B.std():.3f}"),
          ("4", "5.477", "0.2", "0.187"))
    ck.close("sum after LN", LN_SUM, [0.730, -1.969, -0.392, 1.630], 6e-4)
    ck.close("on-screen \"about ±1.6 after LN\": max absolute value of the two branches after LN is 1.604", np.abs(np.r_[LN_A2, LN_B2]).max(), 1.604, 6e-4)
    ck.eq("on-screen ln 2.153 = 0.767 (reuses the illustrative assumption of shot 08, m = s)", (f"{S:.3f}", f"{np.log(S):.3f}"), ("2.153", "0.767"))
    ck.eq("16 layers = 8 G + 8 F, starting with G", "".join("G" if j % 2 == 0 else "F" for j in range(16)), "GF" * 8)
    popup_check(ck, POPS, DUR)
    ck.true("original timeline reconnects after the insert (end = original 77.2 s); the insert falls before P7 retracts", abs(tm(DUR) - 77.2) < 1e-9
            and abs(T_INS - P7.t_out) < 1e-9)
    c = MA["configs"]
    ck.eq("on-screen scale errors (72.3 / 71.9 / 9.5 %)",
          [f"{100 * abs(c[k]['s_pred_m'] / MA['s_gt_m'] - 1):.1f}" for k in ("A", "B", "C")], ["72.3", "71.9", "9.5"])
    ck.record(scale_err={k: c[k]["metric_scale_rel_err"] for k in c}, shape_err={k: c[k]["shape_err_normalized"]
                                                                                 for k in c},
              zd2=round(ZD2, 4), star_norm_depth=round(DR / ZD2, 4), log_zd2=round(float(np.log(ZD2)), 4),
              ln_a=LN_A2.round(4), ln_b=LN_B2.round(4), ln_sum=LN_SUM.round(4), depth_valid_px=int(RD_OK.sum()))


def T(sub, lt, t0, xy, s, size=21, fill=style.TEXT, anchor="lm", dur=0.4):
    a = ss(lt, t0, t0 + dur)
    if a > 0.01:
        sub.rich(s, xy, size, alpha=a, fill=fill, anchor=anchor)
    return a


def mini_grid(sub, box, color, alpha, n=6, fill=0.25, progress=1.0):
    """Icon of the 37×37 feature grid (draws only an n×n set of lines); progress fills rows with color."""
    x0, y0, x1, y1 = box
    if alpha <= 0.01:
        return
    rows = int(np.ceil(n * progress))
    for r in range(rows):
        sub.s.rect(x0, y0 + (y1 - y0) * r / n, x1, y0 + (y1 - y0) * (r + 1) / n, color, fill * alpha)
    for k in range(n + 1):
        x = x0 + (x1 - x0) * k / n
        y = y0 + (y1 - y0) * k / n
        sub.s.line((x, y0), (x, y1), color, 0.7 * alpha, 1)
        sub.s.line((x0, y), (x1, y), color, 0.7 * alpha, 1)


# ---------------------------------------------------------------- P1 image encoder
def body_p1(sub, lt, box):
    T(sub, lt, 0.3, (24, 80), r"Input: one image $3\times518\times518$   →   Output $F_I$: one 1536-dim vector per 14×14 patch", 22)
    a = ss(lt, 0.4, 0.8)
    IMG_P1.draw(sub, a, border=style.GREY)
    IMG_P1.grid(sub, cell=14, alpha=0.35 * a, progress=ss(lt, 2.6, 4.2), th=1)
    if lt > 3.4:
        IMG_P1.cell(sub, PR, PC, 14, GOLD, ss(lt, 3.4, 3.8), glow=0.4)
    # two lines under the image, so it does not run together with the "last 16 layers..." text on the right
    T(sub, lt, 2.8, (40, 310), "518 ÷ 14 = 37", 19, style.TEXT_DIM, "la")
    T(sub, lt, 3.0, (40, 334), "37×37 = 1369 patches", 19, style.TEXT_DIM, "la")
    # ViT-G: 40 layers, the first 24 lit
    a_s = ss(lt, 0.8, 1.2)
    if a_s > 0.01:
        x0 = 300
        cur = int(24 * ((lt - 5.2) / 1.6)) if 5.2 < lt < 6.8 else -1          # sentence 3: the feature climbs layer by layer
        for j in range(40):
            x = x0 + j * 5.5
            on = j < 24
            sub.s.rect(x, 150, x + 4, 290, CYAN if on else GREY, a_s * (0.85 if on else 0.35))
            if j == cur:
                sub.g.rect(x - 1, 150, x + 5, 290, CYAN, 0.8 * a_s)
        sub.text((x0 + 110, 142), "DINOv2 ViT-G: 40 layers", 19, alpha=a_s, fill=style.TEXT, anchor="mb")
        sub.text((x0 + 24 * 5.5 - 4, 298), "First 24 layers: image encoder", 17, alpha=a_s, fill=style.LABEL_CYAN, anchor="ra")
        # on the line below the "...1369 patches" text on the left, joined to the last 16 bars with a bracket line + vertical line
        xb0, xb1 = x0 + 24 * 5.5, x0 + 39 * 5.5 + 4
        sub.s.line((xb0, 296), (xb1, 296), GREY, 0.8 * a_s, 1)
        sub.s.line(((xb0 + xb1) / 2, 296), ((xb0 + xb1) / 2, 350), GREY, 0.8 * a_s, 1)
        sub.text((xb0, 354), "Last 16 layers: initialize the later transformer", 17, alpha=a_s, fill=style.TEXT_DIM, anchor="la")
        arrow(sub, (238, 205), (292, 205), WHITE, a_s, th=2, head=10)
    a_b = ss(lt, 5.1, 5.6)
    if a_b > 0.01:
        p = arrow(sub, (528, 220), (604, 220), WHITE, a_b, th=2, head=10)
        pulse_dot(sub, p, ss(lt, 6.6, 7.0), CYAN, a_b)
        tensor_block(sub, (676, 170), 110, 110, 46, CYAN, a_b, labels={"h": "37", "w": "37", "d": "1536"},
                     label_size=19)
        sub.rich("$F_I$: layer-24 patch features", (740, 142), 18, alpha=a_b, fill=style.LABEL_CYAN,                 anchor="mb")
    T(sub, lt, 5.6, (24, 400), "Why: vs. CroCo v2, DUSt3R encoder, RADIO and random init, DINOv2 was best in results, convergence, generalization",
      17, style.TEXT_DIM)
    T(sub, lt, 5.9, (24, 424), "ViT-G has 40 layers (DINOv2 Tab. 17): the first 24 are the image encoder, the last 16 initialize the multi-view transformer",
      17, style.TEXT_DIM)


# ---------------------------------------------------------------- P2 ray encoder
def body_p2(sub, lt, box):
    T(sub, lt, 0.3, (24, 80), r"Input: ray map $R_i$, $3\times518\times518$ (one unit direction per pixel) → Output $F_R$: $1536\times37\times37$",
      21)
    a = ss(lt, 0.4, 0.8)
    RAY_P.draw(sub, a, border=style.GREY)
    T(sub, lt, 0.6, (24, 292), "Color = direction x / y / z (illustrative colors)", 16, style.TEXT_DIM, "la")
    cb = RAY_P.cell(sub, PR, PC, 14, GOLD, ss(lt, 1.0, 1.4), glow=0.5) if lt > 1.0 else None
    # zoom into this patch: 3 channels of 14×14 each; in sentence 2 they line up into a column of 588 numbers
    a_z = ss(lt, 1.2, 1.6)
    if a_z > 0.01:
        u = ss(lt, 2.8, 4.0)
        chans = [(255, 90, 90), (90, 230, 120), (110, 150, 255)]
        for c in reversed(range(3)):
            src = np.array([238 + 10 * c, 128 - 10 * c, 112, 112], float)
            dst = np.array([410, 116 + 64 * c, 22, 64], float)
            x, y, w, h = src + (dst - src) * u
            sub.s.rect(x, y, x + w, y + h, chans[c], (0.30 + 0.25 * u) * a_z)
            sub.s.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], chans[c], 0.9 * a_z, 1, closed=True)
            if u < 0.05 and c == 0:
                for k in range(1, 14):
                    sub.s.line((x + w * k / 14, y), (x + w * k / 14, y + h), chans[c], 0.35 * a_z, 1)
                    sub.s.line((x, y + h * k / 14), (x + w, y + h * k / 14), chans[c], 0.35 * a_z, 1)
        if cb is not None and u < 0.5:
            bx0, by0, bx1, by1 = cb
            sub.s.dashed((bx1, by0), (238, 128), GOLD, a_z * (1 - 2 * u), 1, 10)
            sub.s.dashed((bx1, by1), (238, 240), GOLD, a_z * (1 - 2 * u), 1, 10)
        T(sub, lt, 1.6, (300, 252), r"This patch: $3\times14\times14$", 17, style.LABEL_GOLD, "ma") if u < 0.3 else None
        a_l = ss(lt, 3.8, 4.2)
        if a_l > 0.01:
            sub.rich(r"$588=3\times14\times14$", (442, 150), 19, alpha=a_l, fill=style.LABEL_GOLD, anchor="lm")
            sub.text((442, 178), "Laid out as this cell's channels", 17, alpha=a_l, fill=style.TEXT_DIM, anchor="lm")
            mini_grid(sub, (446, 200, 536, 290), GOLD, a_l, n=6, fill=0.12)
            sub.s.rect(446 + 15 * 3, 200 + 15 * 4, 446 + 15 * 4, 200 + 15 * 5, GOLD, 0.8 * a_l)
            sub.rich(r"$588\times37\times37$", (448, 300), 18, alpha=a_l, fill=style.LABEL_GOLD, anchor="la")  # clear of the blue bars
    # sentence 3: shallow conv (layers from the MA weight file)
    steps = [r"$3\times3$ conv 588→588", "Residual block →768 (GELU)", "Residual block →1024 (GELU)", r"$1\times1$ conv →1536, LN",
             r"$F_R$: $1536\times37\times37$"]
    a_c = ss(lt, 5.1, 5.5)
    if a_c > 0.01:
        sub.text((790, 100), "Shallow conv (layers from the MA weight file)", 17, alpha=a_c, fill=style.TEXT_DIM, anchor="ma")
        for k, s in enumerate(steps):
            y = 126 + 42 * k
            ak = a_c * ss(lt, 5.2 + 0.25 * k, 5.5 + 0.25 * k)
            col = GOLD if k < 4 else CYAN
            module_box(sub, (640, y, 940, y + 32), s, None, col, ak, title_size=18, r=8)
            if k:
                arrow(sub, (790, y - 10), (790, y - 1), WHITE, ak, th=2, head=6)
        arrow(sub, (542, 245), (636, 245), WHITE, a_c, th=2, head=10)
    T(sub, lt, 5.8, (24, 400), "Why: same grid (37×37) and width (1536) as DINOv2 features, so they add patch by patch (paper §3.1)", 18,
      style.TEXT_DIM)


# ---------------------------------------------------------------- P3 depth split first
def body_p3(sub, lt, box):
    T(sub, lt, 0.3, (24, 80), r"Input: depth map $\hat D_i$ (along the ray, $1\times518\times518$); example = camera 2's ground-truth depth (m)", 21)
    a = ss(lt, 0.4, 0.8)
    if a > 0.01:
        DEP_P.draw(sub, a, border=style.GREY)
        x0, y0, x1, y1 = DEP_P.box
        colorbar(sub, (x1 + 6, y0, x1 + 16, y1), 1.3, 4.1, unit="m", fmt="{:.1f}", alpha=a, size=15)
        DEP_P.mark(sub, UV[0], UV[1], WHITE, 4, a, label=f"{DR:.3f} m", label_offset=(8, -10), size=17)
    a1 = ss(lt, 0.9, 1.5)
    if a1 > 0.01:                                    # split into two parts
        arrow(sub, (262, 170), (324, 182), WHITE, a1, th=2, head=9)
        arrow(sub, (262, 240), (322, 290), WHITE, a1, th=2, head=9)
        NRM_P.draw(sub, a1, border=style.GREY)
        NRM_P.mark(sub, UV[0], UV[1], WHITE, 3, a1, label=f"{DR / ZD2:.3f}", label_offset=(6, -8), size=16)
        # the label sits above the image so it is not read as a caption of the "mean depth" box
        sub.rich(r"$\hat D_2/\hat z_{d,2}$: normalized depth (mean = 1)", (330, 128), 17, alpha=a1, fill=style.LABEL_GOLD,
                 anchor="lb")
        module_box(sub, (330, 274, 600, 310), r"Mean depth $\hat z_{d,2}$ = " + f"{ZD2:.3f} m", None, GREEN, a1,
                   title_size=18, r=8)
    a2 = ss(lt, 2.5, 2.9)
    if a2 > 0.01:                                    # take the log of the scale; encode only if metric
        arrow(sub, (602, 292), (650, 292), WHITE, a2, th=2, head=9)
        sub.rich(r"$\log\hat z_{d,2}$ = " + f"{np.log(ZD2):.3f}", (656, 292), 19, alpha=a2, fill=style.LABEL_GOLD,
                 anchor="lm")
        sub.text((656, 318), "→ MLP in the next window", 16, alpha=a2, fill=style.TEXT_DIM, anchor="lm")
        flick = 1.0 if lt < 3.4 else float(np.floor((lt - 3.4) / 0.6) % 2 == 0)
        on_m = flick if lt < 4.9 else 1.0
        sub.badge((330, 344), "Metric: encode log" if on_m > 0.5 else "Not metric: set to 0", 16, a2, anchor="la",
                  fill=GREEN if on_m > 0.5 else style.TEXT_DIM, border=GREEN if on_m > 0.5 else GREY)
        sub.text((560, 357), "Scale varies widely → log (paper)", 16, alpha=a2, fill=style.TEXT_DIM, anchor="lm")
    a3 = ss(lt, 5.0, 5.4)
    if a3 > 0.01:                                    # normalized depth goes through a conv (same structure, a different set of weights)
        arrow(sub, (438, 158), (488, 158), WHITE, a3, th=2, head=8)
        # The code applies log(1+x) first, before the encoder (model.py L1011-1013 apply_log_to_norm, geometry.py L1666-1679)
        sub.rich(r"Take $\log(1+x)$ first (code): the white dot " + f"{DR / ZD2:.3f} → {NRM_LOG:.3f}", (494, 158), 18, alpha=a3,
                 fill=style.TEXT, anchor="lm")
        sub.rich(r"pixel unshuffle 14: $1\times518\times518\to196\times37\times37$", (494, 186), 18,
                 alpha=a3 * ss(lt, 5.2, 5.6), fill=style.TEXT, anchor="lm")
        sub.rich(r"($196=14\times14$) → conv (first layer 196→588)", (494, 214), 18, alpha=a3 * ss(lt, 5.5, 5.9),
                 fill=style.TEXT, anchor="lm")
        sub.rich(r"→ $F_D$: $1536\times37\times37$ (same grid as ray)", (494, 242), 18, alpha=a3 * ss(lt, 5.8, 6.2),
                 fill=style.LABEL_CYAN, anchor="lm")
    T(sub, lt, 5.9, (24, 404), "Why split: accepts metric or up-to-scale inputs; depth and pose may come apart, so each is normalized alone (§3.1)", 17,
      style.TEXT_DIM)


# ---------------------------------------------------------------- P4 global quantities MLP
ROWS4 = [("$Q_i$: unit quaternion (4)", ORANGE, r"$F_Q$"), (r"$\hat T_i/\hat z_p$: norm. translation (3)", ORANGE, r"$F_T$"),
         (r"$\log\hat z_{d,i}$: depth scale (1)", GREEN, r"$F_{\hat z_d}$"),
         (r"$\log\hat z_p$: pose scale (1)", GREEN, r"$F_{\hat z_p}$")]


def body_p4(sub, lt, box):
    T(sub, lt, 0.3, (24, 80), r"Split $\hat T_i$ first: pose scale $\hat z_p$ = mean $\|\hat T_i\|$ over views that give translation; then $\hat T_i/\hat z_p$", 19)
    T(sub, lt, 0.9, (24, 108), r"$\hat T_i/\hat z_p$ is the translation divided by the mean length, not a unit vector (the paper calls it translation directions)", 17,
      style.TEXT_DIM)
    for r, (lab, col, fo) in enumerate(ROWS4):
        y = 170 + 42 * r
        ar = ss(lt, 1.2 + 0.25 * r, 1.6 + 0.25 * r)
        if ar <= 0.01:
            continue
        sub.rich(lab, (24, y), 19, alpha=ar, fill=col, anchor="lm")
        # each quantity has its own 4-layer MLP: 4 bars stand for the output widths of the 4 layers (quaternion: 128 / 256 / 512 / 1536)
        g = ss(lt, 2.6 + 0.3 * r, 3.4 + 0.3 * r)
        for j, dim in enumerate(MLP_Q[1:]):
            hgt = (8 + 26 * np.log(dim / 64) / np.log(1536 / 64)) * g
            x = 372 + 30 * j
            sub.s.rect(x, y - hgt / 2, x + 16, y + hgt / 2, col, 0.75 * ar)
        if g > 0.01:
            arrow(sub, (342, y), (366, y), WHITE, ar * g, th=2, head=7)
            p = arrow(sub, (496, y), (530, y), WHITE, ar * g, th=2, head=7)
            sub.s.rect(536, y - 5, 536 + 150 * ss(lt, 3.2 + 0.3 * r, 3.8 + 0.3 * r), y + 5, col, 0.85 * ar)
            sub.rich(fo + ": 1536", (696, y), 18, alpha=ar * ss(lt, 3.6 + 0.3 * r, 4.0 + 0.3 * r), fill=col,
                     anchor="lm")
    T(sub, lt, 2.6, (372, 128), "One 4-layer MLP each (GeLU)", 17, style.TEXT_DIM, "la")
    # sentence 3: broadcast to every patch of the 37×37 grid
    a3 = ss(lt, 5.0, 5.4)
    if a3 > 0.01:
        arrow(sub, (788, 170), (814, 196), ORANGE, a3, th=2, head=8)
        mini_grid(sub, (818, 196, 914, 292), ORANGE, a3, n=6, fill=0.35, progress=ss(lt, 5.3, 6.3))
        sub.text((866, 298), "Copied to all 37×37", 15, alpha=a3, fill=style.LABEL_ORANGE, anchor="ma")
        sub.text((866, 318), "(code: broadcast, then add)", 14, alpha=a3, fill=style.TEXT_DIM, anchor="ma")
    T(sub, lt, 2.9, (24, 352), "Both scales only when the input is metric, else 0 (paper + code); quaternion MLP: 4→128→256→512→1536 (MA weight file)", 16,
      style.TEXT_DIM)
    T(sub, lt, 5.6, (24, 384), "Why encode rotation and translation separately: it still works with only IMU (rotation) or only GPS (translation) (§3.1)", 17,
      style.TEXT_DIM)


# ---------------------------------------------------------------- P5 LN
P5_PITCH, P5_W = 74, 30                                  # spacing 74 and readout size 18 keep neighbouring numbers ("−0.365", "−1.461") apart
P5_X = (36, 352, 684)                                    # left ends of the three bar groups: image token, ray token, sum
P5_MID = [x + (3 * P5_PITCH + P5_W) / 2 for x in P5_X]


def bars(sub, x0, vals, scale, col, alpha, y0=232, pitch=P5_PITCH, w=P5_W, label_col=None):
    """4 bars (positive ones point up) + a readout under each (readout.number, so changing digits do not jitter)."""
    for j, v in enumerate(vals):
        x = x0 + pitch * j
        h = v * scale
        sub.s.rect(x, min(y0, y0 - h), x + w, max(y0, y0 - h) + 0.5, col, 0.8 * alpha)
        number(sub, (x + w / 2, 322), float(v), "{:.3f}", size=18, color=label_col or col, alpha=alpha)
    sub.s.line((x0 - 8, y0), (x0 + pitch * (len(vals) - 1) + w + 8, y0), GREY, 0.7 * alpha, 1)


def body_p5(sub, lt, box):
    a = ss(lt, 0.3, 0.7)
    F_LN.draw(sub, (210, 96), alpha=a, part_glow={1: bump(lt, 2.8, 1.0)}, shadow=False)
    T(sub, lt, 0.6, (430, 84), r"$\mu$, $\sigma$: mean and std of this token's 1536 numbers", 18, style.TEXT_DIM)
    T(sub, lt, 0.9, (430, 110), r"$\gamma$, $\beta$: learnable, one per dim (1 and 0 in the example)", 18, style.TEXT_DIM)
    u = ss(lt, 3.0, 4.2)                                      # sentence 2: before LN → after LN
    sc_ = lerp(6.0, 38.0, u)
    va, vb = lerp(LN_A, LN_A2, u), lerp(LN_B, LN_B2, u)
    if a > 0.01:
        sub.text((P5_MID[0], 144), "Image token (4 shown)", 17, alpha=a, fill=style.LABEL_CYAN, anchor="ma")
        sub.text((P5_MID[1], 144), "ray token (4 shown)", 17, alpha=a, fill=style.LABEL_GOLD, anchor="ma")
        bars(sub, P5_X[0], va, sc_, CYAN, a, label_col=style.LABEL_CYAN)
        bars(sub, P5_X[1], vb, sc_, GOLD, a, label_col=style.LABEL_GOLD)
        a_st = fade(lt, 2.5, 4.4, 0.3, 0.3)
        if a_st > 0.01:
            sub.rich(r"$\mu=4$, $\sigma=" + f"{LN_A.std():.3f}" + "$", (P5_MID[0], 348), 17, alpha=a_st,
                     fill=style.TEXT, anchor="ma")
            sub.rich(r"$\mu=0.2$, $\sigma=" + f"{LN_B.std():.3f}" + "$", (P5_MID[1], 348), 17, alpha=a_st,
                     fill=style.TEXT, anchor="ma")
        tag = "After LN" if u > 0.5 else "Before LN"
        sub.badge(((P5_MID[0] + P5_MID[1]) / 2, 144), tag, 16, a, anchor="mm", fill=style.TEXT,
                  border=WHITE if u > 0.5 else GREY)            # no room between the two bar groups, so it sits between the two group titles
    a3 = ss(lt, 5.0, 5.4)                                     # sentence 3: add, then LN again
    if a3 > 0.01:
        sub.text((P5_MID[2], 144), "Sum (then one more LN)", 17, alpha=a3, fill=style.TEXT, anchor="ma")
        bars(sub, P5_X[2], LN_SUM * ss(lt, 5.2, 5.9), 38.0, WHITE, a3, label_col=style.TEXT)
        sub.text(((P5_X[1] + 3 * P5_PITCH + P5_W + P5_X[2]) / 2, 232), "+", 30, "light", a3, style.TEXT, "mm")
    T(sub, lt, 5.6, (24, 404), "Before LN: 20×+ apart (10 vs 0.4); after LN: both within ~±1.6, so neither dominates the sum (our reading)", 18,
      style.TEXT_DIM)                                         # the max absolute value is 1.604, so "~" rather than "within"


# ---------------------------------------------------------------- P6 additive fusion
GRIDS6 = [(r"$\mathrm{LN}(F_I)$", CYAN, 1), (r"$\mathrm{LN}(F_R)$", GOLD, 1), (r"$\mathrm{LN}(F_D)$", GOLD, 0),
          (r"$\mathrm{LN}(F_Q),\mathrm{LN}(F_T)$", ORANGE, 1), (r"$\mathrm{LN}(F_{\hat z})$", GREEN, 0)]


def body_p6(sub, lt, box):
    T(sub, lt, 0.2, (24, 80), "$F_{I,i}$: image feature; $F_{k,i}$: feature of the $k$-th geometric input (ray, depth, pose, scale)", 19)
    T(sub, lt, 0.5, (24, 108), r"$a_{k,i}\in\{0,1\}$: did view $i$ give input $k$; if not, ×0 (code). Example: this view has no depth and isn't metric", 18,
      style.TEXT_DIM)
    u = ss(lt, 2.7, 3.7)                                      # sentence 2: the five maps merge into one
    for k, (lab, col, a_k) in enumerate(GRIDS6):
        ak = ss(lt, 0.4 + 0.2 * k, 0.8 + 0.2 * k)
        if ak <= 0.01:
            continue
        x0 = 40 + 150 * k
        gx = lerp(x0, 420, u)
        gy = 150
        on = 1.0 if a_k else (1.0 - ss(lt, 1.2, 1.6))
        mini_grid(sub, (gx, gy, gx + 84, gy + 84), col, ak * (0.35 + 0.65 * on) * (1 - 0.6 * u), n=6,
                  fill=0.3 if k >= 3 else 0.15)
        if u < 0.3:
            sub.rich(lab, (x0 + 42, 246), 16, alpha=ak * (1 - u / 0.3), fill=col, anchor="ma")
            note = "Always given" if k == 0 else ("Not metric: set to 0" if k == 4 else f"$a={a_k}$")
            sub.rich(note, (x0 + 42, 272), 16, alpha=ak * (1 - u / 0.3), fill=RED if not a_k else style.TEXT_DIM,
                     anchor="ma")
            if k:
                sub.text((x0 - 24, 192), "+", 26, "light", ak * (1 - u / 0.3), style.TEXT, "mm")
            if not a_k:
                sub.text((x0 + 42, 192), "× 0", 24, "light", ak * ss(lt, 1.2, 1.6) * (1 - u / 0.3), RED, "mm")
    a2 = ss(lt, 3.5, 3.9)
    if a2 > 0.01:
        mini_grid(sub, (420, 150, 504, 234), WHITE, a2, n=6, fill=0.25)
        sub.rich(r"$F_{E,i}$: $1536\times37\times37$", (462, 246), 18, alpha=a2, fill=style.TEXT, anchor="ma")
        sub.text((462, 272), "→ 1369 tokens", 17, alpha=a2, fill=style.TEXT_DIM, anchor="ma")
        sub.rich(r"However many inputs are given, the sum is $1536\times37\times37$", (480, 330), 20, alpha=a2, fill=style.TEXT,
                 anchor="mm")
    T(sub, lt, 3.9, (480, 362), "→ the later transformer and heads need no change (our reading)", 18, style.TEXT_DIM, "mm")


# ---------------------------------------------------------------- P7 scale token
def body_p7(sub, lt, box):
    T(sub, lt, 0.2, (24, 82), "scale token: one learnable 1536-dim token, appended after the tokens of the $N$ views (paper §3.1)", 20)
    a = ss(lt, 0.3, 0.7)
    jf = 16 * ss(lt, 0.6, 5.2)                                # the cursor walks layer by layer
    jc = min(int(jf), 15)
    if a > 0.01:
        for j in range(16):
            x = 40 + 54 * j
            g = j % 2 == 0
            col = CYAN if g else ORANGE
            cur = (j == jc and lt < 5.4)
            sub.s.rect(x, 112, x + 46, 236, col, (0.30 if cur else 0.14) * a, r=6)
            sub.s.polyline([(x, 112), (x + 46, 112), (x + 46, 236), (x, 236)], col, (1.0 if cur else 0.45) * a, 1,
                           closed=True)
            sub.text((x + 23, 124), "G" if g else "F", 18, "light", a, style.TEXT, "mm")
            for r in range(3):
                sub.s.rect(x + 8, 146 + 18 * r, x + 38, 156 + 18 * r, ec.CAM_COLORS[r], 0.7 * a)
            st = (x + 23, 214)
            if g:
                token(sub, st, 16, style.TOKEN["scale"], a * (1.0 if j <= jf else 0.4), glow=0.5 * cur)
                if cur:
                    for r in range(3):
                        sub.s.line(st, (x + 23, 156 + 18 * r), GREEN, 0.9 * a, 1)
            else:
                sub.s.polyline([(st[0] - 8, st[1] - 8), (st[0] + 8, st[1] - 8), (st[0] + 8, st[1] + 8),
                                (st[0] - 8, st[1] + 8)], GREY, 0.6 * a, 1, closed=True)
        # 4108 is the paper algorithm; the code adds 1 CLS per view (use_register_tokens_from_encoder), 4111 in total
        sub.rich("G = global layer: tokens of 3 views + scale token attend together (paper: 4108; the code adds 1 CLS per view, 4111 in total)",
                 (24, 260), 17,
                 alpha=a * ss(lt, 0.8, 1.2), fill=style.LABEL_CYAN, anchor="lm")
        sub.rich("F = frame layer: each view on its own, scale token excluded (gray box); 8 G + 8 F in total, starting with G (code)", (24, 286), 17,
                 alpha=a * ss(lt, 1.4, 1.8), fill=style.LABEL_ORANGE, anchor="lm")
        T(sub, lt, 2.6, (24, 316), "In G layers, the scale token attends with the tokens of all views, so it sees the whole scene", 18, style.TEXT)
    a3 = ss(lt, 5.1, 5.5)                                     # sentence 3: → MLP → exp → m
    if a3 > 0.01:
        y = 400
        token(sub, (52, y), 26, style.TOKEN["scale"], a3, glow=0.5 * bump(lt, 5.4, 0.8))
        arrow(sub, (70, y), (112, y), WHITE, a3, th=2, head=8)
        module_box(sub, (116, y - 30, 396, y + 30), "MLP (paper: 2 layers, ReLU)", "Code: 1536→196→196→196→1", GREEN, a3,
                   title_size=18, sub_size=15, r=8)
        a4 = ss(lt, 5.7, 6.1)
        arrow(sub, (400, y), (440, y), WHITE, a4, th=2, head=8)
        sub.text((446, y), f"{np.log(S):.3f}", 24, "light", a4, style.LABEL_GOLD, "lm")
        a5 = ss(lt, 6.2, 6.6)
        arrow(sub, (520, y), (580, y), WHITE, a5, th=2, head=8)
        sub.text((550, y - 16), "exp", 18, "light", a5, style.TEXT, "mm")
        sub.rich(r"$m$ = " + f"{S:.3f}", (588, y), 26, alpha=a5, fill=style.LABEL_GOLD, anchor="lm")
        # exp curve: x in [-1.5, 1.5], the dot stops at ln m
        X0, Y0, Wd, Hd = 730, 470, 200, 130
        xs = np.linspace(-1.5, 1.5, 60)
        pts = np.c_[X0 + (xs + 1.5) / 3 * Wd, Y0 - np.exp(xs) / np.exp(1.5) * Hd]
        sub.s.line((X0, Y0), (X0 + Wd, Y0), GREY, 0.7 * a3, 1)
        sub.s.line((X0 + Wd / 2, Y0), (X0 + Wd / 2, Y0 - Hd), GREY, 0.7 * a3, 1)
        sub.s.polyline(pts, GREEN, a3, 2)
        xv = lerp(-1.5, np.log(S), ss(lt, 5.8, 6.6))
        pv = (X0 + (xv + 1.5) / 3 * Wd, Y0 - np.exp(xv) / np.exp(1.5) * Hd)
        sub.s.dot(pv, GOLD, 5, a3)
        sub.g.dot(pv, GOLD, 5, 0.6 * a3)
        sub.text((X0 + Wd / 2, Y0 + 6), "0", 15, "light", a3, style.TEXT_DIM, "ma")
        sub.rich("$y=e^x$: always positive", (X0 + Wd / 2, Y0 + 26), 15, alpha=a3, fill=style.TEXT_DIM, anchor="ma")
        T(sub, lt, 6.4, (24, 462), r"Illustration: reusing $m$ = " + f"{S:.3f}" + r" from the last shot, so before exp it is $\ln 2.153$ = " + f"{np.log(S):.3f}",
          17, style.TEXT_DIM)
        T(sub, lt, 6.6, (24, 486), "Why exp: metric scale varies a lot across scenes, so the network outputs its log (paper §3.2)", 17, style.TEXT_DIM)
        # The definition sentence for m is the same as in 08 (e01_common.M_DEF); it is held for INS seconds before P7 retracts so it stays lit
        a_d = ss(lt, 6.8, 7.2)
        if a_d > 0.01:
            g = bump(REAL[0], T_INS + 0.2, 1.6)
            if g > 0.01:
                sub.s.rect(16, 498, 520, 546, GOLD, 0.12 * g * a_d, r=6)
            sub.rich(ec.M_DEF_1, (24, 511), 20, alpha=a_d, fill=style.LABEL_GOLD, anchor="lm")
            sub.rich(ec.M_DEF_2, (24, 535), 20, alpha=a_d, anchor="lm")


BODY = {id(P1): body_p1, id(P2): body_p2, id(P3): body_p3, id(P4): body_p4, id(P5): body_p5, id(P6): body_p6,
        id(P7): body_p7}


@shot.frame
def frame(cv, tr):
    REAL[0] = tr
    t = tm(tr)                                            # original timeline (held for INS seconds before P7 retracts)
    a_main = ss(t, 0.3, 0.8) * (1 - ss(t, 73.0, 73.4))
    if a_main > 0:
        a_dg = a_main * (1 - P7.on(t))                       # when P7 covers the encoder diagram, hide the whole diagram (no edges peek out)
        cv.text((186, 340), "Inputs (per view)", 24, alpha=a_dg, fill=style.TEXT_DIM, anchor="mb")
        cv.text((530, 340), "Encoders", 24, alpha=a_dg * ss(t, 5.4, 5.8), fill=style.TEXT_DIM, anchor="mb")
        flick = flicker(t, 4, p=0.5, period=0.55, seed=3, t0=67.9) if 67.9 < t < 73.2 else None
        any_pop = max(p.on(t) for p in (P1, P2, P3, P4))
        for k, (lab, opt, y, col) in enumerate(SLOTS):
            a_s = a_dg * (1.0 if k == 0 else ss(t, 2.9 + 0.4 * k, 3.3 + 0.4 * k))
            focus = max(p.on(t) for p in ROW_POP[k])
            dimk = 1.0 - 0.5 * max(0.0, any_pop - focus)          # dim this row while another row is being explained
            on = 1.0
            if k in OPEN_PLAN:                                   # consistent with the example in P6: this view gave no depth and is not metric
                on = 1.0 - ss(t, OPEN_PLAN[k], OPEN_PLAN[k] + 0.4) * (1 - ss(t, 51.6, 52.0))
            if flick is not None and k > 0:
                on = float(flick[k - 1])
            # "p = 0.5" sits above the connecting line (not on it); slot right end 312, leaving it 88 px
            b = input_slot(cv, (60, y - 34, 312, y + 34), lab, optional=opt, on=on, color=col, alpha=a_s * dimk,
                           size=24)
            if k in (1, 2, 3) and 67.8 < t < 73.2:
                cv.text((356, y - 9), "p = 0.5", 18, alpha=a_s * dimk * (0.35 + 0.65 * on), fill=style.TEXT_DIM,
                        anchor="mb")
            a_e = a_dg * ss(t, 5.5 + 0.3 * k, 5.9 + 0.3 * k) * dimk
            if a_e > 0:
                lit = 0.45 + 0.55 * ss(t, min(p.t_in for p in ROW_POP[k]), min(p.t_in for p in ROW_POP[k]) + 0.4)
                eb = module_box(cv, (400, y - 34, 660, y + 34), ENC[k][0], ENC[k][1], col, a_e * lit, title_size=20,
                                sub_size=18, glow=0.6 * focus)
                link(cv, b["r"], eb["l"], on, alpha=a_e * lit)
                link(cv, eb["r"], (735, 570), on, alpha=a_e * ss(t, 39.2, 39.6))
        a_f = a_dg * ss(t, 39.2, 39.7)
        if a_f > 0:
            module_box(cv, (735, 520, 1020, 620), "LN → sum → LN", "Missing input: $a_{k,i}=0$", WHITE, a_f, title_size=24,
                       glow=0.5 * max(P5.on(t), P6.on(t)))
        FUSE.draw(cv, (540, 250), alpha=a_main * ss(t, 39.2, 39.7),
                  state={"img": "lit", "sum": "hl" if 47.2 < t < 52.2 else "lit"})
        # token rows: 1369 per view, the count does not change with the kind of input
        a_tok = a_main * ss(t, 52.3, 52.8)
        if a_tok > 0:
            STRIP.draw(cv, lambda r, i: "patch", a_tok, band={0: (255, 160, 90)} if t > 54.9 else None)
            for r in range(3):
                cv.text((STRIP.x[0] - 36, STRIP.y[r]), f"View {r + 1}", 22, alpha=a_tok, fill=ec.CAM_COLORS[r],
                        anchor="rm")
            cv.rich(r"Each view has $37\times37=1369$ tokens, $C=1536$ (the code adds 1 CLS per view)", (540, 860), 22,
                    alpha=a_tok * (1 - P7.on(t)), fill=style.TEXT_DIM, anchor="ma")   # the lower edge of P7 at 880 would cut through this line, hide it while covered
            a_ref = a_main * ss(t, 54.9, 55.4)
            cv.rich("View 1: add one fixed reference vector (code: row 0 of the sine table); VGGT uses a dedicated camera token", (540, 1060), 22,
                    alpha=a_ref * (1 - ss(t, 57.2, 57.6)), fill=(255, 190, 140), anchor="ma")
            a_sc = a_main * ss(t, 57.5, 58.0)
            if a_sc > 0:
                p = np.array([STRIP.x[max(STRIP.shown)] + 70, STRIP.y[1]])
                token(cv, p, 34, style.TOKEN["scale"], a_sc, glow=0.6 * bump(t, 57.8, 1.0))
                cv.text((p[0] + 28, p[1]), "scale token ×1", 22, alpha=a_sc, fill=GREEN, anchor="lm")
        a_p = a_main * fade(t, 67.9, 73.2, 0.4, 0.3)
        if a_p > 0:
            cv.rich("Training: geometric inputs given with prob. 0.9 overall; ray, depth, pose 0.5 each; each view 0.95", (540, 1090), 22,
                    alpha=a_p, anchor="ma")
            cv.rich(r"About 21% of samples have images only ($0.1+0.9\times0.5^3$)", (540, 1130), 22, alpha=a_p * ss(t, 70.6, 71.0),
                    fill=style.TEXT_DIM, anchor="ma")

    # last sentence: measured on our test scene
    a_r = ss(t, 73.3, 73.8)
    if a_r > 0:
        c = MA["configs"]
        # recomputed from the 4-decimal mean distance: 0.7225 in the json rounded to 1 decimal would show 72.2%, the true value is 72.26% (the on-screen 72.3% is rounded from the metrics file)
        vals = [100 * abs(c[k]["s_pred_m"] / MA["s_gt_m"] - 1) for k in ("A", "B", "C")]
        bar_chart(cv, (300, 420, 960, 760), vals, ["Images only", "+ intrinsics", "+ intrinsics + metric pose"], vmax=80, horizontal=True,
                  highlight=2, progress=ss(t, 73.5, 75.0), fmt="{:.1f}%",
                  title="Our test scene, 3 images: metric scale error\n(predicted vs GT mean scene distance 2.153 m)", better="lower", alpha=a_r)
        # group C = intrinsics + metric pose
        cv.rich("Images only: predicted mean distance %.2f m; with intrinsics + metric pose: %.2f m" % (c["A"]["s_pred_m"], c["C"]["s_pred_m"]),
                (540, 840), 24, alpha=a_r * ss(t, 74.8, 75.2), anchor="ma")
        cv.rich("With intrinsics + metric pose, shape error rises %.3f → %.3f (inputs checked; cause unknown)"
                % (c["A"]["shape_err_normalized"]["mean"], c["C"]["shape_err_normalized"]["mean"]), (540, 890), 22,
                alpha=a_r * ss(t, 75.4, 75.8), fill=style.TEXT_DIM, anchor="ma")
        cv.rich("Our test scene has no object of known size, so images alone give no scale cue (inferred)", (540, 935), 22,
                alpha=a_r * ss(t, 75.8, 76.2), fill=style.TEXT_DIM, anchor="ma")

    for p in POPS:
        p.draw(cv, t, BODY[id(p)])
