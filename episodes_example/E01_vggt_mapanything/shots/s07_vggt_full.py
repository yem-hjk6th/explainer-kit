"""E01-07 A walk through the full VGGT architecture (with shapes) + training and normalization (L, 62.3 s). Storyboard storyboard.md section E01-07.
Each time a flow-chart node lights up, the "node view" window at top right switches to that node's picture (same box, title = node name,
"Measured" / "Illustration" corner tag, consistent style); the three DPTs are written in full (depth DPT, point DPT, track-feature DPT); the small text
(per-frame small boxes, trunk notes, the 48 layers) lives in the node view, font size not below 18 px.
Shapes: paper + official code (N = 3 images). Measured: predicted cameras, depth / confidence, point map, white-dot track, mean distance of points to the origin.
Computed for real: s = 2.153 m, normalization of the white-dot depth and camera 2 translation, scale invariance.
Timeline: part A (architecture diagram + node view, 0-34.9 s) uses its own timing; parts B (loss) and C (normalization) are written on a base 60 s timeline (from 32.6 s),
shifted back by SH seconds overall (in frame, to = t - SH). All captions are in CAPS, at final times.
"""
import numpy as np

import e01_common as ec
import scene_e01 as sc
from animkit import (CYAN, GOLD, GREY, ORANGE, VIOLET, WHITE, Formula, ImagePanel, Shot, TokenStrip, arrow,
                     arrow_path, attn_blocks, bump, colorize, counter, elbow, fade, fold, module_box, number,
                     pulse_dot, rrect_pts, ss, style, tensor_block, token, vector)

NB = ec.nb()
V = ec.vggt_out()
M = ec.vggt_metrics()
S = NB["s"]
Z = NB["zdepth"]
CAM = sc.EX_CAM
T_GT = sc.cameras()[CAM][1]
TRK = M["track_white_dot"]
KP, KT, W_IDX = NB["K_patch"], NB["K_tok"], NB["tok_idx_code"]
N_ROUNDS = 24                      # 24 alternating rounds (section C)
C_TRK, HW_TRK = 128, sc.W // 2     # output of the track-feature DPT: 128 channels, 518 / 2 = 259 (official track_head.py L22, L55)

SH = 2.3                           # seconds that parts B and C are shifted back relative to the base timeline
DUR = 60.0 + SH
TA = dict(img=0.3, dino=3.3, cat=6.3, aa=9.1, rounds=11.9, cam=14.5, dep=17.5, pts=20.5, trk=23.5, tracker=26.3,
          indep=29.3, back=32.1, end=34.9)

shot = Shot("s07_vggt_full", DUR)
shot.title("VGGT, End to End", "Full forward pass · shapes · loss · normalization", 0.2, 2.4)
for _txt, _a, _b in [("Shapes: paper + official code, N = 3", 0.3, TA["end"]),
                     ("Scene ground truth (m), normalized per paper §3.4", 43.2 + SH, 51.4 + SH)]:
    shot.tag(_txt, _a, _b)
ec.scene_tag(shot, TA["cam"], TA["indep"], "Top-right “Measured” tag = VGGT-1B output")
ec.scene_tag(shot, 51.4 + SH, 54.4 + SH)
CAPS = [(0.3, 3.3, "Input: N images, N×3×518×518"),
        (3.3, 6.3, "DINOv2 encoding: N×1369×1024"),
        (6.3, 9.1, "Add camera, register tokens: N×1374"),
        (9.1, 11.9, "One frame layer plus one global layer is a round"),
        (11.9, 14.5, "24 rounds in total"),
        (14.5, 17.5, "Camera head: 9 numbers per image"),
        (17.5, 20.5, "Depth DPT: per-pixel depth + confidence"),
        (20.5, 23.5, "Point DPT: one 3D point per pixel"),
        (23.5, 26.3, "Track-feature DPT: features for tracking"),
        (26.3, 29.3, "The tracker finds the white dot in frames 2, 3"),
        (29.3, 32.1, "Each DPT sees only its own image"),
        (32.1, 34.9, "All cross-image information comes from the backbone"),
        (34.9, 37.7, "Training sums four loss terms"),
        (37.7, 39.9, "The tracking term is scaled by just 0.05"),
        (39.9, 42.7, "Depth error is first multiplied by confidence Σ"),
        (42.7, 45.5, "Then subtract α log Σ to stop Σ collapsing to 0"),
        (45.5, 48.3, "Ground truth is first divided by mean distance s"),
        (48.3, 50.9, f"Our test scene: s = {S:.3f} m"),
        (50.9, 53.7, f"Depth {Z:.3f} m → {Z / S:.3f}"),
        (53.7, 56.7, "So VGGT's output has no meters"),
        (56.7, DUR, "Scene and cameras scale together; the 3 images look identical")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

# ---------------------------------------------------------------- architecture diagram (left column + four heads at the bottom)
# (box, title, shape, color, light-up time)
BOXES = {
    "img": ((40, 310, 470, 375), "images", "$(N,3,518,518)$", WHITE, TA["img"]),
    "dino": ((40, 405, 470, 470), "DINOv2 (24-layer ViT-L)", "$(N,1369,1024)$", CYAN, TA["dino"]),
    "cat": ((40, 500, 470, 565), "+ camera / register token", "$(N,1374,1024)$", GOLD, TA["cat"]),
    "aa": ((40, 595, 470, 745), "Alternating attention", "frame $(3,1374,1024)$\nglobal $(1,4122,1024)$", ORANGE,
           TA["aa"]),
    "cam": ((30, 808, 255, 888), "camera head", "$(N,9)$", GOLD, TA["cam"]),
    "dep": ((275, 808, 520, 888), "depth DPT", "$(N,518,518)$ + Σ", CYAN, TA["dep"]),
    "pts": ((540, 808, 785, 888), "point DPT", "$(N,3,518,518)$ + Σ", CYAN, TA["pts"]),
    "trk": ((805, 808, 1050, 888), "track-feature DPT", f"$(N,{C_TRK},{HW_TRK},{HW_TRK})$", VIOLET, TA["trk"]),
    "tracker": ((805, 928, 1050, 998), "tracker (CoTracker2)", "$(M,N,2)$", VIOLET, TA["tracker"]),
}
CX = {k: (b[0][0] + b[0][2]) / 2 for k, b in BOXES.items()}
DPTS = ("dep", "pts", "trk")

# ---------------------------------------------------------------- node view at top right
PB = (590, 300, 1050, 762)                      # box of the node view (x0, y0, x1, y1)
PANELS = [  # (node, appearance time, title, corner tag, border color)
    ("img", TA["img"], "images: 3 inputs", "Our test scene", WHITE),
    ("dino", TA["dino"], "DINOv2 per image", "Illustration", CYAN),
    ("cat", TA["cat"], "+ cam / register tokens", "Illustration", GOLD),
    ("aa", TA["aa"], "Alternating attention", "Illustr.: 4/frame", ORANGE),
    ("cam", TA["cam"], "camera head → 3 cameras", "Measured", GOLD),
    ("dep", TA["dep"], "depth DPT → depth + conf.", "Measured", CYAN),
    ("pts", TA["pts"], "point DPT → point map", "Measured", CYAN),
    ("trk", TA["trk"], "track-feat. DPT → $T_i$", "Illustration", VIOLET),
    ("tracker", TA["tracker"], "Tracker: white-dot track", "Measured", VIOLET),
    ("indep", TA["indep"], "DPT: one image at a time", "Illustration", CYAN),
    ("back", TA["back"], "Cross-image: global attn", "Illustration", ORANGE),
]
P_T1 = [p[1] for p in PANELS[1:]] + [TA["end"] + 1.0]
X3 = [606 + 146 * i for i in range(3)]          # left edges of a row of three thumbnails (width 136)


def panel_ws(t):
    return [ss(t, p[1], p[1] + 0.35) * (1 - ss(t, t1, t1 + 0.35)) for p, t1 in zip(PANELS, P_T1)]


def row(imgs, y, w=136, nearest=False):
    return [ImagePanel(im, (X3[i], y), w=w, nearest=nearest) for i, im in enumerate(imgs)]


OK = [np.isfinite(sc.full(i)[1]) for i in range(3)]          # valid pixels (background excluded)
RGB_THUMBS = [ec.view_rgb(i) for i in range(3)]
CONF_STOPS = [(0.0, (30, 45, 100)), (0.5, (90, 220, 230)), (1.0, (250, 250, 250))]
_LC = np.log(np.concatenate([V["depth_conf"][i][OK[i]] for i in range(3)]))
LC_LO, LC_HI = (float(v) for v in np.percentile(_LC, [2, 98]))     # color scale = 2%-98% percentiles of log Sigma over the three images
DEP_THUMBS, CONF_THUMBS, PTS_THUMBS = [], [], []
for _i in range(3):
    _ok = OK[_i]
    _d = V["depth"][_i]
    _lo, _hi = _d[_ok].min(), _d[_ok].max()
    DEP_THUMBS.append(colorize(np.where(_ok, _d, np.nan), _lo, _hi))          # near bright, far dark (same as 01/05/08/09)
    CONF_THUMBS.append(colorize(np.where(_ok, np.log(V["depth_conf"][_i]), np.nan), LC_LO, LC_HI, stops=CONF_STOPS))
    _p = V["world_points"][_i]
    _q = (_p - _p[_ok].min(0)) / (_p[_ok].max(0) - _p[_ok].min(0))
    PTS_THUMBS.append(np.where(_ok[..., None], np.clip(_q * 255, 0, 255), 28).astype(np.uint8))

IP_IN = row(RGB_THUMBS, 398)
IP_D = ImagePanel(RGB_THUMBS[CAM], (612, 360), w=259)
PC, PR = int(TRK[CAM]["gt"][0] // sc.PATCH), int(TRK[CAM]["gt"][1] // sc.PATCH)   # patch of the white dot in frame 2 (column, row)
COL_X = 966
COL = [(0, 392), (1, 426), (2, 460), (None, 494), (NB["patch_idx"], 528), (None, 562), (KP - 1, 596)]
STRIP = TokenStrip(KT, show=fold(KT, head=6, tail=1, force=[W_IDX]), rows=3, center=(862, 534), size=24, pitch=30,
                   row_pitch=62)

# camera head: the 3 cameras predicted by VGGT (world = camera 1, unit = s)
CAM_R = [V["extrinsic"][i][:, :3].astype(float) for i in range(3)]
CAM_C = [-CAM_R[i].T @ V["extrinsic"][i][:, 3].astype(float) for i in range(3)]
FR_D = 0.3


def _corners(i, depth):
    px = np.array([[0, 0, 1], [sc.W, 0, 1], [sc.W, sc.H, 1], [0, sc.H, 1]], float)
    return (depth * (px @ np.linalg.inv(V["intrinsic"][i].astype(float)).T)) @ CAM_R[i] + CAM_C[i]


_FIT_C = np.r_[np.array(CAM_C), np.concatenate([_corners(i, FR_D) for i in range(3)])]
OC_C = ec.viewer(az=-15, el=28, r=3.0, fit_pts=_FIT_C, box=(360, 210), center=(820, 500), target=_FIT_C.mean(0))

IP_DEP = row(DEP_THUMBS, 384)
IP_CONF = row(CONF_THUMBS, 562)
IP_PTS = row(PTS_THUMBS, 384)
_v, _u = np.mgrid[0:sc.H:3, 0:sc.W:3]
PTS_PRED = np.concatenate([V["world_points"][i][_v, _u][OK[i][_v, _u]] for i in range(3)]).astype(float)
PTS_RGB = np.concatenate([RGB_THUMBS[i][_v, _u][OK[i][_v, _u]] for i in range(3)]).astype(float)
OC_P = ec.viewer(az=-20, el=20, r=4.0, fit_pts=PTS_PRED, box=(400, 150), center=(820, 624), target=PTS_PRED.mean(0))

IP_TK = [ImagePanel(RGB_THUMBS[i], (616, 360 + 94 * i), w=84) for i in range(3)]

CROP = 48                                         # tracking zoom: 48x48 pixels around the GT
CROPS = []
for _i in range(3):
    _cx, _cy = (int(round(c)) for c in TRK[_i]["gt"])
    CROPS.append((RGB_THUMBS[_i][_cy - CROP // 2:_cy + CROP // 2, _cx - CROP // 2:_cx + CROP // 2],
                  _cx - CROP // 2, _cy - CROP // 2))
IP_TR = row(RGB_THUMBS, 362)
IP_CR = [ImagePanel(CROPS[i][0], (X3[i], 528), w=136, nearest=True) for i in range(3)]
TRK_PRED = V["track"][:, 0, :].astype(float)      # (N, 2): query point = the white dot in frame 1

IP_IND = [ImagePanel(DEP_THUMBS[i], (922, 370 + 100 * i), w=80) for i in range(3)]

# ---------------------------------------------------------------- parts B, C (kept as is)
L_ALL = Formula([(r"\mathcal{L}=", "L"), (r"\mathcal{L}_{\mathrm{camera}}", "c"), (r"+\mathcal{L}_{\mathrm{depth}}", "d"),
                 (r"+\mathcal{L}_{\mathrm{pmap}}", "p"), (r"+0.05\,", "lam"), (r"\mathcal{L}_{\mathrm{track}}", "tr")],
                size=46)
L_DEP = Formula([(r"\mathcal{L}_{\mathrm{depth}}=\sum_i\ ", "s"), (r"\Vert\Sigma_i\odot(\hat D_i-D_i)\Vert", "v"),
                 (r"+\Vert\Sigma_i\odot(\nabla\hat D_i-\nabla D_i)\Vert", "g"), (r"-\alpha\log\Sigma_i", "reg")], size=31)
L_S = Formula([(r"s=\frac{1}{|\mathcal{V}|}\sum_{(i,y)\in\mathcal{V}}\Vert P_i(y)\Vert_2", "s"),
               (r"\quad t\leftarrow t/s,\ \ P\leftarrow P/s,\ \ D\leftarrow D/s", "div")], size=40)

X_GT, RGB_GT, CID, _ = ec.cloud(4)
OC = ec.viewer(az=-12, el=30, r=7.0, fit_pts=np.r_[X_GT, [c for _, _, c in sc.cameras()]], box=(600, 420),
               center=(700, 720))
RNG = np.random.default_rng(1)
LINE_IDX = RNG.choice(len(X_GT), 40, replace=False)


@shot.check
def check(ck):
    ck.close("s (mean distance of the GT, m)", S, 2.152518, 1e-6)
    ck.close("white-dot depth / s", Z / S, 0.9177, 1e-4)
    ck.close("camera 2 translation / s", T_GT / S, [0.53849, -0.03734, 0.13937], 1e-4)
    for k in (0.5, 2.0):
        R, t, c = sc.cameras()[CAM]
        uv0 = sc.project(R, t, NB["X_star"])[0]
        uvk = sc.project(R, -R @ (k * c), k * NB["X_star"])[0]
        ck.close(f"scale invariance: pixels unchanged after scaling everything by x{k}", uvk, uv0, 1e-9)
    ck.close("measured: mean distance of predicted points (point head, valid pixels)", M["points"]["point_head"]["mean_norm_pred"], 1.0014, 1e-4)
    full = float(np.linalg.norm(V["world_points"].reshape(-1, 3), axis=1).mean())
    ck.close("measured: mean distance over the full image incl. background (1.147 on screen)", round(full, 3), 1.147, 1e-9)
    # node view
    ck.eq("518 / 14 = 37 patches per side", sc.W // sc.PATCH, 37)
    ck.eq("tokens per frame = 1 + 4 + 1369", KT, 1 + 4 + KP)
    ck.eq("global sequence = 3 x 1374", 3 * KT, 4122)
    ck.eq("patch of the frame-2 white dot = row x 37 + column (no. 794 on screen)", PR * 37 + PC, NB["patch_idx"])
    ck.eq("white-dot token = 5 + 794 = 799", W_IDX, 5 + NB["patch_idx"])
    ck.true("the token strip draws no. 799", W_IDX in STRIP.shown)
    ck.eq("24 rounds x 2 = 48 attention layers (called layers, not blocks)", 2 * N_ROUNDS, 48)
    ck.eq("9 numbers = q 4 + t 3 + f 2", V["pose_enc"].shape[1], 4 + 3 + 2)
    ck.le("measured: predicted camera 1 is at the origin (norm of c1)", float(np.linalg.norm(CAM_C[0])), 1e-3)
    ck.eq("track feature map side = 518 / 2", HW_TRK, 259)
    ck.true("tracker output = (N frames, M = 1 query point, 2)", V["track"].shape == (3, 1, 2), list(V["track"].shape))
    for i in (1, 2):
        e = float(np.linalg.norm(TRK_PRED[i] - np.asarray(TRK[i]["gt"])))
        ck.close(f"measured: frame {i + 1} white-dot tracking error (px, {TRK[i]['err_px']:.1f} on screen)", e, TRK[i]["err_px"], 0.01)
    ck.le("the tracking zoom fits the predicted point (px from window center)",
          float(np.max(np.abs(TRK_PRED - np.array([[c[1] + CROP / 2, c[2] + CROP / 2] for c in CROPS])))),
          CROP / 2 - 4)
    ck.true("zoom windows lie inside the image", all(c[0].shape[:2] == (CROP, CROP) for c in CROPS))
    ck.record(track=TRK, mean_norm=M["points"], depth_conf_log_range=[LC_LO, LC_HI], n_pts_panel=len(PTS_PRED))


# ---------------------------------------------------------------- node view content (lt = seconds since this view appeared, a = alpha)
def p_img(cv, lt, a):
    for i, ip in enumerate(IP_IN):
        ai = a * ss(lt, 0.15 + 0.25 * i, 0.5 + 0.25 * i)
        ip.draw(cv, ai, border=ec.CAM_COLORS[i])
        cv.s.dot(ip.to_screen(*TRK[i]["gt"]), WHITE, 3, ai)
        cv.text((ip.box[0] + 68, ip.box[3] + 8), f"Frame {i + 1}", 22, alpha=ai, fill=ec.CAM_COLORS[i], anchor="ma")
    a2 = a * ss(lt, 1.0, 1.4)
    cv.rich(r"$N=3$ images, each $3\times518\times518$", (820, 590), 24, alpha=a2, anchor="ma")
    cv.rich("3 = RGB channels; 518 × 518 = H × W (px)", (820, 632), 20, alpha=a2, fill=style.TEXT_DIM,
            anchor="ma")
    cv.rich("White dot: the 3D point we follow all episode", (820, 666), 20, alpha=a * ss(lt, 1.6, 2.0), fill=style.TEXT_DIM,
            anchor="ma")


def p_dino(cv, lt, a):
    IP_D.draw(cv, a, border=ec.CAM_COLORS[CAM])
    IP_D.grid(cv, cell=sc.PATCH, alpha=0.5 * a, progress=ss(lt, 0.2, 1.3))
    aw = a * ss(lt, 1.6, 1.9)
    if aw > 0.01:
        IP_D.cell(cv, PR, PC, cell=sc.PATCH, color=GOLD, alpha=aw, glow=0)
    a_t = a * ss(lt, 0.6, 1.0)
    cv.rich(r"$518\div14=37$ cells per side", (741, 630), 20, alpha=a_t, anchor="ma")
    cv.rich(f"$37\\times37={KP}$ patches", (741, 658), 20, alpha=a_t, anchor="ma")
    arrow(cv, (878, 490), (940, 490), WHITE, a * 0.7 * ss(lt, 1.0, 1.3), th=2, head=10)
    for k, (idx, y) in enumerate(COL):
        ak = a * ss(lt, 1.1 + 0.08 * k, 1.4 + 0.08 * k)
        if idx is None:
            for dy in (-8, 0, 8):
                cv.s.dot((COL_X, y + dy), style.TEXT_DIM, 2, ak)
            continue
        token(cv, (COL_X, y), 24, CYAN, ak)
        cv.text((COL_X + 24, y), f"#{idx}", 16, "light", ak, style.TEXT_DIM, "lm")
    if aw > 0.01:
        p = IP_D.to_screen((PC + 0.5) * sc.PATCH, (PR + 0.5) * sc.PATCH)
        cv.s.dashed(p, (COL_X - 17, 528), GOLD, aw, 2, n=18)
        s = 17
        cv.s.polyline([(COL_X - s, 528 - s), (COL_X + s, 528 - s), (COL_X + s, 528 + s), (COL_X - s, 528 + s)], GOLD, aw,
                      2, closed=True)
    cv.rich(f"{KP} vectors\n1024 dims each", (COL_X, 624), 18, alpha=a * ss(lt, 1.5, 1.9), fill=style.TEXT_DIM,
            anchor="ma", align="m")
    cv.rich("Each image passes DINOv2 alone; patches attend within", (820, 692), 20, alpha=a * ss(lt, 2.0, 2.4), anchor="ma")
    cv.rich(f"Gold box: the dot's patch → vector no. {NB['patch_idx']}", (820, 724), 18, alpha=aw,
            fill=style.LABEL_GOLD, anchor="ma")


def _kind(r, i):
    if i == 0:
        return "camera1" if r == 0 else "camera"
    if i <= 4:
        return "register1" if r == 0 else "register"
    return "patch"


def p_cat(cv, lt, a):
    for j, (kd, txt, x) in enumerate([("camera", "camera ×1", 622), ("register", "register ×4", 762),
                                      ("patch", f"patch ×{KP}", 906)]):
        aj = a * ss(lt, 0.2 + 0.2 * j, 0.5 + 0.2 * j)
        token(cv, (x, 400), 16, style.TOKEN[kd], aj)
        cv.text((x + 14, 400), txt, 18, "text", aj, style.TEXT_DIM, "lm")
    x_lab = min(STRIP.x.values()) - 24
    for r in range(3):
        ar = a * ss(lt, 0.1 + 0.15 * r, 0.5 + 0.15 * r)
        u = ss(lt, 0.7 + 0.15 * r, 1.1 + 0.15 * r)
        for i in STRIP.shown:
            p = STRIP.xy(i, r)
            if i < 5:
                token(cv, p + [0, -18 * (1 - u)], STRIP.size, style.TOKEN[_kind(r, i)], a * u)
            else:
                token(cv, p, STRIP.size, style.TOKEN["patch"], ar)
        for gx in STRIP.gap_x:
            cv.text((gx, STRIP.y[r]), "…", 22, "light", ar, style.TEXT_DIM, "mm", shadow=False)
        cv.text((x_lab, STRIP.y[r]), f"Frame {r + 1}", 20, alpha=ar, fill=ec.CAM_COLORS[r], anchor="rm")
    aw = a * ss(lt, 1.5, 1.8)
    if aw > 0.01:
        p = STRIP.xy(W_IDX, 1)
        s = 17
        cv.s.polyline([p + [-s, -s], p + [s, -s], p + [s, s], p + [-s, s]], GOLD, aw, 2, closed=True)
    cv.rich(f"$1+4+{KP}={KT}$ per frame, 1024 dims each", (820, 636), 20, alpha=a * ss(lt, 1.0, 1.4), anchor="ma")
    cv.rich("Frame 1: separate camera / register set (marks the reference)", (820, 670), 18, alpha=a * ss(lt, 1.2, 1.6),
            fill=(255, 180, 140), anchor="ma")
    cv.rich(f"Gold box: frame 2's dot token, no. ${W_IDX}=5+{NB['patch_idx']}$", (820, 702), 18, alpha=aw,
            fill=style.LABEL_GOLD, anchor="ma")


def p_aa(cv, lt, a):
    r0 = TA["rounds"] - TA["aa"]
    act_f = min(1.0, 1 - ss(lt, 1.4, 1.7) + ss(lt, r0, r0 + 0.3))
    act_g = ss(lt, 1.4, 1.7)
    af = a * ss(lt, 0.1, 0.5) * (0.3 + 0.7 * act_f)
    ag = a * ss(lt, 0.3, 0.7) * (0.3 + 0.7 * act_g)
    attn_blocks(cv, (656, 402), 3, 4, 14, mode="frame", alpha=af, label_size=16, block_labels=False)
    attn_blocks(cv, (862, 402), 3, 4, 14, mode="global", alpha=ag, label_size=16, frame_labels=False,
                block_labels=False)
    for b in range(3):
        cv.text((862 + (b + 0.5) * 56, 394), f"Frame {b + 1}", 16, alpha=ag, anchor="mb")
    cv.rich("Frame layer", (740, 586), 20, alpha=af, anchor="ma")
    cv.rich("$(3,1374,1024)$", (740, 614), 20, alpha=af, fill=style.TEXT_DIM, anchor="ma")
    cv.rich("Global layer", (946, 586), 20, alpha=ag, anchor="ma")
    cv.rich(f"$(1,{3 * KT},1024)$", (946, 614), 20, alpha=ag, fill=style.TEXT_DIM, anchor="ma")
    a_lg = a * ss(lt, 0.8, 1.2)
    cv.rich("Each cell = does this token pair attend?", (820, 644), 18, alpha=a_lg, fill=style.TEXT_DIM, anchor="ma")
    cv.rich("Cyan = within frame, violet = cross-frame, grey = skipped", (820, 670), 18, alpha=a_lg, fill=style.TEXT_DIM, anchor="ma")
    ar = a * ss(lt, r0 + 0.9, r0 + 1.3)
    cv.rich(f"{N_ROUNDS} rounds × 2 = {2 * N_ROUNDS} attention layers", (820, 700), 22, alpha=ar, fill=style.LABEL_GOLD,
            anchor="ma")
    cv.rich("(excl. DINOv2's 24 layers; each layer 1024 dims, 16 heads)", (820, 732), 18, alpha=ar, fill=style.TEXT_DIM,
            anchor="ma")


def p_cam(cv, lt, a):
    P = OC_C.at(az=-25 + 20 * ss(lt, 0.0, 3.0))
    for i in range(3):
        ai = a * ss(lt, 0.2 + 0.3 * i, 0.6 + 0.3 * i)
        ec.frustum_pose(cv, P, CAM_R[i], CAM_C[i], FR_D, ai, ec.CAM_COLORS[i], image=RGB_THUMBS[i], tex_alpha=1.0,
                        label=ec.CAM_NAMES[i], K=V["intrinsic"][i].astype(float))
    a2 = a * ss(lt, 1.2, 1.6)
    cv.rich("9 numbers per image → drawn as one frustum", (820, 654), 20, alpha=a2, anchor="ma")
    cv.rich("$q$ rotation 4 + $t$ translation 3 + $f$ FoV 2", (820, 686), 20, alpha=a2, fill=style.LABEL_GOLD,
            anchor="ma")
    cv.rich("Code: 4-layer trunk ×4; camera tokens attend to each other", (820, 722), 18, alpha=a * ss(lt, 1.8, 2.2),
            fill=style.TEXT_DIM, anchor="ma")


def p_dep(cv, lt, a):
    cv.rich("Depth $D_i$: per-pixel z-distance (near = bright)", (606, 356), 20, alpha=a * ss(lt, 0.1, 0.5), anchor="la")
    for i, ip in enumerate(IP_DEP):
        ip.draw(cv, a * ss(lt, 0.2 + 0.2 * i, 0.6 + 0.2 * i), border=ec.CAM_COLORS[i])
    cv.rich(r"Confidence $\Sigma_i$: how sure each pixel is (bright = high)", (606, 534), 20, alpha=a * ss(lt, 1.2, 1.6),
            anchor="la")
    for i, ip in enumerate(IP_CONF):
        ip.draw(cv, a * ss(lt, 1.3 + 0.2 * i, 1.7 + 0.2 * i), border=ec.CAM_COLORS[i])
    cv.rich(r"Each $518\times518$; $\Sigma_i$ is used in the training loss (later)", (820, 714), 18,
            alpha=a * ss(lt, 2.0, 2.4), fill=style.TEXT_DIM, anchor="ma")


def p_pts(cv, lt, a):
    cv.rich("Point map $P_i$: $(x,y,z)$ per pixel, shown as color", (606, 356), 20, alpha=a * ss(lt, 0.1, 0.5), anchor="la")
    for i, ip in enumerate(IP_PTS):
        ip.draw(cv, a * ss(lt, 0.2 + 0.2 * i, 0.6 + 0.2 * i), border=ec.CAM_COLORS[i])
    a3 = a * ss(lt, 1.1, 1.5)
    if a3 > 0.01:
        ec.splat(cv, OC_P.at(az=-30 + 25 * ss(lt, 1.1, 3.2)), PTS_PRED, PTS_RGB, a3, size=2)
    cv.rich("All 3 point maps lie in camera 1's frame → one point cloud", (820, 714), 18, alpha=a * ss(lt, 1.5, 1.9),
            fill=style.TEXT_DIM, anchor="ma")


def _shade(kh, kw, kd):
    return tuple(int(c * (0.55 + 0.15 * kd)) for c in VIOLET)


def p_trk(cv, lt, a):
    for i in range(3):
        y = 360 + 94 * i
        ai = a * ss(lt, 0.15 + 0.3 * i, 0.5 + 0.3 * i)
        IP_TK[i].draw(cv, ai, border=ec.CAM_COLORS[i])
        arrow(cv, (708, y + 42), (752, y + 42), WHITE, 0.7 * ai, th=2, head=9)
        tensor_block(cv, (770, y + 20), 64, 54, 46, VIOLET, ai, grid=(1, 1, 4), gap=(0, 0, 4), colors=_shade, th=1)
        cv.rich(f"$T_{i + 1}$", (890, y + 40), 22, alpha=ai, anchor="lm")
    a2 = a * ss(lt, 1.2, 1.6)
    cv.rich(f"One feature map $T_i$ per frame: ${C_TRK}\\times{HW_TRK}\\times{HW_TRK}$ (code)", (820, 650), 20, alpha=a2,
            anchor="ma")
    cv.rich(f"{C_TRK} = channels; {HW_TRK} = half of 518", (820, 680), 18, alpha=a2, fill=style.TEXT_DIM, anchor="ma")
    cv.rich(r"The paper just writes $C\times H\times W$", (820, 706), 18, alpha=a2, fill=style.TEXT_DIM, anchor="ma")
    cv.rich("Only passed to the tracker; it outputs no tracks itself", (820, 732), 18, alpha=a * ss(lt, 1.8, 2.2), fill=style.LABEL_GOLD,
            anchor="ma")


def _ring(cv, p, r, color, alpha, th=2):
    ang = np.linspace(0, 2 * np.pi, 33)
    cv.s.polyline(np.c_[p[0] + r * np.cos(ang), p[1] + r * np.sin(ang)], color, alpha, th, closed=True)


def p_tracker(cv, lt, a):
    a1 = a * ss(lt, 0.1, 0.5)
    for i in range(3):
        IP_TR[i].draw(cv, a1, border=ec.CAM_COLORS[i])
    dots = [IP_TR[i].to_screen(*TRK_PRED[i]) for i in range(3)]
    cv.s.dot(dots[0], WHITE, 4, a1)
    for j in (1, 2):
        pr = ss(lt, 0.5 + 0.3 * (j - 1), 1.0 + 0.3 * (j - 1))
        arrow(cv, dots[0], dots[j], GOLD, 0.9 * a, th=2, head=10, bend=-0.22, progress=pr)
        cv.s.dot(dots[j], CYAN, 4, a * ss(lt, 0.9 + 0.3 * (j - 1), 1.1 + 0.3 * (j - 1)))
    a2 = a * ss(lt, 1.2, 1.6)
    for i in range(3):
        crop, x0c, y0c = CROPS[i]
        q0, q1 = IP_TR[i].to_screen(x0c, y0c), IP_TR[i].to_screen(x0c + CROP, y0c + CROP)
        cv.s.polyline([q0, (q1[0], q0[1]), q1, (q0[0], q1[1])], WHITE, 0.8 * a2, 1, closed=True)
        b = IP_CR[i].box
        cv.s.line((q0[0], q1[1]), (b[0], b[1]), GREY, 0.5 * a2, 1)
        cv.s.line(q1, (b[2], b[1]), GREY, 0.5 * a2, 1)
        IP_CR[i].draw(cv, a2, border=ec.CAM_COLORS[i])
        g = IP_CR[i].to_screen(TRK[i]["gt"][0] - x0c, TRK[i]["gt"][1] - y0c)
        p = IP_CR[i].to_screen(TRK_PRED[i][0] - x0c, TRK_PRED[i][1] - y0c)
        _ring(cv, g, 9, WHITE, a2)
        if i > 0:
            cv.s.line(p - [9, 0], p + [9, 0], CYAN, a2, 2)
            cv.s.line(p - [0, 9], p + [0, 9], CYAN, a2, 2)
        lab = "Query point (input)" if i == 0 else f"Error {TRK[i]['err_px']:.1f} px"
        cv.text(((b[0] + b[2]) / 2, b[3] + 8), lab, 18, alpha=a2, fill=ec.CAM_COLORS[i], anchor="ma")
    a3 = a * ss(lt, 1.8, 2.2)
    cv.rich(f"Bottom: {CROP}×{CROP} px zoom; white circle = GT, cyan cross = pred.", (820, 702), 18, alpha=a3, anchor="ma")
    cv.rich("Output $(M,N,2)=(1,3,2)$: $M$ query points, $N$ frames", (820, 732), 18, alpha=a3, fill=style.TEXT_DIM,
            anchor="ma")


def p_indep(cv, lt, a):
    for i in range(3):
        yc = 410 + 100 * i
        ai = a * ss(lt, 0.15 + 0.3 * i, 0.5 + 0.3 * i)
        cv.text((612, yc), f"Frame {i + 1}", 20, alpha=ai, fill=ec.CAM_COLORS[i], anchor="lm")
        for k in range(4):
            kd = "patch" if k else ("camera1" if i == 0 else "camera")
            token(cv, (672 + 22 * k, yc), 18, style.TOKEN[kd], ai)
        arrow(cv, (748, yc), (784, yc), WHITE, 0.7 * ai, th=2, head=9)
        module_box(cv, (788, yc - 24, 878, yc + 24), "DPT", None, CYAN, ai, title_size=20, title_kind="light", r=10)
        arrow(cv, (882, yc), (916, yc), WHITE, 0.7 * ai, th=2, head=9)
        IP_IND[i].draw(cv, ai, border=ec.CAM_COLORS[i])
        cv.rich(f"$D_{i + 1}$", (1008, yc), 20, alpha=ai, anchor="lm")
        if i < 2:
            cv.s.dashed((612, yc + 50), (1030, yc + 50), GREY, 0.45 * ai, 1, n=40)
    a2 = a * ss(lt, 1.2, 1.6)
    cv.rich("depth DPT runs per image; the three paths never meet", (820, 676), 20, alpha=a2, anchor="ma")
    cv.rich("Same for point DPT and track-feature DPT (paper p.10)", (820, 708), 18, alpha=a2, fill=style.TEXT_DIM,
            anchor="ma")


def p_back(cv, lt, a):
    attn_blocks(cv, (712, 382), 3, 4, 18, mode="global", alpha=a, label_size=18)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * lt / 1.2)
    cv.g.polyline([(712, 382), (928, 382), (928, 598), (712, 598)], ORANGE, 0.45 * a * g, 4, closed=True)
    a2 = a * ss(lt, 0.4, 0.8)
    cv.rich(f"global layer: every pair of the {3 * KT} tokens", (820, 622), 20, alpha=a2, anchor="ma")
    cv.rich("DPT's input tokens were already mixed across images", (820, 654), 20, alpha=a * ss(lt, 0.9, 1.3), anchor="ma")
    cv.rich("camera head and tracker also go cross-frame themselves (code)", (820, 692), 18, alpha=a * ss(lt, 1.4, 1.8),
            fill=style.TEXT_DIM, anchor="ma")


PANEL_FN = dict(img=p_img, dino=p_dino, cat=p_cat, aa=p_aa, cam=p_cam, dep=p_dep, pts=p_pts, trk=p_trk,
                tracker=p_tracker, indep=p_indep, back=p_back)


def panels(cv, t, a, ws):
    """Node view at top right: one box, border color = current node color, title + corner tag (Measured / Illustration), content switches per node."""
    if a <= 0.01:
        return
    tot = sum(ws)
    col = (sum(w * np.asarray(p[4], float) for w, p in zip(ws, PANELS)) / tot) if tot > 1e-3 else np.asarray(WHITE)
    col = tuple(float(c) for c in col)
    x0, y0, x1, y1 = PB
    pts = rrect_pts(x0, y0, x1, y1, 14)
    cv.s.fill(pts, (19, 24, 32), 0.92 * a)
    cv.s.polyline(pts, col, 0.85 * a, 2, closed=True)
    cv.s.line((x0 + 2, y0 + 44), (x1 - 2, y0 + 44), col, 0.4 * a, 1)
    for w, (k, t0, title, badge, _c) in zip(ws, PANELS):
        aw = a * w
        if aw <= 0.01:
            continue
        cv.rich(title, (x0 + 16, y0 + 23), 24, "bold", aw, anchor="lm")
        cv.badge((x1 - 12, y0 + 9), badge, 18, aw, anchor="ra")
        PANEL_FN[k](cv, t - t0, aw)


def diagram(cv, t, a, ws):
    if a <= 0.01:
        return
    w = {p[0]: wk for p, wk in zip(PANELS, ws)}
    for k, (box, title, sub, col, t0) in BOXES.items():
        lit = ss(t, t0, t0 + 0.4)
        cur = w[k] + (w["indep"] if k in DPTS else 0.0) + (w["back"] if k == "aa" else 0.0)
        state = 1.0 if cur > 0.99 else 0.3 + 0.45 * lit + 0.25 * min(cur, 1.0)
        glow = 0.6 * bump(t, t0 + 0.3, 1.0)        # the "three DPTs are independent" part only lights up fully, no glow (STYLE: glow <= 2)
        module_box(cv, box, title, sub, col, a, state=state, glow=glow, title_size=22 if k == "tracker" else 26,
                   sub_size=22,
                   title_kind="light" if k in ("img", "cam") else "text")
    for s0, s1 in [("img", "dino"), ("dino", "cat"), ("cat", "aa")]:
        arrow(cv, (CX[s0], BOXES[s0][0][3] + 2), (CX[s1], BOXES[s1][0][1] - 2), WHITE, a * 0.6, th=2, head=9)
    for k in ("cam", "dep", "pts", "trk"):
        arrow_path(cv, elbow((CX["aa"], 747), (CX[k], 805), first="v", mid=778), WHITE,
                   a * 0.5 * ss(t, BOXES[k][4] - 0.4, BOXES[k][4]), th=2, head=10)
    arrow(cv, (CX["trk"], 890), (CX["tracker"], 926), WHITE, a * 0.5 * ss(t, TA["tracker"] - 0.4, TA["tracker"]),
          th=2, head=10)
    # 24-round counter (right of the AA box; the explanation is in the node view)
    if t > TA["rounds"]:
        au = a * ss(t, TA["rounds"], TA["rounds"] + 0.3)
        counter(cv, (532, 670), ss(t, TA["rounds"] + 0.1, TA["rounds"] + 1.6), 1, N_ROUNDS, size=44, alpha=au,
                color=style.LABEL_GOLD, anchor="rm")
        cv.text((536, 670), "rounds", 24, alpha=au, anchor="lm")
    # track head = feature DPT + tracker (the dashed box encloses both)
    a_th = a * ss(t, TA["tracker"] + 0.4, TA["tracker"] + 0.9)
    if a_th > 0.01:
        x0, y0, x1, y1 = 797, 800, 1058, 1006
        for p, q in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            cv.s.dashed(p, q, VIOLET, 0.6 * a_th, 1, n=30)
        cv.rich("track head = feature DPT + tracker", (788, 963), 20, alpha=a_th, fill=(205, 185, 255),
                anchor="rm")
    # each DPT sees only its own image: code reference
    a_ind = a * w["indep"]
    if a_ind > 0.01:
        cv.rich("Code: 3 DPTHeads (vggt.py L25–26, track_head.py L50–57)", (540, 1046), 20, alpha=a_ind,
                fill=style.TEXT_DIM, anchor="ma")
    # cross-image information comes from the backbone: light dots go from the three DPTs back to the AA
    a_back = a * w["back"]
    if a_back > 0.01:
        for k in DPTS:
            pp = np.array([[CX[k], 805.0], [CX["aa"] + 140, 748.0]])
            pulse_dot(cv, pp, ((t - TA["back"] - 0.1) % 1.2) / 1.2, CYAN, a_back)


# ---------------------------------------------------------------- B: loss (original timeline "to")
def loss(cv, t):
    a_l = fade(t, 32.7, 43.3, 0.4, 0.4)
    if a_l <= 0:
        return
    st = {"L": "lit", "c": "lit" if t > 33.0 else "dim", "d": "lit" if t > 33.4 else "dim",
          "p": "lit" if t > 33.8 else "dim", "tr": "lit" if t > 34.2 else "dim",
          "lam": "hl" if 35.5 < t < 37.6 else ("lit" if t > 35.5 else "dim")}
    L_ALL.draw(cv, (540, 420), alpha=a_l, state=st)
    cv.rich("First three terms have similar ranges, unweighted; tracking term cut to 0.05 (§3.4)", (540, 480), 22, alpha=a_l * ss(t, 35.8, 36.2),
            fill=style.TEXT_DIM, anchor="ma")
    a_d = ss(t, 37.7, 38.2) * a_l
    st2 = {"s": "lit", "v": "hl" if 37.9 < t < 40.4 else "lit", "g": "lit" if t > 39.0 else "dim",
           "reg": "hl" if 40.5 < t < 43.2 else ("lit" if t > 40.5 else "dim")}
    L_DEP.draw(cv, (540, 620), alpha=a_d, state=st2,
               notes={"v": r"$\Sigma$: confidence map, multiplies the error; large = confident", "reg": r"$\alpha$ not given in the paper"})
    cv.rich(r"$\nabla$: gradient along pixel directions, penalizes only wrong local shape", (540, 760), 22, alpha=a_d * ss(t, 39.0, 39.4),
            fill=style.TEXT_DIM, anchor="ma")
    cv.rich(r"Without $-\alpha\log\Sigma$, the network drives all $\Sigma$ to 0 to shrink the loss", (540, 820), 24,
            alpha=a_l * ss(t, 40.8, 41.3), anchor="ma")


# ---------------------------------------------------------------- C: normalization (original timeline "to")
def normalization(cv, t):
    a_n = ss(t, 43.3, 43.8)
    if a_n <= 0:
        return
    k_sc = 1.0
    if t > 54.4:
        u = (t - 54.6) / 4.6
        k_sc = 1.0 + (0.5 * np.sin(2 * np.pi * np.clip(u, 0, 1)))       # 1 -> 1.5 -> 1 -> 0.5 -> 1
    P = OC.at(az=-12 + 8 * ss(t, 43.3, 60.0))
    for i in range(3):
        r = ec.frustum(cv, P, i, 0.45, a_n, scale=k_sc)
        if r is not None:          # when the scene is scaled to 1.5x, the camera-3 label runs off the right edge (in both languages) -> clamp label x inside the frame
            C = r[0]
            cv.text((float(np.clip(C[0], 70, 1010)), C[1] + 14), ec.CAM_NAMES[i], 26, alpha=a_n,
                    fill=ec.CAM_COLORS[i], anchor="ma")
    ec.splat(cv, P, k_sc * X_GT, RGB_GT, 0.8 * a_n, size=2)
    o = P(ec.disp(np.zeros(3)))
    a_lines = fade(t, 43.4, 48.6, 0.4, 0.4)
    for j in LINE_IDX:
        cv.s.line(o, P(ec.disp(X_GT[j])), GOLD, 0.35 * a_lines, 1)
    L_S.draw(cv, (540, 340), alpha=a_n * (1 - ss(t, 54.2, 54.6)),
             state={"s": "hl" if 43.5 < t < 46.0 else "lit", "div": "lit" if t > 48.6 else "dim"},
             notes={"s": r"$\mathcal{V}$: all valid pixels of the 3 images (363,918)"})
    a_s = ss(t, 46.1, 46.6) * (1 - ss(t, 54.2, 54.6))
    if a_s > 0:
        cv.rich("$s$ =", (120, 470), 40, alpha=a_s, anchor="lm")
        counter(cv, (200, 470), ss(t, 46.1, 47.2), 0.0, S, fmt="{:.3f}", size=44, alpha=a_s, color=style.LABEL_GOLD,
                anchor="lm", unit="m")
    a_v = ss(t, 48.7, 49.2) * (1 - ss(t, 54.2, 54.6))
    if a_v > 0:
        u = ss(t, 49.2, 50.6)
        cv.rich("Dot depth:", (120, 540), 28, alpha=a_v, anchor="lm")
        number(cv, (290, 540), Z * (1 - u) + Z / S * u, "{:.3f}", 36, style.LABEL_GOLD, a_v,
               anchor="lm", unit="m" if u < 0.5 else None)
        cv.rich("Cam 2 t:", (120, 600), 28, alpha=a_v, anchor="lm")
        vector(cv, (300, 600), T_GT * (1 - u) + T_GT / S * u, "{:.3f}", 30, style.LABEL_GOLD, a_v, anchor="lm")
    a_m = fade(t, 51.5, 54.4, 0.4, 0.3)
    if a_m > 0:
        cv.rich("Measured on our test scene: point-head mean distance to camera 1 = %.3f"
                % M["points"]["point_head"]["mean_norm_pred"], (540, 1050), 26, alpha=a_m, anchor="ma")
        cv.rich("(363,918 valid pixels; the full image including background gives 1.147)", (540, 1140), 20, alpha=a_m, fill=style.TEXT_DIM,
                anchor="ma")
        cv.rich("Matches the training convention “divide by mean distance” (the reason is inferred); the unit is $s$, not meters", (540, 1095), 22, alpha=a_m,
                fill=style.TEXT_DIM, anchor="ma")
    a_k = ss(t, 54.5, 55.0)
    if a_k > 0:
        for i in range(3):
            ImagePanel(RGB_THUMBS[i], (150 + i * 270, 990), w=240).draw(cv, a_k, border=ec.CAM_COLORS[i])
        number(cv, (540, 440), k_sc, "×{:.2f}", 44, style.LABEL_GOLD, a_k, anchor="mm")
        cv.rich("Scene points and camera centers are multiplied by the same number", (540, 490), 24, alpha=a_k, fill=style.TEXT_DIM, anchor="ma")


@shot.frame
def frame(cv, t):
    a_dia = ss(t, 0.1, 0.6) * (1 - ss(t, TA["end"] - 0.2, TA["end"] + 0.3))
    if a_dia > 0.01:
        ws = panel_ws(t)
        diagram(cv, t, a_dia, ws)
        panels(cv, t, a_dia, ws)
    to = t - SH                                     # base timeline for parts B and C
    loss(cv, to)
    normalization(cv, to)
