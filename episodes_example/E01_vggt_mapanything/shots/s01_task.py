"""E01-01 Task: three images -> cameras, depth, point maps, tracks (S, 31.5 s = 20 s + an 11.5 s inserted popup defining "our test scene"). Storyboard storyboard.md section E01-01.
Computed for real: the three images (ray-cast rendering). Measured: the 3D cameras (camera head), depth thumbnails (depth head),
point cloud (P_i from the point head) and the white dot's position in the three images (track head) are all real VGGT-1B outputs on these 3 images (units / s).
The formula is labeled "Paper Eq. (1)"; every term has the same color as its element on screen, and the frame lights up with each term. After sentence 1, a popup of INS seconds defines "our test scene"
(a test we built, not a paper experiment; where the ground truth comes from, how the error is computed, what the white dot is). During the insert the original timeline is held at T_INS (t -> tm(t) in frame),
the popup content uses real time; the original constants (T_EQ, ...) are values on the original timeline.
"""
import numpy as np
from PIL import Image

import e01_common as ec
import scene_e01 as sc
from animkit import CYAN, GOLD, ORANGE, Formula, Shot, arrow, bump, colorize, fade, ss, style
from e01_common import Popup, popup_check

T_INS, INS = 2.6, 11.5                       # "our test scene" definition insert: starts when sentence 1 ends, 5 sentences x 2.3 s
CAM_SHORT = ["Cam 1", "Cam 2", "Cam 3"]          # short labels: this shot's boxes are narrow
DUR = 20.0 + INS


def tm(t):
    """Real time -> original timeline (held at T_INS during the insert)."""
    return t if t < T_INS else (T_INS if t < T_INS + INS else t - INS)


def sh(x):
    """Time on the original timeline -> real time (everything at or after T_INS shifts later by INS)."""
    return x if x < T_INS - 1e-9 else x + INS


shot = Shot("s01_task", DUR)
shot.title("Multi-view → 3D", "E01 · VGGT × MapAnything", 0.2, 2.5)
shot.tag(ec.SCENE_TAG, T_INS + 4 * INS / 5 + 1.0, DUR)           # unified corner tag: appears when the popup's last line reaches it
shot.tag("VGGT-1B output (units ÷ s)", sh(10.0), DUR)

# Time each term starts lighting up (= start of its caption; original timeline)
T_EQ, T_I, T_G, T_D, T_P, T_T = 5.0, 7.6, 10.0, 12.4, 14.8, 17.2
CAPS_ORIG = [(0.3, 2.6, "Three photos of the same scene"), (2.6, T_EQ, "VGGT outputs all of it in one forward pass"),
             (T_EQ, T_I, "Paper Eq. (1): the input and output of f"), (T_I, T_G, "I: the 3 input images, N = 3"),
             (T_G, T_D, "g: camera parameters per image"), (T_D, T_P, "D: depth at each pixel"),
             (T_P, T_T, "P: a 3D point per pixel"), (T_T, 20.0, "T: track features, used to find the same point")]
# The 5 inserted sentences (real time): what our test scene is -> ground truth -> real models -> error -> white dot
_d = INS / 5
CAPS_DEF = [(T_INS + k * _d, T_INS + (k + 1) * _d, s) for k, s in enumerate([
    "Our test scene: our own test, not a paper experiment", "Ground truth fully known: cameras, depth, 3D points", "Feed the 3 images to real VGGT and MapAnything",
    "Error = prediction minus known ground truth", "The white dot: our chosen marker, tracked throughout"])]
CAPS = CAPS_ORIG[:1] + CAPS_DEF + [(sh(a), sh(b), s) for a, b, s in CAPS_ORIG[1:]]
for a, b, s in CAPS:
    shot.caption(a, b, s)

NB = ec.nb()
V = ec.vggt_out()
VM = ec.vggt_metrics()

# One color per term: the term in the formula, the legend and the matching on-screen element share it
ACC = {"I": (235, 235, 235), "g": ORANGE, "D": (240, 110, 140), "P": CYAN, "T": GOLD}
T_ON = {"I": T_I, "g": T_G, "D": T_D, "P": T_P, "T": T_T}
EQ = Formula([(r"f\,(", "f"), (r"(I_i)_{i=1}^{N}", "I"), (r")=(", "eq"), (r"g_i,", "g"), (r"D_i,", "D"),
              (r"P_i,", "P"), (r"T_i", "T"), (r")_{i=1}^{N}", "end")], size=54, gap=0.12)
EQ_XY = (560, 210)
EQ_BASE = (205, 205, 205)                    # f, brackets and the equals sign use neutral gray so the five term colors stand out
LEGEND = [("I", r"I_i", "Input images"), ("g", r"g_i", "Camera parameters"), ("D", r"D_i", "Depth maps"), ("P", r"P_i", "Point maps"),
          ("T", r"T_i", "Track features")]
LEGEND_X = [95, 300, 500, 690, 880]
LEG_Y = 128                                  # legend above the formula (the space below is for the formula -> three-image connector lines)
NOTE_Y = 278

TH_W, TH_Y = 300, 320
TH_X = [70, 390, 710]

# The point head's point map P_i (not depth + camera unprojection), taken at the same pixels as the three images
_, RGB, CID, UV = ec.cloud(4)
_ui = UV.astype(int)
X = np.stack([V["world_points"][i][v, u] for i, (u, v) in zip(CID, _ui)])
R_P = [V["extrinsic"][i][:, :3] for i in range(3)]
TR_P = [V["extrinsic"][i][:, 3] for i in range(3)]
C_P = [-R_P[i].T @ TR_P[i] for i in range(3)]
K_P = [V["intrinsic"][i] for i in range(3)]
PLANE_D = 0.45 / NB["s"]
START = np.empty_like(X)
for i in range(3):
    m = CID == i
    uv1 = np.c_[UV[m], np.ones(m.sum())]
    START[m] = (PLANE_D * (uv1 @ np.linalg.inv(K_P[i]).T) - TR_P[i]) @ R_P[i]

AZ0, AZ1 = -12.0, 10.0
_FIT = np.r_[X, C_P]
OC = ec.viewer(az=0.5 * (AZ0 + AZ1), el=28, r=7.0 / NB["s"], fit_pts=_FIT, box=(800, 400), center=(540, 950),
               target=ec.scene_center() / NB["s"])

TRACK = V["track"][:, 0, :]                  # (3, 2) predicted pixels of the white dot in the three images (query = the white dot in frame 1)
TRK_ERR = [d["err_px"] for d in VM["track_white_dot"]]
WD_P = np.array(VM["white_dot"]["point_head"])    # white dot: the point head's point on camera 2


def depth_img(i):
    """Depth output by VGGT (background pixels outside the scene are not drawn); near = bright yellow, far = dark purple."""
    ok = np.isfinite(sc.full(i)[1])
    d = V["depth"][i]
    lo, hi = float(d[ok].min()), float(d[ok].max())
    return Image.fromarray(colorize(np.where(ok, d, np.nan), lo, hi))


THUMBS = [ec.view_pil(i) for i in range(3)]
DEPTHS = [depth_img(i) for i in range(3)]

# ---------------- Popup: what "our test scene" is ----------------
POP_S = Popup((60, 668, 960, 572), t_in=T_INS, t_out=T_INS + INS - 0.4,
              title="Our test scene = a test we built (not a paper experiment)", corner="tl", accent=GOLD)
GT_X, GT_RGB, GT_CID, _ = ec.cloud(6)                    # ground-truth scene points (from the scene script's ray casting), thinned
GT_C = [c for _, _, c in sc.cameras()]
OC_S = ec.viewer(az=-25, el=26, r=4.5, fit_pts=np.r_[GT_X, GT_C], box=(290, 320), center=(205, 270),
                 target=ec.scene_center())
_rng = np.random.default_rng(7)
RAY_PTS = [GT_X[_rng.choice(np.flatnonzero(GT_CID == i), 5, replace=False)] for i in range(3)]   # 5 illustrative rays per camera
DEF_LINES = [  # (popup-local time, y, text, font size, color)
    (0.3, 84, "A scripted toy scene: ground, red box, two balls", 22, style.TEXT),
    (0.6, 116, "3 virtual cameras ray-cast three 518×518 images", 22, style.TEXT),
    (_d + 0.2, 166, "All GT known: cameras, depth, a 3D point per pixel", 22, style.LABEL_GOLD),
    (2 * _d + 0.2, 214, "These 3 images go into real VGGT-1B, MapAnything", 22, style.TEXT),
    # separate measured from illustrative (08's 0.922 and 09's m = 2.153 are illustrative)
    (2 * _d + 0.6, 244, "“Measured on our test scene” tag: real model output", 21, style.LABEL_CYAN),
    (2 * _d + 0.9, 272, "“Illustration” tag: example from GT via paper formula", 21, style.TEXT_DIM),
    (3 * _d + 0.2, 320, "Error = model prediction minus known ground truth", 22, style.TEXT),
    # "public datasets": MapAnything's evaluation includes the synthetic TartanAirV2-WB
    (3 * _d + 0.6, 348, "Paper experiments use public datasets", 20, style.TEXT_DIM),          # one line does not fit (runs past the popup's right edge), so split into two lines
    (3 * _d + 0.6, 374, "(e.g., VGGT Tab. 3 = ETH3D, real captures)", 20, style.TEXT_DIM),
    (4 * _d + 0.2, 418, "The white dot = one marker point we chose:", 22, style.TEXT),
    # "the same point", not pixel: the white dot lands on different pixels in the three images
    (4 * _d + 0.5, 450, "Follow this same point through the whole pipeline", 22, style.TEXT),
]


def body_scene(sub, lt, box):
    """Left: small 3D view of the ground-truth scene (GT cameras + rays + GT point cloud + white dot); right: definitions appearing sentence by sentence."""
    P = OC_S.at(az=-25 + 20 * ss(lt, 0.0, INS))
    a0 = ss(lt, 0.2, 0.6)
    a_pc = ss(lt, _d, _d + 0.8)                              # sentence 2: the GT point cloud lights up
    ec.splat(sub, P, GT_X, GT_RGB, (0.25 + 0.75 * a_pc) * a0, size=2)
    for i in range(3):
        ec.frustum(sub, P, i, 0.45, a0 * ss(lt, 0.3 + 0.2 * i, 0.7 + 0.2 * i), tex_alpha=0.8, label=CAM_SHORT[i])
    # sentence 1: rays go from the camera center to the scene surface (ray-cast rendering, a few shown)
    a_ray = ss(lt, 0.6, 1.0) * (1 - ss(lt, _d, _d + 0.5))
    if a_ray > 0.01:
        for i in range(3):
            c = P(ec.disp(GT_C[i]))
            for k, X in enumerate(RAY_PTS[i]):
                u = ss(lt, 0.6 + 0.1 * k + 0.15 * i, 1.4 + 0.1 * k + 0.15 * i)
                q = P(ec.disp(X))
                sub.s.line(c, c + (np.asarray(q) - c) * u, GOLD, 0.7 * a_ray, 1)
    if a_pc > 0.01:
        sub.text((205, 462), "GT: known from the scene script", 20, alpha=a_pc, fill=style.LABEL_GOLD, anchor="ma")
    a_w = ss(lt, 4 * _d, 4 * _d + 0.4)
    if a_w > 0.01:
        ec.star_marker(sub, P(ec.disp(NB["X_star"])), a_w, glow=0.6 + 0.6 * bump(lt, 4 * _d + 0.5, 1.2), rad=5)
    for t0, y, s, size, col in DEF_LINES:
        a = ss(lt, t0, t0 + 0.4)
        if a > 0.01:
            sub.rich(s, (400, y), size, alpha=a, fill=col, anchor="lm")
    a_n = ss(lt, 4 * _d + 1.0, 4 * _d + 1.4)
    if a_n > 0.01:
        sub.rich("From here on, “our test scene” and the tag “Measured on our test scene” mean this test", (480, 530), 22, alpha=a_n,
                 fill=style.LABEL_GOLD, anchor="mm")


@shot.check
def check(ck):
    ck.eq("three cameras", len(sc.cameras()), 3)
    ck.true("camera 1 = world frame", np.allclose(sc.cameras()[0][0], np.eye(3)) and np.allclose(sc.cameras()[0][1], 0))
    ck.true("white dot visible in all three images", all(v for _, v in NB["vis"]))
    ck.le("VGGT camera 1 extrinsic ≈ identity", np.abs(V["extrinsic"][0] - np.c_[np.eye(3), np.zeros(3)]).max(), 1e-3)
    # The P_i drawn is the point head's: mean distance over all valid pixels (about 1.0014)
    norms = np.concatenate([np.linalg.norm(V["world_points"][i][np.isfinite(sc.full(i)[1])], axis=-1)
                            for i in range(3)])
    ck.close("mean distance of point-head points to camera 1 (valid pixels)", norms.mean(),
             VM["points"]["point_head"]["mean_norm_pred"], 1e-3)
    ck.close("track query point = white dot in frame 1", TRACK[0], NB["vis"][0][0], 1e-2)
    for i in (1, 2):
        e = np.linalg.norm(TRACK[i] - NB["vis"][i][0])
        ck.close(f"frame {i + 1} tracking error (px)", e, TRK_ERR[i], 1e-2)
    ck.close("white-dot point-head point = bilinear sample on camera 2", WD_P, VM["white_dot"]["point_head"], 1e-9)
    ws = ec.s01_s02_weight_shapes()
    if ws is None:
        ck.record(weight_header="no VGGT-1B weight file on this machine; T_i channel-count assertion skipped")
    else:
        ck.eq("T_i is 128-dim per pixel (output conv of the track-feature DPT)",
              ws["track_head.feature_extractor.scratch.output_conv1.weight"][0], 128)
    popup_check(ck, [POP_S], DUR)
    ck.true("original timeline resumes correctly after the insert (tm continuous, end = original 20 s)",
            abs(tm(T_INS + INS) - T_INS) < 1e-9 and abs(tm(DUR) - 20.0) < 1e-9)
    ck.true("GT cameras in the popup = the scene script's cameras (camera 1 = world origin)", np.allclose(GT_C[0], 0))
    ck.record(n_points_drawn=int(len(X)), s=NB["s"], track_pred=TRACK.round(2), track_err_px=TRK_ERR,
              white_dot_point_head=WD_P, scene_def_insert=[T_INS, INS])


def thumb_xy(i, uv, dy=0.0):
    return np.array([TH_X[i] + uv[0] * TH_W / sc.W, TH_Y + dy + uv[1] * TH_W / sc.H])


def lit(k, t):
    return ss(t, T_ON[k], T_ON[k] + 0.4)


@shot.frame
def frame(cv, tr):
    t = tm(tr)                                   # original timeline (held at T_INS during the insert)
    draw_main(cv, t)
    # insert sentence 5: the white dot flashes in all three thumbnails (real time)
    a_w = bump(tr, T_INS + 4 * _d + 0.4, 1.6)
    if a_w > 0.01:
        for i in range(3):
            ec.star_marker(cv, thumb_xy(i, NB["vis"][i][0]), a_w, glow=1.0, rad=5)
    POP_S.draw(cv, tr, body_scene)


def draw_main(cv, t):
    # ---- Three images (thumbnails): swap to depth maps at the D sentence, back to the originals at the T sentence ----
    to_depth = ss(t, T_D + 0.1, T_D + 0.8) * (1 - ss(t, T_T + 0.1, T_T + 0.7))
    for i in range(3):
        a = ss(t, 0.3 + 0.3 * i, 0.9 + 0.3 * i)
        dy = 60 * (1 - a)
        if a <= 0.01:
            continue
        x0, y0 = TH_X[i], TH_Y + dy
        ec.paste_img(cv, THUMBS[i], (x0, y0), TH_W, a)
        if to_depth > 0:
            ec.paste_img(cv, DEPTHS[i], (x0, y0), TH_W, a * to_depth)
        col = ec.CAM_COLORS[i]
        box = [(x0, y0), (x0 + TH_W, y0), (x0 + TH_W, y0 + TH_W), (x0, y0 + TH_W)]
        cv.s.polyline(box, col, 0.8 * a, 2, closed=True)
        # When I_i / D_i in the formula lights up, the frames of the three images flash in the matching color
        for k in ("I", "D"):
            fl = bump(t, T_ON[k] + 0.2, 1.4)
            if fl > 0.01:
                cv.s.polyline(box, ACC[k], fl * a, 3, closed=True)
                cv.g.polyline(box, ACC[k], 0.7 * fl * a, 5, closed=True)
        cv.text((x0 + 8, y0 + TH_W + 8), CAM_SHORT[i] + (" (depth)" if to_depth > 0.5 else ""), 26,
                alpha=a, fill=col)
        # Symbol badges on the images: I_1... switch to D_1..., then back to I_1...
        aI = a * lit("I", t) * (1 - to_depth)
        aD = a * to_depth
        if aI > 0.01:
            cv.badge((x0 + 8, y0 + 8), f"$I_{i + 1}$", 28, aI, anchor="la", fill=ACC["I"])
        if aD > 0.01:
            cv.badge((x0 + 8, y0 + 8), f"$D_{i + 1}$", 28, aD, anchor="la", fill=ACC["D"])
        if t < T_T and to_depth < 0.03:              # a low-alpha dot turns into a black dot on a bright image: remove it as soon as the swap starts
            p = thumb_xy(i, NB["vis"][i][0], dy)
            ec.star_marker(cv, p, a * (0.35 + 0.65 * bump(t, 1.6 + 0.15 * i, 0.9)), glow=0.9, rad=4)

    # ---- Paper card ----
    ec.paper_card(cv, (540, 860), fade(t, 2.7, T_EQ, 0.4, 0.4), "2025", "CVPR 2025",
                  "VGGT: Visual Geometry Grounded Transformer", "Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny",
                  "One transformer, one forward pass:\n1 to hundreds of images → cameras, depth, points, track features")

    # ---- Formula (paper Eq. (1)), terms light up one by one; each keeps its color once lit ----
    a_eq = ss(t, T_EQ, T_EQ + 0.5)
    if a_eq > 0:
        pa, pc, pg = {}, {}, {}
        cur = None
        for k in ACC:
            u = lit(k, t)
            pa[k] = 0.3 + 0.7 * u
            pc[k] = ACC[k] if u > 0.5 else EQ_BASE
            pg[k] = 0.6 * bump(t, T_ON[k] + 0.2, 1.0)
            if T_ON[k] <= t < T_ON[k] + 2.0:
                cur = k
        for k in ("f", "eq", "end"):
            pc[k] = EQ_BASE
        box = {cur: min(1.0, 2 * bump(t, T_ON[cur], 2.2))} if cur else None
        EQ.draw(cv, EQ_XY, alpha=a_eq, part_alpha=pa, part_color=pc, part_glow=pg, box=box,
                hl_color=ACC[cur] if cur else ORANGE)
        x0, y0, x1, y1 = EQ.bbox(EQ_XY)
        cv.badge((x0 - 18, EQ_XY[1]), "Paper Eq. (1)", 26, a_eq, anchor="rm", fill=style.LABEL_GOLD,
                 border=(120, 110, 80))
        cv.math(r"N", (x1 + 22, EQ_XY[1]), 30, style.LABEL_GOLD, a_eq, "lm")
        cv.text((x1 + 52, EQ_XY[1]), "images\n= 3", 22, alpha=a_eq, fill=style.TEXT_DIM, anchor="lm", spacing=2)
        # sentence 3: what this equation is (the paper has only two numbered equations)
        a_n = a_eq * (1 - ss(t, T_I - 0.2, T_I + 0.2))
        if a_n > 0.01:
            cv.rich("$f$ = the whole VGGT network. Only two equations are numbered: (1) this definition, (2) the training loss", (540, NOTE_Y), 22,
                    alpha=a_n, fill=style.TEXT_DIM, anchor="mm")
        # Legend: symbol + label, same color, appears with its term
        for (k, sym, lab), x in zip(LEGEND, LEGEND_X):
            al = a_eq * lit(k, t)
            if al > 0.01:
                cv.math(sym, (x, LEG_Y), 30, ACC[k], al, "rm")
                cv.text((x + 8, LEG_Y), lab, 24, alpha=al, fill=ACC[k], anchor="lm")
        # When I_i lights up: three lines from I_i in the formula to the three images
        a_lnk = fade(t, T_I + 0.1, T_G - 0.2, 0.4, 0.4)
        if a_lnk > 0.01:
            bx0, by0, bx1, by1 = EQ.part_box("I", EQ_XY)
            src = ((bx0 + bx1) / 2, by1 + 6)
            for i in range(3):
                arrow(cv, src, (TH_X[i] + TH_W / 2, TH_Y - 6), ACC["I"], 0.7 * a_lnk, 2, 10,
                      progress=ss(t, T_I + 0.1 + 0.15 * i, T_I + 0.7 + 0.15 * i))
        a_lnk = fade(t, T_D + 0.1, T_P - 0.2, 0.4, 0.4)
        if a_lnk > 0.01:
            bx0, by0, bx1, by1 = EQ.part_box("D", EQ_XY)
            src = ((bx0 + bx1) / 2, by1 + 6)
            for i in range(3):
                arrow(cv, src, (TH_X[i] + TH_W / 2, TH_Y - 6), ACC["D"], 0.7 * a_lnk, 2, 10,
                      progress=ss(t, T_D + 0.1 + 0.15 * i, T_D + 0.7 + 0.15 * i))

    # ---- 3D: measured VGGT camera frusta (g_i) + point-head point cloud (P_i) ----
    az = AZ0 + (AZ1 - AZ0) * ss(t, T_G, DUR)
    P = OC.at(az=az)
    for i in range(3):
        a = ss(t, T_G + 0.2 + 0.3 * i, T_G + 0.8 + 0.3 * i)
        C = ec.frustum_pose(cv, P, R_P[i], C_P[i], PLANE_D, a, ec.CAM_COLORS[i], image=ec.view_rgb(i),
                            tex_alpha=0.9 * (1 - 0.6 * ss(t, T_P + 0.4, T_P + 1.2)), label=CAM_SHORT[i],
                            K=K_P[i])
        if C is not None:
            fl = bump(t, T_G + 0.5 + 0.3 * i, 1.4)
            cv.g.dot(C, ACC["g"], 9, 0.9 * fl * a)
            left = C[0] < 540                       # label on the side away from the scene center, so it does not cover the frustum
            cv.rich(f"$g_{i + 1}$", (C[0] + (-14 if left else 14), C[1] - 6), 32, alpha=a, fill=ACC["g"],
                    anchor="rm" if left else "lm")
    a_pc = ss(t, T_P + 0.1, T_P + 0.5)
    if a_pc > 0:
        u = np.clip((t - T_P - 0.2 - 0.25 * CID) / 1.3, 0, 1)
        u = u * u * (3 - 2 * u)
        pts = START + (X - START) * u[:, None]
        ec.splat(cv, P, pts, RGB, a_pc, size=2)
        cv.rich("$P_i$: one 3D point per pixel; points of all three images are in camera 1's coordinate frame", (540, 700), 24,
                alpha=a_pc * ss(t, T_P + 0.6, T_P + 1.0) * (1 - ss(t, T_T, T_T + 0.4)), fill=ACC["P"], anchor="mm")
    a_star = ss(t, T_P + 1.6, T_P + 2.0)
    if a_star > 0:
        ec.star_marker(cv, P(ec.disp(WD_P)), a_star, glow=0.9, rad=5)

    # ---- T_i: the white dot's position in the three images (measured track head), joined by a gold trajectory ----
    a_t = ss(t, T_T + 0.3, T_T + 0.8)
    if a_t > 0:
        pts2 = [thumb_xy(i, TRACK[i]) for i in range(3)]
        for i in range(2):
            arrow(cv, pts2[i], pts2[i + 1], ACC["T"], a_t, 2, 12, bend=-0.18,
                  progress=ss(t, T_T + 0.6 + 0.6 * i, T_T + 1.2 + 0.6 * i))
        for i in range(3):
            ai = a_t * ss(t, T_T + 0.4 + 0.6 * i, T_T + 0.8 + 0.6 * i)
            ec.star_marker(cv, pts2[i], ai, glow=0.6, rad=4)
            cv.badge((TH_X[i] + TH_W - 8, TH_Y + 8), f"$T_{i + 1}$", 28, ai, anchor="ra", fill=ACC["T"])
        a_l = ss(t, T_T + 1.4, T_T + 1.8)
        cv.rich("$T_i$: per-pixel track features of image $i$ (128 dims), not the track itself", (540, 690), 23,
                alpha=a_l, fill=ACC["T"], anchor="mm")
        cv.rich(f"Gold line = white dot found by the tracker from $T_i$ (measured error: frame 2 ${TRK_ERR[1]:.1f}$ px, "
                f"frame 3 ${TRK_ERR[2]:.1f}$ px)", (540, 724), 22, alpha=a_l, fill=style.TEXT_DIM, anchor="mm")
