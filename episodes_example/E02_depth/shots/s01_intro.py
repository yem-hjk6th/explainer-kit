"""E02-01 This episode: the same scene, depth only (S, about 20.5 s). Storyboard storyboard.md section E02-01.
Computed for real: the three images (ray-cast rendering), the white dot's pixel in each image (projection + ray cast hits the red box), the small GT 3D view in the popup (GT cameras + rays + GT point cloud).
Illustration: none. Paper numbers: none (the two paper cards are text only: Depth Pro Abstract, DA3 Abstract + section 3).
Tag: c2.GT_TAG is shown from #1a to the end (everything drawn here is GT, not model output).
"""
import numpy as np

import e02_common as c2
from animkit import GOLD, Popup, Shot, arrow, bump, fade, popup_check, ss, style

ec, sc = c2.ec, c2.sc
NB = ec.nb()                                 # numbers from the scene script (the white dot's pixel in the three cameras is in NB["vis"])

DUR = 22.5
T_POP0, T_POP1 = 3.0, 10.6                   # popup: from #1a to the end of #1b
T_1B = 7.4                                   # #1b starts (white dots flash together)
T_C2, T_C3 = 10.6, 14.6                      # two paper cards

CAPS = [(0.3, 3.0, "Same three images; only depth this time"),
        (3.0, 7.2, "E01 test, not a paper result"),
        (7.4, 10.4, "White dot (tracked)"),
        (10.6, 14.4, "Depth Pro: one image, metric depth"),
        (14.6, 18.8, "DA3: N images, depth + rays"),
        (19.0, 22.3, "Measured on our scene: official weights")]
T_MEAS = 19.0                                # the "Measured on our scene" note line appears

shot = Shot("s01_intro", DUR)
shot.title("Depth only", "E02 · Depth Pro × Depth Anything 3", 0.2, 2.5)
shot.tag(c2.GT_TAG, T_POP0, DUR)
for a, b, s in CAPS:
    shot.caption(a, b, s)

TH_W, TH_Y = 300, 320
TH_X = [70, 390, 710]

# ---------------- popup: what "our test scene" is ----------------
POP = Popup((60, 668, 960, 572), t_in=T_POP0, t_out=T_POP1, title="Our test scene = a test we built (not a paper experiment)",
            corner="tl", accent=GOLD)
GT_X, GT_RGB, GT_CID, _ = ec.cloud(6)
GT_C = [c for _, _, c in sc.cameras()]
OC_S = ec.viewer(az=-25, el=26, r=4.5, fit_pts=np.r_[GT_X, GT_C], box=(290, 320), center=(205, 270),
                 target=ec.scene_center())
_rng = np.random.default_rng(7)
RAY_PTS = [GT_X[_rng.choice(np.flatnonzero(GT_CID == i), 5, replace=False)] for i in range(3)]
LT_1B = T_1B - T_POP0                        # start of #1b in popup-local time
ROWS = [  # (local time, y, text, font size, color)
    (0.4, 80, "Toy scene: floor, red box, 2 balls", 22, style.TEXT),
    (1.2, 130, "3 virtual cameras, ray casting\nRenders three 518×518 images", 22, style.TEXT),
    (2.2, 210, "Full GT: depth of every pixel,\neach camera's pose", 22, style.LABEL_GOLD),
    (3.4, 300, "No model involved: these are\nknown up front: the answer key", 21, style.TEXT_DIM),
    (LT_1B + 0.3, 400, "White dot = a 3D point on the red box front,", 22, style.TEXT),
    (LT_1B + 0.6, 432, "the same one tracked in E01", 22, style.TEXT),
]


def body_scene(sub, lt, box):
    P = OC_S.at(az=-25 + 20 * ss(lt, 0.0, T_POP1 - T_POP0))
    a0 = ss(lt, 0.2, 0.6)
    a_pc = ss(lt, 2.2, 3.0)
    ec.splat(sub, P, GT_X, GT_RGB, (0.25 + 0.75 * a_pc) * a0, size=2)
    for i in range(3):
        ec.frustum(sub, P, i, 0.45, a0 * ss(lt, 0.3 + 0.2 * i, 0.7 + 0.2 * i), tex_alpha=0.8, label=ec.CAM_NAMES[i])
    a_ray = ss(lt, 1.0, 1.4) * (1 - ss(lt, 2.2, 2.7))
    if a_ray > 0.01:
        for i in range(3):
            c = P(ec.disp(GT_C[i]))
            for k, X in enumerate(RAY_PTS[i]):
                u = ss(lt, 1.0 + 0.1 * k + 0.15 * i, 1.8 + 0.1 * k + 0.15 * i)
                q = P(ec.disp(X))
                sub.s.line(c, c + (np.asarray(q) - c) * u, GOLD, 0.7 * a_ray, 1)
    if a_pc > 0.01:
        sub.text((205, 462), "GT: known from the scene script", 20, alpha=a_pc, fill=style.LABEL_GOLD, anchor="ma")
    a_w = ss(lt, LT_1B, LT_1B + 0.4)
    if a_w > 0.01:
        ec.star_marker(sub, P(ec.disp(NB["X_star"])), a_w, glow=0.6 + 0.8 * bump(lt, LT_1B + 0.2, 1.6), rad=5)
    for t0, y, s, size, col in ROWS:
        a = ss(lt, t0, t0 + 0.4)
        if a > 0.01:
            for k, line in enumerate(s.split("\n")):
                sub.rich(line, (400, y + k * (size + 10)), size, alpha=a, fill=col, anchor="la")


@shot.check
def check(ck):
    ck.eq("three cameras", len(sc.cameras()), 3)
    ck.eq("image size = 518 (popup text '518×518')", f"{sc.W}×{sc.H}", "518×518")
    for i in range(3):
        uv, vis = NB["vis"][i]
        R, t, _ = sc.cameras()[i]
        p, Xc = sc.project(R, t, sc.X_STAR)
        ck.true(f"camera {i + 1} white-dot pixel is inside the image", 0 <= p[0] < sc.W and 0 <= p[1] < sc.H, f"{p.round(1)}")
        ck.close(f"camera {i + 1} white-dot pixel = NB", p, uv, 1e-6)
        ck.true(f"camera {i + 1} depth is finite", np.isfinite(Xc[2]) and Xc[2] > 0, f"z={Xc[2]:.3f}")
        _, tz, Xh = sc.render(i, p[None])
        ck.true(f"camera {i + 1} ray cast hits the red box (hit point = white dot)",
                np.isfinite(tz[0]) and np.allclose(Xh[0], sc.X_STAR, atol=1e-6), f"z={tz[0]:.3f}")
        ck.true(f"camera {i + 1} white dot is not occluded", bool(vis))
    ck.true("camera 1 = world frame", np.allclose(sc.cameras()[0][0], np.eye(3)) and np.allclose(GT_C[0], 0))
    ck.true("#1b start lies within the popup window", 0 < LT_1B < T_POP1 - T_POP0)
    popup_check(ck, [POP], DUR)
    ck.record(white_dot_px=[np.round(NB["vis"][i][0], 1).tolist() for i in range(3)])


def card(cv, alpha, year, venue, title, authors, line, color, cy=880):
    """Same layout as ec.paper_card; the left color bar = the paper's accent color."""
    w, h = 960, 380
    x0, y0 = 540 - w / 2, cy - h / 2
    cv.s.rect(x0, y0, x0 + w, y0 + h, (22, 26, 34), 0.92 * alpha, r=18)
    cv.s.polyline([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], (90, 100, 115), 0.8 * alpha, 2,
                  closed=True)
    cv.s.rect(x0 + 258, y0 + 40, x0 + 266, y0 + 140, color, alpha, r=3)
    cv.text((x0 + 40, y0 + 34), year, 96, "light", alpha, style.LABEL_GOLD)
    cv.text((x0 + 290, y0 + 62), venue, 34, "light", alpha, style.TEXT_DIM)
    cv.text((x0 + 290, y0 + 108), authors, 24, "text", 0.85 * alpha, style.TEXT_DIM)
    cv.text((x0 + 40, y0 + 172), title, 34, "bold", alpha, spacing=8)
    cv.text((x0 + 40, y0 + 240), line, 32, "text", alpha, style.TEXT, spacing=12)


def thumb_xy(i, uv):
    return np.array([TH_X[i] + uv[0] * TH_W / sc.W, TH_Y + uv[1] * TH_W / sc.H])


@shot.frame
def frame(cv, t):
    # ---- the three images slide in one after another ----
    flash = bump(t, 1.6, 1.0) + bump(t, T_1B + 0.2, 1.8)
    for i in range(3):
        a = ss(t, 0.3 + 0.3 * i, 0.9 + 0.3 * i)
        if a <= 0.01:
            continue
        dy = 60 * (1 - a)
        x0, y0 = TH_X[i], TH_Y + dy
        ec.paste_img(cv, ec.view_pil(i), (x0, y0), TH_W, a)
        col = ec.CAM_COLORS[i]
        box = [(x0, y0), (x0 + TH_W, y0), (x0 + TH_W, y0 + TH_W), (x0, y0 + TH_W)]
        cv.s.polyline(box, col, 0.8 * a, 2, closed=True)
        cv.text((x0 + 8, y0 + TH_W + 8), ec.CAM_NAMES[i], 26, alpha=a, fill=col)
        p = thumb_xy(i, NB["vis"][i][0]) + np.array([0, dy])
        if flash > 0.01:
            ec.star_marker(cv, p, a * min(1.0, flash), glow=1.0, rad=5)

    # ---- two paper cards ----
    a2 = fade(t, T_C2, T_C3, 0.4, 0.4)
    if a2 > 0.01:
        card(cv, a2, "2024", "ICLR 2025", "Depth Pro", "Bochkovskii et al. (Apple)",
             "One image, no camera intrinsics,\n1536×1536 metric depth in 0.3 s", c2.DP_COLOR)
    a3 = fade(t, T_C3, DUR, 0.4, 0.0)
    if a3 > 0.01:
        card(cv, a3, "2025", "arXiv 2511.10647", "Depth Anything 3", "Lin et al. (ByteDance Seed)",
             "Any number of images, pose optional,\na plain ViT outputs depth + rays", c2.DA3_COLOR)

    a4 = ss(t, T_MEAS, T_MEAS + 0.5)
    if a4 > 0.01:
        cv.text((540, 1105), "Measured on our scene = feeding these renders to", 30, "text", a4, style.LABEL_GOLD, anchor="ma")
        cv.text((540, 1150), "the official Depth Pro and DA3 weights, as run", 30, "text", a4, style.LABEL_GOLD, anchor="ma")

    POP.draw(cv, t, body_scene)
