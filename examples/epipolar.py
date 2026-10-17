"""Regression shot: an epipolar-geometry scene on animkit (the formula goes through mathtext).
Check: rendered epipolar points lie on F x1 to ~1e-13 px.
  python -I -X utf8 tools/render_shot.py examples/epipolar.py
"""
import numpy as np

from animkit import (CYAN, GOLD, GREY, ORANGE, PANEL, WHITE, Formula, OrbitCam, PinholeCam, Shot, draw_camera, fade,
                     floor_grid, fundamental, ss, style)

DUR = 16.0
shot = Shot("epipolar", DUR)

cam1 = PinholeCam((-1.4, 0.0, 0.0), (0.0, 0.2, 2.4))
cam2 = PinholeCam((1.4, 0.0, 0.0), (0.0, 0.2, 2.4))
X1_PIX = np.array([420.0, 205.0])
FAN_PIX = [np.array([250.0, 120.0]), np.array([300.0, 380.0]), np.array([400.0, 300.0])]
S_FAR = 6.0

F = fundamental(cam1, cam2)
e2_pix, _ = cam2.pix(cam1.C)
E2_WORLD = cam2.plane_pt(*e2_pix)


def s_range(cam_src, pix, s_far=S_FAR, z_min=0.3):
    """Depths along cam_src's ray that sit in front of cam2 and project inside image 2."""
    d = cam_src.ray_dir(pix)
    ok = []
    for s in np.linspace(0.2, s_far, 400):
        p, z = cam2.pix(cam_src.C + s * d)
        if z > z_min and 0 <= p[0] <= cam2.IMG_W and 0 <= p[1] <= cam2.IMG_H:
            ok.append(s)
    return (min(ok), max(ok)) if ok else (None, None)


d1 = cam1.ray_dir(X1_PIX)
S_LO, S_HI = s_range(cam1, X1_PIX)
fan = [(px, cam1.ray_dir(px), *s_range(cam1, px)) for px in FAN_PIX]

view = OrbitCam((0.0, 0.35, 2.0), 6.8, el=30)
_key = [cam1.C, cam2.C, *cam1.corners(), *cam2.corners(), cam1.C + d1 * S_FAR, E2_WORLD] + \
       [cam1.C + f[1] * S_FAR for f in fan]
view.fit(_key, 900, 760, (style.W / 2, 760), az=-7)
GRID = floor_grid()
EQ = Formula([r"x_2^{\mathsf{T}}", "F", "x_1", "=", "0"], size=84)

shot.title("Epipolar Geometry", "Two-view geometry", 0.2, 3.8)
for a, b, s in [
    (1.2, 4.4, "A left pixel says only:\nthe 3D point is on this ray"),
    (4.4, 8.6, "Slide the point: its right image\ntraces a line, the epipolar line"),
    (8.6, 11.2, "Centers and ray span a plane,\ncutting image 2 in a line"),
    (11.2, 13.6, "New pixel, new line;\nall lines meet at the epipole"),
    (13.6, 16.0, "So search along the line,\nnot the whole image: 2D becomes 1D"),
]:
    shot.caption(a, b, s)
shot.sheet_times = [1.0, 2.5, 3.6, 5.5, 7.0, 8.5, 9.8, 11.0, 12.0, 13.2, 14.5, 15.8]


@shot.check
def check(ck):
    assert S_LO is not None and all(f[2] is not None for f in fan), "a ray is not visible in image 2"
    ck.eq("rank F", int(np.linalg.matrix_rank(F, tol=1e-9)), 2)
    lines = []
    for name, px, d, lo, hi in [("x1", X1_PIX, d1, S_LO, S_HI)] + [(f"fan{k}", *f) for k, f in enumerate(fan)]:
        l2 = F @ np.r_[px, 1]
        l2 /= np.linalg.norm(l2[:2])
        pts = [cam2.pix(cam1.C + s * d)[0] for s in np.linspace(lo, hi, 50)]
        dist = max(abs(l2 @ np.r_[p, 1]) for p in pts)      # pixel distance of rendered points to F x1
        e_dist = abs(l2 @ np.r_[e2_pix, 1])
        ck.le(f"{name}: max px dist of rendered epipolar points to F x1", dist, 1e-9)
        ck.le(f"{name}: px dist of epipole e2 to F x1", e_dist, 1e-9)
        lines.append({"pixel": px.tolist(), "s_range": [round(lo, 2), round(hi, 2)], "max_px_dist_to_Fx1": dist,
                      "e2_px_dist_to_line": e_dist})
    ck.record(F_rank=int(np.linalg.matrix_rank(F, tol=1e-9)), e2_pix=e2_pix.round(1), lines=lines)


@shot.frame
def frame(cv, t):
    P = view.at(az=-14 + 14 * t / DUR)
    a_cam = ss(t, 0.0, 1.2)
    for a, b in GRID:
        cv.s.line(P(a), P(b), GREY, 0.10 * a_cam, 1)
    for cam in (cam1, cam2):
        draw_camera(cv, cam, P, a_cam)

    x1w = cam1.plane_pt(*X1_PIX)
    a_x1 = ss(t, 1.2, 1.7)
    grow = ss(t, 2.0, 3.8)
    if grow > 0:
        end = cam1.C + d1 * (S_FAR * grow)
        cv.s.line(P(cam1.C), P(end), GOLD, 0.9, 2)
        cv.g.line(P(cam1.C), P(end), GOLD, 0.8, 5)

    slide = ss(t, 4.6, 8.2)
    if t >= 4.4:
        s_now = S_LO + (S_HI - S_LO) * slide
        trail = [P(cam2.hit_plane(cam1.C + s * d1)[0]) for s in np.linspace(S_LO, s_now, 120)]
        for p, q in zip(trail[:-1], trail[1:]):
            cv.s.line(p, q, CYAN, 1.0, 3)
            cv.g.line(p, q, CYAN, 1.0, 7)
        for s in np.linspace(S_LO, S_HI, 7):
            if s <= s_now:
                cv.s.dot(P(cam1.C + s * d1), GOLD, 3, 0.6)
        X = cam1.C + s_now * d1
        hit = cam2.hit_plane(X)[0]
        a_proj = fade(t, 4.4, 9.0)
        cv.s.line(P(X), P(cam2.C), CYAN, 0.45 * a_proj, 1)
        cv.g.dot(P(X), GOLD, 9, a_proj)
        cv.s.dot(P(X), (255, 255, 255), 4, a_proj)
        cv.g.dot(P(hit), CYAN, 7, a_proj)
        cv.s.dot(P(hit), (255, 255, 255), 3, a_proj)

    a_pl = ss(t, 8.6, 9.6)
    if a_pl > 0:
        far = cam1.C + d1 * S_FAR
        ext = [cam2.plane_pt(e2_pix[0] - 60, 0), cam2.plane_pt(0, 0), cam2.plane_pt(0, cam2.IMG_H),
               cam2.plane_pt(e2_pix[0] - 60, cam2.IMG_H)]
        cv.s.add_poly([P(c) for c in ext], PANEL, 0.18 * a_pl)
        cv.s.add_poly([P(cam1.C), P(cam2.C), P(far)], ORANGE, 0.16 * a_pl * (1 - 0.6 * ss(t, 11.2, 11.8)))
        cv.s.dashed(P(cam1.C), P(E2_WORLD), ORANGE, 0.9 * a_pl, 2)
        start = cam2.hit_plane(cam1.C + S_LO * d1)[0]
        cv.s.dashed(P(start), P(E2_WORLD), CYAN, 0.8 * a_pl, 2, n=12)
        cv.g.dot(P(E2_WORLD), ORANGE, 10, a_pl)
        cv.s.dot(P(E2_WORLD), (255, 255, 255), 4, a_pl)

    for k, (px, d, lo, hi) in enumerate(fan):
        a0 = 11.3 + 0.55 * k
        a_f = ss(t, a0, a0 + 0.5)
        if a_f <= 0:
            continue
        cv.g.dot(P(cam1.plane_pt(*px)), ORANGE, 6, a_f)
        cv.s.line(P(cam1.C), P(cam1.C + d * S_FAR * a_f), GOLD, 0.35, 1)
        a_hit = cam2.hit_plane(cam1.C + lo * d)[0]
        b_hit = cam2.hit_plane(cam1.C + hi * d)[0]
        cv.s.line(P(a_hit), P(b_hit), CYAN, a_f, 2)
        cv.g.line(P(a_hit), P(b_hit), CYAN, 0.7 * a_f, 5)
        cv.s.dashed(P(a_hit), P(E2_WORLD), CYAN, 0.5 * a_f, 1, n=10)

    cv.g.dot(P(x1w), ORANGE, 8 + 2 * np.sin(t * 6) * (t < 4.4), a_x1)
    cv.s.dot(P(x1w), (255, 255, 255), 3.5, a_x1)

    # labels
    cv.text(P(cam1.C) + [-30, 18], "camera 1", 30, alpha=a_cam, anchor="ra")
    cv.text(P(cam2.C) + [24, 10], "camera 2", 30, alpha=a_cam)
    cv.math("x_1", P(x1w) + [30, -26], 40, style.LABEL_GOLD, a_x1)
    if 4.6 <= t <= 8.6:
        Xs = cam1.C + (S_LO + (S_HI - S_LO) * ss(t, 4.6, 8.2)) * d1
        cv.math("X\\,?", P(Xs) + [36, -26], 36, (255, 220, 150), fade(t, 4.6, 8.6))
    mid = cam2.hit_plane(cam1.C + 0.5 * (S_LO + S_HI) * d1)[0]
    cv.text(P(mid) + [16, 4], "epipolar", 30, alpha=ss(t, 7.8, 8.4), fill=style.LABEL_CYAN)
    cv.math("e_2", P(E2_WORLD) + [30, -30], 40, style.LABEL_ORANGE, ss(t, 9.0, 9.6))
    cv.text(P(0.5 * (cam1.C + cam2.C)) + [0, 14], "baseline", 30, alpha=fade(t, 9.0, 11.4), fill=style.LABEL_ORANGE)
    a_eq = ss(t, 13.6, 14.3)
    # light the terms one by one: x2^T, F, x1, then "= 0"
    lit = {i: ss(t, 13.6 + 0.25 * i, 13.9 + 0.25 * i) for i in range(5)}
    EQ.draw(cv, (style.W / 2, 200), alpha=a_eq, part_alpha=lit, part_glow={1: 0.5 * lit[1]})
    cv.text((style.W / 2, 262), "Matches obey (F = fundamental matrix):", 30, alpha=a_eq, fill=style.TEXT_DIM, anchor="ma")
