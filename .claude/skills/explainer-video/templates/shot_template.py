"""Template shot: rotate a 2D vector by a rotation matrix (12 s). Copy to <EP>/shots/sNN_<name>.py and replace.

Storyboard: <EP>-NN (section title in storyboard.md contains the id).
Computed: v' = R(theta) v, its coordinates, length preserved (asserted in check()).
Illustrative: none.  Paper numbers: none.

Run:  python -I -B -X utf8 tools/render_shot.py <this file> --check-only | --still 6 11 | --sheet-only | --jobs 2
"""
import numpy as np

from animkit import CYAN, GOLD, GREY, ORANGE, Formula, Shot, arrow, bump, number, ss

DUR = 12.0
shot = Shot("rotate2d", DUR)                    # name, seconds; 1080x1440 @ 30 fps
shot.title("2D rotation", "Rotating a vector by a matrix")   # title (bold), subtitle (light)

# Captions: final wording from the storyboard table, one per visual change, <= 12 words and one frame width
# (render warns "a line is wider than the frame"; long lines without \n are split into 2 automatically)
CAPS = [
    (0.4, 3.0, "Vector v points to (2, 1)"),
    (3.0, 5.8, "Rotate by 30 degrees: the length stays the same"),
    (5.8, 8.8, "New coordinates = rotation matrix R times v"),
    (8.8, DUR, "Result: (1.232, 1.866)"),
]
for a, b, s in CAPS:
    shot.caption(a, b, s)

# ---- numbers: computed once at full precision, formatted only for display ----
V = np.array([2.0, 1.0])
THETA = np.deg2rad(30.0)


def R(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


VP = R(THETA) @ V
SHOW = (f"{VP[0]:.3f}", f"{VP[1]:.3f}")         # the exact strings that appear on screen

# ---- layout: math coordinates -> screen pixels (y up on screen) ----
O, S = np.array([440.0, 900.0]), 170.0


def px(p):
    return (O[0] + S * p[0], O[1] - S * p[1])


EQ = Formula([("v'", "vp"), "=", (r"R(\theta)", "R"), ("v", "v")], size=72)   # built at module level (cached)


@shot.check
def check(ck):
    ck.close("rotation keeps length", float(np.linalg.norm(VP)), float(np.linalg.norm(V)), 1e-12)
    ck.close("R is orthonormal", R(THETA) @ R(THETA).T, np.eye(2), 1e-12)
    ck.eq("x' on screen", SHOW[0], "1.232")
    ck.eq("y' on screen", SHOW[1], "1.866")
    ck.true("caption 4 shows the same numbers", f"({SHOW[0]}, {SHOW[1]})" in CAPS[3][2])


@shot.frame
def frame(cv, t):
    a0 = ss(t, 0.3, 0.9)
    # axes and grid (white / grey = scaffolding)
    cv.s.line(px((-1.2, 0)), px((3.0, 0)), GREY, 0.8 * a0, 2)
    cv.s.line(px((0, -0.6)), px((0, 2.6)), GREY, 0.8 * a0, 2)
    cv.text(px((3.05, -0.05)), "x", 30, kind="math", alpha=a0, anchor="lm")
    cv.text(px((0.05, 2.65)), "y", 30, kind="math", alpha=a0, anchor="lb")

    # the object being asked about: v (gold, glowing = protagonist)
    av = ss(t, 0.6, 1.4)
    arrow(cv, px((0, 0)), px(V), GOLD, alpha=av, th=4, progress=av, glow=0.8)
    cv.rich("$v=(2,\\ 1)$", (px(V)[0] + 16, px(V)[1] - 10), 34, alpha=av, anchor="lb")

    # rotation by theta: the rotated copy sweeps from 0 to 30 degrees
    u = ss(t, 3.2, 5.4)
    if u > 0.02:
        vp = R(u * THETA) @ V
        arrow(cv, px((0, 0)), px(vp), CYAN, alpha=1.0, th=4, glow=0.6)
        cv.rich("$v'$", (px(vp)[0] - 50, px(vp)[1] - 8), 34, alpha=u, anchor="rb")
        arc = [px(0.9 * R(k * u * THETA) @ (V / np.linalg.norm(V))) for k in np.linspace(0, 1, 24)]
        cv.s.polyline(arc, ORANGE, 0.9, 3)
        cv.rich(f"$\\theta={u * 30:.0f}^\\circ$", (px(0.95 * V / np.linalg.norm(V))[0] + 40, px(V)[1] - 70), 30,
                alpha=u, anchor="lb")
        cv.rich("$|v'| = |v|$", (760, 1060), 32, alpha=ss(t, 4.6, 5.2), anchor="mm")

    # formula after the picture, lit term by term, each term in the color of its picture element
    af = ss(t, 5.8, 6.4)
    if af > 0:
        st = {"vp": "lit" if t > 8.8 else "dim", "R": "hl" if 6.4 < t < 8.8 else "lit", "v": "lit"}
        EQ.draw(cv, (540, 330), alpha=af, state=st, part_color={"vp": CYAN, "v": GOLD},
                part_glow={"R": 0.8 * bump(t, 6.6, 0.8)},
                notes={"R": "$R(\\theta)$: the $2\\times2$ matrix that turns the plane by $\\theta$"})

    # computed result, shown with the same strings the check asserted
    an = ss(t, 9.0, 9.6)
    if an > 0:
        tip = px(VP)
        number(cv, (tip[0] + 80, tip[1] - 30), VP[0], fmt="{:.3f}", size=36, color=CYAN, alpha=an)
        number(cv, (tip[0] + 220, tip[1] - 30), VP[1], fmt="{:.3f}", size=36, color=CYAN, alpha=an)
