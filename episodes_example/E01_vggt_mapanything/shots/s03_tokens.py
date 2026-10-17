"""E01-03 Token composition: 1 + 4 + 1369 = 1374 per image (S, 15 s). Storyboard storyboard.md section E01-03.
Computed for real: token counts, the white-dot token's index 799 = 5 + 794 (official code order: camera, 4x register, patch).
Illustration: what is inside a token (color only marks the category); the small camera at the left of each row --
      color = frame number (fixed colors of cameras 1/2/3, ec.CAM_COLORS); shape and orientation do not correspond to real poses.
"""
import numpy as np

import e01_common as ec
from animkit import GOLD, VIOLET, Formula, Shot, TokenStrip, arrow, bump, fade, fold, ss, style, token

DUR = 15.0
shot = Shot("s03_tokens", DUR)
shot.title("Tokens per image", "Tokens · VGGT §3.3", 0.2, 2.6)
shot.tag("Illustration: token color = type; small camera color = frame no.", 0.3, DUR)
shot.tag("Ordered as in the official code (the paper writes 794 / 2168)", 8.0, DUR)
CAPS = [(0.3, 2.8, "Add 5 more tokens in front of each image"), (2.8, 5.4, "1 camera token collects camera info"),
        (5.4, 8.0, "4 register tokens, dropped at the end"), (8.0, 10.6, "The white dot's token is at position 799"),
        (10.6, 13.0, "Image 1 uses a different set of tokens"), (13.0, DUR, "The network uses it to identify the reference camera")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

NB = ec.nb()
KP, KT = NB["K_patch"], NB["K_tok"]
W_IDX = NB["tok_idx_code"]
STRIP = TokenStrip(KT, show=fold(KT, head=9, tail=3, force=[W_IDX]), rows=3, center=(615, 720), size=38,
                   row_pitch=110)
ROW2 = 1                                                    # frame 2 = row 1 (0-based)

EQ = Formula([(r"\tilde K", "Kt"), ("=1", "cam"), ("+4", "reg"), ("+1369", "K"), ("=1374", "res")], size=60)
EQ_XY = (540, 330)
SHAPE = Formula([(r"(N,\ \tilde K,\ C)=(3,\ 1374,\ 1024)", "sh")], size=44)


ICON_X, ICON_S = 92, 1.5                                  # center x and scale of the small camera at the left of each row (left of the "Fr. N" label)


def cam_icon(cv, c, color, alpha, axes=0.0, s=ICON_S):
    """Small camera (2D frustum sketch): optical center on the left, image plane on the right; with axes > 0 it draws three short x, y, z axes at the optical center (reference camera)."""
    if alpha <= 0.01:
        return
    c = np.asarray(c, float)
    A = c + np.array([-24, 0]) * s
    quad = [c + np.array(q) * s for q in ([8, -19], [24, -13], [24, 13], [8, 19])]
    cv.s.add_poly(quad, style.PANEL, 0.5 * alpha)
    for q in quad:
        cv.s.line(A, q, color, 0.6 * alpha, 1)
    cv.s.polyline(quad, color, alpha, 2, closed=True)
    cv.s.dot(A, color, 3.5, alpha)
    if axes > 0.01:
        # Side view: z toward the image plane (right), y down, x out of the screen (drawn as an up-left diagonal), OpenCV convention
        for d, col, lab in (((-9, -9), ec.AXIS_COLORS[0], "x"),((0, 22), ec.AXIS_COLORS[1], "y"),
                            ((56, 0), ec.AXIS_COLORS[2], "z")):
            e = A + np.asarray(d, float) * s
            cv.s.line(A, e, col, axes * alpha, 2)
            cv.text(e + [2 if d[0] >= 0 else -2, 0], lab, 16, "light", axes * alpha, col, "lm" if d[0] >= 0 else "rm")


def kind(r, i):
    if i == 0:
        return "camera1" if r == 0 else "camera"
    if i <= 4:
        return "register1" if r == 0 else "register"
    return "patch"


@shot.check
def check(ck):
    ck.eq("tokens per frame = 1 + 4 + 1369", KT, 1 + 4 + KP)
    ck.eq("white-dot token index = 5 + patch index", W_IDX, 5 + NB["patch_idx"])
    ck.eq("global index = 1×1374 + 799", NB["tok_idx_global"], 1 * KT + W_IDX)
    ck.true("token 799 is still drawn after folding", W_IDX in STRIP.shown)
    ck.record(K_tok=KT, white_token=W_IDX, white_global=NB["tok_idx_global"], paper_order=[794, KT + 794])


def draw_row(cv, r, t, a_patch, a_spec, spec_dy=0.0, hl=None):
    for i in STRIP.shown:
        p = STRIP.xy(i, r)
        if i < 5:
            token(cv, p + [0, spec_dy], STRIP.size, style.TOKEN[kind(r, i)], a_spec,
                  glow=0.5 * (hl or {}).get(i, 0.0))
        else:
            token(cv, p, STRIP.size, style.TOKEN[kind(r, i)], a_patch)
    for gx in STRIP.gap_x:
        cv.text((gx, STRIP.y[r]), "…", 26, "light", a_patch, style.TEXT_DIM, "mm", shadow=False)


@shot.frame
def frame(cv, t):
    # formula K~ = 1 + 4 + 1369 = 1374
    a_eq = ss(t, 1.8, 2.3)
    st = {"Kt": "lit", "K": "lit" if t > 1.8 else "dim", "cam": "lit" if t > 3.0 else "dim",
          "reg": "lit" if t > 5.6 else "dim", "res": "hl" if 7.0 < t < 8.6 else ("lit" if t > 7.0 else "dim")}
    EQ.draw(cv, EQ_XY, alpha=a_eq, state=st, notes={"Kt": "Tokens per image", "K": "patches (02)"})

    # The frame-2 row: patches are already there, the 5 special tokens drop in from above
    a_patch = ss(t, 0.3, 0.8)
    for r in range(3):
        if r == ROW2:
            a_spec = ss(t, 1.0, 1.8)
            hl = {0: bump(t, 3.4, 1.2)}
            hl.update({k: bump(t, 6.0, 1.2) for k in range(1, 5)})
            draw_row(cv, r, t, a_patch, a_spec, -40 * (1 - a_spec), hl)
        else:
            a_r = ss(t, 10.8 + 0.3 * (r == 2), 11.4 + 0.3 * (r == 2))
            draw_row(cv, r, t, a_r, a_r, 0.0, {k: bump(t, 11.6, 1.2) for k in range(5)} if r == 0 else None)
    x_left = min(STRIP.x.values()) - 40
    for r in range(3):
        a_l = a_patch if r == ROW2 else ss(t, 10.8, 11.4)
        cv.text((x_left, STRIP.y[r]), f"Fr. {r + 1}", 28, alpha=a_l, fill=ec.CAM_COLORS[r], anchor="rm")

    # A small camera at the left of each row (color = frame number), camera token -> camera head -> this frame's camera g_i
    for r in range(3):
        y = STRIP.y[r]
        a_ic = ss(t, 3.4, 3.9) if r == ROW2 else ss(t, 11.0 + 0.3 * (r == 2), 11.5 + 0.3 * (r == 2))
        if a_ic <= 0.01:
            continue
        col = ec.CAM_COLORS[r]
        cam_icon(cv, (ICON_X, y), col, a_ic, axes=ss(t, 13.1, 13.6) if r == 0 else 0.0)
        pulse = bump(t, 3.6, 1.2) if r == ROW2 else bump(t, 11.8, 1.2)
        cv.g.dot((ICON_X - 24 * ICON_S, y), col, 9, 0.8 * pulse * a_ic)
        cv.rich(f"$g_{r + 1}$", (ICON_X + 14 * ICON_S, y + 21 * ICON_S), 24, alpha=a_ic, fill=col, anchor="la")
        # Arc from the camera token to the small camera: bright while the camera token is discussed, a faint one stays afterwards
        if r == ROW2:
            strong = max(fade(t, 3.6, 5.4, 0.4, 0.4), fade(t, 11.6, 13.0, 0.4, 0.4))
            base, prog = 0.35 * ss(t, 5.0, 5.4), ss(t, 3.6, 4.2)
        else:
            strong = fade(t, 11.6, 13.0, 0.4, 0.4)
            base, prog = 0.35 * ss(t, 12.6, 13.0), ss(t, 11.6, 12.2)
        a_arc = a_ic * max(strong, base)
        if a_arc > 0.01:
            p = STRIP.xy(0, r)
            arrow(cv, (p[0] - 20, p[1] - 14), (ICON_X + 26 * ICON_S, y - 22), col, a_arc, 2, 10, bend=-0.35,
                  progress=prog)
    a_ref = ss(t, 13.1, 13.6)
    if a_ref > 0.01:
        cv.text((ICON_X - 4, STRIP.y[0] - 36), "World origin", 20, alpha=a_ref, fill=ec.CAM_COLORS[0], anchor="mb")

    # sentence 2: camera token
    a_c = fade(t, 2.9, 5.4, 0.4, 0.3)
    if a_c > 0:
        p = STRIP.xy(0, ROW2)
        cv.s.line(p - [0, 24], p - [0, 120], GOLD, a_c, 2)
        cv.rich("camera token: after attention, the\ncamera head reads this frame's camera $g_i$ (9 numbers)", p - [10, 128], 26,
                alpha=a_c,
                fill=style.LABEL_GOLD, anchor="lb")
    # sentence 3: register token
    a_r = fade(t, 5.5, 8.0, 0.4, 0.3)
    if a_r > 0:
        p1, p4 = STRIP.xy(1, ROW2), STRIP.xy(4, ROW2)
        cv.s.line((p1[0] - 14, p1[1] + 26), (p4[0] + 14, p4[1] + 26), VIOLET, a_r, 2)
        cv.rich("register token ×4:\nonly joins attention; output discarded", ((p1[0] + p4[0]) / 2 - 20, p1[1] + 40), 26,
                alpha=a_r, fill=(205, 185, 255), anchor="la")
    # sentence 4: white-dot token
    a_w = ss(t, 8.1, 8.6)
    if a_w > 0:
        p = STRIP.xy(W_IDX, ROW2)
        cv.s.polyline([p + [-20, -20], p + [20, -20], p + [20, 20], p + [-20, 20]], GOLD, a_w, 2, closed=True)
        cv.g.polyline([p + [-20, -20], p + [20, -20], p + [20, 20], p + [-20, 20]], GOLD,
                      a_w * (0.4 + 0.6 * bump(t, 8.6, 1.0)), 4, closed=True)
        ec.star_marker(cv, p, a_w * (1 - ss(t, 10.6, 11.0)), glow=0.6, rad=4)
        a_lab = a_w * (1 - ss(t, 10.4, 10.8))
        cv.rich("No. $799$ (counting from 0) $=5+794$", p + [0, 44], 28, alpha=a_lab, fill=style.LABEL_GOLD, anchor="ma")
    # sentences 5-6: frame 1 uses a different set of tokens -> reference camera
    a_ref = ss(t, 11.6, 12.1)
    if a_ref > 0:
        p = STRIP.xy(0, 0)
        cv.rich("Frame 1's camera / register tokens: a separate set", (STRIP.xy(4, 0)[0] + 30, STRIP.y[0] - 46), 24,
                alpha=a_ref, fill=(255, 180, 140), anchor="lb")
    a_w2 = ss(t, 13.1, 13.6)
    if a_w2 > 0:
        cv.rich("Frame 1 = reference frame: output cameras and point maps are all in camera 1's frame", (540, 940), 28, alpha=a_w2, anchor="ma")
    a_sh = ss(t, 12.3, 12.8)
    SHAPE.draw(cv, (540, 1060), alpha=a_sh)
    if a_sh > 0:
        cv.rich(r"$N$: no. of images   $\tilde K$: tokens per image   $C$: token dim", (540, 1110), 24, alpha=a_sh,
                fill=style.TEXT_DIM, anchor="ma")
