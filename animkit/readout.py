"""Numbers on screen: counters, vectors, bracketed matrices. Numeric entries use richtext.Digits (Computer Modern
glyphs laid out per frame, no re-typesetting), symbolic entries use mathtext."""
import numpy as np

from . import style
from .formula import anchor_xy, get, paste_mask, request
from .richtext import digits


def _fmt(fmt, v):
    return fmt.format(int(round(v))) if fmt.endswith("d}") else fmt.format(v)


def number(cv, xy, value, fmt="{:.3f}", size=40, color=style.MATH, alpha=1.0, anchor="mm", unit=None,
           unit_size=None, glow=0.0):
    """One number (+ optional unit as rich text, e.g. "m" or "px"). Returns bbox of number + unit."""
    d = digits(size)
    s = _fmt(fmt, value)
    x0, y0, x1, y1 = d.draw(cv, xy, s, color, alpha, anchor, glow=glow)
    if unit:
        _, _, asc, _ = d._glyphs(s)
        ub = cv.rich(unit, (x1 + 0.22 * size, y0 + asc), unit_size or int(size * 0.75), alpha=alpha,
                     fill=style.TEXT_DIM, anchor="ls")
        x1 = ub[2]
    return x0, y0, x1, y1


def counter(cv, xy, u, a, b, fmt="{:d}", **kw):
    """Number running from a to b as u goes 0 -> 1 (drive u with ss / lin). '{:d}' formats as integer jumps."""
    return number(cv, xy, a + (b - a) * float(np.clip(u, 0, 1)), fmt, **kw)


def vector(cv, xy, values, fmt="{:.3f}", size=40, color=style.MATH, alpha=1.0, anchor="mm", part_alpha=None,
           part_color=None, slot=None, brackets="()"):
    """(v0, v1, v2) with each component right-aligned in a fixed slot so morphing values do not jitter.
    slot: px per component (default: fits the current values with a sign). Returns list of component centres."""
    d = digits(size)
    vals = [float(v) for v in values]
    strs = [_fmt(fmt, v) for v in vals]
    sw = slot or max(d.width(_fmt(fmt, -abs(v))) for v in vals)
    lw, cw, rw = d.width(brackets[0]), d.width(", "), d.width(brackets[1])
    W = lw + len(vals) * sw + (len(vals) - 1) * cw + rw
    _, _, asc, dep = d._glyphs("(0.,-)")
    x0, y0 = anchor_xy(xy, W, asc + dep, asc, anchor)
    base = y0 + asc
    d.draw(cv, (x0, base), brackets[0], color, alpha, "ls")
    x = x0 + lw
    centres = []
    for k, s in enumerate(strs):
        a = alpha * (part_alpha or {}).get(k, 1.0)
        c = (part_color or {}).get(k, color)
        d.draw(cv, (x + sw, base), s, c, a, "rs")
        centres.append(np.array([x + sw / 2, base - asc / 2]))
        x += sw
        if k < len(strs) - 1:
            d.draw(cv, (x, base), ",", color, alpha, "ls")
            x += cw
    d.draw(cv, (x, base), brackets[1], color, alpha, "ls")
    return centres


def matrix(cv, xy, M, fmt="{:.2f}", size=36, color=style.MATH, alpha=1.0, anchor="mm", part_alpha=None,
           part_color=None, box=None, col_gap=0.9, row_gap=0.4, bracket_color=None):
    """Bracketed matrix. M: nested lists of numbers (formatted with fmt) or mathtext strings (e.g. "f_x", "c_x").
    part_alpha / part_color / box: {(i, j): value} per entry (box = highlight amount 0-1).
    Returns fn(i, j) -> entry centre, and the bbox."""
    d = digits(size)
    rows = len(M)
    cols = max(len(r) for r in M)
    ent = {}
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            if isinstance(v, str):
                mask, m = get(request(v, size))
                ent[i, j] = ("m", mask, m["width"], m["asc"], max(m["desc"], 0))
            else:
                s = _fmt(fmt, v)
                _, w, a, dp = d._glyphs(s)
                ent[i, j] = ("d", s, w, a, dp)
    cw = [max(ent[i, j][2] for i in range(rows) if (i, j) in ent) for j in range(cols)]
    ra = [max(ent[i, j][3] for j in range(cols) if (i, j) in ent) for i in range(rows)]
    rd = [max(ent[i, j][4] for j in range(cols) if (i, j) in ent) for i in range(rows)]
    g, rg, pad = col_gap * size, row_gap * size, 0.45 * size
    W = sum(cw) + g * (cols - 1) + 2 * pad
    H = sum(a + b for a, b in zip(ra, rd)) + rg * (rows - 1)
    x0, y0 = anchor_xy(xy, W, H, H / 2, anchor)
    bc = bracket_color or color
    if alpha > 0.01:
        tick = 0.28 * size
        for side in (0, 1):
            x = x0 if side == 0 else x0 + W
            s = 1 if side == 0 else -1
            cv.s.polyline([(x + s * tick, y0 - 4), (x, y0 - 4), (x, y0 + H + 4), (x + s * tick, y0 + H + 4)], bc,
                          alpha, 2)
    centres = {}
    y = y0
    for i in range(rows):
        base = y + ra[i]
        x = x0 + pad
        for j in range(cols):
            cx = x + cw[j] / 2
            centres[i, j] = np.array([cx, base - ra[i] / 2])
            if (i, j) in ent:
                kind, p, w, a, _ = ent[i, j]
                al = alpha * (part_alpha or {}).get((i, j), 1.0)
                c = (part_color or {}).get((i, j), color)
                if kind == "d":
                    d.draw(cv, (cx, base), p, c, al, "ms")
                else:
                    paste_mask(cv, p, int(round(cx - w / 2)), int(round(base - a)), c, al)
                amt = (box or {}).get((i, j), 0)
                if amt > 0 and alpha > 0.01:
                    bx = [(cx - w / 2 - 6, base - ra[i] - 6), (cx + w / 2 + 6, base - ra[i] - 6),
                          (cx + w / 2 + 6, base + rd[i] + 6), (cx - w / 2 - 6, base + rd[i] + 6)]
                    cv.s.polyline(bx, style.ORANGE, alpha * amt, 2, closed=True)
                    cv.g.polyline(bx, style.ORANGE, 0.6 * alpha * amt, 4, closed=True)
            x += cw[j] + g
        y = base + rd[i] + rg
    return (lambda i, j: centres[i, j]), (x0, y0, x0 + W, y0 + H)
