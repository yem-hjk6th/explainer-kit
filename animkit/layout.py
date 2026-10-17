"""Layout pieces: paper card, comparison table, optional / required input slots with on-off links, split screen,
multi-scale grid pyramid."""
import numpy as np

from . import style
from .arch import arrow
from .draw import rrect_pts
from .ease import ss
from .richtext import rich, wrap


def paper_card(cv, box, year, venue, title, authors=None, contribution=None, image=None, alpha=1.0,
               accent=style.GOLD, title_size=30):
    """Text paper card: big year, venue / arXiv id, wrapped title, authors, one-line contribution. image: optional
    first-page picture (path / array / PIL) shown in a slot on the right; None leaves the card text-only."""
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = box
    cv.s.fill(rrect_pts(x0, y0, x1, y1, 14), (24, 27, 33), 0.92 * alpha)
    cv.s.polyline(rrect_pts(x0, y0, x1, y1, 14), accent, 0.8 * alpha, 2, closed=True)
    tx1 = x1 - 24
    if image is not None:
        iw = (x1 - x0) * 0.32
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        cv.image(image, (x1 - 20, y0 + 20), w=iw, alpha=alpha, anchor="ra")
        tx1 = x1 - iw - 40
    pad = 26
    cv.text((x0 + pad, y0 + pad), str(year), 72, "bold", alpha, accent)
    cv.rich(venue, (x0 + pad + 190, y0 + pad + 46), 26, alpha=alpha, fill=style.TEXT_DIM, anchor="ls")
    y = y0 + pad + 100
    t = wrap(title, tx1 - x0 - 2 * pad, title_size, "bold")
    b = rich(t, title_size, "bold").draw(cv, (x0 + pad, y), style.TEXT, alpha, "la")
    y = b[3] + 12
    if authors:
        b = cv.rich(wrap(authors, tx1 - x0 - 2 * pad, 22), (x0 + pad, y), 22, alpha=alpha, fill=style.TEXT_DIM)
        y = b[3] + 16
    if contribution:
        cv.s.line((x0 + pad, y), (x0 + pad + 60, y), accent, alpha, 2)
        cv.rich(wrap(contribution, tx1 - x0 - 2 * pad, 26), (x0 + pad, y + 14), 26, alpha=alpha)


def table(cv, xy, col_w, header, rows, row_h=64, size=26, reveal=None, highlight=None, alpha=1.0,
          header_color=style.GOLD, line_color=style.GREY, col_colors=None):
    """Comparison table: header + rows of rich-text cells (Chinese / English / $math$, \\n allowed).
    reveal: rows visible (float, the last one fades in); highlight: row index outlined. Returns row boxes."""
    x0, y0 = xy
    W = float(np.sum(col_w))
    xs = np.r_[x0, x0 + np.cumsum(col_w)]
    boxes = []
    if alpha <= 0.01:
        return boxes
    cv.s.line((x0, y0 + row_h), (x0 + W, y0 + row_h), header_color, 0.9 * alpha, 2)
    for j, h in enumerate(header):
        cv.rich(h, ((xs[j] + xs[j + 1]) / 2, y0 + row_h / 2), size, "bold", alpha, header_color, "mm", "m")
    for i, r in enumerate(rows):
        ra = alpha * (1.0 if reveal is None else float(np.clip(reveal - i, 0, 1)))
        ty = y0 + row_h * (i + 1)
        boxes.append((x0, ty, x0 + W, ty + row_h))
        if ra <= 0.01:
            continue
        cv.s.line((x0, ty + row_h), (x0 + W, ty + row_h), line_color, 0.5 * ra, 1)
        for j, c in enumerate(r):
            col = (col_colors or {}).get(j, style.TEXT)
            cv.rich(c, ((xs[j] + xs[j + 1]) / 2, ty + row_h / 2), size, alpha=ra, fill=col, anchor="mm", align="m")
        if highlight == i:
            pts = [(x0 - 6, ty + 2), (x0 + W + 6, ty + 2), (x0 + W + 6, ty + row_h - 2), (x0 - 6, ty + row_h - 2)]
            cv.s.polyline(pts, style.ORANGE, ra, 2, closed=True)
            cv.g.polyline(pts, style.ORANGE, 0.6 * ra, 4, closed=True)
    return boxes


def input_slot(cv, box, label, optional=True, on=1.0, prob=None, color=style.CYAN, alpha=1.0, size=26):
    """Input box: dashed border = optional input, solid = always given. on (0-1): 0 greys it out.
    prob: small label beside it (e.g. "p = 0.5"). Returns dict of edge midpoints like module_box."""
    x0, y0, x1, y1 = box
    a = alpha * (0.35 + 0.65 * on)
    c = tuple(g + (cc - g) * on for cc, g in zip(color, style.GREY))
    if a > 0.01:
        pts = rrect_pts(x0, y0, x1, y1, 10)
        if optional:
            for p, q in zip(pts, pts[1:] + pts[:1]):
                L = np.hypot(q[0] - p[0], q[1] - p[1])
                if L > 18:
                    cv.s.dashed(p, q, c, a, 2, n=max(2, int(L / 10) // 2 * 2))
                else:
                    cv.s.line(p, q, c, a, 2)
        else:
            cv.s.polyline(pts, c, a, 2, closed=True)
        if on > 0.5:
            cv.s.fill(pts, tuple(cc * 0.3 for cc in c), 0.5 * a)
        cv.rich(label, ((x0 + x1) / 2, (y0 + y1) / 2), size, alpha=a, anchor="mm", align="m")
        if prob:
            cv.rich(prob, (x1 + 10, (y0 + y1) / 2), 20, alpha=a, fill=style.TEXT_DIM, anchor="lm")
    return {"l": np.array([x0, (y0 + y1) / 2]), "r": np.array([x1, (y0 + y1) / 2]),
            "t": np.array([(x0 + x1) / 2, y0]), "b": np.array([(x0 + x1) / 2, y1])}


def link(cv, a, b, on=1.0, color=style.WHITE, alpha=1.0, th=2):
    """Connection that is a solid arrow when on and breaks open (grey, dashed, gap in the middle) when off."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if on >= 0.5:
        arrow(cv, a, b, color, alpha * (0.4 + 0.6 * on), th, head=12)
    else:
        m1, m2 = a + (b - a) * 0.4, a + (b - a) * 0.6
        cv.s.dashed(a, m1, style.GREY, 0.6 * alpha, th, n=6)
        cv.s.dashed(m2, b, style.GREY, 0.6 * alpha, th, n=6)


def flicker(t, n, p=0.5, period=0.6, seed=0, t0=0.0):
    """Deterministic random on/off pattern for n inputs: re-drawn every `period` s from rng(seed + step);
    p = probability each input is on (scalar or per input). Returns a bool array."""
    step = int(max(t - t0, 0) // period)
    rng = np.random.default_rng(seed + step)
    return rng.random(n) < np.broadcast_to(np.asarray(p, float), (n,))


def split(cv, mode="lr", pos=0.5, alpha=1.0, color=style.GREY, margin=(60, 300, 60, 1230)):
    """Divider for a split screen; returns the two halves (x0, y0, x1, y1). margin = content area x0, y0, x1, y1."""
    X0, Y0, X1, Y1 = margin
    if mode == "lr":
        xm = X0 + (X1 - X0) * pos
        if alpha > 0.01:
            cv.s.line((xm, Y0), (xm, Y1), color, 0.6 * alpha, 1)
        return (X0, Y0, xm - 20, Y1), (xm + 20, Y0, X1, Y1)
    ym = Y0 + (Y1 - Y0) * pos
    if alpha > 0.01:
        cv.s.line((X0, ym), (X1, ym), color, 0.6 * alpha, 1)
    return (X0, Y0, X1, ym - 20), (X0, ym + 20, X1, Y1)


def grid_square(cv, c, side, n_lines=8, label=None, color=style.CYAN, alpha=1.0, size=24, glow=0.0):
    """Square feature map of `side` px centred at c: outline + n_lines sparse grid lines each way + label below."""
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2
    cv.s.fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], tuple(v * 0.25 for v in color), 0.8 * alpha)
    for k in range(1, n_lines):
        u = k / n_lines
        cv.s.line((x0 + u * side, y0), (x0 + u * side, y1), color, 0.25 * alpha, 1)
        cv.s.line((x0, y0 + u * side), (x1, y0 + u * side), color, 0.25 * alpha, 1)
    cv.s.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, alpha, 2, closed=True)
    if glow:
        cv.g.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, glow * alpha, 4, closed=True)
    if label:
        cv.rich(label, (c[0], y1 + 10), size, alpha=alpha, anchor="ma", align="m")


def grid_pyramid(cv, y, sides_px, labels=None, x0=90, gap=50, alpha=1.0, color=style.CYAN, progress=1.0,
                 fuse=True, n_lines=None):
    """Feature maps of decreasing size in a row (e.g. 148 / 74 / 37 / 19 cells drawn as sides_px), bottoms aligned
    at y. progress reveals them left to right; fuse draws arrows from each map to the next. Returns centres."""
    cs, x = [], x0
    for s in sides_px:
        cs.append(np.array([x + s / 2, y - s / 2]))
        x += s + gap
    n = len(sides_px)
    for k, (c, s) in enumerate(zip(cs, sides_px)):
        a = alpha * ss(progress, k / n, (k + 1) / n)
        grid_square(cv, c, s, n_lines[k] if n_lines else 8, labels[k] if labels else None, color, a)
        if fuse and k + 1 < n:
            arrow(cv, c + [s / 2 + 6, 0], cs[k + 1] - [sides_px[k + 1] / 2 + 6, 0], style.WHITE,
                  alpha * ss(progress, (k + 1) / n, (k + 1.5) / n), 2, 12)
    return cs
