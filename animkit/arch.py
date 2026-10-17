"""Architecture diagram pieces: module boxes, arrows (straight / bent / elbow / branching, growable), pulses
travelling along them, dimension lines."""
import numpy as np

from . import style
from .draw import rrect_pts

STATE = {"hidden": 0.0, "dim": 0.35, "lit": 1.0}


def module_box(cv, box, title=None, sub=None, color=style.CYAN, alpha=1.0, fill=0.16, glow=0.0, th=2, r=16,
               title_size=32, sub_size=24, state="lit", title_kind="text", sub_kind="text"):
    """Rounded box (x0, y0, x1, y1) with a title and an optional second line (both rich text: Chinese / English /
    $math$, e.g. sub="$(N, 1374, 1024)$"). state: 'lit' | 'dim' | float. glow > 0 lights the border (drive it with
    ease.bump for a highlight pulse). Returns edge midpoints l / r / t / b / c."""
    x0, y0, x1, y1 = box
    lvl = STATE[state] if isinstance(state, str) else float(state)
    a = alpha * lvl
    if a > 0.01:
        pts = rrect_pts(x0, y0, x1, y1, r)
        if fill:
            cv.s.fill(pts, tuple(c * 0.35 for c in color), min(1.0, fill * 2.5) * a)
        cv.s.polyline(pts, color, a, th, closed=True)
        if glow > 0 and lvl >= 1:
            cv.g.polyline(pts, color, min(1.0, glow) * alpha, th + 4, closed=True)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if title and sub:
            cv.rich(title, (cx, cy - 2), title_size, title_kind, a, style.TEXT, "mb", "m")
            cv.rich(sub, (cx, cy + 6), sub_size, sub_kind, a, style.TEXT_DIM, "ma", "m")
        elif title:
            cv.rich(title, (cx, cy), title_size, title_kind, a, style.TEXT, "mm", "m")
    return {"l": np.array([x0, (y0 + y1) / 2]), "r": np.array([x1, (y0 + y1) / 2]),
            "t": np.array([(x0 + x1) / 2, y0]), "b": np.array([(x0 + x1) / 2, y1]),
            "c": np.array([(x0 + x1) / 2, (y0 + y1) / 2])}


# ---------------------------------------------------------------- paths
def path(a, b, bend=0.0, n=40):
    """Points from a to b; bend != 0 makes a quadratic curve bulging bend * |b - a| to the left of a->b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if bend == 0:
        return np.stack([a, b])
    d = b - a
    c = (a + b) / 2 + bend * np.array([d[1], -d[0]])
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * a + 2 * (1 - u) * u * c + u ** 2 * b


def elbow(a, b, first="v", mid=None):
    """Orthogonal route a -> b. first='v': down/up to the middle row, across, then to b; 'h': across first.
    mid: the row (for 'v') or column (for 'h') where it turns (default halfway)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if first == "v":
        m = (a[1] + b[1]) / 2 if mid is None else mid
        pts = [a, (a[0], m), (b[0], m), b]
    else:
        m = (a[0] + b[0]) / 2 if mid is None else mid
        pts = [a, (m, a[1]), (m, b[1]), b]
    out = [np.asarray(pts[0], float)]
    for p in pts[1:]:
        if np.linalg.norm(np.asarray(p, float) - out[-1]) > 1e-6:
            out.append(np.asarray(p, float))
    return np.array(out)


def branches(a, targets, split=None):
    """One source fanning out to several targets (e.g. one trunk -> four heads), vertical layout:
    trunk down from a to row `split`, across to each target's x, down into it. Returns one path per target."""
    a = np.asarray(a, float)
    ys = [np.asarray(t, float)[1] for t in targets]
    m = (a[1] + min(ys)) / 2 if split is None else split
    return [elbow(a, t, "v", m) for t in targets]


def along(pts, u):
    """Point at arc-length fraction u of polyline pts, and the unit tangent there."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    target = np.clip(u, 0, 1) * s[-1]
    i = int(np.clip(np.searchsorted(s, target) - 1, 0, len(seg) - 1))
    w = (target - s[i]) / max(seg[i], 1e-9)
    tan = (pts[i + 1] - pts[i]) / max(seg[i], 1e-9)
    return pts[i] + w * (pts[i + 1] - pts[i]), tan


def _sub(pts, u):
    """Polyline truncated at arc-length fraction u."""
    pts = np.asarray(pts, float)
    if u >= 1:
        return pts
    p, _ = along(pts, u)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    keep = pts[s < u * s[-1]]
    return np.vstack([keep, p])


def arrow_path(cv, pts, color=style.WHITE, alpha=1.0, th=3, head=18, progress=1.0, dashed=False, glow=0.0):
    """Arrow along any polyline (from path / elbow / branches); progress < 1 draws it growing. Returns pts."""
    P = np.asarray(pts, float)
    if alpha <= 0 or progress <= 0:
        return P
    Q = _sub(P, progress)
    if dashed:
        for p, q in zip(Q[:-1], Q[1:]):
            cv.s.dashed(p, q, color, alpha, th, n=max(2, int(np.linalg.norm(q - p) / 9) // 2 * 2))
    else:
        cv.s.polyline(Q, color, alpha, th)
    if glow > 0:
        cv.g.polyline(Q, color, min(1.0, glow) * alpha, th + 4)
    if head:
        tip, tan = along(P, progress)
        nrm = np.array([-tan[1], tan[0]])
        cv.s.fill([tip, tip - tan * head + nrm * head * 0.45, tip - tan * head - nrm * head * 0.45], color, alpha)
    return P


def arrow(cv, a, b, color=style.WHITE, alpha=1.0, th=3, head=18, bend=0.0, progress=1.0, dashed=False, glow=0.0):
    """Arrow a -> b (bend != 0 curves it). Returns the full path (for pulse_dot)."""
    return arrow_path(cv, path(a, b, bend), color, alpha, th, head, progress, dashed, glow)


def pulse_dot(cv, pts, u, color=style.GOLD, alpha=1.0, rad=7):
    """Glowing dot at fraction u along a path (data flowing through an arrow); nothing outside 0 <= u <= 1."""
    if alpha <= 0 or not 0 <= u <= 1:
        return
    p, _ = along(pts, u)
    cv.g.dot(p, color, rad * 1.6, alpha)
    cv.s.dot(p, (255, 255, 255), rad * 0.55, alpha)


def measure(cv, a, b, label=None, color=style.TEXT_DIM, alpha=1.0, offset=0.0, th=2, tick=8, label_size=26,
            dashed=False, label_gap=14):
    """Dimension line between screen points a and b, shifted `offset` px to the left of a->b, with end ticks and a
    label at the middle (rich text, e.g. "along the ray $1.984$"). Use with P(X) for 3D lengths. Returns label anchor."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = max(np.linalg.norm(d), 1e-9)
    n = np.array([d[1], -d[0]]) / L
    A, B = a + n * offset, b + n * offset
    if alpha > 0.01:
        (cv.s.dashed if dashed else cv.s.line)(A, B, color, alpha, th)
        for p in (A, B):
            cv.s.line(p - n * tick, p + n * tick, color, alpha, th)
        if label:
            m = (A + B) / 2 + n * label_gap * np.sign(offset or 1)
            anchor = "mm" if abs(n[1]) < 0.5 else ("mb" if n[1] < 0 else "ma")
            cv.rich(label, m, label_size, alpha=alpha, fill=color, anchor=anchor)
    return (A + B) / 2
