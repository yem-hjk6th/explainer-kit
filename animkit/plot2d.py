"""2D coordinate plot: axes with rich-text ticks / labels, curves, scatter points, least-squares line y = s*x + t,
residual bars. All numbers come from the caller; this module only maps data -> screen and draws."""
import numpy as np

from . import style
from .ease import lerp, ss


def nice_ticks(lo, hi, n=6):
    """4-6 round tick values inside [lo, hi] (steps 1 / 2 / 2.5 / 5 x 10^k)."""
    span = float(hi - lo)
    mag = 10.0 ** np.floor(np.log10(span / n))
    for m in (1, 2, 2.5, 5, 10, 20, 25, 50):
        step = m * mag
        if span / step <= n:
            break
    k0, k1 = int(np.ceil(lo / step - 1e-9)), int(np.floor(hi / step + 1e-9))
    return [round(k * step, 10) for k in range(k0, k1 + 1)]


def _clip_seg(a, b, rect):
    """Liang-Barsky: part of segment a-b inside rect (x0, y0, x1, y1), or None."""
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


class Plot2D:
    """box = (x0, y0, x1, y1) screen rectangle of the plotting area (axes are drawn on its left and bottom edge).
    xlim / ylim = data ranges. xticks / yticks: None = automatic 4-6 ticks, list of values, or list of
    (value, "rich label"). xlabel / ylabel are rich ($math$, units, Chinese); ylabel sits above the y axis."""

    def __init__(self, box, xlim, ylim, xlabel=None, ylabel=None, xticks=None, yticks=None, fmt_x="{:g}",
                 fmt_y="{:g}", grid=True, zero_line=True, tick_size=22, label_size=26):
        self.box = tuple(float(v) for v in box)
        self.xlim, self.ylim = tuple(map(float, xlim)), tuple(map(float, ylim))
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xticks = self._ticks(xticks, self.xlim, fmt_x)
        self.yticks = self._ticks(yticks, self.ylim, fmt_y)
        self.grid, self.zero_line = grid, zero_line
        self.tick_size, self.label_size = tick_size, label_size

    @staticmethod
    def _ticks(spec, lim, fmt):
        vals = nice_ticks(*lim) if spec is None else spec
        out = []
        for v in vals:
            if isinstance(v, (tuple, list)):
                out.append((float(v[0]), v[1]))
            else:
                out.append((float(v), fmt.format(v).replace("-", "−")))
        return out

    def to_screen(self, x, y):
        """Data -> screen pixels; x, y scalars or arrays; returns (..., 2)."""
        x0, y0, x1, y1 = self.box
        x, y = np.asarray(x, float), np.asarray(y, float)
        sx = x0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (x1 - x0)
        sy = y1 - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (y1 - y0)
        return np.stack([sx, sy], -1)

    def inside(self, p, pad=0.5):
        x0, y0, x1, y1 = self.box
        return x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad

    # ---- axes ----
    def axes(self, cv, alpha=1.0, color=style.GREY, label_color=style.TEXT_DIM):
        if alpha <= 0.01:
            return
        x0, y0, x1, y1 = self.box
        ts = self.tick_size
        if self.grid:
            for v, _ in self.xticks:
                x = self.to_screen(v, self.ylim[0])[0]
                cv.s.line((x, y0), (x, y1), color, 0.16 * alpha, 1)
            for v, _ in self.yticks:
                y = self.to_screen(self.xlim[0], v)[1]
                cv.s.line((x0, y), (x1, y), color, 0.16 * alpha, 1)
        if self.zero_line:
            if self.ylim[0] < 0 < self.ylim[1]:
                y = self.to_screen(0, 0)[1]
                cv.s.dashed((x0, y), (x1, y), color, 0.55 * alpha, 1, n=int((x1 - x0) / 8) // 2 * 2)
            if self.xlim[0] < 0 < self.xlim[1]:
                x = self.to_screen(0, 0)[0]
                cv.s.dashed((x, y0), (x, y1), color, 0.55 * alpha, 1, n=int((y1 - y0) / 8) // 2 * 2)
        cv.s.line((x0, y1), (x1, y1), color, alpha, 2)
        cv.s.line((x0, y0), (x0, y1), color, alpha, 2)
        for v, lab in self.xticks:
            x = self.to_screen(v, self.ylim[0])[0]
            cv.s.line((x, y1), (x, y1 + 7), color, alpha, 2)
            cv.rich(lab, (x, y1 + 12), ts, alpha=alpha, fill=label_color, anchor="ma", align="m")
        for v, lab in self.yticks:
            y = self.to_screen(self.xlim[0], v)[1]
            cv.s.line((x0 - 7, y), (x0, y), color, alpha, 2)
            cv.rich(lab, (x0 - 12, y), ts, alpha=alpha, fill=label_color, anchor="rm", align="r")
        if self.xlabel:
            cv.rich(self.xlabel, ((x0 + x1) / 2, y1 + 12 + ts * 1.5 + 10), self.label_size, alpha=alpha,
                    fill=label_color, anchor="ma", align="m")
        if self.ylabel:
            cv.rich(self.ylabel, (x0, y0 - 16), self.label_size, alpha=alpha, fill=label_color, anchor="lb")

    # ---- series ----
    def _runs(self, P):
        """Split a screen polyline into runs that stay inside the plot box (segments clipped at the border)."""
        runs, cur = [], []
        for a, b in zip(P[:-1], P[1:]):
            seg = _clip_seg(a, b, self.box)
            if seg is None:
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
                continue
            if cur and np.allclose(cur[-1], seg[0], atol=1e-6):
                cur.append(seg[1])
            else:
                if len(cur) > 1:
                    runs.append(cur)
                cur = [seg[0], seg[1]]
        if len(cur) > 1:
            runs.append(cur)
        return runs

    def _stroke(self, cv, P, color, alpha, th, glow, dashed):
        for run in self._runs(P):
            if dashed:
                for a, b in zip(run[:-1], run[1:]):
                    cv.s.dashed(a, b, color, alpha, th, n=max(2, int(np.hypot(*(np.subtract(b, a))) / 9) // 2 * 2))
            else:
                cv.s.polyline(run, color, alpha, th)
                if glow:
                    cv.g.polyline(run, color, glow * alpha, th + 2)

    def curve(self, cv, x, y, color=style.CYAN, progress=1.0, alpha=1.0, th=3, glow=0.6, dashed=False):
        """Polyline through (x, y) data, drawn from the left up to `progress` (0-1 of the arc length).
        Parts outside the box are clipped. Returns the screen position of the leading tip."""
        P = self.to_screen(x, y)
        if alpha <= 0.01 or progress <= 0 or len(P) < 2:
            return P[0] if len(P) else None
        seg = np.hypot(*(np.diff(P, axis=0).T))
        cum = np.concatenate([[0], np.cumsum(seg)])
        cut = float(np.clip(progress, 0, 1)) * cum[-1]
        k = int(np.searchsorted(cum, cut, side="right")) - 1
        k = min(max(k, 0), len(P) - 2)
        f = (cut - cum[k]) / max(seg[k], 1e-9)
        tip = P[k] + (P[k + 1] - P[k]) * np.clip(f, 0, 1)
        Q = np.vstack([P[:k + 1], tip[None]])
        self._stroke(cv, Q, color, alpha, th, glow, dashed)
        return tip

    def scatter(self, cv, x, y, colors=None, progress=1.0, alpha=1.0, rad=7, labels=None, label_size=24,
                label_offset=(12, -12), label_color=style.TEXT):
        """Points appear one after another as progress goes 0 -> 1. colors: one RGB or a list per point (mark a bad
        point in RED); labels: list of rich strings (None = skip). Returns the (n, 2) screen positions."""
        P = self.to_screen(x, y).reshape(-1, 2)
        n = len(P)
        cols = [style.CYAN] * n if colors is None else (
            [colors] * n if np.ndim(colors[0]) == 0 else list(colors))
        for k in range(n):
            u = ss(progress * (n + 1) - k, 0, 1)
            a = alpha * u
            if a <= 0.01 or not self.inside(P[k]):
                continue
            cv.g.dot(P[k], cols[k], rad * 1.7, 0.8 * a)
            cv.s.dot(P[k], cols[k], rad, a)
            cv.s.dot(P[k], (255, 255, 255), rad * 0.35, 0.8 * a)
            if labels is not None and labels[k]:
                cv.rich(labels[k], P[k] + np.asarray(label_offset, float), label_size, alpha=a, fill=label_color,
                        anchor="lb")
        return P

    def stem(self, cv, x, y, color=style.CYAN, progress=1.0, alpha=1.0, base=0.0, th=3, rad=6):
        """Vertical stems from y = base to each y, growing left to right."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        out = []
        for k in range(n):
            u = ss(progress * (n + 1) - k, 0, 1)
            a = alpha * u
            p0, p1 = self.to_screen(x[k], base), self.to_screen(x[k], base + (y[k] - base) * u)
            out.append(p1)
            if a > 0.01:
                cv.s.line(p0, p1, color, a, th)
                cv.g.line(p0, p1, color, 0.5 * a, th + 2)
                cv.s.dot(p1, color, rad, a)
        return np.array(out)

    # ---- fitted line y = s x + t ----
    @staticmethod
    def fit_line(x, y, w=None):
        """Least-squares (s, t) of y = s x + t (optional weights w); the same numbers a check should assert."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        w = np.ones_like(x) if w is None else np.asarray(w, float)
        A = np.stack([x, np.ones_like(x)], 1) * np.sqrt(w)[:, None]
        s, t = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        return float(s), float(t)

    @staticmethod
    def morph(params_a, params_b, u):
        """Parameter interpolation (s, t) for a line that moves from a hypothesis to the refit; u = 0 -> 1."""
        return tuple(lerp(a, b, u) for a, b in zip(params_a, params_b))

    def line(self, cv, s, t, color=style.GOLD, progress=1.0, alpha=1.0, th=3, glow=0.6, dashed=False, x_range=None):
        """y = s x + t across x_range (default: whole x axis), drawn left to right up to progress.
        Returns (screen left end, screen leading tip)."""
        xa, xb = x_range or self.xlim
        xs = np.array([xa, xb], float)
        tip = self.curve(cv, xs, s * xs + t, color, progress, alpha, th, glow, dashed)
        return self.to_screen(xa, s * xa + t), tip

    def residuals(self, cv, x, y, s, t, color=style.RED, progress=1.0, alpha=1.0, th=2):
        """Vertical bars from each point to the line y = s x + t (drawn in turn); returns residuals y - (s x + t)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        r = y - (s * x + t)
        n = len(x)
        for k in range(n):
            u = ss(progress * (n + 1) - k, 0, 1)
            if alpha * u <= 0.01:
                continue
            a = self.to_screen(x[k], s * x[k] + t)
            b = self.to_screen(x[k], s * x[k] + t + r[k] * u)
            if self.inside(a) or self.inside(b):
                seg = _clip_seg(a, b, self.box)
                if seg:
                    cv.s.line(seg[0], seg[1], color, alpha * u, th)
                    cv.g.line(seg[0], seg[1], color, 0.5 * alpha * u, th + 2)
        return r
