"""Charts: bar charts (vertical / horizontal, growing), frame-wise vs global attention block pattern, one-row
softmax computed from q and keys."""
import numpy as np

from . import style
from .ease import ss
from .readout import number
from .tensor import heatmap, softmax_rows


def row_softmax(q, keys, mask=None, scale=None):
    """Weights of one query over n keys: softmax(q . k_j * scale), scale default 1/sqrt(d); mask False -> 0."""
    q, keys = np.asarray(q, float), np.asarray(keys, float)
    scale = 1 / np.sqrt(q.shape[-1]) if scale is None else scale
    L = (keys @ q * scale)[None]
    if mask is not None:
        L = np.where(np.asarray(mask)[None], L, -np.inf)
    return softmax_rows(L)[0]


def bar_chart(cv, box, values, labels=None, vmax=None, horizontal=False, color=style.CYAN, highlight=None,
              hl_color=style.ORANGE, progress=1.0, stagger=0.4, fmt="{:.2f}", title=None, better=None, note=None,
              alpha=1.0, value_size=26, label_size=24, unit=None):
    """Bars inside box (x0, y0, x1, y1). progress 0 -> 1 grows them (staggered); the value label counts up with
    the bar. better: 'lower' / 'higher' appends (lower is better) / (higher is better) to the title. note: source line under the chart
    (e.g. "VGGT Tab. 1"). highlight: index (or list) drawn in hl_color. Returns bar end points."""
    if alpha <= 0.01:
        return []
    x0, y0, x1, y1 = box
    v = np.asarray(values, float)
    n = len(v)
    vmax = float(v.max()) * 1.12 if vmax is None else vmax
    hl = set([highlight] if isinstance(highlight, int) else (highlight or []))
    if title:
        t = title + {"lower": " (lower is better)", "higher": " (higher is better)"}.get(better, "")
        cv.rich(t, ((x0 + x1) / 2, y0 - 16), 30, alpha=alpha, anchor="mb", align="m")
    if note:
        cv.rich(note, (x1, y1 + (label_size * 2.4 if not horizontal else 14)), 20, alpha=0.8 * alpha,
                fill=style.GREY, anchor="ra")
    ends = []
    span = 1 - stagger
    if horizontal:
        cv.s.line((x0, y0), (x0, y1), style.GREY, 0.8 * alpha, 1)
        slot = (y1 - y0) / n
        for k in range(n):
            g = ss(progress, stagger * k / max(n - 1, 1), stagger * k / max(n - 1, 1) + span)
            c = hl_color if k in hl else color
            yc = y0 + (k + 0.5) * slot
            L = (x1 - x0) * v[k] / vmax * g
            bh = slot * 0.62
            if L > 0.5:
                cv.s.rect(x0, yc - bh / 2, x0 + L, yc + bh / 2, tuple(ch * 0.8 for ch in c), alpha, r=3)
                if k in hl:
                    cv.g.rect(x0, yc - bh / 2, x0 + L, yc + bh / 2, c, 0.5 * alpha, r=3)
            number(cv, (x0 + L + 10, yc), v[k] * g, fmt, value_size, style.TEXT, alpha * min(1, g * 3), "lm",
                   unit=unit)
            if labels:
                cv.rich(labels[k], (x0 - 12, yc), label_size, alpha=alpha, anchor="rm", align="r")
            ends.append(np.array([x0 + L, yc]))
    else:
        cv.s.line((x0, y1), (x1, y1), style.GREY, 0.8 * alpha, 1)
        slot = (x1 - x0) / n
        for k in range(n):
            g = ss(progress, stagger * k / max(n - 1, 1), stagger * k / max(n - 1, 1) + span)
            c = hl_color if k in hl else color
            xc = x0 + (k + 0.5) * slot
            Hh = (y1 - y0) * v[k] / vmax * g
            bw = slot * 0.58
            if Hh > 0.5:
                cv.s.rect(xc - bw / 2, y1 - Hh, xc + bw / 2, y1, tuple(ch * 0.8 for ch in c), alpha, r=3)
                if k in hl:
                    cv.g.rect(xc - bw / 2, y1 - Hh, xc + bw / 2, y1, c, 0.5 * alpha, r=3)
            number(cv, (xc, y1 - Hh - 10), v[k] * g, fmt, value_size, style.TEXT, alpha * min(1, g * 3), "mb",
                   unit=unit)
            if labels:
                cv.rich(labels[k], (xc, y1 + 10), label_size, alpha=alpha, anchor="ma", align="m")
            ends.append(np.array([xc, y1 - Hh]))
    return ends


def attn_blocks(cv, xy, N, K, cell, mode="frame", alpha=1.0, A=None, vmax=None, diag_color=style.CYAN,
                cross_color=style.VIOLET, off_color=(58, 62, 70), label_size=22, frame_labels=True, block_labels=True):
    """Attention pattern over N frames x K tokens as an (NK x NK) grid of N x N blocks.
    mode='frame': only diagonal blocks are computed; off-diagonal blocks are greyed and marked "skipped".
    mode='global': every block is computed; off-diagonal blocks get cross_color and "cross-frame".
    A: optional (NK, NK) weights drawn as a heatmap inside the computed blocks. Returns fn(i, j) -> cell centre."""
    if alpha <= 0.01:
        return None
    x0, y0 = xy
    B = K * cell
    if A is not None:
        centre = heatmap(cv, A, (x0, y0), cell, 0, vmax if vmax is not None else float(np.max(A)), alpha)
    else:
        def centre(i, j):
            return np.array([x0 + (j + 0.5) * cell, y0 + (i + 0.5) * cell])
    for bi in range(N):
        for bj in range(N):
            bx, by = x0 + bj * B, y0 + bi * B
            pts = [(bx, by), (bx + B, by), (bx + B, by + B), (bx, by + B)]
            diag = bi == bj
            if diag:
                if A is None:
                    cv.s.fill(pts, tuple(c * 0.45 for c in diag_color), alpha)
                cv.s.polyline(pts, diag_color, alpha, 2, closed=True)
            elif mode == "frame":
                cv.s.fill(pts, off_color, 0.92 * alpha)
                if block_labels:
                    cv.text((bx + B / 2, by + B / 2), "skipped", label_size, alpha=0.8 * alpha, fill=style.GREY,
                            anchor="mm", shadow=False)
            else:
                if A is None:
                    cv.s.fill(pts, tuple(c * 0.4 for c in cross_color), alpha)
                cv.s.polyline(pts, cross_color, alpha, 2, closed=True)
                if block_labels:
                    cv.text((bx + B / 2, by + B / 2), "cross-frame", label_size, alpha=alpha, fill=style.TEXT, anchor="mm")
    if frame_labels:
        for b in range(N):
            cv.text((x0 - 10, y0 + (b + 0.5) * B), f"Frame {b + 1}", label_size, alpha=alpha, anchor="rm")
            cv.text((x0 + (b + 0.5) * B, y0 - 8), f"Frame {b + 1}", label_size, alpha=alpha, anchor="mb")
    return centre
