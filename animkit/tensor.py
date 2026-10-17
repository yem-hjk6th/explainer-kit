"""Tensor pictures: cuboid tensor blocks, token tiles, index-exact reshape moves, attention heatmaps.

Everything that shows a number or a position here is computed: reshape moves come from numpy reshaping an id
tensor, attention weights from softmax(q k^T / sqrt(d)). Check helpers return values for the shot's check().
"""
import cv2
import numpy as np

from . import style
from .draw import rrect_pts
from .ease import ss


def _shade(color, k):
    return tuple(min(255.0, c * k) for c in color)


# ---------------------------------------------------------------- tensor blocks
def box3(cv, x, y, w, h, d, color, alpha=1.0, depth_dir=style.DEPTH_DIR, th=2, shade=(0.5, 0.75, 0.32),
         edge=1.0):
    """One cuboid: front face (x, y)-(x+w, y+h), receding d px along depth_dir. Faces occlude ('over' fill)."""
    if alpha <= 0:
        return
    dv = np.asarray(depth_dir, float) * d
    F = [np.array(p, float) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
    T = [F[0], F[1], F[1] + dv, F[0] + dv]
    R = [F[1], F[1] + dv, F[2] + dv, F[2]]
    cv.s.fill(T, _shade(color, shade[1]), alpha)
    cv.s.fill(R, _shade(color, shade[2]), alpha)
    cv.s.fill(F, _shade(color, shade[0]), alpha)
    for face in (F, T, R):
        cv.s.polyline(face, color, alpha * edge, th, closed=True)


def log_px(n, a=30.0, b=90.0):
    """Display length for a tensor dimension of size n: a + b * log10(n) px (shapes are not drawn to scale)."""
    return a + b * np.log10(max(float(n), 1.0))


def block_labels(cv, a, labels, size=30, color=style.TEXT, alpha=1.0):
    """Rich-text labels around a block from tensor_block's anchors: keys h (left), w (below), d (right), top."""
    if not labels or alpha <= 0.01:
        return
    g = size * 0.6
    if "w" in labels:
        cv.rich(labels["w"], a["bottom"] + [0, g], size, alpha=alpha, fill=color, anchor="ma", align="m")
    if "h" in labels:
        cv.rich(labels["h"], a["left"] - [g, 0], size, alpha=alpha, fill=color, anchor="rm", align="r")
    if "d" in labels:
        cv.rich(labels["d"], a["depth_mid"] + [g, g * 0.3], size, alpha=alpha, fill=color, anchor="lm")
    if "top" in labels:
        cv.rich(labels["top"], a["top_edge"] - [0, g], size, alpha=alpha, fill=color, anchor="mb", align="m")


def tensor_morph(cv, xy, px_a, px_b, u, labels_a=None, labels_b=None, label_size=30, **kw):
    """Block morphing between two display sizes px = (w, h, d) as u goes 0 -> 1, labels cross-fading
    (e.g. (3, 1374, 1024) -> (1, 4122, 1024)). Extra kwargs go to tensor_block."""
    w, h, d = (np.asarray(px_a, float) + (np.asarray(px_b, float) - np.asarray(px_a, float)) * u)
    alpha = kw.pop("alpha", 1.0)
    a = tensor_block(cv, xy, w, h, d, alpha=alpha, **kw)
    block_labels(cv, a, labels_a, label_size, alpha=alpha * (1 - ss(u, 0.3, 0.5)))
    block_labels(cv, a, labels_b, label_size, alpha=alpha * ss(u, 0.5, 0.7))
    return a


def tensor_block(cv, xy, w, h, d, color=style.CYAN, alpha=1.0, grid=(1, 1, 1), gap=(0, 0, 0), colors=None,
                 labels=None, label_size=30, label_color=style.TEXT, label_alpha=None, depth_dir=style.DEPTH_DIR,
                 th=2, glow=0.0):
    """A tensor drawn as a w x h x d (px) cuboid whose front-top-left corner is xy, cut into grid = (n_h, n_w, n_d)
    sub-blocks separated by gap = (g_h, g_w, g_d) px. Animate gap 0 -> g to 'split along a dimension'.
    colors: optional fn(kh, kw, kd) -> RGB per sub-block. labels: {"h": ..., "w": ..., "d": ..., "top": ...} rich
    text (words + $math$, e.g. "$N$ (images) = 3"). Use log_px() for display sizes. Returns anchor points."""
    nh, nw, nd = grid
    gh, gw, gd = gap
    sh, sw, sd = h / nh, w / nw, d / nd
    dd = np.asarray(depth_dir, float)
    x0, y0 = xy
    for kd in reversed(range(nd)):              # far layers first
        for kh in reversed(range(nh)):          # lower slices first
            for kw in range(nw):                # left first
                off = kd * (sd + gd) * dd
                c = colors(kh, kw, kd) if colors else color
                box3(cv, x0 + kw * (sw + gw) + off[0], y0 + kh * (sh + gh) + off[1], sw, sh, sd, c, alpha, depth_dir,
                     th)
    W_ = nw * sw + (nw - 1) * gw
    H_ = nh * sh + (nh - 1) * gh
    D_ = nd * sd + (nd - 1) * gd
    dv = dd * D_
    a = {"front_tl": np.array([x0, y0]), "front_br": np.array([x0 + W_, y0 + H_]),
         "back_tr": np.array([x0 + W_, y0]) + dv, "center": np.array([x0 + W_ / 2, y0 + H_ / 2]) + dv / 2,
         "left": np.array([x0, y0 + H_ / 2]), "right": np.array([x0 + W_, y0 + H_ / 2]) + dv / 2,
         "top": np.array([x0 + W_ / 2, y0]) + dv / 2, "bottom": np.array([x0 + W_ / 2, y0 + H_]),
         "depth_mid": np.array([x0 + W_, y0 + H_]) + dv / 2, "top_edge": np.array([x0 + W_ / 2, y0]) + dv}
    if glow > 0:
        cv.g.polyline([a["front_tl"], (x0 + W_, y0), a["back_tr"], np.array([x0, y0]) + dv], color, glow * alpha, 4,
                      closed=True)
    block_labels(cv, a, labels, label_size, label_color, alpha if label_alpha is None else label_alpha)
    return a


# ---------------------------------------------------------------- tokens
def token(cv, c, size, color, alpha=1.0, glow=0.0, radius=0.2):
    """One token tile (rounded square) centred at c."""
    if alpha <= 0.01:
        return
    s = size / 2
    pts = rrect_pts(c[0] - s, c[1] - s, c[0] + s, c[1] + s, radius * size)
    cv.s.fill(pts, _shade(color, 0.8), alpha)
    cv.s.polyline(pts, color, alpha, 1, closed=True)
    if glow > 0:
        cv.g.fill(pts, color, min(1.0, glow * alpha))


def tokens(cv, xy, colors, size, alpha=1.0, glow=0.0, radius=0.2):
    """Many tiles: xy (n,2); colors one RGB or (n,3); alpha / glow scalar or (n,)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    n = len(xy)
    cols = np.broadcast_to(np.asarray(colors, float), (n, 3))
    al = np.broadcast_to(np.asarray(alpha, float), (n,))
    gl = np.broadcast_to(np.asarray(glow, float), (n,))
    for p, c, a, g in zip(xy, cols, al, gl):
        token(cv, p, size, tuple(c), a, g, radius)


def kind_colors(kinds, palette=None):
    """Array of kind names (any shape) -> (..., 3) colors from palette (default style.TOKEN)."""
    pal = palette or style.TOKEN
    k = np.asarray(kinds)
    return np.array([pal[s] for s in k.ravel()], float).reshape(k.shape + (3,))


def grid_xy(rows, cols, center, pitch, pitch_y=None):
    """Centres of a rows x cols grid of tiles -> (rows, cols, 2)."""
    py = pitch if pitch_y is None else pitch_y
    xs = (np.arange(cols) - (cols - 1) / 2) * pitch + center[0]
    ys = (np.arange(rows) - (rows - 1) / 2) * py + center[1]
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X, Y], -1)


def fold(n, head=4, tail=2, force=()):
    """Indices to show for a sequence of n tokens: the first `head`, the last `tail`, plus forced ones (e.g. 799)."""
    s = set(range(min(head, n))) | set(range(max(n - tail, 0), n)) | {int(i) for i in force if 0 <= i < n}
    return sorted(s)


class TokenStrip:
    """Rows of tokens with folding. n = real token count per row (e.g. 1374); show = indices that get a tile (see
    fold()); each gap between non-consecutive shown indices becomes a '…' slot. rows = frames stacked vertically.
    append = extra tokens at the end of each row (indices n, n+1, ..., e.g. a scale token)."""

    def __init__(self, n, show=None, rows=1, center=(540, 600), size=28, pitch=None, row_pitch=None, gap_w=1.6,
                 append=0):
        self.n, self.rows, self.size = n, rows, size
        self.pitch = pitch or size * 1.25
        self.row_pitch = row_pitch or size * 1.9
        show = list(range(n)) if show is None else sorted(show)
        slots, prev = [], None
        for i in show:
            if prev is not None and i > prev + 1:
                slots.append(("gap", (prev, i)))
            slots.append(("tok", i))
            prev = i
        slots += [("tok", n + k) for k in range(append)]
        widths = [self.pitch * (gap_w if kind == "gap" else 1.0) for kind, _ in slots]
        xs = np.cumsum(widths) - np.asarray(widths) / 2
        xs += center[0] - (np.sum(widths)) / 2
        self.x = {v: x for (kind, v), x in zip(slots, xs) if kind == "tok"}
        self.gap_x = [x for (kind, _), x in zip(slots, xs) if kind == "gap"]
        self.shown = [v for kind, v in slots if kind == "tok"]
        self.y = [center[1] + (r - (rows - 1) / 2) * self.row_pitch for r in range(rows)]
        self.width = float(np.sum(widths))

    def xy(self, idx, row=0):
        return np.array([self.x[idx], self.y[row]])

    def all_xy(self, row=0):
        """Centres of the shown tokens of one row, in index order: (k, 2), plus the index list."""
        return np.array([[self.x[i], self.y[row]] for i in self.shown]), list(self.shown)

    def draw(self, cv, kinds, alpha=1.0, rows=None, glow=0.0, palette=None, highlight=None, labels=None,
             label_size=None, band=None, row_alpha=None):
        """kinds: fn(row, idx) -> kind name, or an array (rows, n + append) of names (palette: style.TOKEN).
        highlight: {idx or (row, idx): color} outlines a token; labels: {idx or (row, idx): text} above it;
        band: {row: color} translucent band over a whole row (e.g. reference-view embedding)."""
        pal = palette or style.TOKEN
        kfn = kinds if callable(kinds) else (lambda r, i: kinds[r][i])
        for r in (range(self.rows) if rows is None else rows):
            ra = alpha * (1.0 if row_alpha is None else row_alpha[r])
            if ra <= 0.01:
                continue
            y = self.y[r]
            if band and r in band:
                x0, x1 = min(self.x.values()) - self.pitch * 0.6, max(self.x.values()) + self.pitch * 0.6
                h = self.size * 0.85
                cv.s.add_poly([(x0, y - h), (x1, y - h), (x1, y + h), (x0, y + h)], band[r], 0.35 * ra)
            for i in self.shown:
                token(cv, (self.x[i], y), self.size, pal[kfn(r, i)], ra, glow)
            for gx in self.gap_x:
                cv.text((gx, y), "…", int(self.size * 0.9), "light", ra, style.TEXT_DIM, "mm", shadow=False)
            for d, kind in ((highlight or {}, "hl"), (labels or {}, "lab")):
                for key, v in d.items():
                    rr, i = key if isinstance(key, tuple) else (r, key)
                    if rr != r or i not in self.x:
                        continue
                    c = np.array([self.x[i], y])
                    if kind == "hl":
                        s = self.size * 0.68
                        pts = [c + [-s, -s], c + [s, -s], c + [s, s], c + [-s, s]]
                        cv.s.polyline(pts, v, ra, 2, closed=True)
                        cv.g.polyline(pts, v, 0.7 * ra, 4, closed=True)
                    else:
                        cv.rich(v, c - [0, self.size * 0.8], label_size or int(self.size * 0.8), alpha=ra,
                                anchor="mb")


def reshape_rows(N, K, u, src_center, dst_center, pitch, row_pitch, stagger=0.5, arc=0.0, group_gap=0.0):
    """(N, K) grid of tokens -> one row of N*K as u goes 0 -> 1 (pass 1 - v for the reverse split).
    Token (i, k) goes to slot i*K + k: the slot comes from numpy-reshaping the id grid, not from a formula.
    group_gap adds space between frames in the merged row. Returns pos (N*K, 2) indexed by i*K + k, progress, ids."""
    ids = reshape_ids((N, K), lambda a: a.reshape(1, N * K))
    src = grid_xy(N, K, src_center, pitch, row_pitch)
    g = np.arange(N * K)
    x = (g - (N * K - 1) / 2) * pitch + (g // K - (N - 1) / 2) * group_gap + dst_center[0]
    dst = np.stack([x, np.full(N * K, float(dst_center[1]))], -1).reshape(1, N * K, 2)
    pos, uk = move_tokens(src, dst, ids, u, stagger, arc)
    return pos, uk, ids


def fly_token(cv, a, b, u, color, size=30, bend=0.3, alpha=1.0, glow=0.6, label=None, label_size=26):
    """One token flying from a to b along a curve (u 0 -> 1, eased), with a rich-text label riding above it."""
    from .arch import along, path
    p, _ = along(path(a, b, bend), ss(u, 0, 1))
    token(cv, p, size, color, alpha, glow)
    if label:
        cv.rich(label, p - [0, size * 0.85], label_size, alpha=alpha, anchor="mb")
    return p


# ---------------------------------------------------------------- reshape
def reshape_ids(src_shape, op):
    """Token ids = C-order flat index in src_shape. op maps the id tensor to the destination arrangement, e.g.
    lambda a: a.reshape(1, S * N) (frame-wise -> global) or lambda a: a.transpose(1, 0, 2). Returns the moved id
    tensor; raises if op is not a pure rearrangement."""
    n = int(np.prod(src_shape))
    ids = np.arange(n).reshape(src_shape)
    dst = np.asarray(op(ids))
    if dst.size != n or not np.array_equal(np.sort(dst.ravel()), np.arange(n)):
        raise ValueError("op must only rearrange elements (reshape / transpose / concat of the id tensor)")
    return dst


def move_tokens(src_xy, dst_xy, dst_ids, u, stagger=0.5, arc=0.0):
    """Positions of every token at progress u in [0, 1].
    src_xy: src_shape + (2,) screen centres, indexed like the source tensor (token id = C-order flat index).
    dst_xy: dst_ids.shape + (2,) screen centres of the destination slots; slot j holds token dst_ids.flat[j].
    stagger: 0 = all move together, ->1 = one after another in id order. arc: px of sideways bulge mid-move.
    Returns pos (n,2) and per-token progress (n,), both indexed by token id."""
    n = dst_ids.size
    P0 = np.asarray(src_xy, float).reshape(n, 2)
    P1 = np.empty((n, 2))
    P1[dst_ids.ravel()] = np.asarray(dst_xy, float).reshape(n, 2)
    k = np.arange(n)
    start = stagger * k / max(n - 1, 1)
    uk = np.array([ss(u, s, s + 1 - stagger) for s in start])
    pos = P0 + (P1 - P0) * uk[:, None]
    if arc:
        d = P1 - P0
        L = np.linalg.norm(d, axis=1, keepdims=True)
        perp = np.c_[-d[:, 1], d[:, 0]] / np.maximum(L, 1e-9)
        pos += perp * (arc * np.sin(np.pi * uk))[:, None]
    return pos, uk


def reshape_check(src_shape, dst_ids):
    """Values for check(): bijectivity and, for every token, its source multi-index and destination multi-index."""
    n = dst_ids.size
    src_idx = np.stack(np.unravel_index(np.arange(n), src_shape), -1)
    where = np.empty((n, dst_ids.ndim), int)
    where[dst_ids.ravel()] = np.stack(np.unravel_index(np.arange(n), dst_ids.shape), -1)
    return {"bijective": bool(np.array_equal(np.sort(dst_ids.ravel()), np.arange(n))),
            "src_index": src_idx, "dst_index": where}


# ---------------------------------------------------------------- attention
def softmax_rows(L):
    """Row softmax; -inf entries (masked) get weight 0."""
    L = np.asarray(L, float)
    m = np.max(np.where(np.isfinite(L), L, -np.inf), axis=1, keepdims=True)
    e = np.where(np.isfinite(L), np.exp(L - m), 0.0)
    return e / e.sum(1, keepdims=True)


def attention(q, k, v=None, mask=None, scale=None):
    """logits = q k^T * scale (default 1/sqrt(d)); mask (bool, True = may attend); A = row softmax.
    Returns logits, A, A @ v (None without v)."""
    q, k = np.asarray(q, float), np.asarray(k, float)
    scale = 1 / np.sqrt(q.shape[1]) if scale is None else scale
    L = q @ k.T * scale
    if mask is not None:
        L = np.where(mask, L, -np.inf)
    A = softmax_rows(L)
    return L, A, (None if v is None else A @ np.asarray(v, float))


def block_diag_mask(S, N):
    """(S*N, S*N) mask of frame-wise attention: token i may attend j only inside its own frame."""
    f = np.repeat(np.arange(S), N)
    return f[:, None] == f[None, :]


def cmap(x, stops=style.CMAP_STOPS):
    """Values in [0,1] -> RGB float array (..., 3)."""
    x = np.clip(np.asarray(x, float), 0, 1)
    vs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], float)
    return np.stack([np.interp(x, vs, cs[:, i]) for i in range(3)], -1)


def heatmap(cv, M, xy, cell, vmin=None, vmax=None, alpha=1.0, rows=None, gap=2, masked_color=(26, 30, 38),
            values=False, fmt="{:.2f}", value_size=None, highlight=None, frame_color=style.GREY):
    """Matrix M (n, m) as cells of `cell` px with top-left at xy. NaN / -inf cells use masked_color.
    rows: how many rows are visible (float, the last one fades in) - for row-by-row reveals.
    highlight: (i, j) cell or ("row", i) / ("col", j) to outline. Returns fn(i, j) -> cell centre."""
    M = np.asarray(M, float)
    n, m = M.shape
    fin = np.isfinite(M)
    lo = np.nanmin(np.where(fin, M, np.nan)) if vmin is None else vmin
    hi = np.nanmax(np.where(fin, M, np.nan)) if vmax is None else vmax
    col = cmap((np.where(fin, M, lo) - lo) / max(hi - lo, 1e-12))
    col[~fin] = masked_color
    x0, y0 = int(round(xy[0])), int(round(xy[1]))
    big = cv2.resize(col.astype(np.float32), (m * cell, n * cell), interpolation=cv2.INTER_NEAREST)
    mask = np.full((n * cell, m * cell), 255, np.uint8)
    if gap:
        for i in range(n + 1):
            mask[max(i * cell - gap // 2, 0):i * cell + (gap + 1) // 2] = 0
        for j in range(m + 1):
            mask[:, max(j * cell - gap // 2, 0):j * cell + (gap + 1) // 2] = 0
    if rows is not None:
        ra = np.clip(rows - np.arange(n), 0, 1)
        mask = (mask * np.repeat(ra, cell)[:, None]).astype(np.uint8)
    # blend per color channel through the mask
    H, W = cv.s.img.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + m * cell, W), min(y0 + n * cell, H)
    if xb > xa and yb > ya and alpha > 0:
        mk = mask[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)[..., None] * (alpha / 255)
        roi = cv.s.img[ya:yb, xa:xb].astype(np.float32)
        out = roi * (1 - mk) + big[ya - y0:yb - y0, xa - x0:xb - x0] * mk
        cv.s.img[ya:yb, xa:xb] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    cv.s.polyline([(x0, y0), (x0 + m * cell, y0), (x0 + m * cell, y0 + n * cell), (x0, y0 + n * cell)], frame_color,
                  0.6 * alpha, 1, closed=True)

    def center(i, j):
        return np.array([x0 + (j + 0.5) * cell, y0 + (i + 0.5) * cell])

    if values and alpha > 0:
        vs = value_size or int(cell * 0.32)
        for i in range(n):
            ra = 1.0 if rows is None else float(np.clip(rows - i, 0, 1))
            for j in range(m):
                if fin[i, j] and ra > 0:
                    dark = (M[i, j] - lo) / max(hi - lo, 1e-12) > 0.6
                    cv.text(center(i, j), fmt.format(M[i, j]), vs, "light", alpha * ra,
                            (20, 20, 20) if dark else style.TEXT, "mm", shadow=False)
    if highlight is not None and alpha > 0:
        if highlight[0] == "row":
            i = highlight[1]
            box = (x0, y0 + i * cell, x0 + m * cell, y0 + (i + 1) * cell)
        elif highlight[0] == "col":
            j = highlight[1]
            box = (x0 + j * cell, y0, x0 + (j + 1) * cell, y0 + n * cell)
        else:
            i, j = highlight
            box = (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell)
        pts = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
        cv.s.polyline(pts, style.ORANGE, alpha, 3, closed=True)
        cv.g.polyline(pts, style.ORANGE, 0.8 * alpha, 5, closed=True)
    return center
