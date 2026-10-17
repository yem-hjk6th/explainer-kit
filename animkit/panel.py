"""Image panels: an RGB array (scene render, depth map, ...) placed on screen, with pixel grid, cell highlight,
ruler, pixel markers; depth/heat maps via colorize() + colorbar()."""
import cv2
import numpy as np

from . import style
from .tensor import cmap


class ImagePanel:
    """img (h, w, 3) uint8 RGB placed with its top-left at xy and width w px (or height h px).
    Image coordinates (u, v) are continuous pixels: pixel (row r, col c) covers [c, c+1) x [r, r+1)."""

    def __init__(self, img, xy, w=None, h=None, nearest=False):
        self.img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        self.ih, self.iw = self.img.shape[:2]
        self.s = (w / self.iw) if w else (h / self.ih)
        self.x0, self.y0 = float(xy[0]), float(xy[1])
        self.nearest = nearest
        self._scaled = None

    @property
    def box(self):
        return self.x0, self.y0, self.x0 + self.iw * self.s, self.y0 + self.ih * self.s

    def to_screen(self, u, v):
        return np.array([self.x0 + u * self.s, self.y0 + v * self.s])

    def draw(self, cv, alpha=1.0, border=style.WHITE, border_alpha=0.8, title=None, title_size=28):
        if alpha <= 0.01:
            return
        if self._scaled is None:
            size = (int(round(self.iw * self.s)), int(round(self.ih * self.s)))
            interp = cv2.INTER_NEAREST if self.nearest else (cv2.INTER_AREA if self.s < 1 else cv2.INTER_LINEAR)
            self._scaled = cv2.resize(self.img, size, interpolation=interp)
        cv.s.blit_rgb(self._scaled, int(round(self.x0)), int(round(self.y0)), alpha)
        x0, y0, x1, y1 = self.box
        if border:
            cv.s.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], border, border_alpha * alpha, 2, closed=True)
        if title:
            cv.rich(title, ((x0 + x1) / 2, y0 - 12), title_size, alpha=alpha, anchor="mb")

    def grid(self, cv, cell=14, color=style.WHITE, alpha=0.45, progress=1.0, th=1):
        """Lines every `cell` image px (ViT patches); progress 0-1 grows the grid from the top-left."""
        if alpha <= 0.01 or progress <= 0:
            return
        x0, y0, x1, y1 = self.box
        xe = x0 + (x1 - x0) * progress
        ye = y0 + (y1 - y0) * progress
        for c in range(0, self.iw + 1, cell):
            x = x0 + c * self.s
            if x <= xe + 0.5:
                cv.s.line((x, y0), (x, ye), color, alpha, th)
        for r in range(0, self.ih + 1, cell):
            y = y0 + r * self.s
            if y <= ye + 0.5:
                cv.s.line((x0, y), (xe, y), color, alpha, th)

    def cell(self, cv, row, col, cell=14, color=style.ORANGE, alpha=1.0, fill=0.25, glow=0.6):
        """Outline (and lightly fill) patch (row, col) of a `cell`-px grid. Returns its screen box."""
        a = self.to_screen(col * cell, row * cell)
        b = self.to_screen(min((col + 1) * cell, self.iw), min((row + 1) * cell, self.ih))
        pts = [(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])]
        if alpha > 0.01:
            if fill:
                cv.s.add_poly(pts, color, fill * alpha)
            cv.s.polyline(pts, color, alpha, 2, closed=True)
            if glow:
                cv.g.polyline(pts, color, glow * alpha, 4, closed=True)
        return a[0], a[1], b[0], b[1]

    def scan(self, cv, u, cell=14, **kw):
        """Highlight the patch reached after a fraction u of a raster scan (row by row)."""
        nr, nc = -(-self.ih // cell), -(-self.iw // cell)
        k = min(int(u * nr * nc), nr * nc - 1)
        return self.cell(cv, k // nc, k % nc, cell, **kw)

    def tiles(self, cv, xs, ys, size, progress=1.0, alpha=1.0, color=style.CYAN, color2=style.VIOLET, highlight=None,
              overlap_color=None, overlap_alpha=0.3, voronoi=0.0, voronoi_fill=None, voronoi_fill_alpha=0.35,
              keep=None, keep_alpha=1.0, keep_color=style.GOLD, stagger=4, drop=18, th=2):
        """Overlapping tiles (sliding windows) on the image. xs / ys = tile start per column / row (image px, see
        tile_starts); size = tile side (int) or (w, h); tiles that run past the image are clipped to it.
        progress 0-1: tiles drop in one by one (raster order); neighbours alternate +-`stagger` screen px in y so
        outlines do not coincide. highlight={(row, col): color}: bolder outline + fill.
        overlap_color: fill where >= 2 tiles cover (stronger where more do), strength overlap_alpha.
        voronoi (0-1): dashed lines midway between neighbouring tile centres = the cell each pixel belongs to if
        it is assigned to the nearest tile centre; voronoi_fill=True (or a list of RGB) tints the cells.
        keep=(row, col): dim everything outside that tile's Voronoi cell (strength keep_alpha).
        Returns {"boxes": {(r, c): screen box}, "centers": {(r, c): screen xy}, "vor_x", "vor_y": cell borders in
        image px, "owner": fn(u, v) -> (row, col) of the nearest tile centre}."""
        xs, ys = [float(v) for v in xs], [float(v) for v in ys]
        sw, sh = (size, size) if np.ndim(size) == 0 else size
        nr, nc = len(ys), len(xs)
        xr = [(a, min(a + sw, self.iw)) for a in xs]
        yr = [(a, min(a + sh, self.ih)) for a in ys]
        cx = np.array([(a + b) / 2 for a, b in xr])
        cy = np.array([(a + b) / 2 for a, b in yr])
        mx, my = (cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2
        bx, by = np.r_[0, mx, self.iw], np.r_[0, my, self.ih]

        def owner(u, v):
            return int(np.searchsorted(my, v, side="right")), int(np.searchsorted(mx, u, side="right"))

        def sbox(r, c):
            a, b = self.to_screen(xr[c][0], yr[r][0]), self.to_screen(xr[c][1], yr[r][1])
            return float(a[0]), float(a[1]), float(b[0]), float(b[1])

        out = {"boxes": {(r, c): sbox(r, c) for r in range(nr) for c in range(nc)},
               "centers": {(r, c): self.to_screen(cx[c], cy[r]) for r in range(nr) for c in range(nc)},
               "vor_x": bx, "vor_y": by, "owner": owner}
        if alpha <= 0.01:
            return out
        X0, Y0 = int(round(self.x0)), int(round(self.y0))
        Wp, Hp = int(round(self.iw * self.s)), int(round(self.ih * self.s))
        need = (overlap_color is not None) or voronoi_fill or (keep is not None)
        if need:
            u = (X0 + np.arange(Wp) + 0.5 - self.x0) / self.s
            v = (Y0 + np.arange(Hp) + 0.5 - self.y0) / self.s
            ci, ri = np.searchsorted(mx, u, side="right"), np.searchsorted(my, v, side="right")
        if voronoi_fill and voronoi_fill_alpha > 0.01:
            pal = voronoi_fill if isinstance(voronoi_fill, (list, tuple)) else [
                style.CYAN, style.GOLD, style.VIOLET, style.GREEN, style.ORANGE]
            lut = np.array([[pal[(c + 2 * r) % len(pal)] for c in range(nc)] for r in range(nr)], np.uint8)
            cv.s.blit_rgb(np.ascontiguousarray(lut[ri[:, None], ci[None, :]]), X0, Y0, voronoi_fill_alpha * alpha)
        if overlap_color is not None and overlap_alpha > 0.01:
            nx = np.array([sum(a <= uu < b for a, b in xr) for uu in u])
            ny = np.array([sum(a <= vv < b for a, b in yr) for vv in v])
            cnt = ny[:, None] * nx[None, :]
            m = (np.clip(cnt - 1, 0, 3) / 3.0 * 255).astype(np.uint8)
            m = np.where(cnt >= 2, np.maximum(m, 85), 0).astype(np.uint8)
            cv.s.blend(m, X0, Y0, overlap_color, overlap_alpha * alpha)
        if keep is not None and keep_alpha > 0.01:
            m = np.where((ri[:, None] != keep[0]) | (ci[None, :] != keep[1]), 255, 0).astype(np.uint8)
            cv.s.blend(m, X0, Y0, (0, 0, 0), 0.62 * keep_alpha * alpha)
            x0_, x1_, y0_, y1_ = bx[keep[1]], bx[keep[1] + 1], by[keep[0]], by[keep[0] + 1]
            a, b = self.to_screen(x0_, y0_), self.to_screen(x1_, y1_)
            pts = [(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])]
            cv.s.polyline(pts, keep_color, keep_alpha * alpha, 3, closed=True)
            cv.g.polyline(pts, keep_color, 0.7 * keep_alpha * alpha, 6, closed=True)
        if voronoi > 0.01:
            va = voronoi * alpha
            for u_ in bx[1:-1]:
                a, b = self.to_screen(u_, 0), self.to_screen(u_, self.ih)
                cv.s.dashed(a, b, style.WHITE, 0.85 * va, 2, n=int(np.hypot(*(b - a)) / 8) // 2 * 2)
            for v_ in by[1:-1]:
                a, b = self.to_screen(0, v_), self.to_screen(self.iw, v_)
                cv.s.dashed(a, b, style.WHITE, 0.85 * va, 2, n=int(np.hypot(*(b - a)) / 8) // 2 * 2)
        n = nr * nc
        hl = highlight or {}
        for k in range(n):
            r, c = divmod(k, nc)
            uk = float(np.clip(progress * (n + 1) - k, 0, 1))
            uk = uk * uk * (3 - 2 * uk)
            a = alpha * uk
            if a <= 0.01:
                continue
            dy = ((r + c) % 2 * 2 - 1) * stagger - (1 - uk) * drop
            x0_, y0_, x1_, y1_ = out["boxes"][(r, c)]
            pts = [(x0_, y0_ + dy), (x1_, y0_ + dy), (x1_, y1_ + dy), (x0_, y1_ + dy)]
            col = hl.get((r, c)) or (color if (r + c) % 2 == 0 else color2)
            if (r, c) in hl:
                cv.s.add_poly(pts, col, 0.22 * a)
                cv.s.polyline(pts, col, a, th + 2, closed=True)
                cv.g.polyline(pts, col, 0.8 * a, th + 4, closed=True)
            else:
                cv.s.add_poly(pts, col, 0.03 * a)
                cv.s.polyline(pts, col, 0.85 * a, th, closed=True)
                cv.g.polyline(pts, col, 0.12 * a, th + 2, closed=True)
        return out

    def ruler(self, cv, side="top", label=None, color=style.TEXT_DIM, alpha=1.0, offset=24, size=26):
        """Dimension line along one side with end ticks and a label (default: '<n> px')."""
        if alpha <= 0.01:
            return
        x0, y0, x1, y1 = self.box
        if side in ("top", "bottom"):
            y = y0 - offset if side == "top" else y1 + offset
            a, b, n = (x0, y), (x1, y), self.iw
            tk = [(0, -7), (0, 7)]
            lab_xy, anchor = ((x0 + x1) / 2, y - 8 if side == "top" else y + 8), "mb" if side == "top" else "ma"
        else:
            x = x0 - offset if side == "left" else x1 + offset
            a, b, n = (x, y0), (x, y1), self.ih
            tk = [(-7, 0), (7, 0)]
            lab_xy, anchor = (x - size * 0.75 if side == "left" else x + size * 0.75, (y0 + y1) / 2), None
        cv.s.line(a, b, color, alpha, 1)
        for p in (a, b):
            cv.s.line(np.add(p, tk[0]), np.add(p, tk[1]), color, alpha, 1)
        if anchor:
            cv.rich(label or f"{n} px", lab_xy, size, alpha=alpha, fill=color, anchor=anchor)
        else:      # vertical side: rotated label reading bottom-to-top
            cv.text_rot(lab_xy, label or f"{n} px", size, alpha=alpha, fill=color)

    def mark(self, cv, u, v, color=style.WHITE, rad=6, alpha=1.0, label=None, label_offset=(14, -14), size=26,
             label_color=style.TEXT):
        """Glowing dot at image coords (u, v) (+ label, $math$ allowed). Returns the screen point."""
        p = self.to_screen(u, v)
        if alpha > 0.01:
            cv.g.dot(p, color, rad * 1.6, alpha)
            cv.s.dot(p, (255, 255, 255), rad * 0.6, alpha)
            if label:
                cv.rich(label, p + np.asarray(label_offset, float), size, alpha=alpha, fill=label_color, anchor="lb")
        return p


def tile_starts(length, size, stride):
    """Start of each sliding-window tile along one axis: 0, stride, 2 stride, ...; the last tile is pushed back so it
    ends exactly at `length` (e.g. tile_starts(1536, 384, 288) = [0, 288, 576, 864, 1152])."""
    n = int(np.ceil(max(length - size, 0) / stride)) + 1
    return [min(k * stride, max(length - size, 0)) for k in range(n)]


def colorize(D, vmin=None, vmax=None, stops=style.CMAP_DEPTH, bad=(28, 30, 38)):
    """2D array -> uint8 RGB with a colormap; inf / nan pixels get `bad` (background)."""
    D = np.asarray(D, float)
    ok = np.isfinite(D)
    lo = np.min(D[ok]) if vmin is None else vmin
    hi = np.max(D[ok]) if vmax is None else vmax
    rgb = cmap((np.where(ok, D, lo) - lo) / max(hi - lo, 1e-12), stops)
    rgb[~ok] = bad
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def colorbar(cv, box, vmin, vmax, unit="", fmt="{:.2f}", stops=style.CMAP_DEPTH, alpha=1.0, size=24, vertical=True,
             label=None):
    """Gradient bar in box (x0, y0, x1, y1) with min / max labels (+ unit). Vertical: vmin at the top."""
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    n = (y1 - y0) if vertical else (x1 - x0)
    ramp = cmap(np.linspace(0, 1, n), stops)
    img = np.repeat(ramp[:, None], x1 - x0, 1) if vertical else np.repeat(ramp[None], y1 - y0, 0)
    cv.s.blit_rgb(np.clip(img + 0.5, 0, 255).astype(np.uint8), x0, y0, alpha)
    cv.s.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style.GREY, alpha, 1, closed=True)
    u = f" {unit}" if unit else ""
    if vertical:
        cv.rich(fmt.format(vmin) + u, (x1 + 10, y0), size, alpha=alpha, fill=style.TEXT_DIM, anchor="lt")
        cv.rich(fmt.format(vmax) + u, (x1 + 10, y1), size, alpha=alpha, fill=style.TEXT_DIM, anchor="lb")
        if label:
            cv.rich(label, ((x0 + x1) / 2, y0 - 10), size, alpha=alpha, fill=style.TEXT_DIM, anchor="mb")
    else:
        cv.rich(fmt.format(vmin) + u, (x0, y1 + 8), size, alpha=alpha, fill=style.TEXT_DIM, anchor="la")
        cv.rich(fmt.format(vmax) + u, (x1, y1 + 8), size, alpha=alpha, fill=style.TEXT_DIM, anchor="ra")
        if label:
            cv.rich(label, ((x0 + x1) / 2, y0 - 8), size, alpha=alpha, fill=style.TEXT_DIM, anchor="mb")
