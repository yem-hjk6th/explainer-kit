"""Frame canvas: two cv2 layers (sharp + glow) composited over a vignette background, plus a PIL text layer on top.

Coordinates are screen pixels (x right, y down), floats allowed: cv2 draws with 1/4-px precision (shift=2).
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import i18n, style
from .text import font, text_on


def ip(p):
    return int(round(p[0] * 4)), int(round(p[1] * 4))


def _c(color, alpha):
    return tuple(float(c) * alpha for c in color)


def _roi(img, P4, pad=3):
    """Clip the bbox of quarter-pixel points P4 to img; returns (x0, y0, x1, y1) or None."""
    H, W = img.shape[:2]
    x0, y0 = np.maximum(P4.min(0) // 4 - pad, 0)
    x1, y1 = np.minimum(P4.max(0) // 4 + pad, [W, H])
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def rrect_pts(x0, y0, x1, y1, r, n=6):
    """Polygon of a rounded rectangle."""
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    if r == 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return pts


class Layer:
    """uint8 RGB image drawn with anti-aliased cv2 calls. line/dot overwrite; add_poly adds; fill blends 'over'."""

    def __init__(self, img):
        self.img = img

    def line(self, a, b, color, alpha=1.0, th=2):
        cv2.line(self.img, ip(a), ip(b), _c(color, alpha), th, cv2.LINE_AA, 2)

    def polyline(self, pts, color, alpha=1.0, th=2, closed=False):
        P = np.round(np.asarray(pts, float) * 4).astype(np.int32)
        cv2.polylines(self.img, [P], closed, _c(color, alpha), th, cv2.LINE_AA, 2)

    def dashed(self, a, b, color, alpha=1.0, th=2, n=28):
        a, b = np.asarray(a, float), np.asarray(b, float)
        for k in range(0, n, 2):
            self.line(a + (b - a) * k / n, a + (b - a) * (k + 1) / n, color, alpha, th)

    def dot(self, p, color, rad, alpha=1.0):
        cv2.circle(self.img, ip(p), int(rad * 4), _c(color, alpha), -1, cv2.LINE_AA, 2)

    def dots(self, pts, colors, rad, alpha=1.0):
        """Many dots; colors is one RGB tuple or an (n, 3) array. Draw order = given order (sort far-to-near first)."""
        cols = np.broadcast_to(np.asarray(colors, float), (len(pts), 3))
        for p, c in zip(pts, cols):
            cv2.circle(self.img, ip(p), int(rad * 4), _c(c, alpha), -1, cv2.LINE_AA, 2)

    def add_poly(self, pts, color, alpha):
        """Translucent fill that adds light (epipolar_demo `poly`): img += alpha * color inside the polygon."""
        P = np.array([ip(p) for p in pts], np.int32)
        box = _roi(self.img, P)
        if box is None or alpha <= 0:
            return
        x0, y0, x1, y1 = box
        roi = self.img[y0:y1, x0:x1]
        over = np.zeros_like(roi)
        cv2.fillPoly(over, [P - [x0 * 4, y0 * 4]], tuple(float(c) for c in color), cv2.LINE_AA, 2)
        self.img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0, over, float(alpha), 0)

    def fill(self, pts, color, alpha=1.0):
        """Opaque-ish fill that covers what is under it in this layer (use for solid objects that occlude)."""
        P = np.array([ip(p) for p in pts], np.int32)
        box = _roi(self.img, P)
        if box is None or alpha <= 0:
            return
        x0, y0, x1, y1 = box
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [P - [x0 * 4, y0 * 4]], 255, cv2.LINE_AA, 2)
        self.blend(mask, x0, y0, color, alpha)

    def blend(self, mask, x, y, color, alpha=1.0):
        """'over' blend of a solid color through a uint8 mask whose top-left sits at integer (x, y)."""
        H, W = self.img.shape[:2]
        h, w = mask.shape
        xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if xb <= xa or yb <= ya:
            return
        m = mask[ya - y:yb - y, xa - x:xb - x].astype(np.float32)[..., None] * (alpha / 255.0)
        roi = self.img[ya:yb, xa:xb].astype(np.float32)
        out = roi * (1 - m) + np.asarray(color, np.float32) * m
        self.img[ya:yb, xa:xb] = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def rect(self, x0, y0, x1, y1, color, alpha=1.0, r=0):
        self.fill(rrect_pts(x0, y0, x1, y1, r), color, alpha)

    def blit_rgb(self, img, x, y, alpha=1.0, mask=None):
        """'over' blend of an RGB uint8 image (h, w, 3) with top-left at integer (x, y); mask (h, w) uint8 optional."""
        H, W = self.img.shape[:2]
        h, w = img.shape[:2]
        xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if xb <= xa or yb <= ya or alpha <= 0:
            return
        src = img[ya - y:yb - y, xa - x:xb - x].astype(np.float32)
        if mask is None and alpha >= 1:
            self.img[ya:yb, xa:xb] = img[ya - y:yb - y, xa - x:xb - x]
            return
        m = (np.full(src.shape[:2], 255.0, np.float32) if mask is None
             else mask[ya - y:yb - y, xa - x:xb - x].astype(np.float32))[..., None] * (alpha / 255.0)
        roi = self.img[ya:yb, xa:xb].astype(np.float32)
        self.img[ya:yb, xa:xb] = np.clip(roi * (1 - m) + src * m + 0.5, 0, 255).astype(np.uint8)

    def warp_quad(self, img, quad, alpha=1.0):
        """Paste an RGB image onto a screen quadrilateral (TL, TR, BR, BL) with a perspective warp."""
        quad = np.asarray(quad, np.float32)
        H, W = self.img.shape[:2]
        x0, y0 = np.floor(quad.min(0)).astype(int)
        x1, y1 = np.ceil(quad.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0 or alpha <= 0:
            return
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, quad - np.float32([x0, y0]))
        out = cv2.warpPerspective(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        self.blit_rgb(out, x0, y0, alpha, m)


def _blur(g, sigma):
    """Gaussian blur; large sigmas run on a downscaled copy (visually the same, ~5x faster)."""
    f = 4 if sigma >= 12 else 2 if sigma >= 5 else 1
    if f == 1:
        return cv2.GaussianBlur(g, (0, 0), sigma)
    H, W = g.shape[:2]
    small = cv2.resize(g, (W // f, H // f), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / f), (W, H), interpolation=cv2.INTER_LINEAR)


_formula_cache = {}


def background(W, H, seed=0, center_y=620):
    """Dark blue-grey vignette + fixed film grain (float32 RGB)."""
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - center_y) / H) ** 2)
    v = (14 * np.exp(-rr ** 2 / 0.18)).astype(np.float32)
    bg = np.stack([v * 0.9, v * 1.0, v * 1.15], -1)
    noise = np.random.default_rng(seed).normal(0, 1.2, (H, W, 1)).astype(np.float32)
    return np.clip(bg + noise, 0, 255)


class Canvas:
    """One frame. Draw geometry on `cv.s` (sharp) and `cv.g` (glow source); text/formulas go on the top layer."""

    def __init__(self, W, H, t, bg, glow=((0.9, 6), (0.7, 22))):
        self.W, self.H, self.t = W, H, t
        self.s = Layer(np.zeros((H, W, 3), np.uint8))
        self.g = Layer(np.zeros((H, W, 3), np.uint8))
        self.bg = bg
        self.glow = glow
        self.layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.dr = ImageDraw.Draw(self.layer)

    # ---- text layer ----
    def text(self, xy, s, size=30, kind="text", alpha=1.0, fill=style.TEXT, anchor="la", shadow=True, align="left",
             spacing=None):
        s0, s = s, i18n.tr(s)
        f = font(i18n.kind_for(kind, s), size)
        if s is not s0 and alpha > 0.01:
            bb = (self.dr.multiline_textbbox if "\n" in s else self.dr.textbbox)(tuple(xy), s, font=f, anchor=anchor)
            i18n.note_box(s0, s, bb, self.W, self.H)
        text_on(self.dr, xy, s, f, alpha, fill, anchor, shadow, align, spacing)

    def math(self, tex, xy, size=40, color=style.MATH, alpha=1.0, anchor="mm", glow=0.0, shadow=True):
        """One-piece formula / symbol (mathtext, Computer Modern). Returns its screen bbox."""
        from .formula import Formula
        k = (tex, size)
        if k not in _formula_cache:
            _formula_cache[k] = Formula([tex], size)
        return _formula_cache[k].draw(self, xy, color, alpha, anchor, glow=glow, shadow=shadow)

    def rich(self, s, xy, size=30, kind="text", alpha=1.0, fill=style.TEXT, anchor="la", align="l", math_color=None,
             shadow=True):
        """Text with inline $math$ (e.g. "$R_i$: unit ray map, $3\\times H\\times W$"), multi-line with \\n."""
        from .richtext import rich
        return rich(s, size, kind).draw(self, xy, fill, alpha, anchor, align, math_color, shadow)

    def text_rot(self, xy, s, size=26, kind="text", alpha=1.0, fill=style.TEXT_DIM, angle=90):
        """Plain text rotated by angle degrees (counter-clockwise), centred at xy (vertical ruler labels)."""
        s = i18n.tr(s)
        if alpha <= 0.01:
            return
        f = font(i18n.kind_for(kind, s), size)
        w, h = int(f.getlength(s)) + 8, int(size * 1.5)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((w / 2, h / 2), s, font=f, fill=tuple(fill[:3]) + (int(255 * alpha),), anchor="mm")
        im = im.rotate(angle, expand=True, resample=Image.BICUBIC)
        self.paste_rgba(np.asarray(im), int(round(xy[0] - im.width / 2)), int(round(xy[1] - im.height / 2)))

    def badge(self, xy, s, size=26, alpha=1.0, anchor="ra", fill=style.TEXT_DIM, bg=(40, 44, 52), border=None):
        """Pill-shaped tag ("schematic" / "example values" / "paper Tab. 3" ...) anchored at xy; returns its box."""
        from .formula import anchor_xy
        from .richtext import rich
        if alpha <= 0.01:
            return None
        r = rich(s, size)
        w, h = r.wh
        px, py = 12, 6
        x0, y0 = anchor_xy(xy, w + 2 * px, h + 2 * py, h + py, anchor)
        self.s.rect(x0, y0, x0 + w + 2 * px, y0 + h + 2 * py, bg, 0.85 * alpha, r=8)
        if border:
            self.s.polyline(rrect_pts(x0, y0, x0 + w + 2 * px, y0 + h + 2 * py, 8), border, alpha, 1, closed=True)
        r.draw(self, (x0 + px, y0 + py), fill, 0.9 * alpha, "la", shadow=False)
        return x0, y0, x0 + w + 2 * px, y0 + h + 2 * py

    def paste_rgba(self, arr, x, y):
        """alpha-composite an (h, w, 4) uint8 array onto the text layer at integer (x, y), clipped to the frame."""
        h, w = arr.shape[:2]
        xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, self.W), min(y + h, self.H)
        if xb <= xa or yb <= ya:
            return
        im = Image.fromarray(np.ascontiguousarray(arr[ya - y:yb - y, xa - x:xb - x]), "RGBA")
        self.layer.alpha_composite(im, (xa, ya))

    def image(self, im, xy, w=None, alpha=1.0, anchor="la", layer="s", nearest=False):
        """Paste a picture (path, PIL image or RGB / RGBA array) scaled to width w. Default layer 's' = the sharp
        layer, so lines / points drawn afterwards stay on top; layer='text' puts it above everything."""
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        im = (Image.open(im) if not isinstance(im, Image.Image) else im).convert("RGBA")
        if w is not None:
            im = im.resize((int(round(w)), int(round(im.height * w / im.width))),
                           Image.NEAREST if nearest else Image.LANCZOS)
        arr = np.asarray(im).copy()
        x, y = xy
        x -= {"l": 0, "m": im.width / 2, "r": im.width}[anchor[0]]
        y -= {"a": 0, "t": 0, "m": im.height / 2, "b": im.height}[anchor[1]]
        x, y = int(round(x)), int(round(y))
        if layer == "text":
            arr[..., 3] = (arr[..., 3] * alpha).astype(np.uint8)
            self.paste_rgba(arr, x, y)
        else:
            self.s.blit_rgb(arr[..., :3], x, y, alpha, arr[..., 3])
        return x, y, x + im.width, y + im.height

    # ---- final image ----
    def composite(self):
        img = self.bg.copy()
        img += self.s.img
        if self.g.img.any():
            gf = self.g.img.astype(np.float32)
            for k, sigma in self.glow:
                img += k * _blur(gf, sigma)
        np.clip(img, 0, 255, out=img)
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
        pil.alpha_composite(self.layer)
        return np.asarray(pil.convert("RGB"))
