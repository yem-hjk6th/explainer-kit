"""Popup: a small window that pops out of a corner of the frame; inside it is an independent small canvas."""
from functools import lru_cache

import cv2
import numpy as np

from . import style
from .draw import Canvas, rrect_pts
from .ease import ss

POPUP_BG = (22, 28, 38)
POPUP_TITLE_H = 50
POPUP_PAD = 16
CAPTION_TOP = style.CAPTION_Y - 15          # 1250: the caption band starts here; popups must stay above


@lru_cache(maxsize=None)
def _popup_bg(w, h):
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    base = np.asarray(POPUP_BG, np.float32)[None, None, :]
    return np.broadcast_to(base * (1.08 - 0.16 * yy), (h, w, 3)).astype(np.float32)


@lru_cache(maxsize=None)
def _popup_mask(w, h, r=16):
    m = np.zeros((h, w), np.uint8)
    pts = np.round(np.asarray(rrect_pts(0, 0, w - 1, h - 1, r, 8))).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
    return m


class Popup:
    """Popup: a small window pops out of a corner to explain a side concept quickly (DPT, quaternions, ...),
    then folds back, without interrupting the main line.

    Popup(box, t_in, t_out, title, corner="tr", pop=0.35, accent=ORANGE, point_to=None, tag=None,
          title_size=28, title_dy=0)
      box      (x0, y0, w, h): position and size of the fully open window on the main frame (px).
               y0 + h must stay out of the caption band (>= 1250).
      t_in     time the window starts to open (main timeline, s); t_out = time it starts to close.
               Opening and closing each take `pop` seconds (smoothstep).
      corner   the corner it grows out of and shrinks back into: "tl" | "tr" | "bl" | "br"
               (that corner stays fixed during the animation).
      title    title-bar text (rich, may contain $math$); accent = border / title-bar color;
               tag = small label at the right end of the title bar (rich).
               title_size / title_dy: title font size and downward offset (px); use 26 / 5 when a tall
               symbol (a tilde, ...) in the title touches the top border.
      point_to a point (x, y) on the main frame: if given, a dashed leader line runs from the nearest window
               corner to it ("this is what the popup is about").
    Call pop.draw(cv, t, body) at the **end** of the frame function; body(sub, lt, box) draws on the small canvas:
      sub  an animkit Canvas of size w x h, origin = window top-left; sub.s / sub.g / text / rich / math /
           Formula / readout work as usual
      lt   window-local time (s, counted from t_in); use ss(lt, a, b) for small animations inside the window
      box  content area (x0, y0, x1, y1) in small-canvas coordinates, title bar and padding already removed
    The window is pasted on the main frame's text layer (it covers lines, points and text drawn before it) and
    sits below captions / titles / corner tags.
    pop.on(t) returns how open the window is, 0-1 (the main frame can dim its background or make room with it);
    pop.local(t) = t - t_in.
    draw returns the window's current box on the main frame (x0, y0, x1, y1), or None while closed.
    """

    def __init__(self, box, t_in, t_out, title="", corner="tr", pop=0.35, accent=None, point_to=None, tag=None,
                 title_size=28, title_dy=0):
        x0, y0, w, h = box
        assert x0 >= 0 and x0 + w <= style.W and y0 >= 0, f"popup {title!r} leaves the frame: {box}"
        assert y0 + h <= CAPTION_TOP, f"popup {title!r} enters the caption band: {box}"
        assert t_out > t_in + 2 * pop, f"popup {title!r} is too short"
        self.box = (int(x0), int(y0), int(w), int(h))
        self.t_in, self.t_out, self.pop = t_in, t_out, pop
        self.title, self.corner, self.point_to, self.tag = title, corner, point_to, tag
        self.title_size, self.title_dy = title_size, title_dy
        self.accent = style.ORANGE if accent is None else accent

    def on(self, t):
        return ss(t, self.t_in, self.t_in + self.pop) * (1.0 - ss(t, self.t_out, self.t_out + self.pop))

    def local(self, t):
        return t - self.t_in

    def content_box(self):
        w, h = self.box[2:]
        return (POPUP_PAD, POPUP_TITLE_H + POPUP_PAD // 2, w - POPUP_PAD, h - POPUP_PAD)

    def draw(self, cv, t, body=None):
        u = self.on(t)
        if u <= 0.01:
            return None
        x0, y0, w, h = self.box
        sub = Canvas(w, h, t, _popup_bg(w, h), cv.glow)
        acc = self.accent
        sub.s.rect(0, 0, w, POPUP_TITLE_H, tuple(int(0.22 * c) for c in acc), 1.0, r=0)
        if body is not None:
            body(sub, t - self.t_in, self.content_box())
        sub.s.line((0, POPUP_TITLE_H), (w, POPUP_TITLE_H), acc, 0.6, 1)
        if self.title:
            sub.rich(self.title, (POPUP_PAD, POPUP_TITLE_H / 2 + self.title_dy), self.title_size, "bold", 1.0,
                     anchor="lm", shadow=False)
        if self.tag:
            sub.badge((w - POPUP_PAD, POPUP_TITLE_H / 2 - 17), self.tag, 22, 1.0, anchor="ra")
        rgb = np.ascontiguousarray(sub.composite()).copy()
        border = np.round(np.asarray(rrect_pts(1, 1, w - 2, h - 2, 15, 8))).astype(np.int32)
        cv2.polylines(rgb, [border], True, acc, 2, cv2.LINE_AA)
        # scale open: the anchor corner stays fixed
        sc_ = 0.6 + 0.4 * u
        sw, sh = max(2, int(round(w * sc_))), max(2, int(round(h * sc_)))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(_popup_mask(w, h), (sw, sh), interpolation=cv2.INTER_AREA)
        px = x0 if self.corner[1] == "l" else x0 + w - sw
        py = y0 if self.corner[0] == "t" else y0 + h - sh
        # leader line: point on the main frame -> nearest window corner
        if self.point_to is not None:
            corners = [(px, py), (px + sw, py), (px, py + sh), (px + sw, py + sh)]
            q = np.asarray(self.point_to, float)
            c = min(corners, key=lambda p: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            n = 14
            for k in range(n):
                if k % 2 == 0:
                    a = q + (np.asarray(c) - q) * k / n
                    b = q + (np.asarray(c) - q) * (k + 1) / n
                    cv.dr.line([tuple(a), tuple(b)], fill=tuple(acc) + (int(220 * u),), width=2)
            cv.dr.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=tuple(acc) + (int(230 * u),))
        # shadow + window
        sh_a = (mask.astype(np.float32) * 0.55 * u).astype(np.uint8)
        cv.paste_rgba(np.dstack([np.zeros((sh, sw, 3), np.uint8), sh_a]), px + 6, py + 8)
        a = (mask.astype(np.float32) * u).astype(np.uint8)
        cv.paste_rgba(np.dstack([small, a]), px, py)
        return px, py, px + sw, py + sh


def popup_check(ck, pops, shot_dur):
    """For @shot.check: every popup ends within the shot, and no two overlap in both time and position."""
    for p in pops:
        ck.true(f"popup {p.title!r} ends within the shot", p.t_out + p.pop <= shot_dur + 1e-6, f"{p.t_in}–{p.t_out}")
    for i, p in enumerate(pops):
        for q in pops[i + 1:]:
            ov_t = p.t_in < q.t_out + q.pop and q.t_in < p.t_out + p.pop
            bx, by, bw, bh = p.box
            cx, cy, cw, chh = q.box
            ov_xy = bx < cx + cw and cx < bx + bw and by < cy + chh and cy < by + bh
            ck.true(f"popup {p.title!r} / {q.title!r} do not overlap", not (ov_t and ov_xy))
