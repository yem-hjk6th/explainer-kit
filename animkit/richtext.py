"""Mixed text: plain runs (PIL font, Chinese / English) and $math$ runs (mathtext) on one baseline, multi-line with \\n.
Also Digits: a glyph set for numbers that change every frame (counters, vectors) without re-typesetting."""
import re

import numpy as np

from . import i18n, style
from .formula import anchor_xy, get, paste_mask, request
from .text import font, text_on

_MATH = re.compile(r"\$([^$]+)\$")
MATH_SCALE = 1.12        # CM looks smaller than msyh at the same px size


class Rich:
    def __init__(self, s, size=30, kind="text", math_scale=MATH_SCALE, line_gap=0.3):
        s0, s = s, i18n.tr(s)
        self._src = s0 if s is not s0 else None          # Chinese original, kept to report off-canvas translations
        self.text = s
        kind = i18n.kind_for(kind, s)
        self.size, self.kind, self.line_gap = size, kind, line_gap * size
        self.msize = int(round(size * math_scale))
        self.lines = []
        for line in s.split("\n"):
            runs, pos = [], 0
            for m in _MATH.finditer(line):
                if m.start() > pos:
                    runs.append(("t", line[pos:m.start()]))
                runs.append(("m", request(m.group(1), self.msize)))
                pos = m.end()
            if pos < len(line) or not runs:
                runs.append(("t", line[pos:]))
            self.lines.append(runs)
        self._lay = None

    def _layout(self):
        if self._lay is None:
            f = font(self.kind, self.size)
            fa, fd = f.getmetrics()
            rows, y, W = [], 0.0, 0.0
            for runs in self.lines:
                placed, x, asc, dep = [], 0.0, fa * 0.8, fd * 0.8
                for kind, p in runs:
                    if kind == "t":
                        w = f.getlength(p)
                        placed.append(("t", p, x, w, 0))
                    else:
                        mask, m = get(p)
                        a = m["asc"]
                        asc, dep = max(asc, a), max(dep, m["desc"])
                        w = m["width"] + 0.12 * self.size
                        placed.append(("m", mask, x + 0.06 * self.size, w, a))
                    x += w
                rows.append((placed, x, y + asc))
                W = max(W, x)
                y += asc + dep + self.line_gap
            self._lay = (rows, W, y - self.line_gap)
        return self._lay

    @property
    def wh(self):
        _, w, h = self._layout()
        return w, h

    def draw(self, cv, xy, fill=style.TEXT, alpha=1.0, anchor="la", align="l", math_color=None, shadow=True):
        rows, W, H = self._layout()
        asc0 = rows[0][2] if rows else 0
        x0, y0 = anchor_xy(xy, W, H, asc0, anchor)
        if alpha > 0.01:
            f = font(self.kind, self.size)
            mc = math_color or fill
            for placed, w, base in rows:
                dx = {"l": 0, "m": (W - w) / 2, "r": W - w}[align]
                for kind, p, x, _, a in placed:
                    if kind == "t":
                        text_on(cv.dr, (x0 + dx + x, y0 + base), p, f, alpha, fill, "ls", shadow)
                    else:
                        paste_mask(cv, p, int(round(x0 + dx + x)), int(round(y0 + base - a)), mc, alpha, shadow)
            if self._src:
                i18n.note_box(self._src, self.text, (x0, y0, x0 + W, y0 + H), cv.W, cv.H)
        return x0, y0, x0 + W, y0 + H


_UNIT = re.compile(r"\$[^$]+\$|[A-Za-z0-9_\-.,:;%()/']+\s*|\s+|.")


def wrap(s, width, size=30, kind="text", math_scale=MATH_SCALE):
    """Greedy line breaking of rich text to `width` px: $math$ spans and English words never split, CJK can break
    between any two characters. Existing \\n are kept. Returns the text with \\n inserted (translated first)."""
    s = i18n.tr(s)
    kind = i18n.kind_for(kind, s)
    f = font(kind, size)
    msize = int(round(size * math_scale))
    out = []
    for para in s.split("\n"):
        line, w = "", 0.0
        for u in _UNIT.findall(para):
            if u.startswith("$"):
                uw = get(request(u[1:-1], msize))[1]["width"] + 0.12 * size
            else:
                uw = f.getlength(u)
            if line and w + uw > width and not u.isspace():
                out.append(line.rstrip())
                line, w = "", 0.0
                u = u.lstrip()
                uw = f.getlength(u) if not u.startswith("$") else uw
            line += u
            w += uw
        out.append(line.rstrip())
    return i18n.Done("\n".join(out))


_rich_cache = {}


def rich(s, size=30, kind="text", math_scale=MATH_SCALE):
    k = (s, size, kind, math_scale)
    if k not in _rich_cache:
        _rich_cache[k] = Rich(s, size, kind, math_scale)
    return _rich_cache[k]


# ---------------------------------------------------------------- digits for changing numbers
_TEX = {"-": "-", "%": r"\%", "×": r"\times", "·": r"\cdot"}
_CHARS = "0123456789.,+-()[]%:=/×·mpxsKMB"


class Digits:
    """Computer Modern glyphs, one mask per character, so a number string can be laid out every frame for free.
    Digits are equal width in CM, so columns of numbers line up."""

    def __init__(self, size=40, chars=_CHARS):
        self.size = size
        self.keys = {ch: request(_TEX.get(ch, ch if not ch.isalpha() else r"\mathrm{%s}" % ch), size) for ch in chars}

    def _glyphs(self, s):
        out, x, asc, dep = [], 0.0, 0.0, 0.0
        for ch in s:
            if ch == " ":
                x += 0.3 * self.size
                continue
            if ch not in self.keys:
                raise ValueError(f"Digits has no glyph for {ch!r}; add it to chars=")
            mask, m = get(self.keys[ch])
            a = m["asc"]
            out.append((mask, x, a))
            asc, dep = max(asc, a), max(dep, m["desc"])
            x += m["width"]
        return out, x, asc, dep

    def width(self, s):
        return self._glyphs(s)[1]

    def draw(self, cv, xy, s, color=style.MATH, alpha=1.0, anchor="mm", shadow=True, glow=0.0):
        g, w, asc, dep = self._glyphs(s)
        x0, y0 = anchor_xy(xy, w, asc + dep, asc, anchor)
        for mask, x, a in g:
            paste_mask(cv, mask, int(round(x0 + x)), int(round(y0 + asc - a)), color, alpha, shadow, glow)
        return x0, y0, x0 + w, y0 + asc + dep


_digits = {}


def digits(size):
    if size not in _digits:
        _digits[size] = Digits(size)
    return _digits[size]


def slot_width(d, fmt, values):
    """Width that fits fmt(v) for every v (digits equal width -> use the longest string)."""
    return max(d.width(fmt.format(v)) for v in np.atleast_1d(values))
