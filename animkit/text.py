"""Fonts and plain text labels. Symbols with sub/superscripts go through formula.py instead: many UI fonts have no
subscript glyphs (x₁ renders as a box)."""
import re

from PIL import ImageFont

from .config import CFG

_fonts = {}


_FALLBACK = {"bold": "text", "light": "text"}           # a missing role falls back to the regular face


def font(kind="text", size=30):
    """kind: text (regular), bold, light (thin face for years, numbers, small labels), math (serif italic),
    math_up (serif upright), text_<lang> / bold_<lang> (optional fonts for a translated render, see i18n.kind_for);
    or a font file path. Paths come from config/animkit.json."""
    path = CFG["fonts"].get(kind) or CFG["fonts"].get(_FALLBACK.get(kind, "")) or kind
    k = (path, int(size))
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, int(size))
    return _fonts[k]


def text_on(dr, xy, s, fnt, alpha=1.0, fill=(240, 240, 240), anchor="la", shadow=True, align="left", spacing=None):
    if alpha <= 0.01:
        return
    a = int(255 * min(alpha, 1.0))
    multi = "\n" in s
    kw = dict(font=fnt, anchor=anchor)
    if multi:
        kw.update(align=align, spacing=spacing if spacing is not None else int(fnt.size * 0.33))
    draw = dr.multiline_text if multi else dr.text
    if shadow:
        draw((xy[0] + 2, xy[1] + 2), s, fill=(0, 0, 0, int(a * 0.7)), **kw)
    draw(tuple(xy), s, fill=tuple(fill[:3]) + (a,), **kw)


_WIDE = re.compile("[\u2e80-\u9fff\uac00-\ud7af]")   # CJK / kana / hangul: no spaces, count characters


def caption_len(s):
    """Caption length in words (a $math$ span or a hyphenated / dotted word counts once). For a translation into a
    language written without spaces, each CJK / kana / hangul character counts as one word. This is a reading-load
    budget, not a width check: Timeline warns separately when a caption line is wider than the frame."""
    n = len(re.findall(r"\$[^$]+\$|[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*", _WIDE.sub(" ", s)))
    return n + len(_WIDE.findall(s))
