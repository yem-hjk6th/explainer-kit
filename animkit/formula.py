"""Typeset math with matplotlib mathtext (Computer Modern), rendered once by a subprocess (config math_python, default: this interpreter) and cached as alpha
masks in .cache/formula/<key>.png. Color is applied at draw time, so recoloring / per-term highlighting does not
re-render.

Formula = segments laid out on one baseline per line; each segment can be dimmed / lit / highlighted on its own
(term by term), get a box or underline, and carry a small note under it.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import style
from .config import CACHE, CFG

FDIR = CACHE / "formula"
_VER = "1"
_mem = {}
_pending = {}

STATE_ALPHA = {"hidden": 0.0, "dim": 0.3, "lit": 1.0, "hl": 1.0}


def _key(tex, size, fontset):
    return hashlib.sha1(f"{_VER}|{fontset}|{int(size)}|{tex}".encode("utf-8")).hexdigest()[:20]


def request(tex, size, fontset="cm"):
    """Register a formula for rendering (batched); returns its cache key."""
    k = _key(tex, size, fontset)
    if k not in _mem and not (FDIR / f"{k}.png").exists():
        _pending[k] = dict(key=k, tex=tex, size=int(size), fontset=fontset)
    return k


def flush():
    """Render every requested-but-missing formula in one subprocess call."""
    if not _pending:
        return
    jobs = list(_pending.values())
    _pending.clear()
    FDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        jp = Path(td) / "jobs.json"
        jp.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
        worker = Path(__file__).with_name("_mathtext_worker.py")
        r = subprocess.run([CFG.get("math_python") or sys.executable, "-I", "-X", "utf8", str(worker), str(jp), str(FDIR)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"mathtext worker failed:\n{r.stderr}")
    errs = json.loads((FDIR / "errors.json").read_text(encoding="utf-8"))
    bad = [e for k, e in errs.items() if any(j["key"] == k for j in jobs)]
    if bad:
        raise ValueError("mathtext could not parse:\n  " + "\n  ".join(bad))


def get(k):
    """(mask uint8 (h, w), meta {width, height, depth}) for a requested key."""
    if k not in _mem:
        if not (FDIR / f"{k}.png").exists():
            flush()
        mask = np.asarray(Image.open(FDIR / f"{k}.png").convert("L"))
        meta = json.loads((FDIR / f"{k}.json").read_text(encoding="utf-8"))
        # mathtext crops the raster to the ink; the baseline sits `depth` px above the image bottom (depth < 0 for
        # glyphs that float above the baseline, e.g. a lone '-' or '='), so ascent = rows - depth.
        meta["asc"] = mask.shape[0] - meta["depth"]
        meta["desc"] = meta["depth"]
        _mem[k] = (mask, meta)
    return _mem[k]


def anchor_xy(xy, w, h, asc, anchor):
    x, y = xy
    x -= {"l": 0, "m": w / 2, "r": w}[anchor[0]]
    y -= {"a": 0, "t": 0, "m": h / 2, "b": h, "s": asc}[anchor[1]]
    return x, y


def paste_mask(cv, mask, x, y, color, alpha, shadow=True, glow=0.0):
    """Tinted mask onto the text layer at integer (x, y) (+ optional drop shadow and glow)."""
    if alpha <= 0.01:
        return
    mh, mw = mask.shape
    if glow > 0:
        cv.g.blend(mask, x, y, color, min(1.0, glow * alpha))
    if shadow:
        sh = np.zeros((mh + 2, mw + 2, 4), np.uint8)
        sh[2:, 2:, 3] = (mask * (0.7 * alpha)).astype(np.uint8)
        cv.paste_rgba(sh, x, y)
    arr = np.empty((mh, mw, 4), np.uint8)
    arr[..., :3] = color[:3]
    arr[..., 3] = (mask * min(alpha, 1.0)).astype(np.uint8)
    cv.paste_rgba(arr, x, y)


class Formula:
    """segments: list of mathtext strings (no $) or (tex, id) pairs; "\\n" as a segment starts a new line.
    size: em size in px. gap: space between segments, in em. align: 'l' | 'm' | 'r' for multi-line blocks.
    Segments are addressed by id (or by index when no id is given)."""

    def __init__(self, segments, size=72, gap=0.22, fontset="cm", align="m", line_gap=0.35):
        segs = [segments] if isinstance(segments, str) else list(segments)
        self.size, self.gap, self.align, self.line_gap = size, gap * size, align, line_gap * size
        self.items = []          # (line no, tex, id, key)
        line = 0
        for s in segs:
            if s == "\n":
                line += 1
                continue
            tex, sid = (s, None) if isinstance(s, str) else s
            sid = len(self.items) if sid is None else sid
            self.items.append((line, tex, sid, request(tex, size, fontset)))
        self.ids = [it[2] for it in self.items]
        self._lay = None

    def _layout(self):
        if self._lay is None:
            nl = self.items[-1][0] + 1
            lines = [[it for it in self.items if it[0] == k] for k in range(nl)]
            boxes, y, W = {}, 0.0, 0.0
            rows = []
            for row in lines:
                got = [(it, *get(it[3])) for it in row]
                asc = max(m["asc"] for _, _, m in got)
                dep = max(max(m["desc"] for _, _, m in got), 0)
                x, placed = 0.0, []
                for it, mask, m in got:
                    placed.append((it[2], x, y + asc - m["asc"], mask, m["width"]))
                    x += m["width"] + self.gap
                w = x - self.gap
                rows.append((placed, w, y + asc + dep))
                W = max(W, w)
                y += asc + dep + self.line_gap
            for placed, w, bottom in rows:
                dx = {"l": 0, "m": (W - w) / 2, "r": W - w}[self.align]
                for sid, bx, by, mask, mw in placed:
                    boxes[sid] = (bx + dx, by, mask, mw, bottom)
            asc0 = max(get(it[3])[1]["asc"] for it in lines[0])   # first baseline
            self._lay = (boxes, W, y - self.line_gap, asc0)
        return self._lay

    @property
    def wh(self):
        _, w, h, _ = self._layout()
        return w, h

    def bbox(self, xy, anchor="mm"):
        _, w, h, asc = self._layout()
        x, y = anchor_xy(xy, w, h, asc, anchor)
        return x, y, x + w, y + h

    def part_box(self, sid, xy, anchor="mm"):
        """Screen bbox (x0, y0, x1, y1) of segment sid (for arrows / brackets pointing at a term)."""
        boxes, w, h, asc = self._layout()
        x, y = anchor_xy(xy, w, h, asc, anchor)
        bx, by, mask, bw, _ = boxes[sid]
        return x + bx, y + by, x + bx + bw, y + by + mask.shape[0]

    def draw(self, cv, xy, color=style.MATH, alpha=1.0, anchor="mm", state=None, part_alpha=None, part_color=None,
             glow=0.0, part_glow=None, box=None, underline=None, notes=None, note_size=None, shadow=True,
             hl_color=style.ORANGE):
        """state: {id: 'hidden'|'dim'|'lit'|'hl' or a float level}; part_alpha multiplies on top of it.
        box / underline: {id: amount 0-1} (or a color tuple for amount 1) drawn around / under a segment;
        'hl' state = lit + box + glow. notes: {id: rich text} small line under the segment ($math$ allowed).
        Returns the whole bbox."""
        boxes, w, h, asc = self._layout()
        x0, y0 = anchor_xy(xy, w, h, asc, anchor)
        if alpha <= 0.01:
            return x0, y0, x0 + w, y0 + h
        state, part_alpha = state or {}, part_alpha or {}
        boxd, uline = dict(box or {}), dict(underline or {})
        glows = dict(part_glow or {})
        for sid in self.ids:
            st = state.get(sid, "lit")
            if st == "hl":
                boxd.setdefault(sid, 1.0)
                glows.setdefault(sid, 0.35)
        for sid in self.ids:
            st = state.get(sid, "lit")
            lvl = STATE_ALPHA[st] if isinstance(st, str) else float(st)
            a = alpha * lvl * part_alpha.get(sid, 1.0)
            if a <= 0.01:
                continue
            bx, by, mask, bw, _ = boxes[sid]
            c = (part_color or {}).get(sid, color)
            gx, gy = int(round(x0 + bx)), int(round(y0 + by))
            paste_mask(cv, mask, gx, gy, c, a, shadow, glows.get(sid, glow))
            pad = 0.12 * self.size
            for d, kind in ((boxd, "box"), (uline, "line")):
                if sid in d:
                    v = d[sid]
                    amt, col = (1.0, v) if isinstance(v, tuple) else (float(v), hl_color)
                    if amt <= 0:
                        continue
                    X0, Y0, X1, Y1 = gx - pad, gy - pad * 0.6, gx + bw + pad, gy + mask.shape[0] + pad * 0.6
                    if kind == "box":
                        cv.s.polyline([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)], col, alpha * amt, 2, closed=True)
                        cv.g.polyline([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)], col, 0.6 * alpha * amt, 4,
                                      closed=True)
                    else:
                        cv.s.line((X0, Y1 + 3), (X1, Y1 + 3), col, alpha * amt, 3)
        # notes under their segments; a note that would overlap an earlier one drops to the next row (with a leader)
        from .richtext import rich
        ns = note_size or max(20, int(self.size * 0.36))
        taken = []
        for sid in self.ids:
            if sid not in (notes or {}):
                continue
            bx, by, mask, bw, bottom = boxes[sid]
            st = state.get(sid, "lit")
            lvl = STATE_ALPHA[st] if isinstance(st, str) else float(st)
            a = alpha * min(lvl, 1.0) * part_alpha.get(sid, 1.0)
            r = rich(notes[sid], ns)
            nw, nh = r.wh
            cx = x0 + bx + bw / 2
            row = 0
            while any(pr == row and cx + nw / 2 + 10 > px0 and cx - nw / 2 - 10 < px1 for px0, px1, pr in taken):
                row += 1
            taken.append((cx - nw / 2, cx + nw / 2, row))
            if a > 0.01:
                top = y0 + bottom + 0.18 * self.size + row * (nh + 8)
                if row:
                    cv.s.line((cx, y0 + by + mask.shape[0] + 4), (cx, top - 4), style.GREY, 0.7 * a, 1)
                r.draw(cv, (cx, top), style.TEXT_DIM, a, "ma", "m")
        return x0, y0, x0 + w, y0 + h
