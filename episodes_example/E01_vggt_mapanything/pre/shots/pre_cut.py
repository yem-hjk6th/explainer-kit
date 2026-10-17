"""Shared code of the cut shells for the E01 English pre version: re-time by "source shot + source-second range", without modifying any source shot.

A pre shot = several Segs (source shot name, a, b) joined end to end; adjacent segments cross-fade for XFADE seconds (each uses XFADE/2 seconds of source picture outside its range).
All captions / titles / corner tags of the source shot are dropped and rewritten from the English in the Seg (times are given in source-shot seconds and converted to the new time axis automatically).
The source shot's checks run as usual, plus a check that the range lies within the source length and the captions lie within the segment; the edit table is recorded in data.edl of check.json.
Source shots whose name ends in _pre are read from pre/src/ (forks that hide the numbers measured on our test scene; each fork's docstring lists what it hides); the others are read from the E01 shots/.
Note: script_sha1 in meta.json tracks only this shell; if a source shot or fork changes, re-render by hand.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from animkit import Shot, i18n
from animkit.ease import ss

EP = Path(__file__).resolve().parents[2]
E01_SHOTS = EP / "shots"
PRE_SRC = EP / "pre" / "src"
if str(E01_SHOTS) not in sys.path:
    sys.path.insert(0, str(E01_SHOTS))

XFADE = 0.25
TITLE_DUR = 3.0
_loaded, _tables = {}, {}


class _source_table:
    """When drawing a source shot, its own translation group takes priority (as in the source-language render); the shell's own captions are English and need no lookup.
    A fork sXX_pre uses the group of sXX."""

    def __init__(self, name):
        stem = name[:-4] if name.endswith("_pre") else name
        if i18n.LANG != i18n.SOURCE and stem not in _tables:
            _tables[stem] = i18n.load_table(i18n._src, i18n.LANG, stem)
        self.tab = _tables.get(stem)

    def __enter__(self):
        self.saved = i18n._table
        if self.tab is not None:
            i18n._table = self.tab

    def __exit__(self, *exc):
        i18n._table = self.saved


def load(name):
    """Source shot (captions / titles / corner tags removed). A shot of the same name is loaded only once per process."""
    if name in _loaded:
        return _loaded[name]
    path = (PRE_SRC if name.endswith("_pre") else E01_SHOTS) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e01pre_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    with _source_table(name):
        spec.loader.exec_module(mod)
    shots = [v for v in vars(mod).values() if isinstance(v, Shot)]
    if len(shots) != 1:
        raise ValueError(f"{path}: expected one Shot, found {len(shots)}")
    sh = shots[0]
    sh.path = path
    sh.captions, sh.titles, sh.tags = [], [], []
    _loaded[name] = sh
    return sh


class Seg:
    """Seconds [a, b] of source shot src. caps = [(a, b, English)], tags = [(English, a, b)] (a, b omitted = whole segment), all in source-shot seconds;
    title = (title, subtitle), shown for TITLE_DUR seconds from the start of this segment. hold = how many seconds to freeze on the picture at second b at the end of the segment (caption times may extend to b + hold)."""

    def __init__(self, src, a, b, caps=(), tags=(), title=None, hold=0.0):
        self.src, self.a, self.b, self.hold = src, float(a), float(b), float(hold)
        self.caps, self.tags, self.title = list(caps), list(tags), title

    @property
    def len(self):
        return self.b - self.a + self.hold

    def src_time(self, x):
        """Local seconds of this segment (counted from a, may cross the segment boundary, used for the cross-fade) -> source-shot seconds; frozen at b during a hold."""
        return min(self.a + x, self.b) if self.hold else self.a + x


class PreShot(Shot):
    def __init__(self, name, segs, tags=(), xfade=XFADE):
        self.segs = segs
        for s in segs:
            s.shot = load(s.src)
        first = segs[0].shot
        super().__init__(name, sum(s.len for s in segs), first.W, first.H, first.fps, first.glow, first.seed)
        self.xfade = xfade
        o = 0.0
        for s in segs:
            s.o = o
            o += s.len
        spans = []                                           # (text, t0, t1) on the new time axis
        for s in segs:
            for a, b, text in s.caps:
                self.caption(self.out(s, a), self.out(s, b), text)
            for tg in s.tags:
                text, a, b = (tg + (s.a, s.b))[:3] if len(tg) == 1 else tg
                spans.append((text, self.out(s, a), self.out(s, b)))
            if s.title:
                self.title(s.title[0], s.title[1], s.o + 0.2, min(s.o + TITLE_DUR, s.o + s.len))
        spans += [(text, 0.0 if a is None else a, self.dur if b is None else b) for text, a, b in tags]
        for text, a, b in _merge(spans):
            self.tag(text, a, b)
        self._checks = [self._check_edl]
        seen = set()
        for s in segs:
            if s.src not in seen:
                seen.add(s.src)
                self._checks += s.shot._checks

    def out(self, s, x):
        """Second x of source shot s.src (must lie in [s.a, s.b]) -> new time axis."""
        return s.o + (x - s.a)

    def _check_edl(self, ck):
        for k, s in enumerate(self.segs):
            ck.true(f"seg {k} {s.src} [{s.a}, {s.b}] inside source (0, {s.shot.dur})",
                    0 <= s.a < s.b <= s.shot.dur + 1e-6)
            for a, b, text in s.caps:
                ck.true(f"seg {k} caption inside segment: {text!r}", s.a - 1e-6 <= a < b <= s.b + s.hold + 1e-6)
        for k, (s0, s1) in enumerate(zip(self.segs, self.segs[1:])):
            ck.true(f"cross-fade {k} has source frames on both sides",
                    s0.b + self.xfade / 2 <= s0.shot.dur + 1e-6 or s0.b >= s0.shot.dur - 1e-6)
        ck.close("duration = sum of segments", self.dur, sum(s.len for s in self.segs), 1e-9)
        ck.record(edl=[[s.src, s.a, s.b, round(s.o, 3)] + ([s.hold] if s.hold else []) for s in self.segs])

    def _at(self, s, x):
        with _source_table(s.src):
            return s.shot.render_frame(min(max(x, 0.0), s.shot.dur - 1e-6)).astype(np.float32)

    def _source_frame(self, t):
        h = self.xfade / 2
        for s0, s1 in zip(self.segs, self.segs[1:]):
            if s1.o - h <= t < s1.o + h:
                w = ss(t, s1.o - h, s1.o + h)
                d = t - s1.o
                return (1 - w) * self._at(s0, s0.src_time(s0.len + d)) + w * self._at(s1, s1.a + d)
        s = [s for s in self.segs if s.o <= t + 1e-9][-1]
        return self._at(s, s.src_time(t - s.o))

    def render_frame(self, t):
        self._bg = self._source_frame(t)                     # the full source frame is the background; only this shell's title / corner tag / captions are drawn on top
        return super().render_frame(t)

    def edl_table(self):
        """[(new start, new end, source shot, a, b)], used for the storyboard / review page."""
        return [(round(s.o, 2), round(s.o + s.len, 2), s.src, s.a, s.b) for s in self.segs]


def _merge(spans):
    """Merge corner tags with the same text that touch end to end (gap < 0.05 s) into one span, so they do not flicker at the cut."""
    out = []
    for text, a, b in sorted(spans, key=lambda x: (x[0], x[1])):
        if out and out[-1][0] == text and a <= out[-1][2] + 0.05:
            out[-1][2] = max(out[-1][2], b)
        else:
            out.append([text, a, b])
    return sorted((tuple(x) for x in out), key=lambda x: x[1])
