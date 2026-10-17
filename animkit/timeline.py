"""Shot = duration + per-frame draw callback + captions/titles/tags + numeric checks. Rendering pipes RGB frames to
ffmpeg (H.264, yuv420p), writes a contact sheet, check.json and meta.json next to the mp4.

Output folder convention: <script dir>/render/<script stem>/<script stem>.mp4 (+ sheet.jpg, check.json, meta.json).
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import i18n, style
from .config import ffmpeg
from .draw import Canvas, background
from .ease import fade, ss
from .text import caption_len, font, text_on

LIB_VERSION = "0.1"


class CheckFailed(RuntimeError):
    pass


def _j(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {k: _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    return v


class Checker:
    """Collects named numeric checks. Every method returns pass/fail and records value, operator and reference."""

    def __init__(self):
        self.items, self.data = [], {}

    def _add(self, name, value, op, ref, ok):
        self.items.append({"name": name, "value": _j(value), "op": op, "ref": _j(ref), "ok": bool(ok)})
        return bool(ok)

    def le(self, name, value, limit):
        return self._add(name, float(value), "<=", limit, float(value) <= limit)

    def ge(self, name, value, limit):
        return self._add(name, float(value), ">=", limit, float(value) >= limit)

    def close(self, name, value, ref, tol):
        err = float(np.max(np.abs(np.asarray(value, float) - np.asarray(ref, float))))
        return self._add(name, err, "max|value-ref| <=", tol, err <= tol)

    def eq(self, name, value, ref):
        return self._add(name, value, "==", ref, np.array_equal(np.asarray(value), np.asarray(ref)))

    def true(self, name, cond, detail=None):
        return self._add(name, detail, "is true", True, bool(cond))

    def record(self, **kw):
        """Data shown in check.json without pass/fail (shapes, sample values, ...)."""
        self.data.update({k: _j(v) for k, v in kw.items()})


class Shot:
    def __init__(self, name, dur, W=style.W, H=style.H, fps=style.FPS, glow=((0.9, 6), (0.7, 22)), seed=0):
        self.name, self.dur, self.W, self.H, self.fps, self.glow, self.seed = name, float(dur), W, H, fps, glow, seed
        self.captions, self.titles, self.tags = [], [], []
        self._checks, self._frame = [], None
        self.sheet_times = None      # default: 12 evenly spaced
        self.audio = None            # optional audio file muxed into the mp4 (voice / music interface)
        self.path = None             # set by load_shot
        self._bg = None
        self._cap_cache = {}

    # ---- declaration ----
    def caption(self, t0, t1, text):
        """Bottom subtitle, at most 2 lines ('\\n'), fades in/out over 0.3 s (no fade-out if it runs to the end)."""
        self.captions.append((float(t0), float(t1), text))

    def title(self, text, sub=None, t0=0.2, t1=3.8, fin=0.6, fout=0.6):
        self.titles.append((text, sub, t0, t1, fin, fout))

    def tag(self, text="Illustration", t0=0.0, t1=None):
        """Corner tag for pictures that are illustrative, not computed: "Illustration", "Dimension-reduced sketch" ..."""
        self.tags.append((text, t0, self.dur if t1 is None else t1))

    def check(self, fn):
        """Decorator: fn(ck: Checker) registers numeric checks; runs before rendering, any failure stops the render."""
        self._checks.append(fn)
        return fn

    def frame(self, fn):
        """Decorator: fn(cv: Canvas, t: float) draws one frame."""
        self._frame = fn
        return fn

    # ---- checks and lint ----
    def run_checks(self):
        ck = Checker()
        for fn in self._checks:
            fn(ck)
        return {"shot": self.name, "ok": all(i["ok"] for i in ck.items), "items": ck.items, "data": ck.data,
                "warnings": self.lint()}

    def _cap_text(self, s):
        """Caption as shown: translated (in a --lang render); a line without an explicit \\n that is wider than the
        frame is broken into 2 lines between words (the split that makes the longer line shortest)."""
        if s in self._cap_cache:
            return self._cap_cache[s]
        t = i18n.tr(s)
        f = font(i18n.kind_for("text", t), style.CAPTION_SIZE)
        if "\n" not in t and f.getlength(t) > self.W - 120 and " " in t.strip():
            ws = t.split()
            k = min(range(1, len(ws)), key=lambda k: max(f.getlength(" ".join(ws[:k])), f.getlength(" ".join(ws[k:]))))
            t = i18n.Done(" ".join(ws[:k]) + "\n" + " ".join(ws[k:]))
        self._cap_cache[s] = t
        return t

    def lint(self):
        w = []
        cap_max = style.CAPTION_MAX_WORDS
        for k, (a, b, s0) in enumerate(sorted(self.captions)):
            s = self._cap_text(s0)
            fnt = font(i18n.kind_for("text", s), style.CAPTION_SIZE)
            lines = s.split("\n")
            if len(lines) > style.CAPTION_MAX_LINES:
                w.append(f"caption {k + 1}: {len(lines)} lines > {style.CAPTION_MAX_LINES}: {s!r}")
            if not i18n.translated(i18n.tr(s0)):
                w.append(f"caption {k + 1}: untranslated: {s0!r}")
            n = caption_len(s)
            if n > cap_max:
                w.append(f"caption {k + 1}: {n} words > {cap_max}: {s!r}")
            if max(fnt.getlength(x) for x in lines) > self.W - 120:
                w.append(f"caption {k + 1}: a line is wider than the frame: {s!r}")
            if b - a < 1.0:
                w.append(f"caption {k + 1}: on screen only {b - a:.1f} s")
        for text, sub, *_ in self.titles:
            for x in (text, sub):
                if x and not i18n.translated(i18n.tr(x)):
                    w.append(f"title untranslated: {x!r}")
            t = i18n.tr(text)
            if font(i18n.kind_for("bold", t), style.TITLE_SIZE).getlength(t) > self.W - 2 * style.TITLE_XY[0]:
                w.append(f"title wider than the frame: {t!r}")
        for text, *_ in self.tags:
            if not i18n.translated(i18n.tr(text)):
                w.append(f"tag untranslated: {text!r}")
        cs = sorted(self.captions)
        for (a0, b0, s0), (a1, b1, s1) in zip(cs, cs[1:]):
            if a1 < b0 - 1e-6:
                w.append(f"captions overlap at {a1:.2f}s: {s0!r} / {s1!r}")
        return w

    # ---- drawing ----
    def render_frame(self, t):
        if self._bg is None:
            self._bg = background(self.W, self.H, self.seed)
        cv = Canvas(self.W, self.H, t, self._bg, self.glow)
        if self._frame:
            self._frame(cv, t)
        for text, sub, t0, t1, fin, fout in self.titles:
            a = fade(t, t0, t1, fin, fout)
            cv.text(style.TITLE_XY, text, style.TITLE_SIZE, "bold", a)
            if sub:
                cv.text(style.SUB_XY, sub, style.SUB_SIZE, "light", a, style.TEXT_DIM)
        for k, (text, t0, t1) in enumerate(self.tags):
            a = fade(t, t0, t1, 0.3, 0.3) if t1 < self.dur else ss(t, t0, t0 + 0.3)
            if a > 0.01:
                # fixed slot: one row below every earlier tag whose time range overlaps this one
                slot = sum(1 for (_, a0, b0) in self.tags[:k] if a0 < t1 and t0 < b0)
                x1, y0 = style.TAG_XY
                cv.badge((x1, y0 + slot * (style.TAG_SIZE + 26)), text, style.TAG_SIZE, a, anchor="ra")
        for a0, b0, s in self.captions:
            al = fade(t, a0, b0, 0.3, 0.3) if b0 < self.dur else ss(t, a0, a0 + 0.3)
            if al > 0:
                s = self._cap_text(s)
                cap = font(i18n.kind_for("text", s), style.CAPTION_SIZE)
                text_on(cv.dr, (self.W / 2, style.CAPTION_Y), s, cap, al, (245, 245, 245), "ma", False, "center",
                        style.CAPTION_SPACING)
        return cv.composite()

    # ---- rendering ----
    def _times_for_sheet(self):
        ts = self.sheet_times or [(k + 0.5) * self.dur / 12 for k in range(12)]
        n = int(round(self.dur * self.fps))
        return [(t, min(n - 1, int(round(t * self.fps)))) for t in ts]

    def _ffmpeg(self, path):
        return subprocess.Popen([ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                 f"{self.W}x{self.H}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                                 "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)],
                                stdin=subprocess.PIPE)

    def render(self, out, frames=None, sheet_only=False, skip_checks=False, quiet=False):
        """Full render to out/. frames=(i0, i1) renders only that frame range into out/_parts/ (parallel parts)."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if not skip_checks:
            chk = self.check_to(out)
            if not chk["ok"]:
                bad = [i for i in chk["items"] if not i["ok"]]
                raise CheckFailed(f"{self.name}: {len(bad)} check(s) failed: {json.dumps(bad, ensure_ascii=False)}")
        n = int(round(self.dur * self.fps))
        sheet = self._times_for_sheet()
        t_start = time.time()
        sfx = i18n.suffix()
        parts = out / f"_parts{sfx}"
        if sheet_only:
            tiles = [(t, self.render_frame(i / self.fps)) for t, i in sheet]
            self._save_sheet(out / f"sheet{sfx}.jpg", tiles)
            return {"render_s": round(time.time() - t_start, 1), "frames": len(tiles)}
        i0, i1 = frames or (0, n)
        if frames:
            parts.mkdir(exist_ok=True)
            mp4 = parts / f"part_{i0:06d}.mp4"
        else:
            mp4 = out / f"{out.name}{sfx}.mp4"
        want = {i: t for t, i in sheet if i0 <= i < i1}
        ff = self._ffmpeg(mp4)
        tiles = []
        for i in range(i0, i1):
            fr = self.render_frame(i / self.fps)
            ff.stdin.write(fr.tobytes())
            if i in want:
                if frames:
                    Image.fromarray(fr).resize((270, 360), Image.LANCZOS).save(parts / f"tile_{i:06d}.png")
                tiles.append((want[i], fr))
            if not quiet and (i - i0) % 90 == 89:
                el = time.time() - t_start
                print(f"  {self.name}: frame {i + 1}/{i1}  {el:.0f}s elapsed", flush=True)
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        render_s = round(time.time() - t_start, 1)
        if frames:
            if i18n.LANG != i18n.SOURCE:
                (parts / f"i18n_{i0:06d}.json").write_text(json.dumps(i18n.report(), ensure_ascii=False),
                                                           encoding="utf-8")
            return {"render_s": render_s, "frames": i1 - i0}
        self._save_sheet(out / f"sheet{sfx}.jpg", tiles)
        if self.audio:
            self._mux_audio(mp4)
        self.write_meta(out, render_s, n, i18n.report() if i18n.LANG != i18n.SOURCE else None)
        return {"render_s": render_s, "frames": n}

    def check_to(self, out):
        chk = self.run_checks()
        (Path(out) / f"check{i18n.suffix()}.json").write_text(json.dumps(_j(chk), indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
        for w in chk["warnings"]:
            print("  warning:", w)
        return chk

    def _save_sheet(self, path, tiles):
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (270 * cols, 360 * rows))
        dr = ImageDraw.Draw(sheet)
        f = font("light", 20)
        for k, (t, fr) in enumerate(tiles):
            tl = fr if isinstance(fr, Image.Image) else Image.fromarray(fr).resize((270, 360), Image.LANCZOS)
            x, y = (k % cols) * 270, (k // cols) * 360
            sheet.paste(tl, (x, y))
            dr.text((x + 8, y + 6), f"{t:.1f}s", font=f, fill=(170, 170, 170))
        sheet.save(path, quality=85)

    def _mux_audio(self, mp4):
        tmp = mp4.with_suffix(".av.mp4")
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(mp4), "-i", str(self.audio), "-c:v", "copy", "-c:a",
                        "aac", "-shortest", "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(mp4)

    def write_meta(self, out, render_s, frames, lang_report=None):
        """meta.json (source language) / meta_<lang>.json. Captions are stored as shown (translated); lang_report = i18n.report()
        merged over all render parts: translations used, untranslated strings, translations drawn off their canvas."""
        caps = [{"n": k + 1, "t0": a, "t1": b, "text": self._cap_text(s)} for k, (a, b, s) in
                enumerate(sorted(self.captions))]
        meta = {"shot": self.name, "dur": self.dur, "fps": self.fps, "size": [self.W, self.H], "frames": frames,
                "captions": caps, "titles": [[x[0], x[1]] for x in self.titles],
                "tags": [{"text": x[0], "t0": x[1], "t1": x[2]} for x in self.tags],
                "script": str(self.path) if self.path else None,
                "script_sha1": hashlib.sha1(Path(self.path).read_bytes()).hexdigest()[:12] if self.path else None,
                "render_s": render_s, "rendered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "animkit": LIB_VERSION}
        if lang_report:
            meta["i18n"] = lang_report
            for k, what in (("untranslated", "untranslated string(s)"), ("off_canvas", "translation(s) drawn off canvas")):
                if lang_report[k]:
                    print(f"  warning: {len(lang_report[k])} {what}, listed in meta{i18n.suffix()}.json i18n.{k}")
        (Path(out) / f"meta{i18n.suffix()}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                                             encoding="utf-8")


def default_out(script):
    script = Path(script).resolve()
    return script.parent / "render" / script.stem


def load_shot(script):
    """Import a shot script and return its single Shot instance (with .path set)."""
    script = Path(script).resolve()
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location(f"shot_{script.stem}", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shots = [v for v in vars(mod).values() if isinstance(v, Shot)]
    if len(shots) != 1:
        raise ValueError(f"{script}: expected exactly one Shot object at module level, found {len(shots)}")
    shots[0].path = script
    return shots[0]
