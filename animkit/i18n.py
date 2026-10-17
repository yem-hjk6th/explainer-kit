"""On-screen languages. Shots are written in English (SOURCE). With --lang <code> every text entry point (captions,
titles, tags, cv.text / cv.text_rot / cv.rich / cv.badge / Rich / wrap / formula notes) looks the drawn string up in
captions.json:
  {"<code>": {"<shot script stem>": {"English source string": "translation", ...}, ...}}
Keys are the exact source strings (\\n and $math$ included): editing the English makes the old translation stop
matching, so the string is reported as untranslated instead of silently keeping an outdated line.
Only strings with a word of two or more letters outside $...$ are looked up (numbers, symbols and math are left alone).
In the source language every function here is the identity.
T("token $799$ of the frame -> token {g} overall", g=2173) = template lookup, then {name} placeholders filled
(LaTeX braces untouched); only needed for f-strings whose runtime value takes many different values.
"""
import json
import re
from pathlib import Path

from .config import CFG

SOURCE = "en"
LANG = SOURCE
_WORD = re.compile(r"[A-Za-z]{2,}")
_MATH = re.compile(r"\$[^$]*\$")
_CJK = re.compile("[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]")
_table, _src, _shot = {}, None, None
_missing, _used, _wide, _wide_seen = {}, {}, [], set()
_seen = None                                          # dict when collecting source strings


class Done(str):
    """A string that already went through T() / wrap(); tr() passes it through untouched."""


def has_cjk(s):
    return bool(_CJK.search(s))


def wants_lookup(s):
    """Source strings that carry words (not just numbers, symbols or $math$)."""
    return bool(_WORD.search(_MATH.sub("", s)))


def find_table(script):
    """captions.json next to the shot script or in its parent (the episode folder)."""
    for d in (Path(script).resolve().parent, Path(script).resolve().parent.parent):
        if (d / "captions.json").exists():
            return d / "captions.json"
    return None


def load_table(path, lang, stem):
    """source -> translation for one shot: every group's non-empty entries, the shot's own group last (it wins)."""
    if not path or not Path(path).exists():
        return {}
    groups = json.loads(Path(path).read_text(encoding="utf-8")).get(lang, {})
    out = {}
    for g, d in groups.items():
        if g != stem:
            out.update({k: v for k, v in d.items() if v})
    out.update({k: v for k, v in groups.get(stem, {}).items() if v})
    return out


def setup(lang, script=None, table=None):
    """Called by render_shot.py before the shot script is imported (scripts call shot.caption at import time)."""
    global LANG, _table, _src, _shot
    LANG, _shot = lang, Path(script).stem if script else None
    _src = Path(table).resolve() if table else (find_table(script) if script else None)
    _table = load_table(_src, lang, _shot) if lang != SOURCE else {}
    if lang != SOURCE and _src is None:
        print(f"warning: --lang {lang} but no captions.json found next to the shot or in its parent")


def suffix():
    """File-name suffix of the current language: '' for the source language, '_<code>' otherwise."""
    return "" if LANG == SOURCE else f"_{LANG}"


def start_collect():
    global _seen
    _seen = {}


def collected():
    return list(_seen or {})


def _lookup(s):
    t = _table.get(s)
    if t:
        _used[s] = t
        return t
    _missing[s] = _missing.get(s, 0) + 1
    return None


def tr(s):
    """Translate one drawn string (identity in the source language, for strings without words, and for Done)."""
    if (LANG == SOURCE and _seen is None) or not isinstance(s, str) or isinstance(s, Done) or not wants_lookup(s):
        return s
    if _seen is not None:
        _seen[s] = _seen.get(s, 0) + 1
    if LANG == SOURCE:
        return s
    t = _lookup(s)
    return Done(t) if t else s


def translated(s):
    """False when s is a source string that the current table could not translate."""
    return LANG == SOURCE or isinstance(s, Done) or not wants_lookup(s) or s not in _missing


def T(s, **kw):
    """Template: the table key is s with its {name} placeholders; values are filled after the lookup."""
    if _seen is not None and wants_lookup(s):
        _seen[s] = _seen.get(s, 0) + 1
    out = s if LANG == SOURCE else (_lookup(s) or s)
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return Done(out)


def kind_for(kind, s):
    """Font kind for a drawn string: in a translated render, "<kind>_<lang>" when config/animkit.json defines it
    (e.g. "text_ja": a font that covers Japanese), else the kind itself."""
    if LANG == SOURCE:
        return kind
    k = f"{kind}_{LANG}"
    return k if k in CFG["fonts"] else kind


def note_box(src, out, box, W, H):
    """Remember a translated string drawn partly outside its canvas (frame or popup); box = (x0, y0, x1, y1)."""
    if LANG == SOURCE or src in _wide_seen:
        return
    x0, y0, x1, y1 = box
    if x0 < -1 or y0 < -1 or x1 > W + 1 or y1 > H + 1:
        _wide_seen.add(src)
        _wide.append({"src": src, "tr": out, "box": [round(v) for v in box], "canvas": [W, H]})


def report():
    """Per-process state for meta.json: what was translated, what is missing, what ran off its canvas."""
    return {"lang": LANG, "table": str(_src) if _src else None, "used": dict(_used),
            "untranslated": list(_missing), "off_canvas": list(_wide)}
