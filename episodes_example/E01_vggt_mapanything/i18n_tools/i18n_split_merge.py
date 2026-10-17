"""E01 translation table: split / merge. Shots are written in English (the source language); captions.json is
{"<lang>": {"<shot stem>": {"<English source string>": "<translation>"}}}.
  python -I -X utf8 i18n_split_merge.py [--lang es] split   -> tr_work/in_<shot>.json (only keys not yet translated, value ""); s04 is carried over from the i18n demo first
  python -I -X utf8 i18n_split_merge.py [--lang es] merge   -> write tr_work/out_<shot>.json back into captions.json (keys must match exactly)
  python -I -X utf8 i18n_split_merge.py [--lang es] patch <file.json>   -> {shot: {source string: new translation}}; unknown keys are reported, not added
  python -I -X utf8 i18n_split_merge.py [--lang es] glyphs  -> replace circled digits (missing in many Latin fonts) by "1." ...
  python -I -X utf8 i18n_split_merge.py [--lang es] table   -> captions_<lang>.md (shot, number, source, translation)
"""
import argparse
import json
from pathlib import Path

EP = Path(__file__).resolve().parents[1]
TR = EP / "tr_work"
CAP = EP / "captions.json"
DEMO = EP.parents[1] / "examples" / "i18n_demo" / "captions.json"   # repo root = EP.parents[1]
LANG = "es"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def split():
    TR.mkdir(parents=True, exist_ok=True)
    cap = load(CAP)
    demo = load(DEMO).get(LANG, {}).get("s04_alt_attn", {})
    n_demo = 0
    for k in cap[LANG].get("s04_alt_attn", {}):
        if not cap[LANG]["s04_alt_attn"][k] and demo.get(k):
            cap[LANG]["s04_alt_attn"][k] = demo[k]
            n_demo += 1
    save(CAP, cap)
    print(f"s04 from demo: {n_demo}")
    for shot, tab in cap[LANG].items():
        if shot.startswith("_"):
            continue
        todo = {k: "" for k, v in tab.items() if not v}
        if todo:
            save(TR / f"in_{shot}.json", todo)
        print(f"{shot}: {len(todo)} to translate")


def merge():
    cap = load(CAP)
    for p in sorted(TR.glob("out_*.json")):
        shot = p.stem[4:]
        out = load(p)
        tab = cap[LANG].get(shot)
        if tab is None:
            print(f"{shot}: not in captions.json, skipped")
            continue
        miss = [k for k in out if k not in tab]
        n = 0
        for k, v in out.items():
            if k in tab and v:
                tab[k] = v
                n += 1
        left = sum(1 for v in tab.values() if not v)
        print(f"{shot}: merged {n}, unknown keys {len(miss)}, still empty {left}")
        for k in miss[:5]:
            print("   unknown:", repr(k))
    save(CAP, cap)


def patch(path):
    """Hand-fix translations: path = {shot: {source string: new translation}}; a missing key is reported, not added."""
    cap = load(CAP)
    fix = load(Path(path))
    for shot, kv in fix.items():
        tab = cap[LANG][shot]
        for k, v in kv.items():
            if k not in tab:
                print(f"{shot}: missing key {k!r}")
                continue
            tab[k] = v
        print(f"{shot}: patched {sum(1 for k in kv if k in tab)}")
    save(CAP, cap)


CIRCLED = {"\u2460": "1.", "\u2461": "2.", "\u2462": "3.", "\u2463": "4.", "\u2464": "5."}   # Segoe UI has no circled digits (drawn as boxes)


def fix_glyphs():
    cap = load(CAP)
    n = 0
    for shot, tab in cap[LANG].items():
        if shot.startswith("_"):
            continue
        for k, v in tab.items():
            w = v
            for a, b in CIRCLED.items():
                w = w.replace(a, b)
            if w != v:
                tab[k] = w
                n += 1
    save(CAP, cap)
    print("circled digits replaced in", n, "values")


def table():
    cap = load(CAP)
    lines = [f"# E01 source / {LANG} table (exported from captions.json; one row per string: shot, number, source, translation)", "",
             f"| shot | # | English | {LANG} |", "|---|---|---|---|"]

    def cell(s):
        return s.replace("|", "\\|").replace("\n", "<br>")

    for shot, tab in cap[LANG].items():
        if shot.startswith("_"):
            continue
        for i, (k, v) in enumerate(tab.items(), 1):
            lines.append(f"| {shot} | {i} | {cell(k)} | {cell(v)} |")
    (EP / f"captions_{LANG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"captions_{LANG}.md:", len(lines) - 4, "rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="es")
    ap.add_argument("cmd", choices=["split", "merge", "table", "glyphs", "patch"])
    ap.add_argument("file", nargs="?")
    a = ap.parse_args()
    LANG = a.lang
    if a.cmd == "patch":
        patch(a.file)
    else:
        {"split": split, "merge": merge, "table": table, "glyphs": fix_glyphs}[a.cmd]()