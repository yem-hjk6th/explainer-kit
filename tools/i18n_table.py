"""Collect every source (English) string an episode draws and merge it into the translation table.
  python -I -X utf8 i18n_table.py <episode_dir> --lang CODE [--shots E01-04 ...] [--table FILE] [--work DIR]
                                        [--fps 10] [--jobs 4]
Runs render_shot.py --collect-text for each shot (frames sampled at --fps, no video; parallel up to --jobs), then
updates <episode_dir>/captions.json (or --table):
  {"<lang>": {"<shot stem>": {"English source": "translation or empty", ...}}, "_unused": {"<lang>": {...}}}
Existing translations are kept; a new string gets the translation it already has in another shot, else "".
Strings a shot no longer draws: translated ones move to "_unused" (so nothing typed is lost), empty ones are dropped.
Prints per shot how many strings are still untranslated, and groups of strings that differ only in numbers
(more than 3 variants: better written as T("…{x}…", x=…) in the shot).
--work DIR puts the per-shot text_src.json under DIR/<stem>/ instead of the shot's render folder.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]          # repo root: animkit/, examples/
sys.path.insert(0, str(LIB))

from animkit.timeline import default_out  # noqa: E402

DOC = ("Translation table for animkit --lang. Keys = the exact English source strings drawn (\\n and $math$ included); "
       "fill in the value, leave '' if not translated yet. Built by tools/i18n_table.py; see animkit/README.md.")


def collect(ep_dir, shots, work, fps, jobs):
    todo, running, res = list(shots), [], {}
    while todo or running:
        while todo and len(running) < jobs:
            s = todo.pop(0)
            script = (ep_dir / s["script"]).resolve()
            out = (work / script.stem) if work else default_out(script)
            cmd = [sys.executable, "-I"] + (["-B"] if sys.flags.dont_write_bytecode else []) + [
                "-X", "utf8", str(Path(__file__).resolve().with_name("render_shot.py")), str(script), "--collect-text", "--fps", str(fps),
                "--out", str(out)]
            running.append((s, script, out, subprocess.Popen(cmd)))
        for item in list(running):
            s, script, out, p = item
            if p.poll() is not None:
                running.remove(item)
                if p.returncode != 0:
                    raise RuntimeError(f"{s['id']}: text collection failed (exit {p.returncode})")
                res[s["id"]] = (script.stem, json.loads((out / "text_src.json").read_text(encoding="utf-8"))["strings"])
        time.sleep(0.2)
    return [(s["id"],) + res[s["id"]] for s in shots]


def merge(table_p, lang, results):
    data = json.loads(table_p.read_text(encoding="utf-8")) if table_p.exists() else {}
    data["_doc"] = DOC
    groups = data.setdefault(lang, {})
    unused = data.setdefault("_unused", {}).setdefault(lang, {})
    known = {k: v for g in list(unused.values()) + list(groups.values()) for k, v in g.items() if v}
    rows = []
    for sid, stem, strings in results:
        old = groups.get(stem, {})
        new = {s: old.get(s) or known.get(s, "") for s in strings}
        gone = {k: v for k, v in old.items() if k not in new and v}
        if gone:
            unused.setdefault(stem, {}).update(gone)
        for g in unused.values():
            for k in new:
                g.pop(k, None)
        groups[stem] = new
        fam = {}
        for s in strings:
            fam.setdefault(re.sub(r"\d+(?:\.\d+)?", "#", s), []).append(s)
        rows.append((sid, stem, len(new), sum(1 for v in new.values() if v), len(gone),
                     [v for v in fam.values() if len(v) > 3]))
    for k in [k for k, g in unused.items() if not g]:
        del unused[k]
    if not unused:
        del data["_unused"][lang]
    if not data["_unused"]:
        del data["_unused"]
    table_p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--lang", required=True, help="target language code, e.g. es")
    ap.add_argument("--shots", nargs="+", help="shot ids from episode.json (default: all)")
    ap.add_argument("--table", help="default <episode_dir>/captions.json")
    ap.add_argument("--work", help="folder for per-shot text_src.json (default: each shot's render folder)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    ep_dir = Path(a.episode_dir).resolve()
    ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    shots = [s for s in ep["shots"] if not a.shots or s["id"] in a.shots]
    table_p = Path(a.table).resolve() if a.table else ep_dir / "captions.json"
    work = Path(a.work).resolve() if a.work else None
    rows = merge(table_p, a.lang, collect(ep_dir, shots, work, a.fps, a.jobs))
    print(f"table: {table_p}")
    for sid, stem, n, done, gone, fams in rows:
        print(f"  {sid} {stem}: {n} strings, {n - done} untranslated" + (f", {gone} moved to _unused" if gone else ""))
        for f in fams:
            print(f"    differ only in numbers ({len(f)}): {f[0]!r} ... {f[-1]!r}")


if __name__ == "__main__":
    main()
