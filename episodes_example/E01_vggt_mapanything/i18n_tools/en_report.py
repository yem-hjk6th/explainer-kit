"""Summary of a translated render: per shot, the warnings in check_<lang>.json and the untranslated / off_canvas
entries in meta_<lang>.json.
  python -I -X utf8 en_report.py [--lang es] [shot name ...]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("shots", nargs="*")
ap.add_argument("--lang", default="es")
a = ap.parse_args()

EP = Path(__file__).resolve().parents[1]
R = EP / "shots" / "render"
shots = a.shots or sorted(p.name for p in R.iterdir() if p.is_dir() and p.name.startswith("s"))
for s in shots:
    d = R / s
    meta = d / f"meta_{a.lang}.json"
    chk = d / f"check_{a.lang}.json"
    if not meta.exists():
        print(f"== {s}: no {meta.name}")
        continue
    m = json.loads(meta.read_text(encoding="utf-8"))
    c = json.loads(chk.read_text(encoding="utf-8")) if chk.exists() else {}
    i18n = m.get("i18n", {})
    unt, off = i18n.get("untranslated", []), i18n.get("off_canvas", [])
    warns = c.get("warnings", [])
    print(f"== {s}: ok={c.get('ok')} warnings={len(warns)} untranslated={len(unt)} off_canvas={len(off)} dur={m.get('dur')}")
    for w in warns:
        print("   W", w)
    for u in unt:
        print("   U", repr(u))
    for o in off:
        print("   O", json.dumps(o, ensure_ascii=False))