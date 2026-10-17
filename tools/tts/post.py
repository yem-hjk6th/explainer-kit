"""whisper round-trip (CPU) + stats + samples.html. Run in render env: python -I -X utf8 post.py"""
import json, re, html
from pathlib import Path
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
segs = {s["id"]: s for s in json.loads((HERE / "segments.example.json").read_text(encoding="utf-8"))}
res = json.loads((HERE / "results_raw.json").read_text(encoding="utf-8"))
m = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
for r in res:
    seg, _ = m.transcribe(str(HERE / "audio" / f"{r['name']}.wav"), language="en")
    r["whisper"] = " ".join(s.text.strip() for s in seg)
    s = segs[r["seg"]]
    r["words"] = sum(len(l.split()) for l in s["lines"])
    r["wps"] = round(r["words"] / r["dur"], 2)
    r["window"] = s["window"]
    r["stretch"] = round(r["dur"] / s["window"], 2)
    # whole-shot estimate: English dur per covered caption x all captions
    r["shot_est"] = round(r["dur"] / s["covered"] * s["caps"], 1)
    r["shot_s"] = s["shot_s"]
(HERE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

rows = []
for sid, s in segs.items():
    rows.append(f"<h2>{sid} {html.escape(s['title'])}</h2><p class=meta>Shot {s['shot_s']} s / {s['caps']} captions; sample covers {s['covered']}, window {s['window']} s</p>")
    rows.append("<blockquote>" + "<br>".join(html.escape(l) for l in s["lines"]) + "</blockquote>")
    for r in [x for x in res if x["seg"] == sid]:
        rows.append(f"<div class=item><b>{r['speaker']} · {r['pace']}</b> "
                    f"<audio controls preload=none src='audio/{r['name']}.wav'></audio>"
                    f"<div class=meta>{r['dur']} s · {r['wps']} words/s · window {r['window']} s → needs ×{r['stretch']} · whole-shot estimate {r['shot_est']} s (original {r['shot_s']} s)</div>"
                    f"<div class=wh>whisper: {html.escape(r['whisper'])}</div></div>")
page = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>E01 English voice-over samples</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;background:#fff;color:#222}
.item{border-top:1px solid #ddd;padding:8px 0}.meta{color:#666;font-size:13px}.wh{color:#555;font-size:13px}
blockquote{background:#f4f4f4;margin:8px 0;padding:8px 12px;border-radius:6px}audio{vertical-align:middle;width:360px;max-width:100%}
@media(prefers-color-scheme:dark){body{background:#181818;color:#ddd}blockquote{background:#262626}.meta,.wh{color:#aaa}.item{border-color:#444}}</style>
<h1>E01 English voice-over samples (Qwen3-TTS, local)</h1>
<p class=meta>The same text with 3 voices × 2 speaking paces. "Window" = the time the matching captions occupy in the original shot; "needs ×" &gt; 1 means the English audio is longer than the picture. Numbers are written out as English words when generating (Arabic digits would be misread); the text shown above uses the on-screen form.</p>"""
(HERE / "samples.html").write_text(page + "\n".join(rows), encoding="utf-8")
for r in res:
    print(r["name"], r["dur"], r["wps"], r["stretch"], "|", r["whisper"][:200])
