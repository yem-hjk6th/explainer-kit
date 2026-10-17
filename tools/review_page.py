"""Review pages for one episode, one folder per review round.
  python -I -X utf8 review_page.py <episode_dir> --round N [--out FILE]
Reads <episode_dir>/episode.json, the storyboard file it names, and each shot's render/<stem>/{mp4, meta, check}.
Writes <episode_dir>/review/roundN/review.html, then <episode_dir>/review.html = jump to the latest round + list of
rounds (open it straight from disk; video paths are relative).
  round 1  : per shot the video, its storyboard section and numbered points, each with a feedback box.
  round 2+ : one overall feedback box at the top; per shot the video, "What changed this round" from
             review/roundN/changes.md and the folded previous-round feedback from review/round{N-1}/feedback.md
             (read-only).
Videos never autoplay and the page needs no sound (reviewers often work over a remote-desktop link).
Drafts stay in the browser, per round; "Copy all feedback" copies them under an 'E01 review round N feedback' header.
changes.md : '## <shot id>' headings (e.g. '## E01-06'), markdown under each; text before the first one goes on top.
feedback.md : lines starting with a shot id ('E01-06: ...', 'E01-02-12: ...'; a full-width colon is also accepted)
              belong to that shot; the rest is general.
A page file that this script did not write (e.g. an old root review.html) is never overwritten: the old root page
becomes review/round1/review.html (if free), anything else is moved to <episode_dir>/_moved/.
"""
import argparse
import html
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import quote

LIB = Path(__file__).resolve().parents[1]          # repo root: animkit/, examples/
sys.path.insert(0, str(LIB))

from concat import shot_status  # noqa: E402

# ---------------------------------------------------------------- tiny markdown -> html (storyboard sections)
_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.S)


def _inline(s):
    keep = []

    def stash(m):
        keep.append(html.escape(m.group(0), quote=False))
        return f"\x00{len(keep) - 1}\x00"

    s = _MATH.sub(stash, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"~~(.+?)~~", r"<s>\1</s>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)


def md_to_html(md):
    lines, out, para, i = md.split("\n"), [], [], 0

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            flush()
            lvl = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
        elif ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            flush()
            rows = [ln]
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j])
                j += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            t = "<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0]) + "</tr>"
            for r in cells[1:]:
                t += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</table>")
            i = j
            continue
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", ln):
            flush()
            m2 = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)", ln)
            lvl = len(m2.group(1).expandtabs(4)) // 2
            mark = "•" if m2.group(2) in "-*+" else m2.group(2)
            out.append(f'<div class="li" style="margin-left:{0.4 + 1.2 * lvl:.1f}em"><span class="mk">{mark}</span>'
                       f"{_inline(m2.group(3))}</div>")
        elif ln.strip() == "":
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    return "\n".join(out)


def section(md, key):
    """Storyboard section whose heading contains key (e.g. 'S03'), up to the next heading of the same or higher level."""
    lines = md.split("\n")
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![0-9])")
    for i, ln in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m and pat.search(m.group(2)):
            lvl, j = len(m.group(1)), i + 1
            while j < len(lines) and not re.match(rf"^#{{1,{lvl}}}\s", lines[j]):
                j += 1
            return "\n".join(lines[i:j])
    return None


def _sid_pat(sid):
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(sid)}(?![0-9])")


def parse_changes(md, sids):
    """changes.md -> (general markdown, {sid: markdown under its heading})."""
    per, lines = {}, md.split("\n")
    first = len(lines)
    for i, ln in enumerate(lines):
        m = re.match(r"^#{1,6}\s+(.*)", ln)
        if m and any(_sid_pat(s).search(m.group(1)) for s in sids):
            first = i
            break
    for s in sids:
        sec = section(md, s)
        if sec:
            body = "\n".join(sec.split("\n")[1:]).strip()
            if body:
                per[s] = body
    general = "\n".join(ln for ln in lines[:first] if not re.match(r"^#\s", ln)).strip()
    return general, per


def parse_feedback(md, sids):
    """feedback.md -> (general lines, {sid: [(key, text)]}); keys like 'E01-02-12' go to shot 'E01-02'."""
    pats = {s: re.compile(rf"^\s*(?:[-*]\s*)?(?:\*\*)?({re.escape(s)}(?:-\d+)?)(?:\*\*)?\s*[:\uff1a]\s*(.*)$") for s in sids}
    general, per = [], {}
    for ln in md.split("\n"):
        if not ln.strip() or re.match(r"^#{1,6}\s", ln):
            continue
        for s, p in pats.items():
            m = p.match(ln)
            if m:
                per.setdefault(s, []).append((m.group(1), m.group(2)))
                break
        else:
            general.append(ln.strip())
    return general, per


def rounds(ep_dir):
    """Round numbers that have a folder under <episode_dir>/review/."""
    rd = ep_dir / "review"
    return sorted(int(m.group(1)) for p in rd.glob("round*") if p.is_dir() and (m := re.fullmatch(r"round(\d+)", p.name)))


GEN = '<meta name="generator" content="animkit review_page {}">'


def _ours(path):
    return path.exists() and 'content="animkit review_page' in path.read_text(encoding="utf-8", errors="replace")[:2000]


def _move_aside(ep_dir, path):
    dst = ep_dir / "_moved" / f"{path.relative_to(ep_dir).as_posix().replace('/', '__')}.{time.strftime('%y%m%d_%H%M%S')}"
    dst.parent.mkdir(exist_ok=True)
    shutil.move(str(path), str(dst))
    print(f"moved {path} -> {dst}")


def adopt_root_page(ep_dir):
    """An old root review.html (written before review rounds existed) becomes review/round1/review.html.
    Its relative video paths keep working through <base href="../../">."""
    root = ep_dir / "review.html"
    if not root.exists() or _ours(root):
        return
    dst = ep_dir / "review" / "round1" / "review.html"
    if dst.exists():
        _move_aside(ep_dir, root)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), str(dst))
    s = dst.read_text(encoding="utf-8")
    dst.write_text(s.replace("<head>", '<head><base href="../../">', 1), encoding="utf-8")
    print(f"old root page -> {dst} (added <base href>)")


def _write_page(ep_dir, path, text):
    if path.exists() and not _ours(path):
        _move_aside(ep_dir, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- page
CSS = """
:root{--bg:#f6f6f4;--fg:#1d1d1f;--mute:#6b6b70;--card:#fff;--line:#deded9;--bad:#b3261e;--ok:#1b7a3a;--acc:#9a6a00}
@media (prefers-color-scheme: dark){:root{--bg:#141518;--fg:#e8e8ea;--mute:#9a9aa2;--card:#1d1f23;--line:#33363c;
--bad:#ff8a80;--ok:#7fd99a;--acc:#ffcf70}}
body{background:var(--bg);color:var(--fg);font:15px/1.6 "Microsoft YaHei","Segoe UI",sans-serif;margin:0;padding:16px}
main{max-width:1200px;margin:0 auto} h1{font-size:22px;margin:4px 0 2px} .mute{color:var(--mute);font-size:13px}
.shot{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:18px 0}
.shot h2{font-size:18px;margin:0 0 8px} .grid{display:grid;grid-template-columns:minmax(0,360px) minmax(0,1fr);gap:16px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
video{width:100%;border-radius:6px;background:#000} .sb{max-height:560px;overflow:auto;border-left:3px solid var(--line);
padding-left:12px} .sb h2,.sb h3,.sb h4{font-size:15px;margin:8px 0 4px} table{border-collapse:collapse;font-size:13px;
display:block;overflow-x:auto} td,th{border:1px solid var(--line);padding:3px 6px;vertical-align:top} pre{white-space:pre-wrap;
font-size:12px} .li{margin:2px 0} .mk{display:inline-block;min-width:1.4em;color:var(--mute)}
.pts{margin-top:10px;width:100%;border-collapse:collapse;display:table} .pts td{border:none;border-top:1px solid var(--line);
padding:6px 4px} .pid{font-weight:bold;white-space:nowrap;color:var(--acc)} .tc{white-space:nowrap;color:var(--mute);
font-size:12px} textarea{width:100%;box-sizing:border-box;min-height:34px;font:inherit;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:4px} .bad{color:var(--bad)} .ok{color:var(--ok)}
.bar{position:sticky;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px;display:flex;gap:10px;
align-items:center;flex-wrap:wrap} button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);
background:var(--bg);color:var(--fg);cursor:pointer} #all{width:100%;min-height:80px;display:none}
.nav{margin:6px 0 4px;font-size:13px} .nav a,.nav b{margin-right:12px} a{color:var(--acc)}
.overall{background:var(--card);border:1px solid var(--acc);border-radius:10px;padding:14px;margin:14px 0}
.overall textarea{min-height:170px} .overall h2{font-size:17px;margin:0 0 6px}
.idx{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px} .chip{display:inline-block;padding:1px 9px;border-radius:10px;
font-size:12px;border:1px solid var(--line);text-decoration:none;color:var(--fg)} .chip.on{border-color:var(--acc);color:var(--acc)}
.chg h3,.prev summary{font-size:15px;margin:2px 0 6px} .prev{margin-top:12px;border-top:1px solid var(--line);padding-top:8px}
.prev summary{cursor:pointer;color:var(--mute)} .fb{margin:3px 0} .fb b{color:var(--acc)}
"""

JS = """
const EP = document.body.dataset.ep, R = document.body.dataset.round, K = EP + ':r' + R + ':';
function load(){document.querySelectorAll('textarea[data-k]').forEach(t=>{try{t.value=localStorage.getItem(K+t.dataset.k)||''}catch(e){}
 t.addEventListener('input',()=>{try{localStorage.setItem(K+t.dataset.k,t.value)}catch(e){}})})}
function collect(){const H=EP+' review round '+R+' feedback\\n',ov=document.getElementById('overall');
 if(ov){const v=ov.value.trim();return H+(v||'(nothing written)')}
 const L=[];document.querySelectorAll('textarea[data-k]').forEach(t=>{const v=t.value.trim();
 if(v)L.push(t.dataset.k+': '+v.replace(/\\n+/g,' / '))});return H+(L.length?L.join('\\n'):'(nothing written)')}
function copyAll(){const s=collect(),a=document.getElementById('all'),ov=document.getElementById('overall');
 a.style.display='block';a.value=s;const n=ov?'':' ('+(s.split('\\n').length-1)+' items)';
 (navigator.clipboard?navigator.clipboard.writeText(s):Promise.reject()).then(()=>{msg('Copied'+n)})
 .catch(()=>{a.select();document.execCommand('copy');msg('Selected: press Ctrl+C to copy')})}
function msg(s){document.getElementById('msg').textContent=s}
load();
"""

MATHJAX = """<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>"""


def fmt_t(x):
    return f"{x:.1f}"


def _video(mp4, out):
    # no autoplay, no sound needed: reviewers often work over a remote-desktop link
    if not mp4.exists():
        return '<p class="bad">Not rendered yet</p>'
    return (f'<video src="{quote(os.path.relpath(mp4, out.parent).replace(os.sep, "/"))}" controls loop '
            f'preload="metadata" playsinline></video>')


def _nav(ep_dir, rnd, out):
    rs = sorted({r for r in rounds(ep_dir) if (ep_dir / "review" / f"round{r}" / "review.html").exists()} | {rnd})
    rel = lambda p: quote(os.path.relpath(p, out.parent).replace(os.sep, "/"))  # noqa: E731
    items = [f"<b>Round {r} (this page)</b>" if r == rnd else
             f'<a href="{rel(ep_dir / "review" / f"round{r}" / "review.html")}">Round {r}</a>' for r in rs]
    return f'<div class="nav">Rounds: {"".join(items)}<a href="{rel(ep_dir / "review.html")}?list">All rounds</a></div>'


def _shot_head(s, meta, probs):
    sid = s["id"]
    head = f"{sid} · {html.escape(meta['shot'] if meta else Path(s['script']).stem)}"
    info = []
    if s.get("size"):
        info.append(f"size {s['size']}")
    if meta:
        info.append(f"{meta['dur']:.1f} s")
        info.append(f"rendered {meta['rendered_at']} ({meta['render_s']} s)")
    status = "".join(f' <span class="bad">[{html.escape(p)}]</span>' for p in probs)
    return head + status, " · ".join(info)


def build(ep_dir, out, rnd=1):
    ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    eid = ep.get("id", ep_dir.name)
    title = html.escape(f"{eid} {ep.get('title', '')}".strip())
    body, total = (build_round1 if rnd == 1 else build_round_n)(ep_dir, ep, out, rnd)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">{GEN.format(f"round={rnd}")}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} review round {rnd}</title><style>{CSS}</style>
{MATHJAX}</head>
<body data-ep="{html.escape(eid)}" data-round="{rnd}"><main><h1>{title} · review round {rnd}</h1>
{_nav(ep_dir, rnd, out)}
<div class="mute">{len(ep['shots'])} shots, {total:.1f} s rendered in total. Videos do not autoplay and have no sound. Generated {time.strftime('%Y-%m-%d %H:%M')}.</div>
{body}
<div class="bar"><button onclick="copyAll()">Copy all feedback</button><span id="msg" class="mute"></span>
<textarea id="all" readonly></textarea></div></main><script>{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out == ep_dir / "review" / f"round{rnd}" / "review.html":
        _write_page(ep_dir, out, page)
    else:
        out.write_text(page, encoding="utf-8")
    return out


def build_round_n(ep_dir, ep, out, rnd):
    """Round 2+: one overall box; per shot the video + "What changed this round" + folded previous-round feedback."""
    sids = [s["id"] for s in ep["shots"]]
    ch_p = ep_dir / "review" / f"round{rnd}" / "changes.md"
    fb_p = ep_dir / "review" / f"round{rnd - 1}" / "feedback.md"
    ch_gen, ch = parse_changes(ch_p.read_text(encoding="utf-8"), sids) if ch_p.exists() else ("", {})
    fb_gen, fb = parse_feedback(fb_p.read_text(encoding="utf-8"), sids) if fb_p.exists() else ([], {})
    if not ch_p.exists():
        print(f"warning: {ch_p} missing; every shot shows 'No changes written up'")
    if not fb_p.exists():
        print(f"warning: {fb_p} missing; no previous-round feedback shown")
    parts, total = [], 0.0
    for s in ep["shots"]:
        sid = s["id"]
        mp4, meta, probs = shot_status(ep_dir, s)
        total += meta["dur"] if meta else 0.0
        head, info = _shot_head(s, meta, probs)
        chg = (md_to_html(ch[sid]) if sid in ch else
               f'<p class="mute">{"Unchanged this round" if ch_p.exists() else "No changes written up (changes.md does not exist yet)"}</p>')
        prev = fb.get(sid, [])
        prev_html = ("".join(f'<div class="fb"><b>{html.escape(k)}</b>: {_inline(v)}</div>' for k, v in prev)
                     if prev else '<p class="mute">No feedback on this shot in the previous round</p>')
        parts.append(f'<section class="shot" id="{sid}"><h2>{head}</h2><div class="mute">{info}</div>'
                     f'<div class="grid"><div>{_video(mp4, out)}</div><div class="chg"><h3>What changed this round</h3>{chg}'
                     f'<details class="prev"><summary>Round {rnd - 1} feedback ({len(prev)} items, read-only)</summary>{prev_html}'
                     f'</details></div></div></section>')
    chips = "".join(f'<a class="chip{" on" if s["id"] in ch else ""}" href="#{s["id"]}">{s["id"]}'
                    f'{" (changed)" if s["id"] in ch else ""}</a>' for s in ep["shots"])
    top = (f'<section class="overall"><h2>Round {rnd} overall feedback</h2>'
           f'<div class="mute">After watching the whole episode, one paragraph is enough. To point at a shot, write its id, e.g. "E01-06: ...". Drafts are kept in this browser.</div>'
           f'<textarea id="overall" data-k="overall" placeholder="Overall feedback"></textarea><div class="idx">{chips}</div>')
    if ch_gen:
        top += f'<div class="chg" style="margin-top:10px">{md_to_html(ch_gen)}</div>'
    if fb_gen:
        top += (f'<details class="prev"><summary>Round {rnd - 1} overall feedback (read-only)</summary>'
                + "".join(f'<div class="fb">{_inline(g)}</div>' for g in fb_gen) + "</details>")
    return top + "</section>" + "".join(parts), total


def build_round1(ep_dir, ep, out, rnd):
    """Round 1: per shot video + storyboard section + numbered points, each with its own feedback box."""
    sb_path = ep_dir / ep.get("storyboard", "storyboard.md")
    sb = sb_path.read_text(encoding="utf-8") if sb_path.exists() else ""
    parts, total = [], 0.0
    for s in ep["shots"]:
        sid = s["id"]
        mp4, meta, probs = shot_status(ep_dir, s)
        chk_p = mp4.parent / "check.json"
        chk = json.loads(chk_p.read_text(encoding="utf-8")) if chk_p.exists() else None
        head, info = _shot_head(s, meta, probs)
        total += meta["dur"] if meta else 0.0
        vid = _video(mp4, out)
        sec = section(sb, s.get("storyboard_key", sid))
        sb_html = md_to_html(sec) if sec else f'<p class="mute">No storyboard section with a heading containing "{html.escape(sid)}"</p>'
        rows = [(f"{sid}-0", "", "Overall (visuals, pacing, wording)")]
        for c in (meta or {}).get("captions", []):
            rows.append((f"{sid}-{c['n']}", f"{fmt_t(c['t0'])}–{fmt_t(c['t1'])} s", html.escape(c["text"]).replace(
                "\n", "<br>")))
        pts = "".join(f'<tr><td class="pid">{k}</td><td class="tc">{tc}</td><td>{txt}</td>'
                      f'<td style="width:42%"><textarea data-k="{k}" placeholder="Feedback"></textarea></td></tr>'
                      for k, tc, txt in rows)
        extra = []
        for tg in (meta or {}).get("tags", []):
            extra.append(f"Corner tag \"{html.escape(tg['text'])}\" {fmt_t(tg['t0'])}–{fmt_t(tg['t1'])} s")
        if chk:
            n_ok = sum(i["ok"] for i in chk["items"])
            cls = "ok" if chk["ok"] else "bad"
            items = "".join(f'<div class="{"ok" if i["ok"] else "bad"}">{html.escape(i["name"])}: '
                            f'{html.escape(json.dumps(i["value"], ensure_ascii=False))} {html.escape(i["op"])} '
                            f'{html.escape(json.dumps(i["ref"], ensure_ascii=False))}</div>' for i in chk["items"])
            warns = "".join(f'<div class="mute">⚠ {html.escape(w)}</div>' for w in chk.get("warnings", []))
            extra.append(f'<details><summary class="{cls}">Numeric self-check {n_ok}/{len(chk["items"])} passed'
                         f'{", " + str(len(chk.get("warnings", []))) + " warnings" if chk.get("warnings") else ""}'
                         f'</summary>{items}{warns}</details>')
        parts.append(f'<section class="shot" id="{sid}"><h2>{head}</h2><div class="mute">{info}'
                     f'</div><div class="grid"><div>{vid}</div><div class="sb">{sb_html}</div></div>'
                     f'<table class="pts">{pts}</table>{"".join(f"<div class=mute>{e}</div>" for e in extra)}</section>')
    hint = '<div class="mute">Write feedback to the right of each point (ids like E01-03-2), then click "Copy all feedback".</div>'
    return hint + "".join(parts), total


def build_index(ep_dir):
    """<episode_dir>/review.html: jump to the latest round that has a page; '?list' stays and lists all rounds."""
    ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    title = html.escape(f"{ep.get('id', ep_dir.name)} {ep.get('title', '')}".strip())
    have = [r for r in rounds(ep_dir) if (ep_dir / "review" / f"round{r}" / "review.html").exists()]
    rows = []
    for r in reversed(have):
        d = ep_dir / "review" / f"round{r}"
        st = time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "review.html").stat().st_mtime))
        bits = [f"page {st}"]
        if (d / "changes.md").exists():
            bits.append("has changes.md")
        bits.append("feedback recorded (feedback.md)" if (d / "feedback.md").exists()
                    else '<span class="mute">feedback not recorded yet</span>')
        rows.append(f'<li><a href="review/round{r}/review.html"><b>Round {r}</b></a> · {" · ".join(bits)}</li>')
    jump = ""
    if have:
        latest = f"review/round{have[-1]}/review.html"
        jump = (f'<script>if(!/[?&]list\\b/.test(location.search))location.replace("{latest}")</script>'
                f'<noscript><meta http-equiv="refresh" content="0; url={latest}"></noscript>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">{GEN.format("index")}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} review</title><style>{CSS}</style>{jump}</head>
<body><main><h1>{title} · review</h1>
<p class="mute">{"Opens the latest round automatically (round " + str(have[-1]) + ")." if have else "No review page for any round yet."}</p>
<ul>{"".join(rows)}</ul></main></body></html>"""
    out = ep_dir / "review.html"
    _write_page(ep_dir, out, page)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--round", type=int, help="review round (1 = per-shot boxes; 2+ = one overall box)")
    ap.add_argument("--out", help="write only this file (no review/ folder, no index)")
    a = ap.parse_args()
    ep_dir = Path(a.episode_dir).resolve()
    if a.round is None:
        have = rounds(ep_dir)
        nxt = (max(have) + 1) if have else 1
        print(f"need --round N (existing rounds: {have or 'none'}; a new round would be --round {nxt})")
        sys.exit(2)
    if a.out:
        print(build(ep_dir, Path(a.out).resolve(), a.round))
        return
    adopt_root_page(ep_dir)
    print(build(ep_dir, ep_dir / "review" / f"round{a.round}" / "review.html", a.round))
    print(build_index(ep_dir))


if __name__ == "__main__":
    main()
