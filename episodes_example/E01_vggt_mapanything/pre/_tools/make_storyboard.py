"""Generate pre/storyboard_pre.md from the render results (the review page finds paragraphs by PRE-n): the edit table comes from data.edl in check.json, the corner tags from meta.json.
  python -I -X utf8 make_storyboard.py <pre dir>"""
import json
import sys
from pathlib import Path

pre = Path(sys.argv[1]).resolve()
ep = json.loads((pre / "episode.json").read_text(encoding="utf-8"))

ABOUT = {
    "c0_open": "Opening card: title + four-part contents.",
    "c1_vggt": "Part 1 title card.",
    "p1_vggt": "Part 1 covers only the VGGT paper architecture: Eq. (1) -> token -> alternating attention -> heads -> two point-cloud routes -> the output has no meters."
               "01 / 05 / 06 use the pre forks (they hide the numbers measured on our test scene); real model-output frames stay, with the tag pointing to Part 4. Captions do not mention the white dot.",
    "c2_mapanything": "Part 2 title card.",
    "p2_mapanything": "Part 2 covers MapAnything: factored representation (08, illustrative values) -> optional inputs and fusion -> scale token -> inputs randomly hidden in training (09).",
    "c3_compare": "Part 3 title card.",
    "p3_compare": "Part 3 side-by-side comparison (pre fork of 10, which hides the white-dot coordinates, 1.001, 72.3% / 9.5%) + paper Tab. 2(a).",
    "c4_test": "Part 4 title card.",
    "p4_our_test": "Part 4 our own test: scene definition -> VGGT depth / confidence -> white dot and error for Method A / B -> the output has no meters (1.001)"
                   " -> MapAnything metric-scale error. The source shots are not forked; all numbers measured on our test scene are here.",
}
FORK = {"s01_task_pre": "the measured tracker error line", "s05_heads_pre": "the 9 measured / GT numbers, the camera-2 top view, R(q), the VGGT field-of-view line",
        "s06_two_routes_pre": "white-dot coordinates, the GT block, Method B step-by-step numbers, the intrinsics row, our-test-scene bars and notes",
        "s10_compare_pre": "white-dot coordinates, 1.001, 72.3% / 9.5%"}

out = ["# E01 English pre version · storyboard (generated from the render results by make_storyboard.py; do not edit by hand)", "",
       "Each segment = a piece of a source shot (time = the source shot's seconds), 0.25 s cross-fade between segments; captions are the pre-version English.", ""]
total = 0.0
for s in ep["shots"]:
    stem = Path(s["script"]).stem
    rd = pre / "shots" / "render" / stem
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    chk = json.loads((rd / "check.json").read_text(encoding="utf-8"))
    total += meta["dur"]
    out += [f"## {s['id']} · {stem}({meta['dur']:.1f} s)", "", ABOUT.get(stem, ""), ""]
    edl = chk.get("data", {}).get("edl")
    if edl:
        out += ["| New time (s) | Source shot | Source seconds | Duration |", "|---|---|---|---|"]
        for row in edl:
            src, a, b, o = row[:4]
            hold = row[4] if len(row) > 4 else 0.0
            note = f" (pre fork: hides {FORK[src]})" if src in FORK else ""
            hn = f" (last frame held {hold:.1f} s)" if hold else ""
            out.append(f"| {o:.1f}–{o + b - a + hold:.1f} | `{src}`{note} | {a:.1f}–{b:.1f}{hn} | {b - a + hold:.1f} |")
        out.append("")
    for tg in meta.get("tags", []):
        out.append(f"- Corner tag \"{tg['text']}\" {tg['t0']:.1f}–{tg['t1']:.1f} s")
    out.append("")
out.insert(3, f"Total length {total:.1f} s ({int(total // 60)}:{total % 60:04.1f}).")
out.insert(4, "")
(pre / "storyboard_pre.md").write_text("\n".join(out), encoding="utf-8")
print(f"{len(ep['shots'])} shots, {total:.1f} s -> {pre / 'storyboard_pre.md'}")
