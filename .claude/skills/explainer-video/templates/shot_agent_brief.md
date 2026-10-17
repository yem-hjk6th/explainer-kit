# Sub-agent brief: write and render ONE shot ({SHOT_ID}, script `shots/{STEM}.py`)

You implement one storyboard section literally: script → checks → contact sheet → fix layout → full render.
Content (captions, what is drawn, numbers, sources) is decided; do not change it. If something cannot be done or looks
wrong, do the closest thing and report it.

## Read (in order; only these parts)
1. `.claude/skills/explainer-video/references/STYLE.md` (all; §E item by item).
2. `animkit/README.md`: "API quick reference", "Writing a shot", "API" (only the elements you use), "3D scenes" /
   "2D side view" if the shot has them, "Conventions", "Pitfalls". Grep for an element before reading.
3. `episodes/{EP}/storyboard.md` §0 (lines {A}–{B}) + your section (lines {C}–{D}). Rows marked as verifier fixes win;
   struck-through text is old — do not use it.
4. `episodes/{EP}/CLAIMS_to_verify.md`: the correction table at the top + the rows your shot uses.
5. `episodes/{EP}/shots/{EP_COMMON}.py` (shared scene / numbers / tags / helpers).
6. Template shot to imitate: `{TEMPLATE_SHOT}` (grep, then read line ranges; do not read it whole if > 400 lines).

## Script rules
- Docstring: what the shot is, storyboard section, one line each for computed / illustrative / paper numbers.
- `Shot("{STEM}", DUR)` with DUR from the section title (±3 s allowed, report it).
- `CAPS = [(t0, t1, "..."), ...]` at the top, text verbatim from the caption column, one row = one caption, 2–4 s, in
  order, no overlap; then `for a, b, s in CAPS: shot.caption(a, b, s)`.
- All on-screen text through animkit entry points (no PIL / OpenCV text), so a later `--lang` translation can reach it.
  An f-string whose runtime value changes every frame → `T("...{x}...", x=...)` (one translation key instead of many).
- Numbers only from the shared numbers module; `@shot.check` asserts every computed / measured number **as the string
  shown on screen**.
- Symbols defined on screen every time; formula terms lit one by one, same color / link line as the picture; vector
  concatenations drawn as concatenation; comparisons with metric and direction.
- Tags exactly as the storyboard says, for the stated time ranges. Pop-ups via `animkit.Popup` + `popup_check`,
  never inside the caption band (y ≥ 1250).
- Do not edit: the library, other shots, shared modules, storyboard, CLAIMS, STATUS, effort log. Helpers go in your
  script. Never delete files.

## Render ({MASK}, 1 process)
```
python -I -B -X utf8 tools/render_shot.py episodes/{EP}/shots/{STEM}.py --check-only     # until all pass
python -I -B -X utf8 tools/render_shot.py episodes/{EP}/shots/{STEM}.py --sheet-only     # look at sheet.jpg
python -I -B -X utf8 tools/render_shot.py episodes/{EP}/shots/{STEM}.py --still 12.5 20  # details (≤ 3 frames per call)
python -I -B -X utf8 tools/render_shot.py episodes/{EP}/shots/{STEM}.py --jobs 1         # full render
```
At most 3 sheet iterations. Judge success by files (`{STEM}.mp4`, `check.json` all pass, `meta.json`), not exit
codes. Fix every warning in `check.json`. No GPU, no installs, no concat / review page (the main session does those).
Context budget ≤ 150k.

## Report (≤ 10 lines)
Script path; DUR; checks passed / total and warnings; deviations from the storyboard and why; anything in the
storyboard or CLAIMS you think is wrong (report only); new tool pitfalls (one per line); start / end time (from
`date`), render seconds (`meta.json`), sheet iterations.
