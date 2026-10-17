# PIPELINE: stages, files, commands, gates

Commands assume the repo root as working directory and `python` = the render env (SETUP.md). Always
`python -I -B -X utf8`. `<EP>` = `episodes/<ID>_<topic>`, e.g. `episodes/E03_pinhole`.

## Episode folder

```
<EP>/
  episode.json            {"id": "E03", "title": "...", "storyboard": "storyboard.md",
                           "shots": [{"id": "E03-01", "script": "shots/s01_intro.py", "size": "S"}, ...]}
  storyboard.md           §0 + one section per shot; the section title contains the shot id (E03-01)
  CLAIMS_to_verify.md     claim table (VERIFY.md)
  elements_needed.md      optional: visual elements the library lacks, with parameters
  STATUS.md               stage / next / waiting for / outputs / questions / time (from `date`)
  effort_log.csv          item,size,start,end,render_s,iterations,note
  shots/                  sNN_name.py, <ep>_common.py (shared scene, numbers, tags, helpers)
  model_runs/             optional: scripts + saved outputs (npz / json) of real models
  verify/                 verifier reports + their recompute scripts
  captions.json           translation table (I18N.md)
  review/round1/          review.html (generated), feedback.md (the reviewer's notes)
  review/round2/          changes.md (what this round changed), review.html, feedback.md ...
  render/<ID>.mp4         the assembled episode
```

Shot ids are `<ID>-NN`; caption-level references `<ID>-NN-k` (k = caption number in the storyboard table).
`size`: S = one geometric picture or one formula process, 15–30 s; M = a multi-step process or a small real computation,
30–60 s; L = a whole architecture with tensor shapes, or real model output, 60–120 s.

## A. Study

- Source material: convert PDFs / pages to text once, write a digest (section → line range → 2-line summary) and read
  only the sections you need. Official code beats the paper's prose for layer counts, token counts, channel orders,
  concatenation order; when they disagree, show the code's version and note the paper's.
- Optional reference videos: `references/REFERENCES.md`.
- Output of this stage, for yourself: the mechanism as input / output → formula → tensor shapes → a small numeric
  example → why it is designed this way; the list of things the audience will not know.

## B. Storyboard (`references/STORYBOARD.md`)

1. `storyboard.md` §0: audience and depth, the through-line object, the numbers table (computed by a script, never by
   hand), notation, computed vs illustrative, tags; then a section per shot.
2. A numbers script (`shots/<ep>_numbers.py` or `<ep>_common.py`) that computes every on-screen number from the scene
   definition and asserts the relations you claim. Storyboard numbers are copied from its full-precision output.
3. `CLAIMS_to_verify.md`: one row per fact / number / shape / formula.
4. Launch the verifier (`templates/verifier_prompt.md`, round 1, scope = all rows). While it runs you may build the
   shared helpers and shots that its findings cannot change much (render stills only).
5. Apply "must fix" items as `~~old~~ → new` in storyboard and CLAIMS; record a correction table at the top of CLAIMS.

## C. Shots

Per shot (`templates/shot_template.py` is a complete minimal one):

```bash
python -I -B -X utf8 tools/render_shot.py <EP>/shots/s02_proj.py --check-only     # until all pass
python -I -B -X utf8 tools/render_shot.py <EP>/shots/s02_proj.py --still 4.0 9.5   # PNGs of chosen moments
python -I -B -X utf8 tools/render_shot.py <EP>/shots/s02_proj.py --sheet-only      # 12-frame contact sheet
python -I -B -X utf8 tools/render_shot.py <EP>/shots/s02_proj.py --jobs 2          # full render
```

Outputs in `<EP>/shots/render/s02_proj/`: `s02_proj.mp4`, `sheet.jpg`, `check.json`, `meta.json` (caption times,
tags, render time, script hash). A failing check stops the render (exit code 2; an exception exits 1, success 0).
Windows `cmd /c start /WAIT` does not pass the exit code through: behind it, judge by the files.

Look at `sheet.jpg` (and stills) after every change: captions not covering the picture, labels inside the frame,
small text not overlapping, tags present, pop-ups readable, no boxes (missing glyphs → use `$...$`). Fix every
`check.json` warning (caption too long / overlapping / shorter than 1 s / wider than the frame).

## D. Assemble

```bash
python -I -B -X utf8 tools/concat.py <EP>                      # -> <EP>/render/<ID>.mp4; refuses stale or failing shots
python -I -B -X utf8 tools/review_page.py <EP> --round 1       # -> <EP>/review/round1/review.html (+ <EP>/review.html)
python -X utf8 tools/render_check.py <EP>/review/round1/review.html   # optional: MathJax errors, overflow at 390 px
```

Before handing over: every shot's `check.json` passes; every number on screen is a CLAIMS row; you looked at all sheets.

## E. Review rounds (`references/REVIEW.md`)

1. The reviewer writes notes on the page and pastes them back; save verbatim as `<EP>/review/roundN/feedback.md`.
2. Answer each note (in `STATUS.md`: note → what you changed). New explanations → new CLAIMS rows → verifier
   (scope = new rows only). General preferences → `episodes/STYLE_local.md`.
3. Write `<EP>/review/round{N+1}/changes.md` (`## <ID>-NN` sections; text before the first section goes on top of
   the page), re-render the changed shots, concat, `review_page.py <EP> --round N+1`.

## F. Optional stages

- Translated version: `references/I18N.md` (`tools/i18n_table.py --lang <code>`, `--lang <code>`).
- Short cut for a talk (≤ N minutes) without touching the approved master: cut shells (see
  `episodes_example/E01_vggt_mapanything/pre/`: `pre/shots/pre_cut.py` and the `pre/src/*_pre.py` forks); prove
  the master unchanged (file hashes + a few frames pixel-identical, e.g. with `pre/_tools/regress_diff.py`).
- Narration script: `references/NARRATION.md`. Voice-over: `references/TTS.md`.
- Real model runs: separate env per model, torch first with the right CUDA index, official code pinned to a commit in
  `third_party/`, weights from the official Hugging Face repo, outputs saved to `model_runs/` as npz + json with a
  `run_meta.json` (versions, commit, timings). Shots read the saved outputs; they never run a model. A verifier
  re-derives the shown numbers from the saved outputs with its own script.

## Timing and cost

Rendering is cheap: a 16 s shot takes roughly 40 s on 2 CPU cores. Writing shots and reviewing are the expensive parts,
and a revision round re-renders only the shots it changes. Make the review page easy to use and each round's changes
easy to find.
