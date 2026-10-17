---
name: explainer-video
description: Make a short, verified math / computer-vision / ML explainer video (computed animation + captions, optional translated version, narration script and TTS voice-over) with the animkit library in this repo. Use when the user asks for an explainer video, an animation of a concept / paper / mechanism, a storyboard for one, or a review / revision round of one.
---

# Explainer video

You make explainer videos that a specific viewer understands and that are *right*. Two things decide the result:
the rules in `references/STYLE.md` (they come from real reviewer feedback) and the habit of computing and checking
every number you show. Rendering is the easy part.

Repo map: library `animkit/` (API reference `animkit/README.md`), tools `tools/`, examples `examples/`, two finished
episodes `episodes_example/` (copy their storyboard and shot formats; the CLAIMS format is in
`references/VERIFY.md`), your work goes in `episodes/<ID>_<topic>/`.

## 0. Before anything

1. Environment works? `python -I -B -X utf8 tools/selftest.py` must print `all OK`. If `config/animkit.json` is
   missing, run `python -X utf8 tools/setup_config.py` (the config is per checkout: a new clone or git worktree
   needs it again, or set `ANIMKIT_CONFIG`). Problems: `SETUP.md`.
2. Ask the user only what you cannot default (one message, short):
   - topic and the source material (lecture notes, a paper, a page); you need a source for every claim;
   - **audience** and what they already know (sets the depth, `references/STORYBOARD.md` §Audience);
   - length; translated version / narration / voice-over wanted?
   - may you run real models (GPU, downloads)? Default: no, use a synthetic scene with known ground truth.
   - Defaults if unanswered: 1080×1440, 30 fps, captions only, no voice, one review round planned.
3. Create `episodes/<ID>_<topic>/` with `STATUS.md` (stage / next step / waiting for / time from `date`).

## 1. The pipeline (details and commands: `references/PIPELINE.md`)

| Stage | You produce | Gate before the next stage |
|---|---|---|
| A. Study | source digest; optional reference-video breakdown (`references/REFERENCES.md`) | you can state the mechanism as input → formula → shapes → small numeric example → why |
| B. Storyboard | `storyboard.md`, `CLAIMS_to_verify.md`, `episode.json` (+ `elements_needed.md` if the library lacks something) | user OK on the outline for long episodes; **independent verification round 1** done and corrections applied (`references/VERIFY.md`) |
| C. Shots | `shots/sNN_*.py`, each with `@shot.check` | `--check-only` passes; contact sheet inspected; full render; `check.json` warnings fixed |
| D. Assemble | `render/<ID>.mp4` (`tools/concat.py`), `review/round1/review.html` (`tools/review_page.py`) | you watched the sheets of every shot; numbers on screen = CLAIMS |
| E. Review rounds | `review/roundN/feedback.md` (user's notes), `review/roundN+1/changes.md`, re-renders | feedback items each answered; new claims verified (max 2 rounds per item) |
| F. Optional | translated version (`references/I18N.md`), narration (`references/NARRATION.md`), voice-over (`references/TTS.md`) | each has its own check |

Write shots directly in the language of the video (English in the examples): captions, titles and labels are plain
strings in the script. `--lang` is only for a second, translated version.

**Micro path (≤ 60 s, 1–3 shots).** Same stages, lighter. Minimal file list in `episodes/<ID>_<topic>/`:

1. `storyboard.md`: one table, one row per caption (time range, caption, what changes on screen, computed or
   illustrative). Skeleton: `templates/storyboard_skeleton.md`.
2. `CLAIMS_to_verify.md`: one row per number / fact on screen (claim, source, how to check, measured / computed /
   inferred / paper).
3. `shots/s01_<name>.py` from `templates/shot_template.py`, with an `@shot.check` that asserts every computed number
   as the string shown.
4. Verification: one fresh sub-agent with `templates/verifier_prompt.md` scoped to all CLAIMS rows; apply its
   corrections (struck-through old wording).
5. Render: `--check-only` → `--sheet-only` (look at `sheet.jpg`) → full render.
6. `episode.json` + `tools/concat.py` only if there are 2+ shots; `tools/review_page.py <EP> --round 1` for the
   review page (one round). Do not skip the claim table, the `@shot.check`, or the contact-sheet look.

## 2. Rules that are never optional

- **Every computable visual is computed** (projections, attention weights, reshapes, back-projections, fitted lines) and
  asserted in `@shot.check`, including the *displayed string*: `ck.eq("depth on screen", f"{z:.3f}", "1.976")`.
  Format from full precision; never re-round a printed value. Illustrative visuals get `shot.tag("Illustration")`; dimension-reduced drawings say so.
- **Every number has a source** (paper § / table / equation, official code file + line, or our own script) in
  `CLAIMS_to_verify.md`, labelled measured / computed / inferred / paper. Comparisons state metric, unit, and
  whether lower or higher is better; percentages state their denominator.
- **Every symbol is defined where it appears**, every time; terms the viewer may not know get a pop-up side window
  (`animkit.Popup`) that explains them briefly and closes.
- **One caption = one visual change**, ≤ 12 words and no wider than the frame (the render warns; at 42 px that is
  often ~8 long words per line, 2 lines max), 2–3 s each, nothing static.
- **Picture first, then the formula**, lit term by term, each term colored or linked to its part of the picture.
- **Our own tests are labelled as ours** (not the paper's experiment) the first time and with the same tag after.
- **Independent verification**: whoever wrote a claim does not verify it; a fix is a new claim; at most two rounds,
  then apply as worded and spot-check. Old wording is struck through (`~~old~~ → new`), never silently deleted.
- **No invented facts in translation or narration**: they may only say what the verified captions / CLAIMS say.
- Full checklist: `references/STYLE.md` (§A content, §B rhythm, §C visual grammar, §D pre-delivery, §E reviewer
  feedback that became rules). Read it before writing the storyboard and again before each review round.

## 3. Working style

- Read the API section you need in `animkit/README.md` (it is long: grep for the element, then read that part).
  Copy patterns from the closest shot in `episodes_example/` instead of inventing layouts. `templates/shot_template.py`
  is a minimal complete shot.
- Iterate on a shot with `--still t1 t2` (PNG frames) and `--sheet-only` (12-frame contact sheet); full renders last.
  Look at the sheet yourself each time (overlaps, text off canvas, boxes = missing glyphs, captions over the picture).
  Stills are not refreshed by later renders: re-run `--still` after every edit.
- 3D scenes: `animkit/README.md` §3D scenes (PinholeCam, OrbitCam az / el convention: az=90, el=0 is a side view).
  2D diagrams such as a pinhole side view: §2D side view (Plot2D with equal scale).
- Run Python as `python -I -B -X utf8`; never edit library files while another session renders (`references/PITFALLS.md`).
- Keep `STATUS.md` short and current; log effort in `effort_log.csv` (`item,size,start,end,render_s,iterations,note`).
- Long episodes or many shots: split the work across sessions / sub-agents as in `references/DISPATCH.md`
  (context budget ≈ 300k per main session, ≤ 150k per one-shot sub-agent, handoff file before switching).

## 4. Delivering

Tell the user: the video path, the review page path (`review/roundN/review.html`, open in a browser; it has a comment
box per shot and a "copy all" button), what is computed vs illustrative, what was verified and by whom, open questions.
When their notes come back, save them verbatim as `review/roundN/feedback.md` (one line per shot id, e.g.
`E03-02: ...`), answer each, and turn any general preference into a new line in your episode's local rules (append to
a `STYLE_local.md` in `episodes/`), so the next episode starts with it.

## Reference index

| File | Read when |
|---|---|
| `references/PIPELINE.md` | starting; stage commands, file formats, gates |
| `references/STYLE.md` | writing the storyboard; before each review round |
| `references/STORYBOARD.md` | stage B: audience depth, storyboard + CLAIMS formats, story patterns |
| `references/REFERENCES.md` | studying reference videos / source papers |
| `references/VERIFY.md` + `templates/verifier_prompt.md` | after CLAIMS, after model runs, after narration / translation |
| `references/REVIEW.md` | stage D–E: review page, feedback file, rounds |
| `references/I18N.md` | a translated version of an English episode (`--lang`) |
| `references/NARRATION.md` | spoken script per visual segment |
| `references/TTS.md` | local TTS voice-over, timing, mixing |
| `references/DISPATCH.md` | multi-session / sub-agent work, cost, shared-machine etiquette |
| `references/PITFALLS.md` | something breaks; before changing the library |
| `templates/` | prompts for sessions and sub-agents, skeleton files, a shot template |
