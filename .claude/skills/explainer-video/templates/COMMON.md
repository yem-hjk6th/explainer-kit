# Shared rules for every line of project {PROJECT_NAME} (coordinator = session {COORDINATOR})

Fill the braces, save as `{PROJECT_ROOT}/_prompts/COMMON.md`, and start every line brief with "Read COMMON.md first".

## What we are making
{ONE_PARAGRAPH: topic, audience, the reviewer, the single success criterion (e.g. "the reviewer understands it")}.
Repo: `{REPO_ROOT}` (animkit + tools). Style rules: `.claude/skills/explainer-video/references/STYLE.md` and
`episodes/STYLE_local.md`. Pipeline: `.claude/skills/explainer-video/references/PIPELINE.md`.

## Environment
- Render: `{PY_RENDER} -I -B -X utf8 tools/render_shot.py ...` (config `config/animkit.json`).
- Optional envs: `{PY_ASR}` (whisper / OCR), `{PY_TTS}` (TTS, GPU), `{PY_MODEL_*}` (real models, GPU).
- Never install large dependencies without telling the coordinator "I am about to install X" first.

## Compute share (the coordinator updates this table; it overrides anything older)
| Line | CPU cores (mask) | threads | max render processes | GPU |
|---|---|---|---|---|
| {LINE} | {CORES} (`{MASK}`) | {N} | {P} | {none / memory fraction f, ≤ T min per job} |
Command form (Windows): `cmd /c start "" /B /WAIT /BELOWNORMAL /AFFINITY {MASK} <python> ...`; set
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` = threads. Before any GPU job check `nvidia-smi`; if someone else is using it,
ask. When the coordinator says higher-priority work started: stop GPU at once, drop to the minimum share.

## Protocol
1. Write only your own folder; read-only elsewhere; never delete (move to `_moved/`).
2. Keep `STATUS.md`: stage / next / waiting for / outputs for others (paths) / questions for the reviewer / time
   (from `date`).
3. Facts and numbers → `CLAIMS_to_verify.md` (claim, source, how to check, measured / computed / inferred / paper).
   Verifier corrections are applied as `~~old~~ → new`.
4. Effort → `effort_log.csv` (`item,size,start,end,render_s,iterations,note`).
5. Questions for the reviewer and anything for another line → message the coordinator. At each stage end send ≤ 8
   lines: what, paths, next, waiting for.
6. Tool pitfalls → append to `{ISSUES_FILE}` (read first; no duplicates; ≤ 10 lines per entry).
7. Context: main session ≤ ~300k (write a HANDOFF before), sub-agents one narrow task ≤ 150k; execution work may use
   a cheaper model.
8. Reply to the coordinator in {LANGUAGE}.
