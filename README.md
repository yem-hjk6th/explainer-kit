# explainer-kit

Make short, *correct* math / computer-vision explainer videos with Claude Code: every picture that can be computed is
computed (and self-checked in the shot script), every number on screen has a source, and a human reviewer watches each
round on a review page. Shots are written in English; optional extras: an on-screen translation into another language
(`--lang`), a spoken voice-over script, and a local TTS voice-over.

Rendering is plain Python: numpy + OpenCV + Pillow + ffmpeg, formulas through matplotlib's mathtext. No Manim, no LaTeX,
no GPU. Output is a 1080x1440 (3:4) 30 fps H.264 video, one short script per shot.

What is in here:

| Path | What |
|---|---|
| `animkit/` | the shot library (timeline, captions, formulas lit term by term, tensors / tokens / reshape, attention maps, 3D cameras and point clouds, architecture diagrams, charts, pop-up side windows, on-screen translation). API reference with a quick-reference table: `animkit/README.md` |
| `tools/` | `render_shot.py` (check, then render one shot), `concat.py` (whole episode), `review_page.py` (round-by-round review page), `i18n_table.py` (collect on-screen text for translation), `selftest.py`, `setup_config.py`, `analyze_explainer.py` (break down a reference video), `render_check.py` (headless check of HTML pages), `tts/` (voice samples), `sanitize_scan.py` |
| `examples/` | small self-test shots (epipolar geometry, tensors, plots, tiles, i18n demo) |
| `episodes_example/` | two complete episodes made with this kit (storyboards, shot scripts, model-run scripts; for E01 also a short cut, its narration script and a voiced variant); see `episodes_example/README.md` |
| `.claude/skills/explainer-video/` | the Claude Code skill: `SKILL.md` (the pipeline and the rules) + `references/` + `templates/` |
| `config/animkit.example.json` | machine paths (fonts, ffmpeg); your copy `config/animkit.json` is git-ignored |

## Quick start

1. **Environment** (details and the optional extra environments in `SETUP.md`):
   ```bash
   conda create -n explainer python=3.11 -y
   conda activate explainer
   pip install numpy opencv-python pillow matplotlib imageio-ffmpeg
   ```
2. **Config** (finds fonts and ffmpeg, writes `config/animkit.json`):
   ```bash
   python -X utf8 tools/setup_config.py
   ```
   The config is per checkout (git-ignored): a fresh clone or a `git worktree` needs this step again, or set
   `ANIMKIT_CONFIG` to an existing `config/animkit.json`.
3. **Self-test** (6 checks, all should print OK; first run typesets formulas into `.cache/`):
   ```bash
   python -I -B -X utf8 tools/selftest.py
   python -I -B -X utf8 tools/render_shot.py examples/epipolar.py
   ```
   Output: `examples/render/epipolar/epipolar.mp4`, `sheet.jpg` (contact sheet), `check.json` (all `pass`).
4. **First episode**: open Claude Code in this folder and ask, for example,
   *"Use the explainer-video skill: make a 25-second explainer of pinhole projection for first-year grad students."*
   The skill (`.claude/skills/explainer-video/SKILL.md`) walks it through: audience and depth → storyboard + claim
   table → independent check → shot scripts with self-checks → contact-sheet review → review page → your feedback rounds.
   Episodes go in `episodes/<id>_<topic>/` (create it; `episodes_example/` shows the finished layout). For a clip under
   ~40 s, the skill's *micro path* is a short checklist (one storyboard table, a few CLAIMS rows, one shot, one check
   by a fresh sub-agent); start a shot from `.claude/skills/explainer-video/templates/shot_template.py`.

Always run Python as `python -I -B -X utf8 ...` (isolated mode, no `.pyc` next to shared scripts, UTF-8 I/O on Windows).
The Markdown files are UTF-8 without BOM: Windows PowerShell 5.1 `Get-Content` shows them garbled; use an editor,
`Get-Content -Encoding utf8`, or PowerShell 7.

## What a "good" result looks like

- One caption sentence per visual change (at most 12 words and one frame width; the render warns when a line is too
  wide), no static slides.
- Geometry or tensor picture first, then the formula, lit term by term, each term colored / linked to its part of the
  picture.
- Every symbol defined every time it appears; every comparison states metric, unit and whether lower or higher is better.
- Computed visuals are computed (and asserted in `@shot.check`); illustrative ones carry a corner tag.
- Each claim on screen is in `CLAIMS_to_verify.md` with a source, and an independent agent has checked it.

The full rule list is `.claude/skills/explainer-video/references/STYLE.md`; it grew out of real review feedback.

## Licenses

Code in this repo: see `LICENSES.md`. Model weights used by the example episodes are *not* included and several are
non-commercial; fonts are not distributed (the config points at fonts installed on your system).
