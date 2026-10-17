# Example episodes

Two complete episodes made with this kit for a graduate computer-vision audience, in English. Each folder has its
own short README (what the episode covers, how to render it, which shots need real model output first).

| Episode | Topic | Shots | Length |
|---|---|---|---|
| [`E01_vggt_mapanything/`](E01_vggt_mapanything/README.md) | multi-view images → cameras, depth, point cloud: VGGT (tokens, alternating attention, heads) vs MapAnything (factored rays × depth × pose × metric scale, optional inputs); plus a ≤ 5 min cut and a voiced version of that cut | 10 (+ 9 in the cut, 9 in the voiced cut) | 7:43 (cut 3:59, voiced cut 4:13) |
| [`E02_depth/`](E02_depth/README.md) | depth vs inverse depth vs disparity, scale ambiguity and affine alignment, Depth Pro (multi-scale tiles, focal length), Depth Anything 3 (depth + ray maps), comparison | 10 | 6:36 |

What is shipped: scripts and text only. No videos, no paper figures, no weights, and no raw model-output arrays
(the `.npz` files some shots read). The measured numbers the episodes quote are in `model_runs/*metrics.json`.

## Files to look at, in pipeline order

1. `storyboard.md`: §0 (audience, the one object followed through the episode, numbers table, notation, which visuals
   are computed vs illustrative, corner tags), then one section per shot: duration, caption table (one caption = one
   visual change), formulas, the source of every number; at the end dependencies, cut order and render-time details.
2. `shots/`: one script per shot (`sNN_*.py`) plus shared helpers (`e01_common.py`, `scene_e01.py` = the synthetic
   3-camera test scene with full ground truth; `e02_common.py`, `e02_numbers.py`). Captions sit in a list at the top of
   each script; `@shot.check` asserts every computed number in the exact string shown on screen.
3. `episode.json`: shot order for `tools/concat.py` and `tools/review_page.py`.
4. `model_runs/`: scripts that run the real models once and save their outputs; shots only read the saved files.
5. E01 only: `pre/` (a ≤ 5 min cut built from thin "cut shells" that load the full-episode shots and hide or swap
   parts, so those shots stay byte-identical), `pre/narration/` (the spoken script, per `references/NARRATION.md`),
   `pre/voiced/` (local TTS voice-over, per `references/TTS.md`), `i18n_tools/` (helpers for a translated render).

## Checking and rendering

```bash
python -I -B -X utf8 tools/render_shot.py episodes_example/E02_depth/shots/s03_scale_align.py --check-only
python -I -B -X utf8 tools/render_shot.py episodes_example/E01_vggt_mapanything/shots/s04_alt_attn.py --jobs 2
python -I -B -X utf8 tools/concat.py episodes_example/E02_depth          # after every shot is rendered
```

On a fresh clone, 29 of the 38 episode shots pass `--check-only` (E01 6 of 10, cut 7 of 9, voiced cut 7 of 9, E02
9 of 10), and so do all 7 shots in `examples/`. The other 9 draw real model output and stop with a
"... missing: run model_runs/..." message until you run the model scripts listed in each episode's README.

Install notes from the original model runs: install torch first with the right `--index-url` (otherwise pip may
install a CPU build), then the official code. Depth Anything 3's package pulls xformers / open3d / gradio and can
replace a CUDA build of torch: `pip install --no-deps -e` the repo and add only the inference dependencies. Check
output channel meanings on real numbers (means / stds) before trusting the docs. Your run should reproduce the
metrics JSON files closely, not bit-for-bit.
