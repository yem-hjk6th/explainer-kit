# E02: depth, Depth Pro × Depth Anything 3

What a depth map stores and how two recent models predict it. Depth vs inverse depth vs disparity; why one image
fixes depth only up to scale and offset, and affine alignment; Depth Pro (multi-scale tiles through a shared encoder,
canonical inverse depth, the field-of-view head, training losses); Depth Anything 3 (a depth map plus a ray map per
image, token layout and the camera token, training and teacher-student); a side-by-side comparison. It reuses E01's
synthetic 3-camera scene and white dot (`../E01_vggt_mapanything/shots/scene_e01.py`, imported read-only); every
number on screen is computed in `shots/e02_numbers.py` with asserts. 10 shots, 6:36.

Files: `storyboard.md`, `episode.json`, `shots/` (`sNN_*.py`, helpers `e02_common.py`, `e02_numbers.py`),
`model_runs/` (`check_env.py`, `download_weights.py`, `run_e02.py`, `analyze_e02.py`, measured `e02_metrics.json`).

## Render

```bash
python -I -B -X utf8 tools/render_shot.py episodes_example/E02_depth/shots/s03_scale_align.py --jobs 2
python -I -B -X utf8 tools/concat.py episodes_example/E02_depth
```

## Shots that need real model output

Only `s08_da3_tokens` (it draws Depth Anything 3's ray map and stops with "da3_B_nopose_raypose.npz missing: ..."
otherwise). The other 9 pass `--check-only` on a fresh clone; the measured numbers they show come from the shipped
`model_runs/e02_metrics.json`.

```bash
# needs the 3 test images E01's VGGT run writes to E01_vggt_mapanything/model_runs/vggt/inputs/*.png
python -X utf8 -B episodes_example/E02_depth/model_runs/check_env.py
python -X utf8 -B episodes_example/E02_depth/model_runs/download_weights.py     # Depth Pro, DA3-Large-1.1, DA3-Metric-Large
python -X utf8 -B episodes_example/E02_depth/model_runs/run_e02.py
python -X utf8 -B episodes_example/E02_depth/model_runs/analyze_e02.py          # rewrites e02_metrics.json
```

With 3+ images and no poses, Depth Anything 3 picks its own reference view; the run passes
`ref_view_strategy="first"` to keep camera 1 as the world frame.
