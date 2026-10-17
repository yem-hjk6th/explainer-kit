# SETUP

Only environment 1 is required. The others are for optional pipeline stages; keep them separate so a heavy install
(torch, a model repo's pinned dependencies) never upgrades the numpy / Pillow the renderer depends on.

| # | Env (suggested name) | Needed for | Install |
|---|---|---|---|
| 1 | `explainer` | rendering, formulas, review pages, concat | `pip install numpy opencv-python pillow matplotlib imageio-ffmpeg` |
| 2 | `explainer` (same) or `asr` | breaking down a reference video (`tools/analyze_explainer.py`), checking a voice-over by transcribing it back | `pip install faster-whisper rapidocr-onnxruntime` (both run on CPU) |
| 3 | `tts` | local voice-over (`tools/tts/`) | torch for your CUDA (`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`, pick the index that matches your driver), then `pip install qwen-tts soundfile librosa` |
| 4 | one env per model, e.g. `vggt`, `mapanything`, `depth-models` | episodes that show a real model's output | torch first (with `--index-url`, otherwise pip may install a CPU build), then the official repo; see `episodes_example/README.md` |
| 5 | `explainer` (same) | `tools/render_check.py` (MathJax / layout check of HTML pages) | `pip install playwright && playwright install chromium` |

Tested versions (any recent ones should work): Python 3.11–3.12, numpy 2.x, opencv-python 4.10+ / 5.0, Pillow 11–12,
matplotlib 3.8+, faster-whisper 1.2, rapidocr-onnxruntime 1.4, torch 2.11 + cu128, qwen-tts 0.1.1, transformers 4.57.

## Machine config: `config/animkit.json`

`python -X utf8 tools/setup_config.py` writes it. Keys:

- `fonts`: five roles. `text` (regular, the default for all labels and captions), `bold` (titles, headers), `light`
  (thin face for years, numbers, small labels), `math` / `math_up` (symbols drawn as plain text). Formulas themselves
  use matplotlib's Computer Modern, no font needed. For a translated render whose script these fonts lack, add
  `text_<code>` / `bold_<code>` entries (e.g. `"text_ja"`: a Noto Sans CJK file for Japanese).
- `math_python`: a python that can `import matplotlib`; `null` = the interpreter running the render. Set it only if you
  keep matplotlib in a separate env.
- `ffmpeg`: `null` = PATH, then the active conda env, then `imageio-ffmpeg`'s bundled binary.
- Env overrides: `ANIMKIT_CONFIG` (another config file), `ANIMKIT_CACHE` (formula cache dir, default `.cache/`).
- The file is per checkout and git-ignored: a new clone or `git worktree` needs `setup_config.py` again (or point
  `ANIMKIT_CONFIG` at an existing one).

Font gotchas: many UI fonts have no subscript digits (x₁ draws as a box): write symbols with subscripts as `$x_1$`
so they go through mathtext. Segoe UI has no circled digits ①②…: write `1.` instead.

## How to run things

```bash
python -I -B -X utf8 tools/render_shot.py <shot.py> [--check-only | --sheet-only | --still 3.5 8 | --jobs N] [--lang es]
python -I -B -X utf8 tools/concat.py <episode_dir> [--lang es]
python -I -B -X utf8 tools/review_page.py <episode_dir> --round N
python -I -B -X utf8 tools/i18n_table.py <episode_dir> --lang es [--jobs N]     # only for a translated version
```

- `-I` isolates from the user's site-packages and PYTHON* variables, so `PYTHONUTF8=1` is ignored: always add `-X utf8`.
- `-B` stops `.pyc` files from landing next to scripts that other sessions are reading.
- `render_shot.py` puts the repo root and the shot's folder on `sys.path`; run shots through it, not directly.
- `--still` PNGs are not refreshed by later renders; after editing a shot, re-run `--still` (each run prints a note
  listing stills older than the script).

## Sharing a machine (CPU / GPU etiquette)

If the machine also runs other work, cap the renderer explicitly; child processes inherit the cap.
- Windows: `cmd /c start "" /B /WAIT /BELOWNORMAL /AFFINITY <hex core mask> python -I -B -X utf8 ...`
  (mask `F` = cores 0–3, `300000` = cores 20–21). `start /WAIT` does not pass the child's exit code through
  (it returned 0 for children that exited 1 or 2): judge success by the outputs (`<shot>.mp4`, `check.json`).
- Linux: `taskset -c 0-3 nice -n 10 python ...`; macOS has no affinity: use `nice` and fewer `--jobs`.
- Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to the number of cores you were given.
- GPU jobs: check `nvidia-smi` first; cap memory with `torch.cuda.set_per_process_memory_fraction(f)`; release it when done.
- Typical cost: a 16 s shot (480 frames) renders in about 40 s on 2 cores; formulas are typeset once and cached.

## Windows notes

- Git Bash heredocs collapse `\\` to `\`: write files that contain backslashes (LaTeX!) or non-ASCII text with an
  editor or Claude's Write / Edit tools, not `cat <<EOF`.
- Windows PowerShell 5.1 strips double quotes from arguments of native programs (`python -c "..."` breaks) and reads /
  writes files in the ANSI code page by default: put test code in a `.py` file; do not edit non-ASCII files with PS 5.1.
- A conda env's ffmpeg lives in `<env>\Library\bin`; `setup_config.py` finds it.
- `review_page.py` and `concat.py` refuse stale shots (script changed after render): re-render, do not force.

## Reference videos

`tools/analyze_explainer.py <dir>` expects `<dir>/video.mp4` and writes `analysis/` (ASR with timestamps, on-screen text
changes, hard cuts, timestamped contact sheets). Use sources you have the right to watch and download, for example a
public link with `yt-dlp`. Study the style; do not put reference footage into your videos or redistribute it.
