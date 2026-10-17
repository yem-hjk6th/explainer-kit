# PITFALLS (each one cost a re-render or a wrong number once)

## Running

- `python -I` ignores every `PYTHON*` variable (`PYTHONUTF8`, `PYTHONDONTWRITEBYTECODE`) and does not put the script's
  folder on `sys.path`: use `python -I -B -X utf8` and run shots through `tools/render_shot.py`.
- Without `-B`, loading another line's shot scripts writes `.pyc` into their `__pycache__/`.
- `render_shot.py` exits 0 on success, 2 when a check fails, 1 on an exception (e.g. missing model output). Wrappers
  can hide that: `cmd /c start /B /WAIT` returned 0 even for a child that exited 1 or 2, and background shell jobs'
  exit codes are unreliable too. Behind a wrapper, judge by `<stem>.mp4`, `check.json`, `meta.json` timestamps.
- Windows PowerShell 5.1 eats double quotes inside native-program arguments (`python -c "..."` → SyntaxError): put
  code in a `.py` file. In PowerShell, `R` is an alias (Invoke-History): do not name a function `R`.
- Git Bash heredocs collapse `\\` → `\`; write LaTeX / non-ASCII files with an editor tool.
- A library error like "object has no attribute ..." right after someone edited `animkit/`: the module was half
  written. Check modification times, wait, re-run; do not patch the library from a shot line.

## Numbers

- Never round a printed value again: 0.53849 printed as 0.5385 and re-rounded became 0.539 (true 0.538); a product of
  three displayed factors did not equal the displayed result. Format every on-screen number from full precision and
  assert the *displayed string* in `@shot.check`; assert that displayed factors multiply to the displayed result.
- Paper prose vs code: layer counts, token counts, which layers are global, concatenation order — read the official
  code; a structure "inferred from a ratio in the paper" was wrong.
- Model outputs: check channel meaning on real statistics before use. One model's ray map stored direction first and
  origin second, opposite to the paper's text; reading it in the paper's order gave a 141° rotation error. Another
  model chose its own reference view when given 3+ images without poses; pass the option that pins view 1.
- Model installs: pyproject dependencies (xformers, open3d, gradio ...) can replace a CUDA torch with another build.
  Install torch first with `--index-url`, then the repo with `--no-deps -e`, then only the inference dependencies.
- `torch.cdist` (matrix-multiply mode) has ~1e-4 error on unit-scale point clouds: use
  `compute_mode="donot_use_mm_for_euclid_dist"` when checking distances.

## Drawing

- Images go on the sharp layer (`cv.s`, `ImagePanel`, `cv.image` default); lines / points drawn after them sit on top.
  An image on the text layer covers everything drawn on `cv.s` / `cv.g`.
- `cv.s.line` etc. overwrite; to let near objects hide far ones, draw far to near and use `fill`. Point clouds have
  their own z-buffer.
- Camera axes: `axes3d` takes axes as **columns** of R; a world→camera R has the camera axes as its **rows** —
  use `camera_axes(cv, P, cam)`.
- A composited sub-canvas (`sub.composite()`) is a read-only array: `.copy()` before drawing on it with OpenCV.
- Pop-ups: box must stay above the caption band (y < 1250); `t_out ≤ DUR − pop` or `popup_check` fails.
- `pulse_dot` draws at u = 0 too (guard with `u > 0.02`); `arch.measure` with a negative offset puts the label on the
  line (swap endpoints, use a positive offset); `readout.matrix` with `\sum_k` entries makes very tall rows;
  `Canvas.rich()` has no line-spacing argument (split lines yourself); `ck.eq` on nested lists of different lengths
  raises; `Plot2D` with `{:.1f}` ticks shows 2.25 as 2.2 (give ticks explicitly).

## Formulas and glyphs

- Formula strings that change every frame start a typesetting subprocess per new string (1–2 s each): use `readout`
  (`number` / `counter` / `vector` / `matrix`) for changing numbers; build static formulas at module level.
- mathtext has no CJK (keep any CJK text of a translation outside `$...$`); `\left(` / `\right)` must be in the same segment (split terms →
  plain parentheses); `\top` is oversized, write `^{\mathsf{T}}`.
- Lone operators in split formulas: baseline handled by the library (self-test 3); if you change `formula.py`, re-run it.
- UI fonts lack subscript digits; Segoe UI lacks circled digits and some arrows → boxes.
- `--still` PNGs are written once; a later render does not refresh them, so an old still can show a fixed bug.
  render_shot prints a note listing stills older than the script.
- The caption word budget (12) is not the width limit: at the default size a line of ~8 long words is already
  wider than the frame; trust the "wider than the frame" warning (long lines without `\n` are split in two).
  Full-width parentheses in a Latin-font line also render as boxes.

## Review and translation

- `off_canvas = 0` does not mean no overlap: look at the densest frames.
- English widths: measure after a first render (`en_fit.py`); agents' guesses were optimistic (85 strings overflowed).
- A cut shell that loads a shot under another name must keep the original translation group.
- Cross-fade cut points that land on a closing pop-up leave a ghost: keep cuts ≥ 0.35 s after `t_out`.
- A status file that says "stage done" can be written before the renders finish: check the outputs.
