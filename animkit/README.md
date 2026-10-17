# animkit: explainer-shot library

numpy + OpenCV + PIL + ffmpeg. One shot = one short script: it declares a duration, captions and self-checks, and defines one function that draws a frame. The library does the compositing, encoding, contact sheet and check report. Visual conventions are in `STYLE.md`.
Shots are written in English. `--lang CODE` renders a translation from a table (see "Languages").
Setup is in `SETUP.md` at the repo root; `config/animkit.json` is per checkout (see "Run").

## API quick reference

Call signatures are abbreviated; `...` marks further optional arguments (see the sections below). All positions and sizes are screen pixels, colors are RGB tuples, `alpha` is 0-1.

| Module | Main names | Purpose | Most used call |
|---|---|---|---|
| `ease` | `ss`, `fade`, `lin`, `bump`, `lerp` | time envelopes | `ss(t, a, b)`; `fade(t, a, b, fin=0.4, fout=0.4)` |
| `timeline` | `Shot`, `Checker` | duration, captions, titles, tags, checks, render | `Shot(name, dur, W, H, fps, glow, seed)`; `shot.caption(t0, t1, text)` |
| `draw` | `Canvas` (`cv.s`, `cv.g`, `cv.text`, `cv.rich`, `cv.math`, `cv.text_rot`, `cv.badge`, `cv.image`) | per-frame canvas: sharp layer, glow layer, text | `cv.s.line(a, b, color, alpha, th)`; `cv.text(xy, s, size=30, kind="text", alpha=1.0, fill=TEXT, anchor="la", shadow=True, align="left")` |
| `draw` (text) | `cv.rich`, `cv.math`, `cv.text_rot`, `cv.badge` | text with `$math$`, formula string, rotated text, pill tag | `cv.rich(s, xy, size=30, kind="text", alpha=1.0, fill=TEXT, anchor="la", align="l", math_color=None)`; `cv.math(tex, xy, size=40, color=MATH, alpha=1.0, anchor="mm", glow=0.0)`; `cv.text_rot(xy, s, size=26, kind="text", alpha=1.0, fill=TEXT_DIM, angle=90)`; `cv.badge(xy, s, size=26, alpha=1.0, anchor="ra", fill=TEXT_DIM, bg=(40, 44, 52), border=None)` |
| `richtext` | `Rich`, `wrap`, `rich`, `Digits` | rich text with inline `$...$`, line wrapping | `wrap(text, width, size=30, kind="text")` |
| `formula` | `Formula` | multi-part formula, per-part state | `Formula(segments, size=72, gap=0.22, align="m").draw(cv, xy, color, alpha, anchor, state=..., notes=...)` |
| `readout` | `number`, `counter`, `vector`, `matrix` | changing numbers without per-frame math rendering | `number(cv, xy, value, fmt="{:.3f}", size=40, color=MATH, alpha=1.0, anchor="mm", unit=None)` |
| `cam3d` | `PinholeCam`, `ViewCam`, `OrbitCam`, `draw_camera`, `point_cloud`, `ray_fan`, `marker3d`, `axes3d`, `camera_axes`, `floor_grid`, `fundamental`, `relative_pose`, `skew` | 3D scenes | `PinholeCam.from_KRt(K, R, t, img_w, img_h, plane_d=0.3)`; `OrbitCam(target, r, az=0, el=30, world="yup")`; `point_cloud(cv, P, X, colors, rad=2.0, ...)` |
| `tensor` | `TokenStrip`, `reshape_rows`, `reshape_ids`, `move_tokens`, `fly_token`, `tensor_block`, `tensor_morph`, `log_px`, `heatmap`, `attention`, `block_diag_mask`, `softmax_rows`, `fold` | token strips, reshape, tensor blocks, attention math | `TokenStrip(n, show, rows, center, size, pitch, row_pitch, append).draw(cv, kinds, alpha, ...)` |
| `arch` | `module_box`, `path`, `elbow`, `branches`, `arrow_path`, `arrow`, `pulse_dot`, `measure` | architecture diagrams | `module_box(cv, box, title, sub, color, alpha, state, glow)` |
| `panel` | `ImagePanel`, `colorize`, `colorbar`, `tile_starts` | images, patch grids, depth maps, tiles | `ImagePanel(img, xy, w=..., nearest=False).draw(cv, alpha, border, title)` |
| `chart` | `attn_blocks`, `bar_chart`, `row_softmax` | attention block grid, bar chart | `bar_chart(cv, box, values, labels, vmax, ...)` |
| `plot2d` | `Plot2D`, `nice_ticks` | 2D axes, curves, scatter, lines | `Plot2D(box, xlim, ylim, ...)`; `p.curve(cv, x, y, color, progress, ...)`; `p.line(cv, s, t, ...)`; `p.scatter(cv, x, y, ...)` |
| `layout` | `grid_pyramid`, `grid_square`, `paper_card`, `table`, `input_slot`, `link`, `flicker`, `split` | pyramids, cards, tables, input slots, split line | `table(cv, xy, col_w, header, rows, row_h, size, ...)`; `split(cv, "lr" or "tb", pos, margin)` |
| `popup` | `Popup`, `popup_check` | corner pop-up window for a side concept | `Popup(box, t_in, t_out, title, corner="tr", ...).draw(cv, t, body)` |
| `i18n` | `T`, `tr`, `SOURCE`, `LANG` | translation hooks | `T("... {x} ...", x=value)` |
| `style` | colors, `TOKEN`, `CAPTION_MAX_WORDS`, ... | palette and constants | `style.GOLD`, `style.CYAN`, ... |
| `config` | paths | machine paths from `config/animkit.json` | (read by the library) |

Color parameters: `cv.text` and `cv.rich` take the text color as **`fill`** (not `color`); `cv.rich` takes `math_color` for the `$...$` parts (default: same as `fill`). `cv.math`, `Formula.draw`, `readout.*` and the line/dot methods take `color`. Anchor strings have two letters: horizontal l/m/r, vertical t(a)/m/b/s (s = baseline).

```
<repo>/
  animkit/               library (timeline, draw, text, richtext, formula, readout, cam3d, tensor, arch, panel, chart, plot2d, layout, popup, i18n, style, ease, config)
  tools/render_shot.py   render one shot
  tools/review_page.py   build the review page of an episode
  tools/concat.py        join an episode from episode.json
  tools/i18n_table.py    collect the on-screen source strings of an episode into captions.json
  tools/selftest.py      library self-test (z-buffer, "cv" world, operator baseline, language hooks, OrbitCam convention); run after changing the library
  tools/setup_config.py  write config/animkit.json for this machine
  config/animkit.json    per-machine paths (python for the formula subprocess, ffmpeg, fonts); git-ignored
  examples/              epipolar.py (regression), tensor_demo.py, p0_demo.py / p1_demo.py / plot2d_demo.py / tiles_demo.py (element self-tests), i18n_demo/ (translation example), episode.json + storyboard.md (format sample)
  .cache/formula/        formula cache (generated; safe to delete)
```

## Run

Rendering and formulas can use the same environment (numpy, opencv-python, pillow, matplotlib). Formulas are typeset by a subprocess that uses matplotlib mathtext and caches the result (`math_python` in `config/animkit.json`; empty = the current interpreter). ffmpeg is looked up in the config, then on PATH, then in the current conda env.

**`config/animkit.json` is per checkout** (it is git-ignored). A new clone or a new git worktree has no config: run `python tools/setup_config.py` there, or set `ANIMKIT_CONFIG` to an existing config file. `ANIMKIT_CACHE` moves the formula cache.

```bash
python -I -B -X utf8 tools/render_shot.py episodes/<episode>/shots/s03_alt_attn.py --jobs 4
```
To cap the cores on a shared machine (child processes inherit the cap): Windows `cmd /c start "" /B /WAIT /BELOWNORMAL /AFFINITY <hex mask> python ...`; Linux `taskset -c <cores> nice -n 10 python ...`; also set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to the core count.

| Command | Effect |
|---|---|
| `render_shot.py <shot.py>` | runs the self-checks first (stops on failure, exit code 2), then renders. Output goes to `<shot dir>/render/<script stem>/`: `<stem>.mp4`, `sheet.jpg`, `check.json`, `meta.json` |
| `... --check-only` | self-checks only |
| `... --still 3.5 8` | PNGs of these times only (the fastest loop while adjusting a picture). See "Stills" below |
| `... --sheet-only` | render only the 12 frames of the contact sheet |
| `... --jobs N` | split the frames into N ranges, N parallel subprocesses, lossless join at the end; children inherit the core cap |
| `... --lang CODE` | translated render; output files get `_<code>` (see "Languages") |
| `... --collect-text` | list the source strings the shot draws -> `text_src.json` (sampling rate `--fps`, default 10) |
| `review_page.py <episode_dir> --round N` | writes `<episode_dir>/review/roundN/review.html` and a root `review.html` (jumps to the latest round; `review.html?list` lists the rounds). See "Review rounds" |
| `concat.py <episode_dir>` | writes `<episode_dir>/render/<episode id>.mp4`; refuses if a shot is not rendered, a self-check failed, or the script changed after the render |
| `concat.py <episode_dir> --lang CODE` | the translated episode, `<episode id>_<code>.mp4` |
| `i18n_table.py <episode_dir> --lang CODE` | collect the on-screen source strings into `captions.json` |

**Stills.** `--still T ...` writes `still_<T>.png` once. Later renders do not refresh existing stills. Each run prints a note that lists the stills older than the shot script, so you know which ones to delete or redo.

Measured on a desktop machine limited to 8 logical cores: the epipolar-geometry shot (16 s, 480 frames) takes 31 s with one process; `tensor_demo` (24 s, 720 frames) takes 30 s with `--jobs 2`; `p0_demo` (20 s, 600 frames, with a 60,000-point cloud and 3 perspective textures) takes 30 s with `--jobs 2`. Formulas used for the first time wait for the math env to typeset them (a batch takes about 5-10 s); later runs read the cache.

## Review rounds (review_page.py)

```
<episode dir>/review.html              redirect only: opens the latest round; ?list lists the rounds
<episode dir>/review/round1/{review.html, feedback.md}
<episode dir>/review/round2/{review.html, changes.md, feedback.md}   ...
```
- **Round 1**: per shot, the video, the storyboard section, numbered points and one comment box per point.
- **Round 2 onward**: one overall comment box at the top; per shot only the video, "what changed this round" (read from `roundN/changes.md`) and the folded, read-only "previous feedback" (read from `round{N-1}/feedback.md`). A row of shot numbers sits at the top; shots that changed are highlighted.
- "Copy all comments" exports text whose first line is "<episode id> round N review comments". Drafts are stored per round in the browser. Videos do not autoplay and have no sound (also over a remote desktop).
- **Per round**: (1) write the changes of the round to `review/roundN/changes.md`, one section per shot (`## E01-06`) with a few bullet lines (text before the first shot heading goes to the top of the page; a shot without a section shows "no change this round"); (2) re-render, then `concat.py`, then `review_page.py <episode dir> --round N`, then `render_check.py review/roundN/review.html`; (3) save the feedback that comes back as `review/roundN/feedback.md`, one line per comment starting with the shot id (`E01-06: ...`, `E01-02-12: ...`); such lines are attached to that shot, the rest counts as overall feedback.
- Without `--round` the tool stops and prints the number of the next round. An old root `review.html` (from before rounds existed) is moved to `review/round1/review.html` on the first run (a `<base href>` is added so the videos still play); a page not written by this script is never overwritten but moved to `<episode dir>/_moved/`.
- Note: a shot's mp4 is overwritten by each render, so the page of an old round plays the **current** video.

## Languages: captions and on-screen text

**Shots are written in English.** That English is the source text. `--lang CODE` (for example `es`) renders a translation.
- **Text entry points** that go through the translation hook: captions, titles, tags, `cv.text` / `text_rot` / `rich` / `badge`, `Rich`, `wrap`, formula `notes`, pop-up titles, tables, bar-chart labels, `Plot2D` labels, paper cards, `T(...)`. In the source language the hooks are the identity (selftest 4: 8 frames of p1_demo are pixel-identical with the hooks and with the hooks replaced by an identity function).
- **Translation table** `<EP>/captions.json` (also allowed next to the shot; `--captions FILE` names one explicitly): `{"<code>": {"<shot stem>": {"<English source string>": "<translation>"}}}`. The key is the exact English string drawn on screen (including `\n` and `$formula$`). Lookup tries the group of this shot first, then the groups of the other shots of the episode. If the English source changes, the key no longer matches and the string is reported as untranslated; an old translation is never reused silently. Why not `T("id")`: keys are the source text, so shot code needs no change.
- **Only strings with a word of at least 2 letters outside `$...$` are looked up.** Numbers, symbols and pure formulas are never looked up.
- **Fonts for a translated render.** Font kinds are `"text"` (regular, default), `"bold"`, `"light"` (thin face: years, numbers, small labels), `"math"`, `"math_up"`. In `config/animkit.json`, optional entries `"text_<code>"` / `"bold_<code>"` are used for a render in that language (for example a CJK font for a Chinese or Japanese translation). Without them the regular and bold faces are used.
- **Captions.** `style.CAPTION_MAX_WORDS = 12` is a reading-load budget: `text.caption_len` counts words (characters for CJK text) and the render warns above 12. The real limit is pixel width: the timeline warns "a line is wider than the frame" (often at about 8 long words). A line without an explicit `\n` that is too wide is broken automatically into two lines (the break that makes the longer line shortest); you can also write `\n` in the text or in the translation. At most 2 lines (`style.CAPTION_MAX_LINES`).
- **`T("...{x}...", x=value)`** (`from animkit import T`): only for a string that embeds a runtime number taking many values. The key in the table is the template with `{x}`; `{x}` is a literal replacement, LaTeX `{}` is not affected. Strings with only a few values (for example `f"frame {i + 1}"`) need no change; the collector records each value.
- **Collect.** `render_shot.py s.py --collect-text` writes `text_src.json` for one shot (sampling at `--fps`, no video). `i18n_table.py <EP> --lang CODE [--shots E01-04 ...] [--jobs 4] [--table FILE] [--work DIR] [--fps 10]` does this for all shots of an episode and merges the strings into `captions.json` under `<code>`. Existing translations are kept, a string translated in another shot is carried over, and entries that a shot no longer draws are moved to `_unused` (not lost). At the end it prints how many strings each shot still lacks, and strings that "differ only in numbers" (more than 3 variants: turn them into a `T` template).
- **Output.** `render_shot.py s.py --lang CODE` writes `render/<stem>/<stem>_<code>.mp4`, `sheet_<code>.jpg`, `check_<code>.json` (warnings: caption over 12 words / too wide / over 2 lines; untranslated title, tag or caption; title too wide) and `meta_<code>.json` (`i18n.used` = the translations used, `untranslated` = strings on screen without translation, `off_canvas` = translated strings that fall outside the canvas or a pop-up). `concat.py <EP> --lang CODE` writes `render/<episode id>_<code>.mp4`; it refuses if strings are untranslated or a used translation changed after the render (`translations changed since render`).

**Workflow for a new language**
1. Shots need no change. Only one case needs an edit: an f-string that embeds a number which changes every frame (step 2 lists it as "differ only in numbers"); turn it into `T("...{x}...", x=...)`.
2. `i18n_table.py <EP> --lang CODE --jobs 4` -> `<EP>/captions.json` (frames are sampled, nothing is encoded).
3. Fill in the translations. Follow the writing rules of STYLE section A (plain wording, symbols come with a definition, numbers say which metric); captions stay within 12 words.
4. `render_shot.py shots/sXX.py --lang CODE --jobs 4`; read `sheet_<code>.jpg`, the warnings of `check_<code>.json`, and `untranslated` / `off_canvas` of `meta_<code>.json`. If a string does not fit, shorten the translation or add `\n` in it; do not edit the shot.
5. When every shot passes: `concat.py <EP> --lang CODE`. After the English source changes, repeat step 2 (it only adds empty entries), fill them in, and re-render the affected shots.

**Example** `examples/i18n_demo/`: `s04_alt_attn.py` is a shell of about ten lines that loads an existing shot unchanged; `captions.json` holds the collected strings with their translations. Command: `render_shot.py examples/i18n_demo/s04_alt_attn.py --lang CODE --jobs 4`.

## Voice-over timeline (design draft, not implemented)
Goal: with a voice-over, the on-screen time of each caption follows the narration of that sentence; the animation is retimed and the shot script does not change.
- **External timetable** (produced by the voice tool, one per shot) `<episode dir>/voice/<lang>/<stem>.json`: `[{"n": 3, "audio": "s04_03.wav", "dur": 2.84}, ...]`. `n` = the number of the caption in `meta_<lang>.json`; `dur` = seconds measured with ffprobe.
- **Time remapping**: caption k occupies script time $[a_k, b_k]$. The new length is $d_k' = \max(b_k - a_k,\ dur_k + 0.3)$; the gaps between captions stay. This gives a monotone piecewise-linear map from output time $t'$ to script time $t = g(t')$. Frame $t'$ is rendered by calling the original `frame(cv, g(t'))`; the t_in / t_out of captions, titles, tags and pop-ups go through the same map; the shot length is the mapped end point.
- **Two modes per segment** (settable per segment): `hold` (default) = the animation plays at the original speed and, if the narration is still running, stays on the last frame of the segment; `stretch` = the animation of the segment is slowed down proportionally. `hold` is the default because slowing changes the approved pacing.
- **Interface**: `render_shot.py s.py --lang CODE --timing voice/<code>/<stem>.json` -> `Shot.retime(map)`; the shot sees script time only, so `ss` / `fade` / checks stay unchanged. The audio clips are joined with ffmpeg at the new start times $a_k'$ into one track, attached to the existing `shot.audio` and mixed into the mp4; the map is written to `meta_<lang>.json`; concat is unchanged.
- **To be decided**: which TTS service and voice; whether the narration reads the captions word for word or follows a separate, longer script (then each timetable entry must carry the narration text, and captions and narration are no longer one to one).

## Writing a shot

```python
import numpy as np
from animkit import Shot, Formula, ss, fade, bump, style, CYAN, GOLD

shot = Shot("alt_attn", 12.0)                 # name, duration in s; default 1080x1440, 30 fps
shot.title("Alternating Attention", "VGGT")   # large title at the top left, fades in/out within 0.2-3.8 s
shot.caption(0.5, 3.0, "Each image first attends within itself")   # bottom caption: at most 2 lines, 12 words (see Languages)
shot.tag("schematic")                         # top-right tag; several tags stack downward by time overlap

EQ = Formula([("A", "A"), "=", (r"\mathrm{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d}}\right)", "sm")], size=60)

@shot.check
def check(ck):                                # runs before rendering; one failed item stops the render
    ck.close("rows sum to 1", A.sum(1), np.ones(len(A)), 1e-12)
    ck.record(shape=[3, 6, 8])                # record only, no pass/fail

@shot.frame
def frame(cv, t):                             # cv = canvas of this frame, t = seconds
    a = ss(t, 0.5, 1.0)
    cv.s.line((100, 700), (980, 700), CYAN, a, 2)        # sharp layer
    cv.g.line((100, 700), (980, 700), CYAN, 0.8 * a, 6)  # glow layer (two Gaussian blurs, added on top)
    EQ.draw(cv, (540, 200), alpha=a, state={"A": "lit", "sm": "hl" if t > 2 else "dim"},
            notes={"sm": "softmax per row; each row sums to 1"})
```

The script is loaded by `render_shot.py`, which puts the repo root and the script directory on `sys.path`. So the script can `from animkit import ...` directly, and helper modules in the same directory can be imported. A script may contain only one `Shot` object.

## API

Parameters common to all elements: positions and sizes are screen pixels (fractions allowed); colors are RGB tuples; `alpha` 0-1 is the appear/disappear progress (use with `ss` / `fade`); `glow` is the glow level (0 = no glow). Every text parameter marked "rich" accepts inline `$math$`, for example `"$R_i$: unit ray map, $3\times H\times W$"`.

**Time (`ease`)**: `ss(t, a, b)` smoothstep; `fade(t, a, b, fin, fout)` fade in, hold, fade out; `lin` linear; `bump(t, t0, dur)` 0 -> 1 -> 0 pulse envelope; `lerp(a, b, u)`.

**Shot (`timeline`)**: `Shot(name, dur, W, H, fps, glow, seed)`; `.caption(t0, t1, text)`, `.title(text, sub, t0, t1)`, `.tag(text, t0, t1)` (corner tag, rich); decorators `@shot.check` (argument `ck: Checker`) and `@shot.frame` (arguments `cv, t`); `.sheet_times`; `.audio` (voice/music hook: path of an audio file, mixed into the mp4 after rendering); `.render_frame(t)`.
`Checker`: `le / ge (name, value, limit)`, `close(name, value, ref, tol)`, `eq(name, value, ref)`, `true(name, cond, detail)`, `record(**data)`.

**Canvas (`draw.Canvas`)**: on `cv.s` (sharp layer) and `cv.g` (glow layer): `line`, `polyline(pts, ..., closed)`, `dashed(a, b, ..., n)`, `dot`, `dots`, `add_poly` (translucent additive fill), `fill` (occluding fill), `rect(..., r)`, `blend(mask, x, y, color)`, `blit_rgb(img, x, y, alpha, mask)`, `warp_quad(img, quad)` (paste an image onto a quadrilateral in perspective). Text layer:
- `cv.text(xy, s, size=30, kind="text", alpha=1.0, fill=TEXT, anchor="la", shadow=True, align="left", spacing=None)`; `kind` is one of `"text"`, `"bold"`, `"light"`, `"math"`, `"math_up"`. The text color parameter is `fill`.
- `cv.rich(s, xy, size=30, kind="text", alpha=1.0, fill=TEXT, anchor="la", align="l", math_color=None, shadow=True)`: text with inline `$math$`, multi-line with `\n`. `fill` colors the text, `math_color` colors the `$...$` parts.
- `cv.math(tex, xy, size=40, color=MATH, alpha=1.0, anchor="mm", glow=0.0, shadow=True)`: a formula string.
- `cv.text_rot(xy, s, size=26, kind="text", alpha=1.0, fill=TEXT_DIM, angle=90)`: plain text rotated counter-clockwise, centered at `xy`.
- `cv.badge(xy, s, size=26, alpha=1.0, anchor="ra", fill=TEXT_DIM, bg=(40, 44, 52), border=None)`: pill-shaped tag at any position (it can follow an element); returns its box.
- `cv.image(im, xy, w=None, alpha=1.0, anchor="la", layer="s", nearest=False)`: an existing picture (path, PIL image or array).
- Anchor: two letters, horizontal l/m/r, vertical t(a)/m/b/s (s = baseline).

**1. Formula (`formula.Formula`)**: `Formula(segments, size=72, gap=0.22, fontset="cm", align="m", line_gap=0.35)`; `segments` are mathtext strings or `(tex, id)`; `"\n"` starts a new line (multi-line formula). `.draw(cv, xy, color, alpha, anchor, state={id: "hidden"|"dim"|"lit"|"hl"|float brightness}, part_alpha, part_color, part_glow, box={id: amount or color}, underline={id: ...}, notes={id: rich small text})`: `hl` = bright + colored box + glow; `notes` hang below the part, move down one line when they overlap the previous note, and get a leader line. `.part_box(id, xy, anchor)` returns the screen box of a part (to draw a connecting line back to the picture). A `\left( ... \right)` pair must be in the same segment; across segments use plain parentheses.

**2. Numbers (`readout`)**: `number(cv, xy, value, fmt, size, color, alpha, anchor, unit)`; `counter(cv, xy, u, a, b, fmt="{:d}")` (u 0 -> 1 runs from a to b; `{:d}` jumps by integers); `vector(cv, xy, values, fmt, size, part_alpha, part_color, slot)` (each component has a fixed slot width and is right-aligned, so a morphing value does not jitter); `matrix(cv, xy, M, fmt, size, part_alpha={(i,j)}, part_color, box={(i,j): amount})` (M holds numbers or mathtext strings such as `"f_x"`; returns `center(i, j)` and the bbox). Changing numbers are assembled glyph by glyph from CM digits, so no math env starts per frame.

**3. Token strip (`tensor.TokenStrip`)**: `TokenStrip(n, show=fold(n, head, tail, force=[799]), rows=N, center, size, pitch, row_pitch, append=1)`: the real count is n, only the entries in `show` are drawn, and a gap in the middle is drawn as "...". `append` adds tokens at the end of each row (indices n, n+1, ..., for example a scale token). `.draw(cv, kinds, alpha, highlight={idx or (row, idx): color}, labels={...: "799"}, band={row: color}, row_alpha)`; `kinds` is `fn(row, idx) -> category name` or an array. Default palette `style.TOKEN`: camera gold, register violet, patch cyan, scale green, camera1 / register1 (frame 1 only) orange / pink. `.xy(idx, row)`, `.all_xy(row)`. Single tiles: `token`, `tokens`, `grid_xy`, `kind_colors`.

**4. Reshape**: `pos, u_k, ids = reshape_rows(N, K, u, src_center, dst_center, pitch, row_pitch, stagger, arc, group_gap)`: $(N,K)\to(1,NK)$; the landing place of token $(i,k)$ comes from the numpy reshape id grid (= $iK+k$); to run it backward pass `1 - v`. General version: `ids = reshape_ids(src_shape, op)` + `move_tokens(src_xy, dst_xy, ids, u, stagger, arc)`; `reshape_check` is for checks. One flying token: `fly_token(cv, a, b, u, color, size, bend, label="799 in frame -> global $1\times1374+799=2173$")` (the numbers in the label are computed by the shot and passed in).

**5. Tensor block**: `tensor_block(cv, xy, w, h, d, color, alpha, grid=(nh, nw, nd), gap=(gh, gw, gd), colors, labels={"h", "w", "d", "top": rich}, label_size)`; use `log_px(n, a, b)` for display sizes (not to true scale); shape morph: `tensor_morph(cv, xy, px_a, px_b, u, labels_a, labels_b)` (sizes interpolate, labels cross-fade). `box3` is a single cube.

**6. Architecture diagram (`arch`)**: `module_box(cv, box, title, sub, color, alpha, state="lit"|"dim"|float, glow)` (title and sub are rich; sub often holds a shape such as `"$(N, 1374, 1024)$"`; returns the edge midpoints l/r/t/b/c). Paths: `path(a, b, bend)`, `elbow(a, b, first="v"|"h", mid)` (polyline), `branches(a, targets, split)` (one box to several boxes); `arrow_path(cv, pts, color, alpha, th, head, progress, dashed, glow)`, `arrow(cv, a, b, ...)`; `pulse_dot(cv, path, u)` a light dot moving along a path; `measure(cv, a, b, label, offset, dashed)` a dimension line (short ticks at both ends, a label in the middle; in 3D pass points projected with `P(X)`).

**7. 3D**: see the section "3D scenes" below.

**8. Image panel (`panel.ImagePanel`)**: `ImagePanel(img, xy, w=..., nearest=False)`: `.draw(cv, alpha, border, title)`; `.grid(cv, cell=14, progress)` (patch grid, grows from the top left with progress); `.cell(cv, row, col, cell)`, `.scan(cv, u, cell)` (row-by-row scan highlight); `.ruler(cv, side, label="518 px")` (text on the left and right sides is set vertically); `.mark(cv, u, v, label)`; `.to_screen(u, v)` (image coordinates -> screen; u, v are continuous pixel coordinates).

**9. Heat map / depth map**: `colorize(D, vmin, vmax, stops=style.CMAP_DEPTH, bad=background color)` (inf / nan = background) -> give the result to `ImagePanel`; `colorbar(cv, box, vmin, vmax, unit="m", fmt, vertical)`; value labels at a pixel: `panel.mark(cv, u, v, label="$1.975$ m")`. For an attention matrix use `heatmap(cv, M, xy, cell, vmin, vmax, rows, values, highlight)`; compute weights for real with `attention(q, k, v, mask)` / `block_diag_mask(S, N)` / `softmax_rows`.

**10. Corner tag**: `shot.tag(text, t0, t1)` (top right, several stack automatically); a tag that follows an element: `cv.badge(xy, text, anchor)`.

**Charts and layout (`chart` / `layout`, example `examples/p1_demo.py`)**
- 11. `attn_blocks(cv, xy, N, K, cell, mode="frame"|"global", alpha, A=weights, vmax)`: the $NK\times NK$ grid drawn as $N\times N$ blocks; in `frame` mode the off-diagonal blocks are greyed and labeled "not computed", in `global` mode they are labeled "cross-frame"; with `A` given, a heat map is drawn in the computed blocks. One softmax row: `w = row_softmax(q, keys, mask, scale)` (assert the result of the same function in the check), then draw it with `bar_chart`.
- 12. `bar_chart(cv, box, values, labels, vmax, horizontal, highlight, progress, fmt, title, better="lower"|"higher", note="source ...", unit)`: bars grow one after another as progress goes 0 -> 1 and the values count up; the title gets "lower is better" / "higher is better" appended automatically.
- 13. `grid_pyramid(cv, y, sides_px, labels, x0, gap, progress, n_lines)` and a single `grid_square(cv, c, side, n_lines, label)`; levels appear one by one with merge arrows.
- 14. `paper_card(cv, box, year, venue, title, authors, contribution, image=None)` (text version, optional image slot; the title wraps automatically).
- 15. `table(cv, xy, col_w, header, rows, row_h, size, reveal=number of visible rows, highlight=row)`; cells are rich text (`\n` allowed).
- 16. `input_slot(cv, box, label, optional=True (dashed)|False (solid), on=0..1, prob="p = 0.5")`, `link(cv, a, b, on)` (when off it is broken and grey), `flicker(t, n, p, period, seed)` (a random on/off sequence with a fixed seed).
- 17. `split(cv, "lr"|"tb", pos, margin)` draws a divider and returns the boxes of the two halves.
- Line-wrapping helper: `wrap(text, width, size, kind)` (`$formula$` and Latin words are not broken; CJK text can break between characters).

**18. Pop-up (`popup.Popup`)**: a small window pops up in a corner, explains a side concept (DPT, quaternion, ...) and retracts, leaving the main line running.
- `Popup(box=(x0, y0, w, h), t_in, t_out, title, corner="tr"|"tl"|"br"|"bl", pop=0.35, accent=ORANGE, point_to=(x, y), tag=None, title_size=28, title_dy=0)`: `box` is the fully open position (it must not enter the caption area y >= 1250; asserted at construction). The window grows from the given `corner` and retracts, each in `pop` seconds. If `point_to` is given, a dashed leader line is drawn from the nearest corner to that point of the main picture. `tag` = small tag at the right of the title bar. If the title contains a tilde or another tall symbol that touches the top border, use `title_size=26, title_dy=5`.
- Call `pop.draw(cv, t, body)` **last** in the frame function. `body(sub, lt, box)` draws on a separate small canvas: `sub` is a w x h Canvas (origin = top-left of the window; `sub.s` / `sub.g` / `text` / `rich` / `Formula` / `readout` work as usual), `lt` = window-local time (from t_in), `box` = the content area without title bar and padding. It returns the current screen box of the window, or None while it is not open.
- `pop.on(t)` is the openness 0-1 (the main picture can dim or make room accordingly); `pop.local(t)` = t - t_in; `pop.content_box()`. In the check use `popup_check(ck, [pop, ...], DUR)`: every pop-up lies within the shot length and pop-ups that are on at the same time do not overlap.
- Constants `POPUP_BG`, `POPUP_TITLE_H` (50), `POPUP_PAD` (16), `CAPTION_TOP` (1250) are in `animkit.popup`.

**19. 2D plot (`plot2d.Plot2D`, example `examples/plot2d_demo.py`)**: curves, scatter, stems, least-squares lines; all values come from the caller, the library invents none.
- `Plot2D(box, xlim, ylim, xlabel=None, ylabel=None, xticks=None, yticks=None, fmt_x="{:g}", fmt_y="{:g}", grid=True, zero_line=True, tick_size=22, label_size=26)`: `box = (x0, y0, x1, y1)` is the plot area in screen pixels. Ticks: `None` = 4-6 tidy values (`nice_ticks`), or a list of values, or `[(value, "rich label")]`. Axis labels are rich (`$D$ (m)`); the y label sits above the y axis. The minus sign is drawn as a proper minus.
- `.axes(cv, alpha)` draws axes, ticks, a faint grid and the dashed line through 0; `.to_screen(x, y)` (scalar or array -> screen points; use it to connect other elements); `.inside(p)`.
- `.curve(cv, x, y, color, progress, alpha, th, glow, dashed)`: a polyline; `progress` 0 -> 1 draws it from left to right by arc length; the part outside the plot area is cut at the border; returns the screen position of the pen tip.
- `.scatter(cv, x, y, colors, progress, alpha, rad, labels, label_size, label_offset, label_color)`: `progress` 0 -> 1 shows the points one by one; `colors` may be one color or one per point (mark outliers with `RED`); `labels` are per-point rich labels (fade in with the point); returns the screen points. `.stem(cv, x, y, color, progress, base=0)`: vertical lines growing from `base` with end points.
- Fitted line $y = s\,x + t$: `Plot2D.fit_line(x, y, w=None)` -> `(s, t)` (least squares; compare with `np.polyfit` in the check); `Plot2D.morph((s0, t0), (s1, t1), u)` interpolates the parameters (assumed line -> refitted line); `.line(cv, s, t, color, progress, ..., x_range)` draws the line and returns (left end point, pen tip); `.residuals(cv, x, y, s, t, color, progress)` draws the vertical lines from points to the line (one by one) and returns the residuals $y-(sx+t)$.
- Not supported: log axes, legend boxes (put legends next to the plot as rich text). For values that change (s and t during a morph) use `readout.number`, not `cv.math(f"...")`.
- For a diagram instead of a data plot, see "2D side view" below.

**20. Image tiles (`ImagePanel.tiles`, example `examples/tiles_demo.py`)**: overlapping sliding windows on an image, the overlap regions, and ownership by nearest center (Voronoi).
- `tile_starts(length, size, stride)`: the tile start positions per direction; the last tile is flush with the edge (`tile_starts(1536, 384, 288) = [0, 288, 576, 864, 1152]`).
- `panel.tiles(cv, xs, ys, size, progress, alpha, color, color2, highlight={(row, col): color}, overlap_color, overlap_alpha, voronoi, voronoi_fill, voronoi_fill_alpha, keep=(row, col), keep_alpha, keep_color, stagger=4, drop=18)`: `xs` / `ys` are tile starts (image pixels), `size` is the side length (or `(w, h)`; parts outside the image are cut).
  - `progress` 0 -> 1: tiles drop in one by one in row-major order; neighboring tiles are offset up/down alternately by `stagger` screen pixels (`color` / `color2` alternate) so their borders do not overlap; a tile in `highlight` is thicker, filled and glowing.
  - `overlap_color`: regions covered by 2 or more tiles are filled (darker where more tiles cover); strength `overlap_alpha`, drive it with `ss` in the shot and let it appear after the tiles have landed.
  - `voronoi` (0-1): perpendicular bisectors of neighboring tile centers (dashed); `voronoi_fill=True` (or a list of RGB colors) + `voronoi_fill_alpha`: each pixel is colored by its nearest tile center (neighbors differ); `keep=(row, col)` + `keep_alpha`: only the ownership region of that tile stays bright, the rest is dimmed, and the region is outlined.
  - Returns `{"boxes": {(r, c): screen box}, "centers": {(r, c): screen point}, "vor_x", "vor_y": ownership boundaries (image pixels), "owner": fn(u, v) -> (row, col)}`. With `alpha=0` it only returns and draws nothing (`cv` may be `None`); use this in a check to assert "each tile center belongs to its own tile", "ownership areas sum to the image area", "owner agrees with a brute-force nearest center".
  - Image scales side by side (1536 / 768 / 384): create several `ImagePanel`s and call `tiles` once on each. The tiles form an equally spaced grid, so the Voronoi boundaries are always horizontal or vertical lines.

**Layer convention**: pictures always go to the sharp layer `cv.s` (`ImagePanel`, `cv.image` by default, frustum textures, color bars), so lines and points drawn afterwards lie on top; the text layer holds only text, formulas and badges (`cv.image(..., layer="text")` puts a picture on top of everything).
**Camera axes**: `axes3d` takes the **columns** of R as the axis directions; the own axes of a scene camera (R of world -> camera) are the **rows** of R: use `camera_axes(cv, P, cam)` (it passes `R.T` internally).
**World "cv"**: `OrbitCam(target, r, az, el, world="cv")` takes OpenCV world coordinates directly (y down) and is pixel-identical to flipping y first with a helper and using `world="yup"` (measured difference 0.0 px). Both ways are fine.

## 3D scenes (`cam3d`)

Two kinds of camera are involved: the **scene camera** (a pinhole camera that is drawn in the scene) and the **viewer** (the virtual camera that decides how the 3D scene is projected onto the 1080x1440 frame).

**Scene camera `PinholeCam`**: world -> camera, OpenCV convention (x right, y down, z forward).
- `PinholeCam(C, target, img_w=640, img_h=480, plane_d=1.0, plane_w=1.6, up=(0, 1, 0))`: centre `C`, looking at `target`; the drawn image plane is `plane_d` in front of the centre and `plane_w` wide (world units); the focal length in pixels follows from these.
- `PinholeCam.from_KRt(K, R, t, img_w, img_h, plane_d=0.3)`: from intrinsics `K` (3x3), extrinsics `R, t` (X_cam = R X + t). `plane_d` is how far from the optical centre the image plane is drawn.
- `.pix(X)` -> `(pixel, depth)` of one world point (depth = camera-frame z); `.pix_many(X)` for `(n, 3)` points -> `(n, 2)` pixels and `(n,)` depths.
- `.unproject(px, depth)`: pixels `(n, 2)` and camera-frame z-depths `(n,)` -> world points `(n, 3)`.
- `.ray_dir(pix)`: unit direction (world) of the ray through a pixel; `.plane_pt(px, py, depth=None)`: the world point on that ray at camera depth `depth` (default `plane_d`); `.corners(depth=None)`: the four image-plane corners (TL, TR, BR, BL) in world coordinates; `.hit_plane(X)`: where the segment X -> C crosses the drawn plane.
- Two-view geometry: `relative_pose(cam1, cam2)` -> `R, t` with `X2 = R X1 + t`; `fundamental(cam1, cam2)`; `skew(v)`; `look_at(C, target, up)`.

**Viewer cameras**
- `ViewCam(eye, target, up=(0, -1, 0), focal=1000.0, cx=W/2, cy=760.0, right_handed=True)`: any pose. `P = ViewCam(...)`; `P(X)` projects world points to screen pixels (single point or `(n, 3)`); `P.depth(X)` gives the distance along the view direction; `P.fit(pts, w=900, h=760, center=(W/2, 760))` scales `focal` and shifts `(cx, cy)` so `pts` fill that box.
- `OrbitCam(target, r, az=0.0, el=30.0, focal=1000.0, cx=W/2, cy=760.0, world="yup")`: the viewer sits on a sphere of radius `r` around `target`. `.at(az=..., el=..., r=...)` returns the `ViewCam` (the projector `P`) for a pose; `.eye(az, el, r)` returns the eye position; `.fit(pts, w, h, center, az=..., el=...)` rescales and centers (returns the `OrbitCam`).
- World conventions: `world="cv"` = OpenCV world (y down), `world="yup"` = y up.

**OrbitCam convention** (measured; `tools/selftest.py` test 6 checks it). The eye is

`eye = target + r * (sin az * cos el, +-sin el, -cos az * cos el)`

with `+sin el` for `world="yup"` and `-sin el` for `world="cv"`. Angles are in degrees.
- `az=0, el=0` looks along +z with +x to the right.
- `az=90, el=0` is a side view (the y-z plane) with +z to the right; `az=-90` puts +z to the left.
- `el>0` looks from above. In world `"cv"` "above" is -y and screen-down is +y; in world `"yup"` "above" is +y.
- `el=90` is degenerate (the view direction is parallel to `up`); use 89 for a top view.
- Tip for a sparse scene: start at `az` 30-60 and `el` 15-30, and call `.fit(points)` so the scene fills the frame. Check a `--still` before rendering the whole shot.

**Drawing**
- `draw_camera(cv, cam, P, alpha=1.0, color=WHITE, fill=PANEL, fill_alpha=0.35, label=None, label_size=28, image=None, image_alpha=1.0, depth=None)`: frustum + translucent image plane + glowing centre; `image` pastes a thumbnail onto the plane by perspective warp; `depth` draws the plane at that camera depth instead of `plane_d`.
- `point_cloud(cv, P, X, colors, rad=2.0, alpha=1.0, ref_depth=None, max_rad=6, glow=0.0, near=1e-3)`: `X` is `(n, 3)`, `colors` is `(n, 3)` or one color; z-buffer, the nearest point wins; with `ref_depth` the point radius scales as `rad * ref_depth / depth`; 60,000 points take about 0.1 s per frame.
- `ray_fan(cv, P, C, dirs, length, color=GOLD, alpha=0.6, th=1, progress=1.0, glow=0.0)`: lines from `C` along unit directions `dirs` of shape `(h, w, 3)` (for example `cam.ray_dir` over a coarse pixel grid), `length` in world units.
- `marker3d(cv, P, X, label=None, color=WHITE, rad=6, alpha=1.0, label_offset=(16, -16), label_size=28, label_color=TEXT)`: glowing dot with a label that follows it; returns the screen point.
- `axes3d(cv, P, origin, R=eye(3), length=0.3, labels=("$x$", "$y$", "$z$"), colors=AXIS, alpha=1.0, th=3, label_size=30, progress=1.0)`: coordinate triad; the COLUMNS of `R` are the axis directions. `camera_axes(cv, P, cam, length=0.3)` draws the axes of a scene camera.
- `floor_grid(x=(-4, 5), z=(-1, 10), y=-1.3, step=1.0)`: returns a list of `(start, end)` world segments; draw them yourself, for example `for a, b in floor_grid(): cv.s.line(P(a), P(b), GREY, 0.3, 1)`.

```python
cam = PinholeCam.from_KRt(K, R, t, 640, 480, plane_d=0.3)
pts_w = cam.unproject(px, depth)                          # (n, 3) world points
view = OrbitCam(target=pts_w.mean(0), r=4.0, az=40, el=20, world="cv").fit(np.vstack([pts_w, cam.C[None]]))
# in frame(cv, t):
P = view.at(az=40 + 10 * t)
draw_camera(cv, cam, P, alpha=a, label="camera 1")
point_cloud(cv, P, pts_w, colors, rad=2.0, alpha=a)
ray_fan(cv, P, cam.C, cam.ray_dir((320, 240))[None, None], length=3.0, progress=ss(t, 1, 2))
```

## 2D side view / diagram (Plot2D with equal scale)

`Plot2D` can draw a geometry diagram, not only data. The one rule: use the **same pixels per unit on both axes**, so angles and ratios are not distorted. Choose the screen box and the limits together: `(x1 - x0) / (xmax - xmin)` must equal `(y1 - y0) / (ymax - ymin)`.

Example: a pinhole side view. The horizontal axis is the depth $z$, the vertical axis is $y$. The camera centre $C$ is at the origin, the image plane is the line $z=f$, the scene point is $P=(Z, Y)$, and its projection is $p=(f, fY/Z)$.

```python
from animkit import Plot2D, ss, GOLD, CYAN, WHITE
f, Z, Y = 1.0, 4.0, 1.2                       # focal distance, depth and height of P
# box 900 x 600 px; limits 6 x 4 units -> 150 px per unit on both axes
p = Plot2D((90, 420, 990, 1020), xlim=(-0.5, 5.5), ylim=(-2, 2), xlabel="$z$", ylabel="$y$", grid=False)
# in frame(cv, t):
p.axes(cv, 1.0)
p.curve(cv, [f, f], [-0.8, 0.8], WHITE, th=4)                         # image plane at z = f
p.line(cv, Y / Z, 0, GOLD, ss(t, 1, 2), x_range=(0, Z))                # ray C -> P: y = (Y/Z) z
p.scatter(cv, [0], [0], colors=GOLD, labels=["$C$"])                   # camera centre
p.scatter(cv, [Z], [Y], colors=WHITE, labels=["$P$"], progress=ss(t, 0.5, 1))
p.scatter(cv, [f], [f * Y / Z], colors=CYAN, labels=["$p$"], progress=ss(t, 2, 2.5))  # projection
```

In the check, assert the equal scale: `ck.close("equal scale", 900 / 6.0, 600 / 4.0, 1e-9)`. `p.to_screen(x, y)` gives the screen position of any data point, so extra labels (for example `f` along the axis) can be placed with `cv.rich(...)`.

## Conventions
- **Frame** 1080x1440, 30 fps, H.264 yuv420p crf 18, `+faststart`. The content area is by default centered at 900x760 (center y = 760).
- **Colors**: always RGB tuples in the API; the palette and its meanings are in `style.py` and `STYLE.md` section C.
- **Fonts**: font kinds `"text"` (regular, default), `"bold"`, `"light"` (thin face for years, numbers and small labels), `"math"` (italic math), `"math_up"` (upright math); formulas use Computer Modern (mathtext). The faces are set in `config/animkit.json`. For a translated render, optional `"text_<code>"` / `"bold_<code>"` entries are used (see "Languages"). Symbols with subscripts only go through `cv.math` or `$...$` in rich text.
- **Captions**: bottom center, from y = 1265, 42 px, at most 2 lines (more is an error), 0.3 s fade in/out, the last caption does not fade out at the end of the shot. More than 12 words (`style.CAPTION_MAX_WORDS`), a line wider than the frame, less than 1 s on screen, or overlapping captions give a render warning (also written to `check.json`). The 12-word limit is a reading-load budget; the real limit is pixel width.
- **Title** at the top left (70, 90): 86 px bold plus a 40 px subtitle; **tags** at the top right.
- **Output folder** `<script dir>/render/<script stem>/`; review and concat find files by this rule; `meta.json` records the caption times, tags, render time and the script hash (used to detect "changed since render").
- **episode.json** (in the episode directory): `{"id": "E01", "title": "...", "storyboard": "storyboard.md", "shots": [{"id": "S01", "script": "shots/s01_input.py", "size": "S"}, ...]}`. The review page looks in the storyboard for the section whose title contains the shot id (override with `"storyboard_key"`). Format sample: `examples/`.

## Pitfalls
- **A formula string that changes every frame** starts a math env for every new string (about 1-2 s each). Use `readout` (`number` / `counter` / `vector` / `matrix`) for changing numbers, not `cv.math(f"{x:.3f}")`. Build static formulas at module level: they are registered at construction and typeset in one batch at first draw.
- mathtext does not support CJK text (put such text outside `$...$`); `\top` is too large in mathtext, write the transpose as `^{\mathsf{T}}`; `\left(` / `\right)` cannot be split across two segments.
- `python -I` ignores `PYTHON*` environment variables and does not put the script directory on `sys.path`; always run through `render_shot.py`, and add `-X utf8`.
- `cv.s.line` and similar calls paint over what is already there (as in the demos). For an earlier solid object to hide what comes later, use `fill` and draw from far to near. Point clouds have their own z-buffer.
- Renamed since v0: token categories `cam` / `reg` -> `camera` / `register`; the labels of `tensor_block` are rich text, write a single symbol as `"$S$"`.
- A `--still` PNG is not refreshed by later renders; delete it or rerun `--still` after editing the shot (the render prints a note listing stale stills).
