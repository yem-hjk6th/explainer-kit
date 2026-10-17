# E02 storyboard: Depth, Depth Pro × Depth Anything 3

Materials: the Depth Pro and Depth Anything 3 papers, plus background notes on depth representations. Every number shown on screen is computed by `shots/e02_numbers.py` (it imports E01's `scene_e01.py` read-only, with asserts in the script).

## 0. Overview

| Item | Value |
|---|---|
| Length | 10 shots, rendered total 6:36 (395.6 s), including pop-ups; §3 lists what to cut for a ≤ 5:00 version |
| Size mix | S × 2 (01, 10) · M × 8 (02–09) |
| Frame / caption rules | Same as E01: 1080×1440, 30 fps; English, ≤ 12 words per caption (animkit `caption_len`: each word / number counts 1, punctuation does not) |
| Explanation order | For every mechanism: input / output → formula → shape → small numeric example → why it is designed this way |
| Captions and on-screen text | Each shot script starts with a `CAPS` list; all on-screen text goes through the animkit text entry point; run `i18n_table.py` to collect `captions.json` (README, "Multi-language") |

### The object that runs through the episode: E01's toy scene + the white dot (the same point)
**Our test scene** = the self-built test from E01 (`scene_e01.py`, referenced read-only): floor + red box + blue ball + yellow ball, three virtual cameras each render one 518×518 image, and the ground truth is fully known. It first appears in E02-01 #1a, where we say plainly that it is our own test and not a paper experiment, where the ground truth comes from, and what it is compared to. From then on the whole episode uses the same phrase "our test scene" and the same corner tag: "Scene GT (own test, not model)" where ground truth is shown, and the wording of `e01_common.SCENE_TAG` where model output is shown (see below).
**White dot** $X^\star$: the same 3D point on the front face of the red box. E02 follows it: its z-depth in camera 2 → inverse depth → disparity in a virtual stereo pair → which tile it falls on in Depth Pro, converted to $C$ and back to meters → written as "ray origin + depth × direction" in DA3 → recovering the camera from the whole ray map → its DA3 token index → one of 6 sparse points when the teacher labels are aligned.

| Quantity (computed by `e02_numbers.py`, rounded for the screen) | Value |
|---|---|
| White dot pixel in camera 2 (sub-pixel point, same as E01) | (239.7, 296.6) |
| z-depth $D$ / distance along the ray $r$ (= E01-08's ray depth) | 1.975 m / 1.984 m ($r=D\,\lVert K^{-1}\tilde p\rVert$, $\lVert K^{-1}\tilde p\rVert=1.0044$) |
| Inverse depth $\rho=1/D$ | 0.506 m⁻¹; over the whole camera 2 image $D\in[1.131,\,3.954]$ m, $\rho\in[0.253,\,0.884]$ m⁻¹ |
| Virtual right camera (camera 2 shifted by $B$=0.12 m along its own x axis) | White dot $u$: 239.7 → 212.5, disparity $\Delta u=27.25$ px $=fB/D$ ($f$ = 448.6 px); the white dot is not occluded in the right camera (assert) |
| $\delta D\approx D^2\,\delta\rho$, $\delta\rho=0.01$ m⁻¹ | $D$=1 m → 1.0 cm; white dot 1.975 m → 3.9 cm; 10 m → 1.0 m |
| Affine alignment illustration (5 pixels) | GT $D$ = (1.975, 2.590, 1.943, 1.184, 2.238) m; illustrative relative depth $\tilde D=(D-1.2)/1.5+\varepsilon$; the normal equations give $\hat s=1.545$, $\hat b=1.175$ m, RMS residual 0.008 m |
| Depth Pro tiling (518 image scaled ×2.965 to 1536) | White dot → (711.9, 880.5) (pixel-center mapping, as in torch `align_corners=False`); scale 1: stride 288, overlap 96 px, covered by the two tiles in rows 2 and 3 (counted from 0), column 2; Voronoi boundary $y$=912 → goes to row 2, column 2 (tile 12); token inside the tile (19, 8) (index 464), in the merged 96×96 map (55, 44) (index 5324) |
| Scale 2 (768, official code) | White dot (355.7, 440.0), stride 192, overlap 192, covered by 2×2 tiles, goes to row 1, column 1 |
| Attention size | Scaled up directly: $96^2=9216$ tokens, per layer per head $9216^2=8.49\times10^7$ entries; Depth Pro: 36 × $576^2$ = $1.19\times10^7$, about 1/7.1; but the number of tokens through the linear layers is 20,736 = 2.25 × 9216 |
| Canonical inverse depth $C$ | $f/w=448.6/518=0.866$; white dot $C=0.866/1.975=0.438$; a 1 m wide vertical bar at the white dot's depth covers 227.1 px = 0.438 × 518 |
| FoV estimated wrong | 60° → 70°: $f/w$ 0.866 → 0.714, white dot 1.975 → 1.629 m (−17.5%) |
| Derivative-loss illustration (row 297, occlusion edge at u = 361 (yellow ball right edge → ground behind), 2.03 → 2.70 m, depth ratio 1.335) | 41 px window: prediction A = whole segment shifted, B = Gaussian blur (σ 3 px), MAE of both is 0.0064; gradient error A = 0, B = 0.0046 |
| DA3 ray (white dot) | $K^{-1}\tilde p=(-0.043,0.084,1)$ → $d=R_{c2w}K^{-1}\tilde p=(0.435,0.111,0.899)$, $\lVert d\rVert=1.004$; origin $t$ = camera 2 center (−1.159, −0.080, 0.300); $t+1.975\,d=(-0.300,0.139,2.075)=X^\star$ |
| Recovering the camera from rays (camera 2, 14×14 = 196 pixels) | Center = mean of origins = (−1.159, −0.080, 0.300); DLT + RQ → $\hat K$: $f$ = 448.6, principal point (259, 259), rotation 30.0° (all exact); with direction lengths randomly ×0.5–2 and σ 0.002 noise → $f$ 448.7 / 448.2, rotation error 0.023° |
| DA3 token (504×504, patch 14) | White dot → (233.2, 288.6), patch row 20 column 16, index 736; each image has 1 + 1296 = 1297 tokens, the white dot token is number 737 within the frame and, in the cross-view layout, $1\times1297+737=2034$ (counted from 0; slot 0: layers 0–7 hold DINOv2's CLS, from layer 8 it is replaced by the camera token; 2034 holds only when camera 1 is the reference view and the order is unchanged) |
| Attention size (3 images) | Per frame $3\times1297^2=5.05\times10^6$, cross-view $3891^2=1.51\times10^7$ = 3 times |
| RANSAC alignment illustration (6 sparse points, the 5th is a bad point: 1.774 → 3.193 m) | Best hypothesis (points 0, 1): $s=1.420,\ b=1.211$, threshold 0.258, 5 inliers; refit on inliers $\hat s=1.502$, $\hat b=1.201$ (the data were built with 1.5, 1.2) |
| Mean distance (E01's normalization, 363,918 valid pixels over 3 images) | 2.153 m |

### Notation (shared across the episode, avoiding letters E01 already uses)
- $D$ = z-depth (meters); $r$ = distance along the ray; $\rho$ = inverse depth; disparity is written **$\Delta u$** (difference of x coordinates between left and right images, in pixels), not $d$, because $d$ is DA3's ray direction in E02-07.
- Affine alignment is written $(\hat s,\hat b)$: $s$ scale, $b$ offset. DA3 Eq. 8 writes the offset as $t$; on screen it is always $b$ ($t$ is already the ray origin / camera translation). E01's mean distance is not given a symbol in E02, only "mean distance 2.153 m".
- Depth Pro's $C$ only means canonical inverse depth; the token dimension is always written as the number 1024, never $C$. DA3's per-image token count is written as the number 1297, not $T$ (E01's $T_i$ is the tracking feature).

### "Computed" vs "illustration"
- **Computed** (asserted in the script): scene depth / inverse depth, disparity of the virtual stereo pair, affine least squares, tiling and Voronoi ownership, attention sizes, $C\leftrightarrow D_m$, FoV error propagation, depth-ray point assembly, solving the camera from the ray map with DLT + RQ, token indices, RANSAC.
- **Illustration**: the "relative depth prediction", "blurred / shifted prediction", "teacher output" and "noisy rays" are fake predictions built from the ground truth (corner tag "Illustration: predictions built from GT"); module internals; the unit and order of the 9 camera-token numbers (the paper does not say).
- **Paper numbers**: Depth Pro Tab. 3 / 9 / 10 / 13, DA3 Tab. 6 / 7 / 11; the screen shows the source, the metric, and whether larger or smaller is better.
- **Measured model output**: values from real Depth Pro / DA3 runs on our test scene (see below). Every other depth map, $C$ map and ray map on screen is **scene ground truth**, with the common corner tag "Scene GT (own test, not model)"; shots that show measured model output use "Measured on our test scene (our own test, not a paper experiment)" (= E01 `SCENE_TAG`).

### Measured model runs: the real Depth Pro / DA3 on our test scene
| | Depth Pro | DA3 |
|---|---|---|
| Repo | `apple/ml-depth-pro` (GitHub; code LICENSE not read) | `ByteDance-Seed/Depth-Anything-3` (GitHub, code Apache-2.0) |
| Weights | HF `apple/DepthPro` `depth_pro.pt`, **1.90 GB** (1,904,446,787 B); license tag `apple-amlr` (Apple ML Research model license; by its name it is research use, the LICENSE text was not read); the model card says this is a retrained reference implementation with performance close to the paper | 1. `depth-anything/DA3-LARGE-1.1` (0.35B, `model.safetensors` **1.64 GB**, CC BY-NC 4.0): multi-view + optional pose, runs on 3 images; 2. `depth-anything/DA3METRIC-LARGE` (0.35B, **1.34 GB**, Apache 2.0): single-image metric, a same-task comparison with Depth Pro. (The paper's main results use Giant: `DA3-GIANT-1.1` 1.15B, CC BY-NC 4.0, size not checked; not used here, since the ablations are ViT-L anyway) |
| Environment | Env `depth-models env` (Python 3.11; torch installed for cu128 with `--index-url` pointing at cu128, GPU required; both repos with `pip install -e`, source cloned into `E02_depth/model_runs/_src/`); weights go in the HF cache (same as E01). DA3 needs `xformers` (README install); if no wheel exists for torch 2.11 / sm_120, first try without it (the README FAQ mentions a path for older cards without xformers); if that fails, split the two repos into two envs | |
| Compute | One forward pass each (Depth Pro 1 image at 1536²; DA3 3 images at 504², once with and once without GT pose; DA3-metric 1 image); expected total GPU time < 5 min, VRAM < 8 GB (not measured) | |
| Shots using the measured output | **05 #8** (own FoV estimate → $D_m$, compared with GT 1.975 m / 60°); **10 #7** | **07 #11** (DLT + RQ on the camera 2 ray map output by DA3, compared with GT $K$ and rotation); **08 #10** (pose with / without GT pose); **09 #1** (normalization scale); **10 #7** |
| Reading of the results (inferred, not on screen) | The toy scene has no real-object size cues, so metric scale can be far off; that is itself evidence for "metric scale comes from priors" and goes on screen as is | Little texture, so the pose can be inaccurate; shown on screen as is |
- All measured values are in `model_runs/e02_metrics.json`.
- 05 #7–#8: the Depth Pro measured **71.0°** field of view (white dot 1.259 m, GT 1.975 m) is shown next to GT.
- 07 #11: DLT on DA3's real ray-map output. 07 #2 / #7 carry code facts (ray map 288×288, direction first).
- 09 #1: with no pose given, DA3 outputs 0.918 for the white dot = 1.975 ÷ 2.153.
- 10 #7: a small measured table of Depth Pro and DA3 results.

---

## 1. Shots

### E02-01 This episode: the same scene, only depth
**S · 22 s**  (input / output + definition of our test scene + two paper cards)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | Same three images; only depth this time | The three rendered images slide in one after another, the white dot flashes once | Computed (ray casting) |
| 1a | E01 test, not a paper result | Pop-up "Our test scene = a test we built (not a paper experiment)" (below the three images): on the left a small GT 3D view (3 cameras casting a few rays to the scene surface); on the right three lines "Toy scene: floor, red box, 2 balls", "3 virtual cameras, ray casting", "Renders three 518×518 images", "Full GT: depth of every pixel, each camera's pose"; after it is dismissed the common top-right corner tag stays from here to the end | Computed |
| 1b | White dot (tracked) | The white dot flashes together on the three images and the small 3D view; "White dot = a 3D point on the red box front, the same one tracked in E01" | Computed |
| 2 | Depth Pro: one image, metric depth | Paper card "2024 · ICLR 2025 · Depth Pro · Bochkovskii et al. (Apple)"; one contribution line "One image, no camera intrinsics, 1536×1536 metric depth in 0.3 s" | Text card (Abstract) |
| 3 | DA3: N images, depth + rays | Paper card "2025 · arXiv 2511.10647 · Depth Anything 3 · Lin et al. (ByteDance Seed)"; one contribution line "Any number of images, pose optional, a plain ViT outputs depth + rays" | Text card (Abstract, §3) |

Sources: Depth Pro Abstract; DA3 Abstract + §3.1–3.2.

---

### E02-02 Depth, inverse depth, disparity
**M · 34 s**  (definition → formula → white dot numbers → why networks like inverse depth)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | Depth D: distance along the optical axis | Side-view cross-section of camera 2 (drawn like E01-08's small figure, code copied): center, optical axis, image plane, white dot; a gold measuring line along the optical axis "$D$ = 1.975 m"; definition line "$D$ ($z$-depth): the point's coordinate along the optical axis" | Computed |
| 2 | Not the straight distance to the center | A second measuring line along the ray "$r$ = 1.984 m"; $r=D\,\lVert K^{-1}\tilde p\rVert=1.975\times1.004$ lights up term by term; small text "$r$ (range): distance to the point along the ray; MapAnything in E01-08 uses $r$" | Computed |
| 3 | Inverse depth: one over depth | Camera 2 depth map (near bright, far dark) → inverse depth map cross-fades in (also near bright); color bar 1.13–3.95 m → 0.25–0.88 m⁻¹; white dot "$\rho=1/D=0.506$ m⁻¹"; definition "$\rho$: inverse depth" | Computed |
| 4 | Imagine a parallel right camera | Small 3D view: a right camera appears beside camera 2's frustum (shifted along camera 2's x axis), orange baseline "$B$ = 0.12 m"; corner tag "Illustration: virtual stereo pair, free baseline" | Position is an illustration, projection computed |
| 5 | White dot differs by 27.3 px between images | Left and right images stacked (the right one rendered from the same scene), an orange horizontal line between the white dots; readout "$u_L=239.7$, $u_R=212.5$"; no pop-up, inline definition "Disparity $\Delta u$: x-coordinate difference of the same point between left and right images (px)" | Computed (assert equals $fB/D$, white dot not occluded in the right image) |
| 6 | Disparity = inverse depth times a constant | Formula $\Delta u=\dfrac{fB}{D}=fB\,\rho$ lights up term by term, same colors on screen: $f$ (focal length, px) = 448.6, $B$ (baseline, m) = 0.12, $D$ = 1.975; $448.6\times0.12/1.975=27.3$ | Computed |
| 7 | Inverse depth off by 0.01: only 1 cm off up close | Curve plot: x axis $D$ (0–10 m), y axis $\delta D=D^2\,\delta\rho$ ($\delta\rho=0.01$ m⁻¹), three points light up in turn: 1 m → 1 cm, white dot 1.975 m → 3.9 cm, 10 m → 1 m | Computed |
| 8 | So inverse depth is more accurate up close | Horizontal bars "Depth Pro paper Tab. 10 (Hypersim, 0–1 m range) / δ1, higher is better": inverse depth 0.730, log depth 0.700, depth 0.657; notes below "δ1 = fraction of pixels whose prediction/GT ratio is within 1.25", "only the output representation changes, trained with MAE only" and "MAE = mean of per-pixel \|prediction − GT\|" | Paper numbers |

Formulas: $r=D\,\lVert K^{-1}\tilde p\rVert$, $\rho=1/D$, $\Delta u=fB/D=fB\rho$, $\delta D\approx D^2\,\delta\rho$ (differentiating $D=1/\rho$). Every symbol on screen comes with its definition each time.
Why: an inverse-depth error of the same size lands as a very small metric error up close; at infinity (sky) $\rho\to0$ stays finite (#3 background pixels $\rho=0$, the color bar marked "background = 0").
Sources: Depth Pro §3.2 p.5 "prioritizes areas close to the camera", B.4 Tab. 10.

---

### E02-03 Relative vs metric depth: scale ambiguity and affine alignment
**M · 36 s**  (phenomenon → definition → formula → numeric example with 5 points → where it is used)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | Scene and cameras scaled up 2x together | Small 3D view: the scene point cloud and the 3 cameras scale ×2 about camera 1; next to it camera 2's image is pixel-identical; white dot depth 1.975 → 3.951 m | Computed (already asserted in E01) |
| 2 | Same photo, double the depth | A depth readout hangs above each of two identical images; "One image holds no geometry that fixes scale" | Computed |
| 3 | Relative depth: off by a scale and an offset | Pop-up "relative depth": $D\approx s\,\tilde D+b$; "$\tilde D$: model output; $s$: scale; $b$: offset; one unknown pair per image", "also called affine-invariant", "metric depth: every pixel directly in meters" | Definition |
| 4 | Offset free too: each dataset has its own zero | Second line of the pop-up "Mixed datasets: disparity has unknown baseline and zero point → neither scale nor offset matches", marked "Background note" | Background |
| 5 | Alignment: solve s and b by least squares | Scatter plot: 5 pixels (white dot / blue ball / yellow ball / near ground / far ground, marked together on a small camera 2 image), x axis $\tilde D$ (illustrative relative depth), y axis $D$ (GT, meters); formula $(\hat s,\hat b)=\arg\min_{s,b}\sum_k\big(s\tilde D_k+b-D_k\big)^2$ | Illustrative prediction, GT computed |
| 6 | Two unknowns, two equations | Normal-equation matrix lights up term by term: $\begin{pmatrix}\sum\tilde D_k^2&\sum\tilde D_k\\\sum\tilde D_k&n\end{pmatrix}\begin{pmatrix}s\\b\end{pmatrix}=\begin{pmatrix}\sum\tilde D_kD_k\\\sum D_k\end{pmatrix}$ → $\begin{pmatrix}1.827&2.624\\2.624&5\end{pmatrix}$, right side $(5.906,\ 9.930)$ → $\hat s=1.545$, $\hat b=1.175$ m; the fitted line is drawn | Computed (check agrees with lstsq) |
| 7 | After alignment only 0.8 cm off | The 5 residual vertical lines light up; "RMS residual 0.008 m (on these 5 illustrative points)" | Computed |
| 8 | Evaluation and labels both need this step | Two small labels appear in turn: "Depth Pro: datasets without meters are normalized before the loss (E02-06)", "DA3: teacher relative depth aligned to sparse depth (E02-09)" | Structure |

Corner tag: #5–#7 "Illustration: relative depth built from GT (GT ÷ 1.5 − 0.8 + small noise)".
Sources: Depth Pro §2 p.3 "ambiguous in scale and shift"; DA3 Eq. 8.

---

### E02-04 Depth Pro 1: multiscale tiling + shared encoder
**M · 42 s**  (input / output → why tile → how to tile (white dot) → shapes → why it is designed this way)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | Input one image; output inverse depth and field of view | Input / output boxes: left "Input: any image is first resized to 1536×1536", camera 2 image enlarged (×2.965); right two output slots "$C$: canonical inverse depth, $1536\times1536$ (covered in E02-05)", "$\theta$: horizontal field of view, 1 number" | Computed (resize) |
| 2 | Whole image into a ViT: too costly | Pop-up "ViT recap": every 16×16 pixels → 1 token (1024-dim), every layer computes attention between all token pairs; 1536 / 16 = 96, $96^2=9216$ tokens, per layer per head $9216^2=8.49\times10^7$ entries; small text "DINOv2 originally uses 14×14 cells; the authors interpolated the weights to 16×16 (B.3)" | Computed |
| 3 | Cut into 384 tiles at three scales | A three-level pyramid appears in turn: 1536 (5×5 tiles), 768 (3×3 tiles), 384 (1 tile); small text next to 768: "The paper omits the middle scale; official code: half size" | Structure from the paper, 768 = official code |
| 4 | Finest level: 5×5 tiles, stride 288 | On the 1536 image the start positions of the 5 tiles, 0 / 288 / 576 / 864 / 1152, drop in one by one, overlap strips of adjacent tiles in orange; "Overlap 96 px = 25% of tile width (Tab. 16)" | Computed |
| 5 | Middle 9 tiles, coarsest 1 tile, 35 in all | $25+9+1=35$ lights up term by term; the 35 tiles stack into one pile with shape $(35,3,384,384)$, "Stacked along the batch dim, fed in at once" | Computed |
| 6 | 35 tiles share one ViT | Box "patch encoder (ViT-L, DINOv2 weights)", the 35 tiles pass through it in turn, the same-weights icon unchanged; output $(35,1024,24,24)$ "Each tile: 24×24 tokens"; small text "The finest level also takes outputs of layers 5 and 11 (official code): 50 more tiles" | Shape computed (384 / 16 = 24); layer numbers = code |
| 7 | Another ViT sees the whole downscaled image | Box "image encoder (separate weights)": the whole image resized to 384 → $(1024,24,24)$; labeled "Gives whole-image context" | Paper |
| 8 | White dot sits in a two-tile overlap | On the 1536 image the white dot (711.9, 880.5) glows (small text "518 → 1536 scaled by pixel centers"); the two tiles in column 2, rows 2 and 3 (counted from 0), are framed together, the overlap strip flashes | Computed |
| 9 | Merge: overlaps cut along the midline | Pop-up "Voronoi merge": each tile keeps only the part closer to its own center; boundary $y$=912 (midpoint between the two tile centers); the white dot is above the line → goes to tile 12, token inside the tile (19, 8) → (55, 44) in the merged 96×96 map; tokens per row $21+18+18+18+21=96=1536/16$ | Computed (Appendix C.1 rule) |
| 10 | Tiling makes attention about 7x cheaper | $36\times576^2=1.19\times10^7$ vs $8.49\times10^7$ → 1/7.1; small text "But multiscale + overlap + whole image make 2.25x as many tokens pass the linear layers"; horizontal bars "Tab. 9: latency 341 vs 392 ms (lower is better); DIS-5k boundary recall 0.080 vs 0.065 (higher is better)", baseline = "the same ViT-L scaled up directly to 1536"; one definition line below the bars: "Boundary recall: fraction of GT depth-jump edges the prediction also finds (details in E02-06)" | Computed + paper numbers |

Shape chain (on screen): $(3,1536,1536)\to35\times(3,384,384)\to(35,1024,24,24)\to$ merged into three feature maps $96^2/48^2/24^2$ (48 = official code).
Why (the last line of #10 on screen): "Shared weights: the authors say they may learn a scale-invariant representation (\"may\" in the paper; not tested separately)"; the ViT structure is unchanged and pretrained weights are swapped in directly (p.4).
Sources: §3.1; Appendix C.1; B.2–B.3 patch 16 (the interpolation sentence = B.3); Tab. 16 "Intersection 25%"; Tab. 9.

---

### E02-05 Depth Pro 2: decoder → canonical inverse depth → meters; FoV head
**M · 38 s**  (what the output is → formula → white dot numbers → meaning of $C$ → where the focal length comes from → what happens when the focal length is wrong)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | Decoder fuses coarse to fine | 5 feature streams stacked vertically: layer 5 and layer 11 intermediate features of the finest scale (96², two streams), outputs of the three scales (96² / 48² / 24²), the coarsest stream is first concatenated with the image encoder output (24²) and passed through a 1×1 conv; the 5 streams are upsampled to 768 / 384 / 192 / 96 / 48 → fused step by step from 48 up to 768 (a light dot travels bottom to top) → output head ×2 → $1536\times1536$; pop-up "DPT (covered in E01-05): feature maps are upsampled, added and convolved step by step, ending in a per-pixel map"; small text "Channels 256 (Tab. 16)", corner tag "Official code" | Code + paper |
| 2 | Output C: canonical inverse depth | Camera 2's $C$ map (computed from GT, 0.22–0.77); white dot "$C=0.438$"; pop-up title "$C$: canonical inverse depth" | Computed (GT, corner tag "Scene GT"; #2–#7 all use GT to explain the formula, the measured result is in #8) |
| 3 | To meters: D equals f divided by w times C | Formula $D_m=\dfrac{f_{px}}{w\,C}$ (paper p.5, unnumbered) lights up term by term: $f_{px}$ (focal length, px) = 448.6, $w$ (image width, px) = 518, $C$ = 0.438 → $D_m$ = 1.975 m | Computed |
| 4 | C: fraction of image width ÷ actual width | Inside the pop-up: a 1 m wide vertical bar (gold) stands at the white dot's depth, covering 227.1 px on the image plane; $227.1/518=0.438=C$ | Computed (derivation) |
| 5 | This ratio is readable from the image | "How wide the object appears in the image ÷ how big it usually is → $C$; no focal length needed"; corner tag "Derivation, not stated this way in the paper" | inferred |
| 6 | Focal length goes to the FoV head | Box "FoV head": two input streams (frozen depth network features + features of the third ViT (FoV only)) → three conv layers (kernel 3, 3, 6, stride 2, 2, 1, channels 128 → 64 → 32 → 1) → $\theta$; small text "Trained separately after depth training, with L2 loss" | Paper (§3.3, C.2) |
| 7 | FoV 60°, focal length 448.6 px | $f_{px}=\dfrac{w/2}{\tan(\theta/2)}=\dfrac{259}{\tan30^\circ}=448.6$ lights up term by term | Computed |
| 8 | Our test scene: estimated 71°, depth 19% shorter still | Measured with Depth Pro on camera 2: $\theta$ = 71.0° (GT 60°) → $f_{px}$ = 363.1 (GT 448.6); same $C$ map, white dot $D_m$: with GT focal length 1.555 m → with its own estimate 1.259 m (× 363.1 / 448.6 = 0.809); one reference line at GT 1.975 m; small text "The whole depth map shrinks by the same ratio", "Even with the GT focal length only 1.555 m: $C$ itself is also too large (0.557 vs 0.438 computed from GT)"; corner tag `SCENE_TAG` | Measured model output + computed |
| 9 | Paper: more images with focal error within 25% | Horizontal bars "Tab. 3 (PPR10K) / δ25% = fraction of images with focal-length relative error < 25%, higher is better": Depth Pro 64.6, second best SPEC 34.6 | Paper numbers |

Why two quantities: $C$ and the focal length can be learned separately from different datasets (the FoV head can use large image sets that have only an EXIF focal length and no depth, §3.3).
Sources: §3.2 p.5; §3.3; C.2; Tab. 3 (p.9); Tab. 16 decoder 256.
**Measured model output**: only #8 shows it; #2–#7 use GT to make the formula clear.

---

### E02-06 Depth Pro 3: training, losses, derivative loss, two stages
**M · 36 s**  (loss definition → an example that MAE cannot separate → derivative loss → why the two stages are ordered this way → evidence)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | All losses act on inverse depth C | Eq. 1: $\mathcal L_{\text{MAE}}=\frac1N\sum_i\lvert\hat C_i-C_i\rvert$, note below the terms "$\hat C$: GT, $C$: prediction, $N$: number of pixels"; small text "Real data: the 20% of pixels with the largest error per image are excluded" | Paper |
| 2 | Datasets without meters: normalize first | "Non-metric datasets: subtract the median from prediction and GT separately, divide by the mean absolute deviation, then compute the MAE", linked back to the small icon of E02-03 | Paper |
| 3 | Two predictions, the same MAE | Curve plot: GT $C$ of the white dot's row ($v$ = 297) at $u$ = 341–381 (occlusion edge at u = 361, the edge is labeled "Yellow ball right edge 2.03 m → ground behind 2.70 m", the same segment highlighted on the small camera 2 image); prediction A = whole segment raised by 0.0064 (edge sharp), prediction B = Gaussian blur (edge blurry); both MAE readouts are 0.0064 | Illustrative predictions, GT computed |
| 4 | Derivative loss: only the blurry one is penalized | Below the curve plot, three derivative curves replace it; readout "Gradient error: A = 0, B = 0.0046"; Eq. 2: $\mathcal L_{*,p,M}=\frac1M\sum_j\frac1{N_j}\sum_i\lvert\nabla_*C^j_i-\nabla_*\hat C^j_i\rvert^p$ lights up term by term ($\nabla_*$ = Scharr first-order / Laplace second-order derivative, $p$ = 1 or 2, $M$ = 6 scales); corner tag "Example uses adjacent differences; the paper uses Scharr / Laplace kernels" | Computed (checks: the two MAEs are equal, A's gradient error = 0) |
| 5 | Three abbreviations: MAGE, MALE, MSGE | Small table: MAGE = Scharr, L1, 6 scales; MALE = Laplace, L1, 6 scales; MSGE = Scharr, L2, 6 scales | Paper (p.6) |
| 6 | Stage 1: real plus synthetic, all data | Left half of the time bar: "Stage 1: all labeled data; MAE (metric) + normalized MAE (non-metric) + normalized gradient losses on synthetic data only"; goal "learn features that generalize" | Paper |
| 7 | Stage 2: synthetic data only, to sharpen edges | Right half of the time bar: "Stage 2: synthetic data only; MAE, MSE, MAGE, MALE, MSGE (Tab. 17)"; comparison text "Synthetic data: GT accurate at every pixel; real data: edges have holes and errors", "The reverse of the usual synthetic-then-real order" | Paper |
| 8 | Metric better, boundaries slightly worse | Small table "Tab. 13 / Hypersim: two-stage vs single-stage" AbsRel 0.235 vs 0.245 (lower is better), boundary F1 0.465 vs 0.478 (higher is better); two definition lines below the table: "AbsRel: mean of per-pixel \|prediction − GT\| ÷ GT", "Boundary F1: harmonic mean of precision and recall of predicted depth-jump edges against GT (how edges are found comes next)"; "Authors: single-stage is a fine default, two-stage: better metric, slightly worse boundaries" | Paper numbers (OCR table) |
| 9 | Boundary metric: adjacent depth ratio exceeds 1+t% | Back to the edge of #3: $2.704/2.025=1.335>1.15$ ($t$=15) → marked as an occlusion contour; one line for Eq. 3 precision / recall; "Matting / segmentation masks as GT: only recall can be computed" | Computed + paper |

Sources: Eq. 1; normalization; Eq. 2 and abbreviations; two stages; Tab. 13; boundary metric.

---

### E02-07 DA3 1: depth-ray, a point = ray origin + depth × direction
**M · 46 s**  (input / output → formula (white dot) → comparison with MapAnything → shapes → recovering the camera from rays → why a camera head is still needed → evidence)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | Any number of images in; depth and rays out | Input: 3 images (small text "1 image = monocular") + an optional pose slot (dashed, covered in E02-08); output two blocks per image: $\hat D$ ($H\times W$), $\hat M$ ($H\times W\times6$, the ray map); gray optional output "Camera (translation, rotation, FoV)" | Paper |
| 2 | Ray map: 6 numbers per pixel | The white dot pixel unfolds into a 6-slot vector, **drawn as a concatenation**: first 3 slots "origin $t$ (camera center)" + last 3 slots "direction $d$", $3+3=6$ (not a product); corner tag "Paper's order; the official code stores direction first" | Paper |
| 3 | Direction d is not normalized | Cross-section: $K^{-1}\tilde p=(-0.043,0.084,1)$ lies on the camera-frame $z=1$ plane; rotated to the world frame $d=R\,K^{-1}\tilde p=(0.435,0.111,0.899)$, $\lVert d\rVert=1.004$; definition line "$R$: rotation from cam 2 to world" | Computed |
| 4 | P equals t plus D times d | Formula $P=t+D(u,v)\cdot d$ (paper §3.1 p.6, unnumbered) lights up term by term: $t$ = (−1.159, −0.080, 0.300), $D$ = 1.975, $d$; the point travels along the ray from the center to the white dot, readout → (−0.300, 0.139, 2.075) = $X^\star$ | Computed (assert) |
| 5 | d has z = 1, so D is the z-depth | Cross-section: walking 1.975 times along $d$ reaches $z$ = 1.975; the same measuring line as $D$ in E02-02 | Computed |
| 6 | MapAnything splits it differently; same point | Side by side (E01-08 style): left "MapAnything: unit ray (−0.043, 0.083, 0.996) × ray depth 1.984"; right "DA3: unnormalized $d$ × z-depth 1.975"; both ends reach the same white dot; small text "MapAnything's direction is in the camera frame, DA3's $d$ in the world frame" | Computed |
| 7 | Whole image: shapes and element-wise operations | Shape bar: $\hat D\ (N,H,W)$, $\hat M\ (N,H,W,6)$ → $P=\hat M[\ldots,{:}3]+\hat D\odot\hat M[\ldots,3{:}]$: $(N,H,W,3)$; camera 2's ray map drawn as two small images (three origin channels = a constant color block, three direction channels → RGB); small text "For a 504 input, DA3-Large's ray map is 288×288 (the ray branch is not interpolated back to input resolution; the official API does not return ray maps by default, so we tapped it from an intermediate stage)"; small text next to the formula "$[\ldots,{:}3]$ = origin is the paper's order; the official code is reversed" | Computed (GT ray map, corner tag "Scene GT"); the ray-map size is read from the real model output |
| 8 | Reverse: average the camera center | Eq. 1: $t_c=\frac1{HW}\sum M(h,w,{:}3)$ → (−1.159, −0.080, 0.300); the same small text "Paper's order: first 3 = origin" | Computed |
| 9 | Directions: solve a matrix with DLT | Pop-up "DLT (direct linear transform)": Eq. 2 $H^*=\arg\min_{\lVert H\rVert=1}\sum\lVert H\tilde p\times M(h,w,3{:})\rVert$; "cross product = 0 ⇔ $H\tilde p$ and $d$ are parallel"; each pixel gives 2 linear equations, 196 pixels stack into 392 × 9 → SVD, take the smallest singular vector | Computed (14×14 pixels) |
| 10 | RQ decomposition splits out K and R | "$H\propto R\,K^{-1}$ → $H^{-1}\propto K R^{\mathsf T}$ (upper triangular × orthogonal) → RQ"; the entries of the $\hat K$ matrix light up one by one: 448.6, 259, 448.6, 259; "Rotation angle = 30.0° (GT 30°)"; corner tag "Derivation: from §3.1's $d=RK^{-1}\tilde p$" | Computed (assert equals GT) |
| 11 | Real model rays solve back too | We run the same DLT + RQ on the cam 2 ray map output by DA3-Large (288×288, 856 pixels used): $\hat f_x$ = 426.4, $\hat f_y$ = 423.1 (GT 436.5, in 504-pixel units; relative error −2.3 % / −3.1 %); cam 2 rotation angle 30.1° (GT 30°), rotation error about 0.1–0.2° (varies with solver); small text "Our DLT: Eq. 2 form, coordinates not normalized, 856 pixels"; the principal point is not shown; corner tag `SCENE_TAG` | Measured model output |
| 12 | A lightweight camera head is added and trained jointly | "Camera head: reads only each image's camera token → translation $t$, rotation $q$, field of view $f$"; "About 0.1% of backbone compute" | Paper (p.7, §7.2.1) |
| 13 | Depth + rays only: more accurate pose | Horizontal bars "Tab. 6 / Auc3, higher is better": HiRoom depth + ray 48.7 vs depth + cam 10.8; ETH3D 25.5 vs 9.9; pop-up "Auc3: for each image pair take the larger of the rotation and translation angle errors, the area under the accuracy curve for thresholds 0–3°" | Paper numbers |

Formula: $P=R\big(D\,K^{-1}\tilde p\big)+t=t+D\cdot d$, $d=RK^{-1}\tilde p$; on screen $R,t$ are defined as camera → world (DA3's convention, the opposite of VGGT's world → camera).
On screen we use the paper's notation $P=t+D\cdot d$ ($t$ origin, $D$ depth, $d$ direction) rather than $P=O+d\cdot r$.
Sources: §3.1; Tab. 6; Auc3 definition §6.2; camera head 0.1%. Notation note: the paper writes $H=KR$ on p.6.
**Measured model output**: #10 uses GT to show that DLT + RQ solves back exactly; #11 shows DA3's real output.

---

### E02-08 DA3 2: a plain ViT (token rearrangement) + camera token + Dual-DPT
**M · 44 s**  (structure → shape switch (white dot token) → why → optional pose → output heads → evidence)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | The backbone is just DINOv2; no extra layers | Left: DA3 as one 24-layer column (ViT-L, all pretrained); right: E01's VGGT column "DINOv2 24 layers + 48 new layers (trained from scratch)", the new part in gray; small text "DA3's claim: two thirds of VGGT's layers are not pretrained" | Paper (p.17) + E01 |
| 2 | First, attention within each image | 3 rows of token strips (each 1 + 1296 = 1297, 504×504, patch 14), shape $(3,\ 1297,\ 1024)$; the white dot token is number 737 within the frame (patch row 20, column 16 → 736, preceded by 1 slot 0: in layers 0–7 slot 0 is DINOv2's CLS (whole-image summary token), from layer 8 it is replaced by the camera token, covered in #8) | Computed |
| 3 | Then the three images go in one row | Reuse E01-04's reshape animation: $(3,1297,1024)\leftrightarrow(1,3891,1024)$; the white dot token flies to $1\times1297+737=2034$; small text "In input order; cam 1 is the reference view" | Computed |
| 4 | Same weights; only token order changes | The same $W_Q,W_K,W_V$ icons light up above both layouts; "Paper: no structural change; cross-view comes from rearranging tokens" + small text "Code: from layer 8 on, QK-norm and 2D RoPE are added; no layers added" | Paper + code |
| 5 | Cross-view layer: 3x the compute | Per frame 3 blocks of 1297² vs cross-view 1 block of 3891² (`attn_blocks`); $5.05\times10^6$ vs $1.51\times10^7$ | Computed |
| 6 | With only one image, it is monocular | $N=1$: the two layouts coincide into one sequence; "One image reduces to monocular depth at no extra cost" | Paper |
| 7 | First 8 layers: own frame only, then alternate | Layer bar with 24 cells (DA3-Large): layers 0–7 per-frame; layers 8–23: odd layers cross-view (orange), even layers per-frame → 8 cross-view layers, 16 per-frame layers; small text "Paper: $L_s:L_g=2:1$; the code can only be read as per-frame : cross-view = 16 : 8" | Code + paper (layer 9 on the bar is labeled "first cross-view layer") |
| 8 | From layer 8, one camera token per image | Slot 0 of the tokens: layers 0–7 show a gray CLS cell, from layer 8 it becomes a gold camera token cell; two input slots: solid "With pose: $c_i=E_c(t_i,q_i,f_i)$: MLP (9 → 512 → 1024) + 4-layer transformer" / dashed "Without: 2 learned tokens, one for the reference view, one shared by the other views"; **drawn as a concatenation**: $t$ (translation, 3) + $q$ (rotation quaternion, 4) + $\theta_v,\theta_h$ (vertical, horizontal field of view, radians, 2) = 9; pop-up "Quaternion: 4 numbers (unit length) represent a 3D rotation" | Paper + code |
| 9 | In training, 20% of samples get a pose | `flicker` with a fixed-seed switch, labeled "$p=0.2$" | Paper (§3.4) |
| 10 | Given pose: better reconstruction | Horizontal bars "Tab. 7 f / g (point cloud fused with GT pose)" HiRoom F1 65.8 → 73.8 (higher is better), DTU CD 3.65 → 2.14 (lower is better); a third bar "7Scenes F1 58.4 → 46.0, drops instead"; definitions below the bars: "F1: harmonic mean of precision and recall of the reconstructed vs. GT point cloud within a distance threshold", "CD (Chamfer distance): mean distance between nearest points of the two clouds; the paper gives no unit"; one measured line on our test scene: "No pose given: cam 2 rotation angle 30.07° (GT 30°); GT pose given: white dot 1.978 m. When a pose is given, the API uses Umeyama to scale the output to the input pose's scale" | Paper numbers + measured model output |
| 11 | Dual-DPT: first half shared, second half separate | Structure diagram: 4 feature streams → shared reassemble → two fusion branches → two output layers: depth + confidence (2 channels), rays + confidence (6 + 1 = 7 channels); pop-up "DPT (E01-05): tokens rearranged to 2D, fused step by step" dismissed after one sentence | Paper (Fig. 3) |
| 12 | Old approach: nearly all metrics drop | Small table "Tab. 7, HiRoom Auc3, higher is better": DA3's design 39.2; two separate DPTs 5.59; VGGT style, two stacked transformers (same parameter count) 3.72; corner tag "The authors' explanation: two thirds of VGGT-style layers are not pretrained" | Paper numbers |

Shape: $(B\cdot N,\ 1297,\ 1024)\leftrightarrow(B,\ N\cdot1297,\ 1024)$ ($B$ = batch size; in our test scene $B=1,N=3$).
Sources: §3.2; §3.4 $p$=0.2; Tab. 7; §7.2.2; 504 / patch 14.
**Measured model output**: one measured line in #10; the rotation error with GT pose is not shown (the output is aligned back to the input by the API, which says nothing about the network).

---

### E02-09 DA3 3: training, normalization, five losses, teacher-student
**M · 36 s**  (how GT is processed → losses → why a teacher → alignment (RANSAC on 6 points) → schedule → evidence)

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | GT divided by mean distance first | Lines from the scene point cloud to the origin, the average counts up to 2.153 m (same as E01-07); white dot depth 1.975 → 0.918; beside it one line "With no pose given, DA3-Large outputs 0.918 for the white dot, consistent with dividing by the mean distance (turning the three depth maps into meters needs a factor of 2.149, 0.16 % below the mean distance of 2.153 m)" (corner tag `SCENE_TAG`); small text "DA3: mean $\ell_2$ length from valid points to the origin; the paper does not say which frame the origin is in, here we use cam 1" | Computed + measured model output + inferred |
| 2 | So the main output has no meters | The "m" on the axis fades out; "Metric output comes from a separate DA3-metric (single-image input)"; corner tag "Inference: not stated directly in the paper" | inferred |
| 3 | The five losses are summed | $\mathcal L=\mathcal L_D+\mathcal L_M+\mathcal L_P+\beta\mathcal L_C+\alpha\mathcal L_{\text{grad}}$ lights up term by term, notes below the terms: depth (error times confidence, minus $\lambda_c\log$ confidence, which keeps confidence from collapsing to 0) / ray map / points built from depth × rays / camera / depth gradient; "$\alpha=\beta=1$; every term is an L1 error" | Paper (§3.3) |
| 4 | Real-data depth is sparse and dirty | 6 sparse points light up on camera 2's image, the 5th is marked red "Bad point: 1.774 → 3.193 m"; corner tag "Illustration: the sparse depth and the bad point were built by us" | Illustration |
| 5 | Teacher: trained on synthetic data only | "Teacher = a single-image relative-depth model (DINOv2 + DPT), trained on synthetic data only" → camera 2's dense relative depth map (illustration) | Paper + illustration |
| 6 | Align the teacher to the sparse depth | Eq. 8: $(\hat s,\hat b)=\arg\min_{s>0,b}\sum_p m_p\big(s\tilde D_p+b-D_p\big)^2$, labeled "Same formula as E02-03, wrapped in RANSAC"; $m_p$ = whether this pixel has depth | Paper |
| 7 | RANSAC: guess from two points, count inliers | Pop-up "RANSAC": pick any two points and solve one $(s,b)$ → residuals → threshold = mean absolute deviation of the residuals from their median; the group from points 0, 1: $s=1.420$, $b=1.211$, threshold 0.258, 5 inliers, the bad point outside | Computed (illustrative data) |
| 8 | Refit on the inliers to get dense labels | Least squares on the 5 inliers: $\hat s=1.502$, $\hat b=1.201$; label $=\hat s\tilde D+\hat b$ fills the whole image | Computed |
| 9 | GT for the first 120k steps, then labels | Time bar 0–200k steps, switch at 120k; "128 H100 GPUs" | Paper (§3.4) |
| 10 | Without teacher: most get worse | Horizontal bars "Tab. 7 e": HiRoom Auc3 39.2 → 11.2 (higher is better); DTU CD 3.82 → 3.29 (lower is better, improves instead); definition lines "Auc3: for each image pair take the larger of the rotation and translation errors, area under the accuracy curve for thresholds 0–3°", "CD: mean distance between nearest points of the two clouds; the paper gives no unit" | Paper numbers |

Sources: §3.3; §4.1–4.2, Eq. 8; §3.4; Tab. 7 e; DA3-metric §4.4.

---

### E02-10 Explanatory comparison + wrap-up
**S · 34 s**  (the same white dot: what each needs, what each outputs, internal representation, where scale comes from, what each suits) [every shot is tagged "Our take"; each table cell says whether it is the paper's own words or our reading]

| # | Caption | Visual | Computed / illustration |
|---|---|---|---|
| 1 | The same white dot: what each method needs | Split screen: left "Depth Pro", right "DA3"; the left takes only camera 2's image (the other two grayed out), the right takes all three; pose slot: left none, right dashed optional | Paper |
| 2 | What it outputs | Left: $D_m$ (meters) + $\theta$; right: $D$ (normalized-scale units) + ray map → camera, point cloud | Paper |
| 3 | How depth is represented inside | Left: canonical inverse depth $C$, converted to meters with $f/w$; right: z-depth × unnormalized rays | Paper |
| 4 | Where meters come from | Left: prior on "how big objects usually are" + the estimated focal length; right: the main model gives no meters, either DA3-metric (which needs the focal length as input: $D$ = focal length × output ÷ 300, README L235), or a metric pose (the API uses Umeyama to scale the output to the pose's scale; the network itself gives no meters) | Paper + code + our reading |
| 5 | What each suits | Table lights up row by row (`table`): "One photo needing real distances and sharp edges (e.g. single-image novel view synthesis) → Depth Pro", "Several images needing consistent geometry and cameras (reconstruction, video, pose estimation) → DA3", "Only one image, no meters needed → either works, DA3 on one image is monocular"; corner tag "Our take" | Our reading |
| 6 | Metric comparisons from the paper | Horizontal bars "DA3 paper Tab. 11 / ETH3D δ1, higher is better": DA3-metric 0.917, Depth Pro 0.386; small text "Reported by the DA3 authors; DA3-metric is a separate monocular model, not the main DA3 model; the paper does not say whether Depth Pro used estimated focal length or GT intrinsics" | Paper numbers |
| 7 | Two sentences: "Our test scene: shapes all correct", "Meters: from priors or external focal length / pose" | "Measured on our test scene" small table (`table`, rows light up; corner tag `SCENE_TAG` + "Our take"): columns = method (where focal length / pose come from) / white dot depth (GT 1.975 m) / whole-image AbsRel (lower is better) / AbsRel after scaling; rows: 1. Depth Pro, own focal estimate: 1.259 m / 0.357 / 0.030; 2. Depth Pro, GT focal length: 1.555 m (−21 %) / 0.205 / —; 3. DA3-metric, GT focal length: 2.136 m (+8.1 %) / 0.082 / —; 4. DA3-Large, no pose: 0.918 (normalized units) / — / 0.0063; 5. DA3-Large, GT pose: 1.978 m / 0.0035 / —; a bracket "Same GT focal length" is drawn between rows 2 and 3; three small lines below the table: "AbsRel: mean of per-pixel \|prediction − GT\| ÷ GT", "Pixel sets: rows 1-3 = camera 2 (518 or 504), rows 4-5 = all three images (504)", "The meters in row 5 come from the API scaling depth to the input pose's scale with Umeyama; the network does not output meters itself" | Measured model output |

Sources: Depth Pro Abstract, §3; DA3 §3, §4.4; DA3 Tab. 11.

---

## 2. Dependencies
- Already in the library, used directly: `Formula`, `readout` (vector / matrix), `ImagePanel` + `colorize` / `colorbar`, `cam3d`, `TokenStrip` + `reshape_rows`, `attn_blocks`, `bar_chart`, `table`, `input_slot` / `flicker`, `paper_card`, `Popup`, `arch` (module_box / arrow / measure).
- Episode-specific elements: a 2D coordinate plot (`plot2d`) and overlapping tiles on an image with Voronoi ownership.
- Copied from E01 (original files untouched): the cross-section drawing (E01-08), the reshape segment (E01-04), the white dot / frustum tools (read-only import from `e01_common.py`).
- Data: `shots/e02_numbers.py` (read-only import of `scene_e01.py`).

## 3. Cut order if over time (target ≤ 5:00)
1. 06 #9 boundary metric (5 s): not on the main line of this episode.
2. 05 #9 FoV paper numbers (3 s).
3. 07 #11 real model rays (3 s): #10 already shows it can be solved back.
4. 03 #4 why the offset is freed (3 s): merge into one line of the #3 pop-up.
5. 08 #12 merged with #1 (3 s).
6. 02 #8 Tab. 10 (4 s): #7 already explains the reason.
