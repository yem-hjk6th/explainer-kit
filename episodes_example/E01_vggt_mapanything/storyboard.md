# E01 storyboard: VGGT × MapAnything — multiple images → cameras, depth, point cloud

## 0. Overview

| Item | Value |
|---|---|
| Duration | 10 shots, rendered total 7:43 (463.1 s; per-shot times in §4); three repeated spots are cut per §3 (each with a reason) |
| Size distribution | S × 5 (01 02 03 06 10) · M × 4 (04 05 08 09) · L × 1 (07) |
| Frame | 1080 × 1440 portrait, 30 fps (library default) |
| Caption rule | English; one sentence ≤ 12 words, counted by animkit `caption_len`: one word / number counts 1, punctuation does not count; at render time the library warns about over-long captions; one sentence per visual change, about 2.5–3 s |
| Explanation order | For each mechanism: input / output → formula → shape → small numeric example → why it is designed this way |
| Notation | The paper writes $N$ = number of images (the VGGT code calls it `S`). Every symbol on screen appears with its definition, e.g. “$N$ (number of images) = 3” |

### The object that runs through the whole episode
A toy scene (red box, blue ball, yellow ball, checkerboard ground), with 3 cameras each taking one 518×518 image. The scene and cameras are defined in `shots/scene_e01.py`, and every number shown on screen is computed from it (ray-cast rendering → ground-truth depth → unprojection).

**The white dot**: a 3D point $X^\star$ on the front face of the red box, visible in all three images. The whole episode follows it: its pixel in the camera 2 image → which patch it falls in → which token → where it sits in the global layer → its depth → unprojection back to 3D → in MapAnything, split into ray × depth × pose × scale.

| Quantity (computed by script, rounded for display) | Value |
|---|---|
| White dot's pixel $(u,v)$ in camera 2 | (239.7, 296.6) (a sub-pixel point; the depth at the nearest pixel center (240, 297) is 1.976, so on-screen readings always refer to the sub-pixel point and are labeled “sub-pixel point”) |
| Patch it lies in (counting from 0) | row 21, column 17 → patch index $21\times37+17=794$ |
| Token index within the frame (code order: camera, 4×register, patch) | $5+794=799$ |
| Index in the global-layer sequence (frame 2 = the 1st frame, counting from 0) | $1\times1374+799=2173$ |
| Camera 2 extrinsics (world→camera, VGGT convention) | $q=(0,\,-0.250,\,-0.067,\,0.966)$ ($x,y,z,w$), $t=(1.159,\,-0.080,\,0.300)$ m |
| Field of view / focal length | 60° × 60°; $f_x=f_y=259/\tan30^\circ=448.6$ px |
| $K^{-1}[u,v,1]^\top$ | (−0.043, 0.084, 1) |
| z-depth $D$ | 1.975 m |
| Point in camera 2's frame | (−0.085, 0.166, 1.975) m |
| Point $X^\star$ in the world frame (= camera 1) | (−0.300, 0.139, 2.075) m |
| Mean distance $s$ (363,918 valid pixels over the 3 images) | 2.153 m |
| After normalization: $t/s$, $D/s$, $X^\star/s$ | (0.538, −0.037, 0.139) (0.53849 before rounding); 0.918; (−0.139, 0.065, 0.964) |
| MapAnything: unit ray $R$ / ray depth $\tilde D$ | (−0.043, 0.083, 0.996) (0.08345 before rounding) / 1.984 m (0.922 after normalization) |
| MapAnything: pose $O=R_{\text{VGGT}}^\top$, $\tilde T$ = camera 2 center | $\tilde T=(-1.159,\,-0.080,\,0.300)$ m (after normalization (−0.538, −0.037, 0.139)) |

### General principle: “computed” vs “illustration”
- **Computed**: scene rendering, ground-truth depth, patch / token indices, the index mapping of the reshape, attention weights (for given small q, k), the 9 camera numbers → $K,R$, unprojection, mean-distance normalization, the MapAnything four-factor assembly. The script checks these with `assert`.
- **Paper numbers**: Tab. 3 / 5 / Tab. 2 etc., quoted directly, with the source shown on screen.
- **Illustration**: the values inside token vectors, the confidence map Σ, the tracking features T, the internal structure of modules. Tagged “Illustration” in the corner.
- **Ground truth ≠ model output**: shots that show the real VGGT or MapAnything output carry the tag “Measured on our test scene”; all other depth maps and point clouds on screen are scene ground truth, labeled “scene ground truth, not model output”.

### Real model runs on these 3 images
The real VGGT and MapAnything models were run on the same 3 test images; the outputs are in `model_runs/`. Shots that show their measured output carry the tag shown on screen.
- **VGGT**: env `vggt` (torch 2.11 + cu128); weights `facebook/VGGT-1B` (`model.safetensors`, 5.03 GB); the code is the `mapanything/models/external/vggt` bundled in the map-anything repo (referenced read-only, not modified). The 3 images are run at 518×518 in one GPU forward pass.
  - Shots that show its measured output: **01** (the output point cloud), **05** (the real 9 numbers and the real confidence map Σ, side by side with ground truth), **06** (the real error of the point head vs depth+camera on our scene), **07** (the real prediction's normalized values).
  - The claim in 06 that “Method B is more accurate” is therefore not only a quote of Tab. 3 but measured on this episode's scene; the toy scene has little texture, so the result may also be poor, and it goes on screen as measured.
- **MapAnything**: weights HF `facebook/map-anything`, with dependencies such as UniCeption. Shots that show its measured output: **08, 09, 10** (how far the metric scale is off with images only vs with intrinsics + poses given).

---

## 1. Shots

Caption numbers in the shot tables are stable ids; a gap in the numbering marks a sentence cut per §3.

### E01-01 Task: three images → cameras, depth, point cloud
**S · 31.5 s**  (input / output)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | Three photos of the same scene | Three rendered images slide in one after another, the white dot flashes | Computed (ray-cast rendering) |
| 1a | Our test scene: our own test, not a paper experiment | Popup “Our test scene = a test we built (not a paper experiment)” (2.6–13.7 s, below the three images): on the left a small 3D view of the ground-truth scene, with 3 ground-truth cameras each casting a few rays to the scene surface; right column “A scripted toy scene: ground, red box, two balls” and “3 virtual cameras ray-cast three 518×518 images”; the unified top-right tag “Measured on our test scene (our own test, not a paper experiment)” hangs here from now until the end of the shot | Computed (the scene script) |
| 1b | Ground truth fully known: cameras, depth, 3D points | The ground-truth point cloud in the small view lights up, labeled “GT: known from the scene script”; right column “All GT known: cameras, depth, a 3D point per pixel” | Computed |
| 1c | Feed the 3 images to real VGGT and MapAnything | “These 3 images go into real VGGT-1B, MapAnything”; “‘Measured on our test scene’ tag: real model output” and “‘Illustration’ tag: example from GT via paper formula” | VGGT / MapAnything runs |
| 1d | Error = prediction minus known ground truth | “Error = model prediction minus known ground truth”; small text “Paper experiments use public datasets” / “(e.g., VGGT Tab. 3 = ETH3D, real captures)”, split into two lines so it stays inside the popup (the MapAnything evaluation also includes synthetic TartanAirV2-WB) | Definition; ETH3D = Tab. 3 |
| 1e | The white dot: our chosen marker, tracked throughout | The white dot in the three images and in the small 3D view flashes together; “The white dot = one marker point we chose:” “Follow this same point through the whole pipeline”; last line “From here on, ‘our test scene’ and the tag ‘Measured on our test scene’ mean this test”, and the unified tag only appears at this point | Computed |
| 2 | VGGT outputs all of it in one forward pass | Paper card: “2025 · CVPR · VGGT · Wang et al.” + one line of contribution | Text card |
| 3 | Paper Eq. (1): the input and output of f | The formula appears, with a “Paper Eq. (1)” badge on the left; one line below: “$f$ = the whole VGGT network. Only two equations are numbered: (1) this definition, (2) the training loss” | Paper |
| 4 | I: the 3 input images, N = 3 | $I_i$ turns white + gets a box; three arrows are drawn from $I_i$ to the three images, the image frames flash white, and badges $I_1/I_2/I_3$ appear in the corners | Computed |
| 5 | g: camera parameters per image | In the 3D view three frustums (VGGT measured cameras) appear, with orange $g_1/g_2/g_3$ beside them; the $g_i$ in the formula turns orange | Measured (VGGT run) |
| 6 | D: depth at each pixel | The three images are replaced by VGGT measured depth (near = bright yellow, far = dark purple), frames flash pink, badges change to $D_1/D_2/D_3$; the $D_i$ in the formula turns pink + three arrows | Measured (VGGT run) |
| 7 | P: a 3D point per pixel | Point cloud = $P_i$ of the point head (`world_points`) rising from the image plane; small cyan text “$P_i$: one 3D point per pixel; points of all three images are in camera 1's coordinate frame”; the $P_i$ in the formula turns cyan | Measured (VGGT run; check asserts mean distance = 1.0014) |
| 8 | T: track features, used to find the same point | The three images switch back to the originals, gold $T_1/T_2/T_3$ in the corners; gold line = white dot position computed by the tracker from $T_i$ (error in frames 2 / 3: 2.3 / 1.2 px); small text “$T_i$: per-pixel track features of image $i$ (128 dims), not the track itself” | Measured (VGGT run) |

Formula (at the top, lit up term by term):
$$f\big((I_i)_{i=1}^{N}\big)=\big(g_i,\ D_i,\ P_i,\ T_i\big)_{i=1}^{N}$$
On-screen labels: $N$ (number of images) = 3; each of the five terms has its own color ($I$ white, $g$ orange, $D$ pink, $P$ cyan, $T$ gold); the term in the formula, the legend above the formula, and the matching element in the picture share the color; when a term lights up, the matching element in the picture flashes with it; f, the brackets, and the equals sign are neutral grey. “Core formula” is only a figure of speech for the narration; the screen says “Paper Eq. (1)” and does not say “the paper says this is the core”.
One-line contribution on the paper card: “One transformer, one forward pass: 1 to hundreds of images → cameras, depth, point maps, track features”.
Sources: Eq. 1 (§3.1 p.3); CVPR 2025; “1 to hundreds” §1 / Abstract.

---

### E01-02 Input: 518 pixels cut into 37 × 37 patches
**S · 35.6 s**  (input → formula → shape → example → why)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | Each image is 518×518 pixels | Camera 2's image is enlarged and centered, with rulers “518 px” on the top and left edges | Computed |
| 2 | Cut into small 14×14 patches | Grid lines sweep from top-left to bottom-right, 37 vertical and 37 horizontal lines | Computed (each cell 14 px) |
| 3 | 37 patches per side, 1369 in total | The formula $K=\frac{518}{14}\cdot\frac{518}{14}=37\times37=1369$ lights up term by term | Computed |
| 4 | The white dot is in row 21, column 17 | The white dot glows and its cell is highlighted; corner tag “counting from 0”; index $21\times37+17=794$ | Computed |
| 5 | DINOv2 (24 layers) encodes the whole image into 1369 vectors | The whole image (with grid) enters a “DINOv2 · 24-layer ViT-L” box, and 37×37 vertical bars come out; the bar for the white dot's cell is highlighted and labeled “1024 dims”. DINOv2 runs the whole image through 24 layers of self-attention, so every output vector has seen the whole image; it is not an independent per-patch mapping | Shape is real; vector values are illustrative |
| 6 | Why is each vector 1024-dimensional? | The overview from sentence 5 fades out; the white dot's cell flies out of the thumbnail and is enlarged (14×14 real pixels, white dot marked inside the cell) | Computed |
| 7 | One cell: 14×14×3 = 588 numbers | The cell slides out R, G, B channel slices; the formula $14\times14\times3=588$ lights up term by term | Computed |
| 8 | Flatten by channel into one column | The three slices are pressed into one column $p$ on the left in turn (R 196 / G 196 / B 196; the order = the flattening order of the conv weight [1024, 3, 14, 14]); small text “pixels normalized by mean, std” (`aggregator.py` L205) | Computed (check asserts the flattening order) |
| 9 | Multiply by W, add b: 588 dims become 1024 | Formula $x=Wp+b$ (note: $W$ is 1024×588, $b$ is 1024 biases); $p$ → box “$W$” → $+b$ → a column $x$ on the right grows to 1024 (column height follows the number of components, with dashed lines marking the top and bottom of $p$ for comparison) | Shape is real; values of $x$ are illustrative |
| 10 | All 1369 cells share the same W | The scan box on the thumbnail jumps over cells 0, 1, 2, 38, 520, 1000, 1368, 794, and the $p$ column changes to that cell's real pixels while $W$ stays fixed; right column “= a 14×14 conv, stride 14 (DINOv2's patch embed)” and “Params: $1024\times588+1024=603{,}136$” | Computed (parameter count and weight-header shape checked) |
| 11 | All 24 layers after that stay at 1024 dims | Shape chain $(3,518,518)\to(1369,588)\to(1369,1024)\to\cdots\to(1369,1024)$; small text below “Backbone: 24 rounds at 1024 dims; before the heads, a round's frame + global outputs are concatenated → 2048” | Computed (weight header: depth_head.projects.0 input is 2048) |
| 12 | Width fixed by choosing the ViT-L size | Popup “Where 1024 comes from” (height 660, fully covering the two “Params: …” lines of the main picture): the same 588 numbers go into ViT-S/B/L/g (patch 14), the four bars have lengths 384 / 768 / 1024 / 1536, and the L bar is labeled “VGGT”; then the ViT-L bar splits into 16 segments: “16 attention heads × 64 dims per head = 1024, 24 layers”; source ViT Tab. 1 | Code (vit_small / base / large / giant2, check compares each) + paper (external) |
| 13 | VGGT's backbone keeps 1024, with no projection | In the popup four weight shapes slide in (patch embed [1024,3,14,14], DINOv2 24 layers 1024, camera / register token 1024, backbone 24 rounds 1024); “Code has no projection after DINOv2 → backbone width is 1024 too” | Weight headers + code |

Formula: $K=\dfrac{H}{14}\cdot\dfrac{W}{14}$, with on-screen labels “$K$ (patches per image)” and “$H,W$ (image height, width, in pixels)”.
Small text (sentence 5): “DINOv2's own CLS and register outputs are dropped; only 1369 patch vectors are kept” (official `aggregator.py` L211–212); “DINOv2 has ~304M parameters, 24% of the full model's 1.26B” (counted from the weight-file headers).
Sources: 518 = §3.4 (p.6) and App. B; patch 14 = App. B “multiple of the 14-pixel patch size”; DINO patchify = §3.2, the code is the full 24-layer DINOv2 ViT-L (official `aggregator.py` L166–181, `vision_transformer.py` L369–377); 1024 = App. B layer width + official `aggregator.py` L65 `patch_embed="dinov2_vitl14_reg"`; square 518 input: official `vggt.py` L18 `img_size=518`, `load_fn.py` L162–175 (size unchanged in crop mode).
Note: the paper's 518 is “longer side 518”; the square input is the code default `img_size=518`; the rectangular example (392×518 → 28×37=1036) is not shown on screen.

---

### E01-03 Token composition: 1 + 4 + 1369 = 1374 per image
**S · 15 s**  (input → shape → why)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | Add 5 more tokens in front of each image | Camera 2's 1369 patch tokens are laid out in one strip (the middle folded into “…”), and 5 cells are inserted in front | Count is computed |
| 2 | 1 camera token collects camera info | The orange cell lights up, labeled “camera token: after attention, the camera head reads this frame's camera $g_i$ (9 numbers)”; a small green camera (frame 2's color) + “$g_2$” sits at the left of the row, and an arc runs from the camera token to it | Illustration (the small camera's color = frame number; its shape does not correspond to the real pose) |
| 3 | 4 register tokens, dropped at the end | 4 grey cells light up, labeled “register token ×4: only joins attention; output discarded” | Illustration |
| 4 | The white dot's token is at position 799 | The white dot's token glows, labeled $5+794=799$ (counting from 0); corner tag “Ordered as in the official code (the paper writes 794 / 2168)” | Computed |
| 5 | Image 1 uses a different set of tokens | The three token rows are stacked; the 5 cells of frame 1 change to another color; small cameras also sit at the left of the frame 1 / frame 3 rows (white / purple, same color as the “Fr. N” label) + “$g_1$”, “$g_3$”, and the three arcs light up together | Illustration |
| 6 | The network uses it to identify the reference camera | Next to frame 1 a coordinate axis appears, “world frame = camera 1”; the axis is drawn at the optical center of frame 1's small camera (z toward the image plane, y down), with “World origin” above it; small camera center x = 92, scale 1.5, the “x” is about 33 px from the left edge | Illustration |

Colors coexist: token cells keep their type colors (camera gold / frame 1 orange, register purple / frame 1 pink, patch cyan); frame numbers use `ec.CAM_COLORS` (camera 1 white, 2 green, 3 purple) and appear only on the “Fr. N” labels, the small cameras, and $g_N$; the top-right tag reads “Illustration: token color = type; small camera color = frame no.”. The token strip is shifted right by 45 px to make room on the left.

Formula: $\tilde K=1+4+K=1374$; shape label “$(N,\tilde K,C)=(3,\,1374,\,1024)$”, with on-screen labels “$\tilde K$ (tokens per image)” and “$C$ (token dim)”.
Why: after attention, the camera token is read by the camera head to give the camera; frame 1 uses its own set of learnable tokens so that the network can tell which frame is the reference and put its 3D outputs in camera 1's coordinate frame (§3.3 p.4–5).
Sources: 1 camera + 4 register: §3.3; separate set for frame 1 (official `aggregator.py` L128–131, L315–338); register outputs discarded; order camera→register→patch: official `aggregator.py` L134, L221 (the paper's concatenation order is $t^I, t^g, t^R$; positions are given explicitly by the RoPE pos array, so the order only affects slice indices).

---

### E01-04 Alternating attention: the shape switches back and forth
**M · 31.4 s**  (input / output → formula → shape → small example → why)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | Tokens of the 3 images, one row each | The three token rows from the end of 03; to keep it readable only 4 cells are drawn per row, corner tag “Not to scale: 4 tokens drawn per frame” | Reduced scale |
| 2 | Frame attention: each row on its own | The box “frame-wise attention” lights up; lines connect tokens within each row | Structure is real |
| 3 | Shape (3, 1374, 1024) | Tensor block label; the screen defines $(N,\tilde K,C)$ | Real |
| 4 | Join the three rows end to end | Reshape animation: token $(i,k)$ flies along a curve to position $i\cdot\tilde K+k$, the paths computed from the index | Computed (index mapping) |
| 5 | Global attention: all tokens together | The box “global attention” lights up; lines connect the whole row | Structure is real |
| 6 | Shape (1, 4122, 1024) | Label; $3\times1374=4122$ | Computed |
| 7 | White-dot token: 799 → 2173 | The white-dot token flies on its own, labeled $1\times1374+799=2173$ | Computed |
| 8 | Frame attention: only the 3 diagonal blocks | A 12×12 attention matrix (reduced scale) with only the 3 diagonal 4×4 blocks lit | Structure is real |
| 9 | Global attention fills the whole table | The off-diagonal blocks also light up (another color, labeled “cross”) | Structure is real |
| 10 | Same query: frame attention sees only its own frame | Small example (below): frame softmax bars 0.670 / 0.330 | Computed |
| 11 | Global: 63% of the weight goes to frame 2 | Global softmax bars 0.249 / 0.123 / 0.505 / 0.123, the two bars of frame 2 total 0.628 | Computed |
| 13 | Same parameter count: alternating has the lowest error | Tab. 5, three bars: 1.061 / 0.827 / 0.709 | Paper numbers |

Formula (shown at sentences 2 and 5, $Z$ lit up term by term):
$$\mathrm{Attn}(Z)=\mathrm{softmax}\!\Big(\frac{(ZW^Q)(ZW^K)^\top}{\sqrt d}\Big)\,ZW^V$$
frame-wise: $Z=X_i$ (the $\tilde K$ tokens of one image); global: $Z=$ the $N\tilde K$ tokens of all images.
Shapes (with batch size $B$, small text on screen): frame-wise $(B\!\cdot\!N,\ \tilde K,\ C)$; global $(B,\ N\!\cdot\!\tilde K,\ C)$.
Small example (computed, script `assert`): $d=2$, the query of token $a_1$ in frame 1 is $\mathbf q=[1,0]$; keys: $a_1=[1,0]$, $a_2=[0,1]$ (frame 1), $b_1=[2,0]$, $b_2=[0,0]$ (frame 2). Scores $\mathbf q^\top k/\sqrt2$ = 0.707, 0, 1.414, 0. Frame softmax (only $a_1,a_2$): 0.670, 0.330; global (all four): 0.249, 0.123, 0.505, 0.123. Corner tag “Example values” on screen.
Real size (small text on screen, computed): with $N=3$, $\tilde K=1374$ the number of scores per head per layer is: frame $3\times1374^2\approx5.7\times10^6$, global $4122^2\approx1.7\times10^7$, exactly $N=3$ times.
Why: the paper's reason for alternating is that it “strikes a balance between integrating information across images and normalizing activations for tokens within each image” (§3.2); evidence Tab. 5 (p.8): ETH3D point-map Overall (Chamfer distance, lower is better), same parameter count: cross-attention 1.061, global-only 0.827, alternating 0.709. The screen states “lower is better” and “same parameter count”.
Sources: 24 blocks, each = 1 frame-wise + 1 global: App. B, §3.2; 1024 dims, 16 heads; code `aa_order=["frame","global"]`, `depth=24` (official L57, L66, L81–113), the reshape (official L267–312, global offset L295–296); Tab. 5 numbers, same parameter count, the alternating row = the point-head result in Tab. 3. The attention example and the real-size counts are asserted in 04's script.

---

### E01-05 Output heads: a 9-number camera + a per-pixel depth
**M · 73.0 s**  (input / output → formula → shape → example → why)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | Take the camera token after 24 rounds | Three orange tokens are pulled out of the token rows | Illustration |
| 2 | Camera head reads out 9 numbers | Box “camera head (paper: 4 self-attention layers + linear layer)”, box width 380 (x 120–500); the 9 cells come out in three segments in code order: “$t$ (3 numbers): translation”, “$q$ (4 numbers): rotation”, “$f$ (2 numbers): FoV”; upper row VGGT measured, lower row GT, with “No. 1–3 / 4–7 / 8–9” below; the $t$ cells show the normalized values (0.538, −0.037, 0.139), the FoV cells show radians 1.047, 1.047; small text “Code: the same 4 layers iterated 4 times; the 3 camera tokens attend to each other; finally a 2-layer MLP (2048→1024→9)” | Measured values + GT |
| 3 | 3 for translation, 4 for rotation, 2 for FoV | The three segments flash in turn | — |
| 4 | Concatenated end to end: 3+4+2=9 | The three segments slide together, a white frame encloses them as one vector of length 9; below, $g=[t,q,f]$ and $3+4+2=9$ (same group, same color) | Real (code `pose_enc.py` L55 `cat([T, quat, fov_h, fov_w])`; check asserts pose_enc[0:3] = extrinsic $t$, [7]/[8] → $f_y$/$f_x$) |
| 5 | Not 4×3×2=24, but concatenation | The grey $4\times3\times2=24$ is struck out in red, with “Not ×” beside it | — |
| 6 | Paper writes [q,t,f]; code stores [t,q,f] | Narrow bars whose widths follow the counts, $q\,t\,f$, swap to $t\,q\,f$; on the left “Paper §3.1 writes” turns into “Code stores pose_enc as”, on the right “(bar above follows the code order)” | Paper + code |
| 7 | First 3 (t): camera translation | The $t$ segment lights up; in the top-view thumbnail camera 2's frustum translates into place (title “Top view: camera 2 placed by …” centered at x 815, right end about 1020); small text “Translation $t$ is divided by $s$ ($s$ = mean distance from scene points to camera 1 = 2.153 m, see 07)” | Measured |
| 8 | Middle 4 (q): rotation, as a quaternion | The $q$ segment lights up; the frustum turns to the predicted orientation; $R(q)$ unfolds + “rotates 30.9°” | Measured |
| 9–16 (popup 1: quaternion) | Quaternion: 4 numbers for one rotation / u is the axis, θ is the angle / Camera 2 ground truth: rotate 30° about u / Measured q from VGGT: rotates 30.9° / Why use it? Here is the general reasoning / Matrix: 9 numbers, must also be orthogonal / Euler angles: 3 numbers, but gimbal lock / A quaternion only needs length 1 | Gold-framed popup (pointer tip toward the $q$ segment, main picture dimmed). Page 1: input (axis $\mathbf u$, angle $\theta$) / output (4 numbers) → $q=(\sin\frac\theta2\,\mathbf u,\ \cos\frac\theta2)$, $w$ last (code convention) → GT $q$=(0, −0.250, −0.067, 0.966), $\cos\frac\theta2=0.966\Rightarrow\theta=30.0°$, $\mathbf u=(x,y,z)/\sin15°=(0,-0.966,-0.259)$ → in the right column a coordinate axis really rotates about $\mathbf u$ (counter 0 → 30.0° → VGGT 30.9°), small text “$R$: world→cam; cam 2 turns opposite”. Page 2: a table with the 3×3 matrix (9 numbers, $R^{\mathsf T}R=I$, $\det R=1$) / Euler angles (3 numbers, gimbal lock, yaw-pitch-roll as the example) / quaternion (4 numbers, $\lvert q\rvert=1$, $q$ and $-q$ are the same rotation), labeled “General reasons; the VGGT paper does not say why quaternions”; VGGT measured $\lvert q\rvert$ 0.9997 → divide by length → 1 | Computed (checks: axis-angle back to $R$, 30.0°, axis, 15°, $R^{\mathsf T}=$ rotation about $\mathbf u$ by $-\theta$, $\lvert q\rvert$); table = general reasons |
| 17 | Last 2 (f): field of view, in radians | The $f$ segment lights up; small text “$f$ = (fov_h, fov_w): vertical, horizontal FoV in radians; 1.047 rad = 60°” | Real |
| 18 | FoV 60° → focal length 448.6 | $f_x=\frac{259}{\tan 30^\circ}=448.6$ px lights up term by term ($1.047$ rad = 60°), the $K$ matrix takes shape | Computed |
| 19 | Patch tokens go to the DPT head | Cut to the right half of the screen: box “DPT head (dense prediction head)” | Illustration |
| 20 | Take the output of 4 of the 24 rounds | 4 token streams are drawn out from the matching positions of the 24-round stack, labeled “Indices 4, 11, 17, 23 (0-based)”; the 2048-dim small text sits to the right of the DPT box | Real (code indices) |
| 21–26 (popup 2: DPT) | DPT: turn tokens back into a per-pixel map / 1369 tokens back into 37×37 / 4 rounds become 148, 74, 37, 19 / 19 interpolated to 37, add round 17's / Then 74, 148, and on to 296 / Conv, interpolate to 518, output 2 numbers | Cyan-framed popup (leader line toward the DPT box, corner tag “Illustration”; height 652, the figure drawn 24 px lower so the column titles are clear of the 148 / fuse / 296 squares): 4 rows = rounds 4/11/17/23 token strips → arranged into 37×37 grids → ×4 / ×2 / ×1 / ÷2 → 148/74/37/19 → “Fuse” column: starting from 19, each level is “add → interpolate to the next size” (19→37→74→148), the last level ×2 to 296 → 3×3 conv 256→128 → interpolate to 518 → 128→32 → 1×1 to 2 channels (the measured depth map + confidence map pasted as two layers); a bottom block of explanation steps through: input / output (tokens of rounds 4, 11, 17, 23, each round 1369×2048; one round = frame layer + global layer, with the two outputs concatenated; rounds are numbered from 0, matching layers 5/12/18/24 of ViT-L in the original DPT paper (numbered from 1 there)), reassemble, resize, fuse (unified to 256 channels so they can be added), output (the point head has the same structure with 4 channels) | Sizes and channels are real (code + weight headers, check asserts); the drawing is an illustration |
| 27 | Camera 2: depth output by VGGT | Camera 2's depth heat map (VGGT measured); white dot “0.925, GT 0.918” (÷ $s$) | Measured |
| 28 | The second channel is confidence Σ | Confidence map (measured, colored by $\log\Sigma$ with CONF_STOPS as in 07, color bar = 2%–98% percentile of $\log\Sigma$ over the three images; legend below “Bright = high confidence” + low→high color band + “Colored by $\log\Sigma$, color bar as in 07”; the output area of popup 2 is colored the same way); small text “Depth unit: ÷ $s$ ($s$ = 2.153 m) … Depth head outputs depth + confidence; the point map comes from another DPT head (point head)” | Measured |

Formula:
$$g=[\,q,\ t,\ f\,]\in\mathbb R^{9},\quad q\in\mathbb R^4,\ t\in\mathbb R^3,\ f\in\mathbb R^2;\qquad f_x=\frac{W/2}{\tan(f_w/2)},\ \ K=\begin{pmatrix}f_x&0&W/2\\0&f_y&H/2\\0&0&1\end{pmatrix}$$
The screen draws $g=[t,q,f]$ in code order ($3+4+2=9$, concatenation); the paper order $[q,t,f]$ is explained in the small bars of #6. Quaternion: $q=(\sin\frac\theta2\,\mathbf u,\ \cos\frac\theta2)$, $(x,y,z,w)$ with the scalar last.
$$\hat t^{I}_i\in\mathbb R^{1369\times2048}\ \xrightarrow{\text{reshape}}\ \mathbb R^{2048\times37\times37}\ \xrightarrow{\text{DPT}}\ D_i\in\mathbb R^{518\times518},\ \ P_i\in\mathbb R^{3\times518\times518}$$
Note: 2048 = the code concatenates the frame-layer output and the global-layer output of each round (1024 + 1024, official `aggregator.py` L254–261, `vggt.py` L24–27; weight `depth_head.projects.0` has shape [256, 2048, 1, 1]); the input of the camera head is also 2048-dimensional. The paper does not state this dimension, so the screen labels it “code”.
Values (camera 2, computed): the output slots follow the convention of VGGT's actual output: $q=(0,-0.250,-0.067,0.966)$, $t/s=(0.538,-0.037,0.139)$, $f=(1.047,1.047)$ rad; side note “translation is divided by the mean distance $s$ (explained in 07)”.
Small text on screen: depth and point map come from **two independent DPT heads** (official `vggt.py` L25–26); each head also outputs a confidence Σ (illustration).
Sources: $g=[q,t,f]$: §3.1; camera head: §3.3; code iteration and cross-frame attention: official `camera_head.py` L29, L51–56, L73, L86–90, L110–139, the final MLP 2048→1024→9 at L71; DPT uses blocks 4/11/17/23: App. B + official `dpt_head.py` L52, `aggregator.py` L71; the four scales 148/74/37/19: official `dpt_head.py` L74–87 (deconv ×4, ×2, identity, stride-2 conv), fusion (L278–287, L432–454): from 19 on, each level “adds this level's feature (residual conv unit) → interpolates to the next size”; 19→37 is bilinear interpolation (not ×2), the last level ×2 to 296; then output_conv1 3×3 256→128 (at 296) → interpolate to 518 → output_conv2 128→32→output channels (L229–242); FoV ↔ focal length: official `pose_enc.py` L53–54, L113–114 (FoV in radians, code order [fov_h, fov_w]); principal point at the image center.

---

### E01-06 Two methods to a point cloud: unprojection vs direct regression
**S · 42.5 s**  (formula → small example → why)
Method A = point head regresses the point map $P_i$ directly; Method B = $D_i$ from the depth head + $g_i$ from the camera head, then unprojection (Ours (Point) / Ours (Depth + Cam) in Tab. 3).

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 2 | Method A: point head outputs points | Left: “Method A: point head” + “Regresses the point map”; the point head outputs the white dot's 3D coordinates (VGGT measured); the white-dot vector in 28 pt, the “)” is about 30 px from the divider; “GT $X^\star/s$:” sits above the GT vector, with the two lines spaced apart | Measured |
| 2a | Ground truth: known scene points, as reference | The picture holds for 2.4 s; left column “GT $X^\star/s$: reference for comparison” + “= known 3D points from the scene script (÷ $s$)” + the GT vector, the whole block flashes with an orange frame | Computed |
| 3 | Method B: depth + camera, unproject | Right: “Method B: depth head + camera head” + “Then unprojects depth to 3D with the cameras”; the formula appears | — |
| 4 | First, the direction of this pixel | $K^{-1}[239.7,\,296.6,\,1]^\top=(-0.043,\,0.084,\,1)$; a line shoots out inside camera 2's frustum | Computed |
| 5 | Multiply by depth 0.925 for the camera-frame point (VGGT measured) | Camera frame (−0.038, 0.074, 0.925); two source lines below: “0.925 = depth predicted by VGGT's depth head at the white dot pixel” and “(bilinear sample of depth map, normalized units; z-depth, GT 0.918)” (so it is not confused with 08's 0.922); the 3rd line (transform back to camera 1's frame) sits 40 px lower to make room | Measured (0.9252; check asserts) |
| 6 | Transform back to camera 1's frame | Subtract $t$, multiply by $R^\top$, the point lands at (−0.300, 0.139, 2.075), coinciding with the left side | Computed (assert) |
| 7 | Unproject all three images into one point cloud | All pixels of the 3 depth maps are lifted into a point cloud, colored by source camera in three colors | Computed (scene ground truth) |
| 8 | Paper, ETH3D: Method B has lower error | Tab. 3 horizontal bars: DUSt3R 1.005, MASt3R 0.826, Method A 0.709, Method B 0.677; title “Paper Tab. 3 (ETH3D) / Chamfer distance, lower is better” | Paper numbers |
| 8b | Our test scene: B is lower without alignment | Top-right horizontal bars A 0.0247 / B 0.0219, title “Our test scene: no alignment / Same-pixel error, lower = better” | Measured (VGGT run) |
| 8c–8h (popup: how each side computes error) | Paper: each point finds its nearest point / Average each direction, then average both / Paper aligns first; no unit given / Our test scene: per-pixel pairing / Averaged over 363,918 valid pixels / Both: lower is better | Orange-framed popup in the lower half of the screen (corner tag “Illustration”), the main picture frames the block being explained. On the left 6 GT points + 6 predicted points (illustrative points). Page 1 Chamfer: each predicted point links to its nearest GT point (Accuracy = 0.483) → each GT point links to its nearest predicted point (Completeness = 0.534) → Overall = average of the two = 0.508; “Umeyama-align first (rotation, translation, scale)”, “No unit in paper; lower is better”, “Defined in §4.2 (DTU); Tab. 3 uses same-named metrics”. Page 2 per-pixel pairing: points are labeled with pixel numbers, two points with the same number are linked (mean over illustrative points 0.541), $\Vert P_{\text{pred}}-X_{\text{gt}}/s\Vert$, “Measured: mean over 363,918 valid pixels”, “Unit: normalized units, 1 = $s$ = 2.153 m”, “Lower is better”, “Unlike the paper: no nearest search; pixels fix pairs”. After the popup retracts, a definition stays under each of the two charts | Illustrative points, values computed per the definition (check); applying the §4.2 definition to Tab. 3 is an inference |
| 8i | Our test scene: align rotation, translation, scale together | After Umeyama alignment with scale, A 0.0075 / B 0.0077 | Measured (VGGT run) |
| 8j | After alignment, A and B are nearly identical | Same as above | Measured (VGGT run) |
| 9 | Authors' view: splitting into subtasks is easier to learn | Diagram of the “point map = depth + camera” split; corner tag “the authors' explanation; not tested separately in the paper” | The paper's claim |

Formula (appears at sentence 3, lit up term by term at sentences 4–6):
$$P_i(y)=R_i^{\top}\Big(D_i(y)\,K_i^{-1}[u,v,1]^{\top}-t_i\Big),\qquad y=(u,v)$$
On-screen definitions: $R_i,t_i$ (extrinsics of camera $i$, world→camera), $K_i$ (intrinsics), $D_i(y)$ (z-depth at pixel $y$).
Why still train a point head: the paper says that explicitly predicting all quantities (even where closed-form relations exist) clearly helps training; the evidence and reading of Tab. 6 are convoluted (there is no “remove the point map loss” row), so it is not shown in this episode.
Sources: two methods: §3.1 Over-complete; Tab. 3 (p.7; times in the table ~7 s / ~9 s; Method A / B = Ours (Point) / Ours (Depth + Cam) in the table note), Overall = Chamfer distance, lower is better (Acc / Comp defined in §4.2; §4.3: 10 random frames per scene, Umeyama alignment, official masks; no unit stated); “We attribute this to the benefits of decomposing…”: §4.3; unprojection: official `geometry.py` L47–84; z-depth: official `geometry.py` L106–112 (`z_cam = depth_map`, integer pixel grid).
Sentences 2, 5, 6, 8b–8j show VGGT's real predictions on this scene (Method A point cloud vs Method B point cloud, each one's mean distance to ground truth, measured), side by side with Tab. 3.

---

### E01-07 A walk through the full VGGT architecture (with shapes) + training and normalization
**L · 62.3 s**  (input / output → shape → formula → small example → why), a portrait flow chart top to bottom, with a light pulse running along it

**Top-right node figure** (#2–#12): each time the flow chart lights a node, the same box at top right (590, 300)–(1050, 762) switches to that node's figure; the box's border color = the node color, title = node name → output, and the top-right corner tag is “Measured” (VGGT-1B's output on our test scene) or “Illustration”, in a consistent style, so every node has a figure. Each figure lasts about 3 s, with a 0.35 s cross-fade. The small texts (DINOv2 note, trunk note, 48 layers, tracker error) live in the node figures, font size ≥ 18 px. Segment A of the script (#2–#12) uses its own timing; segments B and C (#13–#21) start 2.3 s later than in the base timeline.

| Node | Node figure | Measured / illustration |
|---|---|---|
| images | 3 input images + white dot; “$N=3$ images, each $3\times518\times518$; 3 = RGB channels” | This episode's scene |
| DINOv2 | A 37 × 37 patch grid grows on frame 2, the white-dot patch (row 21, column 17) gets a gold frame → the 794th of the 1369 vectors in a vertical stack; “Each image passes DINOv2 alone; patches attend within” | Grid is real, vectors are illustrative |
| + camera / register | 3 rows of token strips (1 camera + 4 register + 1369 patch), frame 1 in another color set, frame 2's 799th gets a gold frame | Illustration |
| frame / global alternating | Two attention grids, frame (block diagonal) / global (fully filled), light up in turn, with the shapes as subscripts; “24 rounds × 2 = 48 attention layers (excl. DINOv2's 24 layers; each layer 1024 dims, 16 heads)” | Illustration (4 tokens drawn per frame) |
| camera head | The 3 frustums predicted by VGGT (extrinsic / intrinsic, pasted in); “9 numbers = $q$ rotation 4 + $t$ translation 3 + $f$ FoV 2”; “Code: 4-layer trunk ×4, camera tokens attend to each other” | Measured |
| depth DPT | 3 depth maps $D_i$ (near bright, far dark) + 3 confidence maps $\Sigma_i$ (log, colored by the 2–98% percentiles) | Measured |
| point DPT | 3 point maps (xyz → color) + the three combined into a point cloud (camera 1 frame, every 3rd pixel, 40,493 points) | Measured |
| track-feature DPT | One feature block $T_i$ per frame: $128\times259\times259$ (code; 259 = 518 / 2; the paper writes $C\times H\times W$); “Only passed to the tracker; it outputs no tracks itself” | Shape is real, drawing is illustrative |
| tracker | Frame 1's white dot on 3 images → arcs to frames 2, 3; bottom row 48 × 48 px zoom: white circle = GT, cyan cross = prediction, error 2.3 / 1.2 px; “Output $(M,N,2)=(1,3,2)$”; in the architecture diagram a dashed frame “track head = feature DPT + tracker” | Measured |
| #11 | Three paths “frame $i$ token → DPT → $D_i$” that never meet | Illustration |
| #12 | Global attention grid (all cross-frame) + a light spot returns from the three DPTs to the backbone | Illustration |

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 2 | Input: N images, N×3×518×518 | Box “images” lights up, label $(3,3,518,518)$, side note “$N$ (number of images) = 3” | Real |
| 3 | DINOv2 encoding: N×1369×1024 | The DINOv2 box lights up, with “24-layer ViT-L: each image passes alone, patches attend within” on the box; the second half of the note lives in the node figure | Real |
| 4 | Add camera, register tokens: N×1374 | The token concatenation box lights up, labeled $(3,1374,1024)$ | Real |
| 5 | One frame layer plus one global layer is a round | The AA block lights up, the shape jumps back and forth between $(3,1374,1024)$ and $(1,4122,1024)$ | Real |
| 6 | 24 rounds in total | Counter 1 → 24 (right of the AA box); “24 rounds × 2 = 48 attention layers” in the node figure | Real |
| 7 | Camera head: 9 numbers per image | The camera branch splits off, labeled $(3,9)$; small text “Code: 4-layer trunk iterated 4 times, camera tokens attend to each other” in the node figure (node figure = the 3 frustums predicted by VGGT) | Real |
| 8 | Depth DPT: per-pixel depth + confidence | Depth branch, labeled $(3,518,518)$ + confidence $(3,518,518)$; node figure = measured $D_i$, $\Sigma_i$ | Real |
| 9 | Point DPT: one 3D point per pixel | Point-map branch, labeled $(3,3,518,518)$ + confidence; node figure = measured point maps + point cloud | Real |
| 10 | Two sentences: “Track-feature DPT: features for tracking” and “The tracker finds the white dot in frames 2, 3” (the feature DPT inside the track head only outputs $T_i$; the tracks are computed later by the tracker from $T_i$) | Fourth branch “Paper notation $T_i\in\mathbb R^{C\times H\times W}$ (the paper writes $C$, this page writes $C_T$); the code has 128×259×259” → tracker → $(M,N,2)$ (code layout [B, N, M, 2]; in the paper's notation $N$ frames, $M$ query points); the shape on the box is $(N,128,259,259)$, and the two boxes share a dashed frame “track head = feature DPT + tracker” | Paper notation + real code values |
| 11 | Each DPT sees only its own image | Three (depth, point, track-feature) DPTs; the node figure draws three unconnected paths “frame $i$ token → DPT → $D_i$”, the three DPT boxes glow; the source “Code: 3 DPTHeads (vggt.py L25–26, track_head.py L50–57)” is placed below the architecture diagram and “paper p.10” in the node figure | Real (paper p.10) |
| 12 | All cross-image information comes from the backbone | A light pulse returns from the three DPTs to AA; node figure = global attention grid + “every pair of the 4122 tokens”; small text “camera head and tracker also go cross-frame themselves” | Real |
| 13 | Training sums four loss terms | Eq. 2 appears, the four terms light up in turn | Real |
| 14 | The tracking term is scaled by just 0.05 | $\lambda=0.05$ lights up | Real |
| 15 | Depth error is first multiplied by confidence Σ | $\Sigma\odot\lvert\hat D-D\rvert$ lights up | Real |
| 16 | Then subtract α log Σ to stop Σ collapsing to 0 | $-\alpha\log\Sigma$ lights up; side note “$\alpha$ not given in the paper” | Real |
| 17 | Ground truth is first divided by mean distance s | Lines from the scene point cloud to the origin appear beside it, the mean $s$ counts up | Computed |
| 18 | Our test scene: s = 2.153 m | $s$ freezes at 2.153 m (363,918 valid pixels) | Computed |
| 19 | Depth 1.975 m → 0.918 | The white dot's number changes; $t$: 1.159 → 0.538 changes at the same time | Computed |
| 20 | So VGGT's output has no meters | The “m” labels on the point cloud's axes fade out; the point head's points' mean distance to camera 1 = 1.001, matching the training normalization convention (the reason is inferred), with small text below “363,918 valid pixels; the full image including background is 1.147” | — |
| 21 | Scene and cameras scale together; the 3 images look identical | The scene point cloud and the three cameras scale together about camera 1 (×0.5, ×2), while the three images beside it stay pixel-identical | Computed (reprojection unchanged under scaling, assert) |

Formula:
$$\mathcal L=\mathcal L_{\text{camera}}+\mathcal L_{\text{depth}}+\mathcal L_{\text{pmap}}+0.05\,\mathcal L_{\text{track}}$$
$$\mathcal L_{\text{depth}}=\sum_{i=1}^{N}\big\lVert\Sigma^D_i\odot(\hat D_i-D_i)\big\rVert+\big\lVert\Sigma^D_i\odot(\nabla\hat D_i-\nabla D_i)\big\rVert-\alpha\log\Sigma^D_i$$
$$s=\frac{1}{|\mathcal V|}\sum_{(i,y)\in\mathcal V}\lVert P_i(y)\rVert_2,\qquad t\leftarrow t/s,\ \ P\leftarrow P/s,\ \ D\leftarrow D/s$$
On-screen definitions: $\Sigma$ (confidence map; the paper calls it uncertainty; it multiplies the error, large = confident), $\alpha$ (regularization coefficient, not given), $\nabla$ (gradient along the pixel directions), $\mathcal V$ (all valid pixels of all images).
If time runs short, merge sentences 15–16.
Sources: shapes see 02–06; DPT runs per frame, relations between frames are handled only in the backbone: §5 (p.10; in the paper this sentence is in the context of the DPT head, while in the code the camera head and the tracker also go cross-frame: official `camera_head.py` L86–90, `base_track_predictor.py` L61–63, L118–145); code shape of T_i: official `track_head.py` L22 (features=128), L55 (down_ratio=2); Eq. 2, λ = 0.05; confidence loss; normalization: §3.4 “average Euclidean distance”; predictions are not normalized: §3.4 + §5; about 1.2B parameters (on-screen corner tag; weight count 1,256,537,516): §3.4.

---

### E01-08 MapAnything: the same white dot, split into ray × depth × pose × scale
**M · 55.6 s**  (input / output → formula → small example → why)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | MapAnything changes the output format | Paper card: “2025 · arXiv 2509.13414 · MapAnything · Keetha et al.” + one line of contribution | Text card |
| 2 | The same white dot, split into four factors | Back to camera 2's frustum and the white dot; the formula appears at the top, with its four terms dimmed | — |
| 3 | ray: this pixel's unit direction | $R=(-0.043,\,0.083,\,0.996)$, an arrow of length 1 shoots out from the camera center; $R_i$ lights up | Computed |
| 4 | The whole image's rays are the intrinsics | 518×518 arrows fan out (drawn thinned), labeled “ray map, 3×518×518” | Computed |
| 5 | ray depth: how far along the ray | A gold line runs from camera 2 along the ray to the white dot; popup “$\tilde D_i$: depth along the ray (ray depth)” pops up (10.8–23.2 s, lower half of the screen) (title size 26, shifted down 5 px so the tilde does not press against the box): definition “Definition: one number per pixel of view $i$, distance along unit ray $R_i$ from the optical center to the 3D point”; small schematic: gold arrow = $R_i$ (length 1), gold dashed line of full length = $\tilde D_i$, the pixel on the image plane | Definition from the paper; schematic is an illustration |
| 6 | One number per pixel: 1×H×W | In the lower left of the popup the whole $\tilde D_2$ map of camera 2 sweeps out from top to bottom (z-depth from ray casting × $\lvert K^{-1}[u,v,1]\rvert$ ÷ s, 0.61–1.87, 123,156 valid pixels); “Shape: $1\times H\times W$ = $1\times518\times518$ (left)” | Computed |
| 7 | Up-to-scale: only relative distance, no units | A second pixel's ray is added to the schematic, the two points scale up and down together while the pixel positions stay; “What up-to-scale means:” “Rescale the whole scene: pixels stay the same; so only relative distance is known, no units” | Illustration (scale invariance) |
| 8 | The white dot's cell is 0.922 | The white-dot pixel is marked 0.922 on the map; “= true 1.984 m ÷ scene mean distance 2.153 m (illustration: mean output distance set to 1)”; the top-right tag starts from this sentence | Computed + illustrative assumption |
| 9 | Known depth vs network output | “$\hat D_i$: known ray depth”, “(can be an input at inference; ground truth in training)”, “$\tilde D_i$: the network's up-to-scale ray depth” (hat = given, not necessarily metric); legend line “$\tilde D_i$: distance from each pixel along its ray to the point, up-to-scale” | Paper |
| 10 | Measured along the ray, not z-depth | Cross-section schematic + a definition line “z-depth = the point's coordinate along the optical axis (VGGT's D)”, definition first, then numbers; readings along the ray $\tilde D$ 0.922 / along the axis (z-depth) 0.918 (VGGT's D); the numbers in the cross-section schematic also use normalized values (quantities with a tilde never get metric numbers; 0.918 = the GT depth of the white dot in shot 05) | Computed |
| 11 | They differ by R's z component | $0.922\times0.996=0.918$ (consistent at three decimals, the check compares strings), with the three terms annotated “$\tilde D$”, “$R_z=\cos5.4^\circ$”, “z-depth”; the white line in the cross-section flashes; $R_z$ is explained as the cosine of the angle between the ray and the optical axis | Computed |
| 12 | The nearer the image edge, the bigger the gap | “Edge pixel (30, 470): ray 34.8° off the axis”; $0.7305\times0.8215=0.6001$ (consistent at four decimals; the metric 1.572 / 1.292 m are not shown); the gold line in the cross-section and its segment along the axis light up | Computed |
| 13 | Multiply to get the point in camera 2's frame | $\tilde L=R\cdot\tilde D$; $\tilde D_i$ lights up | Computed |
| 14 | Pose O, T moves it into camera 1's frame | The point moves from camera 2's frame to the world frame; $O_i,\tilde T_i$ light up; labeled “$O=R_{\text{VGGT}}^\top$, $\tilde T$ = camera 2 center” | Computed (assert) |
| 15 | Only the scale is missing now | Coordinates show the normalized values (−0.139, 0.065, 0.964) | Computed |
| 16 | m: the metric scale factor, scales the scene into meters | $m$ lights up; popup “$m$: the metric scale factor shared by all views” (39.6–52.2 s, middle of the screen, the two readout lines at the bottom stay): definition sentence “$m$: the metric scale factor shared by all views in one forward pass: scales the whole up-to-scale scene from the network into meters at once” (= `e01_common.M_DEF`, the same sentence in 09; two lines, size 24); $X_i^{\text{metric}}=m\cdot\tilde X_i,\ i=1,\dots,N$; “$m$: one number (a scalar) that turns the up-to-scale $\tilde X_i$ into meters”; the shape comparison table appears row by row: $R_i$ $3\times H\times W$, $\tilde D_i$ $1\times H\times W$ (per view, per pixel), $O_i$: quaternion $Q_i$, 4 numbers; $\tilde T_i$: translation, 3 numbers (one row each, per view; the network outputs $Q_i$ and $O_i$ is computed from it), $m$: scalar, shared by all views, 1 number (whole scene) boxed; legend “$m$: the metric scale factor, shared by all views; multiply to get meters”; the content below sits 24 px lower | Paper |
| 17 | Shared across views: one m for all | Three boxes for views 1 / 2 / 3 ($m\tilde X_1$, $m\tilde X_2$, $m\tilde X_3$), with arrows all coming from the same $m$; “Shared by all views”, “One $m$ per pass” (Eq. (1) outputs $\{m,(\cdot)_{i=1}^N\}$) | Illustration (structure = paper) |
| 18 | Training pushes m ≈ GT distance ÷ output distance | “How it is learned: the scale loss $L_{\mathrm{scale}}$ makes $m\cdot\tilde z\approx\hat z$ in training, so $m\approx\hat z/\tilde z$”; “$\tilde z$ = mean distance of network output points (up-to-scale); $\hat z$ = mean distance of GT points (meters)” | Paper |
| 19 | Assume mean output distance 1: m = 2.153 | “Illustrative assumption: set $\tilde z$ = 1, then $m=\hat z$ = true mean scene distance = 2.153”; the white dot's readout rolls from (−0.139, 0.065, 0.964) to (−0.300, 0.139, 2.075) m, with “×2.153” beside it | Illustrative assumption + computed |
| 20 | Relative-scale-only data can train it too | “Why separate: datasets with only relative scale can still train $R,\tilde D,O,\tilde T$ (paper §1); such data just doesn't train $m$ (this half-sentence is inferred)” | Paper + inferred |
| 21 | m = 2.153; multiply back to get meters | The popup retracts; $m=2.153$ appears, the scene scales back up to metric, and the coordinates return to (−0.300, 0.139, 2.075) m | Computed |

Formula (lit up term by term):
$$X_i^{\text{metric}}=m\cdot\Big(O_i\,\big(R_i\cdot\tilde D_i\big)+\tilde T_i\Big),\qquad \tilde L_i=R_i\cdot\tilde D_i,\quad \tilde X_i=O_i\tilde L_i+\tilde T_i$$
On-screen definitions: $R_i$ (unit ray map, 3×H×W), $\tilde D_i$ (depth along the ray; tilde = without scale), $O_i$ (rotation of camera $i$ in camera 1's frame, converted from the quaternion $Q_i$), $\tilde T_i$ (translation, without scale), $m$ (one metric scale for the whole scene).
Comparison with VGGT: VGGT's extrinsics are world→camera ($R,t$), MapAnything's poses are camera→world ($O,\tilde T$), with $O=R^\top$, $\tilde T=-R^\top t$ = camera center (script assert).
Assumption note (corner tag from sentence 8 on): the network's $\tilde X$ can have any scale and $m$ follows, so the product is unchanged; here the mean distance of $\tilde X$ is taken as 1, so $m=s$.
Why split: known intrinsics / poses / depth can be fed directly as input in the same format; datasets with only relative scale can still train $R,\tilde D,O,\tilde T$ (§1), they just don't train $m$ (this half-sentence is inferred).
Sources: Eq. 1 and the three-line assembly: §3; unit ray, ray depth; relation between ray depth and z-depth: MapAnything's wrapper around VGGT, `external/vggt/__init__.py` L157–160 (`convert_z_depth_to_depth_along_ray`); the premise $m=s$ is consistent with the scale loss: $L_{\text{scale}}$ pushes $m$ toward $\hat z/\tilde z$; arXiv number: `meta.json`.
The tag adds “the measured metric scale error is at the end of the next shot”: MapAnything's measured result (images only vs with intrinsics + poses, compared with 2.153) is shown in E01-09 #29.

---

### E01-09 MapAnything inputs and tokens: optional inputs are added + the scale token
**M · 79.6 s**  (input → shape → why)
Popups P1–P6 sit below the encoder figure (y 795–1245); the row being explained has its input slot and encoder box lit and glowing while the other rows are dimmed; P7 covers the encoder figure (y 320–880), with the token rows staying below.

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | Anything we already know can be fed in | Input slots on the left: image (always given), ray, depth, pose, scale (dashed box = optional) | — |
| 2 | Intrinsics as rays; depth and pose work too | The optional slots light up in turn | — |
| 3 | Each input is encoded into 1536 dims | Five encoder boxes appear in turn (dimmed), the one being explained lights up; shape labels below; small text “ViT-G has 40 layers: the first 24 serve as the encoder, the last 16 initialize the multi-view transformer” (config `keep_first_n_layers: 24`) | Shape is real |
| 4 | Image: through DINOv2 ViT-G | Popup P1 (8.1–15.4 s) “Image encoder: DINOv2 ViT-G”: “Input: one image $3\times518\times518$ → Output $F_I$: one 1536-dim vector per 14×14 patch”; view 1's image + 40 layer bars (the first 24 lit) | Paper |
| 5 | Split into 37×37 patches, 14 px each | A 14 px grid grows on the image, the patch holding the white dot lights up; “518 ÷ 14 = 37, 37×37 = 1369 patches” on one line under the image; “last 16 layers…” on the next line, with a bracket line back to the last 16 bars | Computed |
| 6 | One 1536-dim feature per patch | A cursor climbs the first 24 layers → a cube $F_I$ (37 × 37 × 1536, “$F_I$: layer-24 patch features”; in the released config `norm_returned_features: false`, normalization is done by the LN($F_I$) at fusion time); small text “Why: vs. CroCo v2, DUSt3R encoder, RADIO and random init, DINOv2 was best in results, convergence, generalization” and “ViT-G has 40 layers (DINOv2 Tab. 17): the first 24 are the image encoder, the last 16 initialize the multi-view transformer” | Paper |
| 7 | ray map: 3 numbers per pixel | Popup P2 (15.8–23.4 s) “ray encoder: pixel unshuffle + shallow conv”: “Input: ray map $R_i$, $3\times518\times518$ (one unit direction per pixel) → Output $F_R$: $1536\times37\times37$”; the whole ray map (computed unit vectors, illustrative colors), the 14×14 patch holding the white dot is enlarged into 3 channels | Computed + illustrative colors |
| 8 | Each 14×14 patch becomes 588 numbers | The 3 channels of 14×14 are collected into one column of length 588 and put into one cell of the 37×37 grid: “$588=3\times14\times14$”, “$588\times37\times37$” (left-aligned to the left edge of the grid, clear of the blue channel bar) | Computed |
| 9 | Then a shallow conv makes it 1536 dims | Conv chain (MA weight file): 3×3 conv 588→588 → residual block →768 (GELU) → residual block →1024 (GELU) → 1×1 conv →1536, LN → $F_R$; small text “Why: same grid (37×37) and width (1536) as DINOv2 features, so they add patch by patch (paper §3.1)” | Code (weight file) + paper |
| 10 | Split depth first: mean depth + normalized depth | Popup P3 (23.8–31.2 s): camera 2's ground-truth depth map $\hat D_2$ (along the ray, meters, white dot 1.984 m) splits into the normalized depth map $\hat D_2/\hat z_{d,2}$ (white dot 0.981) + the mean depth $\hat z_{d,2}$ = 2.023 m (valid pixels); the label of the normalized depth sits above the thumbnail | Computed (GT used as the input, illustrative) |
| 11 | Take the log of the scale; 0 if not metric | $\log\hat z_{d,2}$ = 0.705 → MLP in the next window; a switch badge flips between “Metric: encode log” and “Not metric: set to 0”; “Scale varies widely → log (paper)” | Paper + code |
| 12 | Normalized depth goes through a conv of the same structure (the weights are a different set, `depth_encoder.conv_in` [588,196,3,3]) | “Take $\log(1+x)$ first (code): the white dot 0.981 → 0.683” (`model.py` L1011–1013 `apply_log_to_norm`, `geometry.py` L1666–1679) → “pixel unshuffle 14: $1\times518\times518\to196\times37\times37$ ($196=14\times14$) → conv (first layer 196→588) → $F_D$: $1536\times37\times37$ (same grid as ray)”; small text “Why split: accepts metric or up-to-scale inputs; depth and pose may come apart, so each is normalized alone (§3.1)” | Paper + weight file |
| 13 | Translation is split too: divide by mean length | Popup P4 (31.6–39.0 s): “Split $\hat T_i$ first: pose scale $\hat z_p$ = mean $\|\hat T_i\|$ over views that give translation; then $\hat T_i/\hat z_p$”, “$\hat T_i/\hat z_p$ is the translation divided by the mean length, not a unit vector (the paper calls it translation directions)” | Paper + code |
| 14 | Each quantity has its own 4-layer MLP | Four rows: $Q_i$ (4 numbers), $\hat T_i/\hat z_p$ (3), $\log\hat z_{d,i}$ (1), $\log\hat z_p$ (1), each grows a group of 4 bars (MLP, GeLU) → each a 1536-long strip $F_Q,F_T,F_{\hat z_d},F_{\hat z_p}$; small text “Both scales only when the input is metric, else 0 (paper + code); quaternion MLP: 4→128→256→512→1536 (MA weight file)” | Paper + code |
| 15 | The 1536-dim vector is added to every patch | $F_Q$ is copied into the 37×37 grid (filled row by row): “Copied to all 37×37 (code: broadcast, then add)”; small text “Why encode rotation and translation separately: it still works with only IMU (rotation) or only GPS (translation) (paper §3.1)” | Code + paper |
| 16 | Before merging, each branch gets its own LN | The box “LN → sum → LN” and the fusion formula at the top appear (corner tag “Fusion formula: as written in the paper”); popup P5 (39.4–46.8 s): $\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sigma}+\beta$; “$\mu$, $\sigma$: mean and std of this token's 1536 numbers; $\gamma$, $\beta$: learnable, one per dim (1 and 0 in the example)” | Paper + background |
| 17 | LN: subtract the mean, then divide by the std | Two illustrative tokens (4 numbers each): image (8, 2, −4, 10), $\mu$ = 4, $\sigma$ = 5.477 → (0.730, −0.365, −1.461, 1.095); ray (0.2, −0.1, 0.4, 0.3), $\mu$ = 0.2, $\sigma$ = 0.187 → (0.000, −1.604, 1.069, 0.535); bar heights and readouts change together (readouts size 18, bar spacing 74; the “Before LN / After LN” badges sit between the two group titles) | Computed (values illustrative) |
| 18 | Match the scales, then add; neither dominates | The two branches are summed to (0.730, −1.969, −0.392, 1.630), then one more LN; “Before LN: 20×+ apart (10 vs 0.4); after LN: both within ~±1.6, so neither dominates the sum (our reading)” (there is a −1.604, so the range is “about ±1.6” rather than “within ±1.6”) | Computed + our reading |
| 19 | Add it if given, skip it if not | Popup P6: “$F_{I,i}$: image feature; $F_{k,i}$: feature of the $k$-th geometric input”, “$a_{k,i}\in\{0,1\}$: did view $i$ give input $k$; if not, ×0 (code)”; five 37×37 grids are added, in the example depth has $a=0$ and scale is “Not metric: set to 0”, dimmed and marked “× 0”; in the main picture the depth and scale branches break at the same time (the same example as the popup) | Real (code mask), example illustrative |
| 20 | Shape unchanged after adding: 1536×37×37 | The five grids merge into one $F_{E,i}$: $1536\times37\times37$ → 1369 tokens; “→ the later transformer and heads need no change (our reading)” | Computed + our reading |
| 21 | Tokens per view stay the same | Token rows appear, 1369 per view | Computed |
| 22 | View 1 gets one fixed vector | Token rows as in sentence 21; view 1 gets one fixed vector | Real (paper + code) |
| 23 | Then append a scale token | After all token rows one green cell appears; the label “scale token ×1” sits to the right of the cell | Illustration |
| 24 | The scale token enters global layers only | Popup P7 (60.1–67.8 s): 16 columns of G / F (8 G + 8 F, starting with G, code), the cursor walks layer by layer; in G columns the scale token is lit, in F columns it is a grey box; “G = global layer: tokens of 3 views + scale token attend together (paper: 4108; the code adds 1 CLS per view, 4111 in total)” (`use_register_tokens_from_encoder: True`; `dinov2.py` L200–205; `model.py` L680–687, L1592–1595) “F = frame layer: each view on its own, scale token excluded”; the 16-layer bar and the MLP → exp block are inside the popup | Real (code) |
| 25 | In global layers it sees all views | When the cursor reaches a G column, the scale token is linked to the three views; “In G layers, the scale token attends with the tokens of all views, so it sees the whole scene” | Follows from the definition of global attention |
| 26 | MLP, then exp, gives the scale factor m | scale token → “MLP (paper: 2 layers, ReLU; code: 1536→196→196→196→1)” → 0.767 → exp → $m$ = 2.153; on the right a small curve $y=e^x$ with the point at $x$ = 0.767; “Illustration: reusing $m$ = 2.153 from the last shot, so before exp it is $\ln2.153$ = 0.767”, “Why exp: metric scale varies a lot across scenes, so the network outputs its log (paper §3.2)” | Paper + code; values illustrative |
| 26a | m: the metric scale factor, scales the scene into meters (the same sentence as 08 #16) | P7 holds for 2.4 s before retracting (the base timeline ends at 67.8 s); the two bottom lines of P7 light up with 08's definition sentence “$m$: the metric scale factor shared by all views in one forward pass: / scales the whole up-to-scale scene from the network into meters at once”, with a gold backing box flashing once; the two small lines above sit 14 / 18 px higher to make room | Paper |
| 27 | Inputs are randomly hidden in training | The four optional slots flicker on and off at random several times (probabilities labeled beside them; “p = 0.5” placed above the link line, the right end of the input slots at x = 312) | Probabilities are real, flicker is an illustration |
| 28 | One model handles every combination | The slot combinations flip like cards and stop | — |
| 29 | Our test scene: images only, scale error 72% / With intrinsics and metric pose: 9.5% (two sentences, 73.2–75.2 / 75.2–77.2 s) | Bars of the MapAnything run's metric scale error for the three input settings; readout line “with intrinsics + metric pose: 2.36 m”, shape-error line “With intrinsics + metric pose, …” (the 9.5% group is intrinsics + metric pose) | Measured |

Formula:
$$F_{E,i}=\mathrm{LN}\Big(\mathrm{LN}(F_{I,i})+\sum_{k}a_{k,i}\,\mathrm{LN}(F_{k,i})\Big),\qquad a_{k,i}\in\{0,1\}\ \text{(whether view } i\text{ gives input } k\text{)}$$
The paper writes that every input is first passed through LN, then summed, then passed through LN again; corner tag “Fusion formula: as written in the paper”.
Shapes (one 518×518 image): image $3\times518\times518\to1536\times37\times37$; ray $3\times518\times518\xrightarrow{\text{unshuffle 14}}588\times37\times37\to1536\times37\times37$; depth $1\times518\times518\to196\times37\times37\to1536\times37\times37$; pose / scale → $1536$, broadcast to 37×37. Global-layer sequence length $N\times1369+1=4108$ ($N=3$, paper algorithm; the code adds 1 CLS per view: $3\times1370+1=4111$). Small text on screen: 16 layers (8 global + 8 frame, starting with global, code), 24 heads, $C=1536$, initialized from the last 16 layers of DINOv2 ViT-G.
Input dropout probabilities (beside sentence 27): geometric inputs given with probability 0.9; ray, depth, pose 0.5 each; each view gives geometric inputs with probability 0.95 (§3.3).
Why add: the number of tokens does not depend on the kinds of inputs, so the later transformer and heads need no change (an inference by the explainer; the paper's limitation (d) itself mentions the cost of early fusion).
Sources: §3.1; scale token; reference-view embedding, the code uses a fixed sine vector (UniCeption `info_sharing_alternating_attention_transformer.py:773–779`); the scale token enters only global layers: same file `:820–824`; scale MLP + exp; input dropout: §3.3; the code also has 1 CLS token per view (`configs/model/mapanything.yaml:17`), not drawn on screen, noted in small text.

---

### E01-10 Side-by-side comparison + closing
**S · 36.6 s**  (the same set of images and the same white dot flow through the two models in two columns, then the paper numbers)

| # | Caption | Visual change | Computed / illustration |
|---|---|---|---|
| 1 | The same three images go into both models | Split screen along the center line, “VGGT” on the left and “MapAnything” on the right, each side with the same three thumbnails | Computed (rendered images) |
| 2 | Input: VGGT needs only images | The three images on the left flash, arrows go into the box “VGGT · 24 rounds × (frame + global), width 1024”, small text below “Reference: frame 1 uses dedicated camera / register tokens” | Paper |
| 3 | MapAnything can also take known inputs | Under the three images on the right, three dashed slots appear: “Rays (calib.)”, “Pose”, “Depth” (slot width 146, text clear of the edge), flickering at random to show they are optional and any subset may be given; arrows go into the box “MapAnything · 16 layers (8 global + 8 frame), width 1536” (model box width in both columns 400, subtitle clear of the edge), small text “Reference: view 1 gets a fixed embedding” | Paper |
| 4 | VGGT: camera, depth, point map, one each | Four blocks on the left: $g_i$ camera (9 numbers), $D_i$ depth (z), $P_i$ point map, $T_i$ track features (colors as in 01) | Paper |
| 5 | Points come twice: P directly, or from D plus g | White dot (camera 2): $P$ direct (−0.131, 0.061, 0.971), $D+g$ unprojected (−0.132, 0.064, 0.972); “Same point computed twice → redundancy” | Measured (VGGT run) + paper |
| 6 | MapAnything: factored into four parts | Four blocks on the right: $R_i$ ray per pixel, $\tilde D_i$ ray depth, $O_i,\tilde T_i$ pose, $m$ one scene scale | Paper |
| 7 | Ray times depth, to camera 1, then times m | The formula $X_i=m\cdot(O_i(R_i\cdot\tilde D_i)+\tilde T_i)$ lights up in three steps (same colors); “Each quantity once; points are built from them” | Paper |
| 8 | VGGT's unit is mean distance, not meters | Left: “Unit: mean distance = 1, not meters”, “Our test scene: point-head mean distance 1.001” | Measured (VGGT run) |
| 9 | MapAnything multiplies by m; unit is meters | Right: “Unit: meters (times $m$)” | Paper |
| 10 | Our test scene, images only: metric scale off 72% | Right: “Our test scene, images only: scale off by 72.3%”, “With intrinsics + metric pose: 9.5%” (the 9.5% group is intrinsics + metric pose) | Measured (MapAnything run) |
| 11 | Our reading: images only, either model works | A bar at the bottom with the badge “Our take”: “Images only: both work (MapAnything paper's own test: better than VGGT even then, Tab. 2)” | Inferred |
| 12 | Need tracking? Use VGGT | Left: “Need point tracking → VGGT (only it outputs $T_i$)” | Inferred |
| 13 | Calibration, pose, or meters? Use MapAnything | Right: “Calibration (incl. fisheye / non-pinhole), depth, IMU / GPS pose, or metric scale → MapAnything” | Inferred |
| 14 | Paper numbers: two images, images only | Tab. 2(a), two bars, title “Two images, images only: share of pixels whose distance to camera 1 has error < 3% (higher is better)”; corner tag “Measured in the MapAnything paper (VGGT via depth + camera route)” | Paper numbers |
| 15 | VGGT 43.2%, MapAnything 57.5% | The bars grow into place; “Same table, adding intrinsics + pose + depth: MapAnything reaches 82.0%”; license tag at the end | Paper numbers |

The five comparison rows below are all covered by the two columns of #2–#10 (input, output, scale, reference frame, backbone); the table itself is not shown on screen:

| | VGGT | MapAnything |
|---|---|---|
| Input | Images only | Images + optional ray (intrinsics), depth, pose, metric scale; any subset of views |
| Output | Camera + depth + point map + track features (redundant) | ray + ray depth + pose + one $m$ (not redundant) |
| Scale | Divided by mean distance, no meters | scale token → $m$, meters |
| Reference frame | Frame 1 uses dedicated camera / register tokens | View 1 gets a fixed embedding |
| Backbone | 24 rounds × (frame + global), $C=1024$ | 16 layers (8 global + 8 frame), $C=1536$ |

Sources: Tab. 2(a) (p.7; VGGT row, MapAnything row, three-dataset average), points τ; the definition of τ (it compares the distance to the origin ‖X‖, criterion max(ratio, 1/ratio) < 1.03, both sides normalized first) comes from MapAnything's evaluation code `utils/metrics.py` L82–117, `benchmark.py` L173–197; in the evaluation VGGT uses only the camera head + depth head (`external/vggt/__init__.py` L113–176).

---

## 2. Dependencies and requirements for the library
All numbers in the shots of this episode are computed uniformly by `shots/scene_e01.py`.

## 3. Cut order when over time
Sentences cut from the planned shots (numbers are the caption numbers of §1; each cut has a reason):

| Shot | What was cut | Duration | Why understanding is not affected |
|---|---|---|---|
| E01-04 | #12 “24 rounds of alternation, 48 layers in total” and its 24-layer stacking picture | −2.6 s | 07 #5–6 explains “one frame layer plus one global layer is a round; 24 rounds in total” with the same architecture diagram, and next to the counter it adds “24 rounds × 2 = 48 attention layers, not counting DINOv2's 24 layers; each layer 1024 dims, 16 heads”, so no information is lost, it is just not stated once earlier in 04 |
| E01-07 | #1 “Walk the whole path once” | −2.7 s | This sentence is only an opener; the title “VGGT, End to End” is on screen at the same time, and the architecture diagram lights up at the first sentence “Input N images” |
| E01-06 | #1 “Two paths to a point cloud” | −2.3 s | As soon as the split screen appears it carries two large titles, “Method A: point head” and “Method B: depth head + camera head”, and the first caption “Method A: point head outputs points” follows immediately |
| E01-05 | The static 148/74/37/19 grid pyramid in the main picture, and the fusion-path small text 19→37→74→148→296→518 under the depth-map sentence | −2.8 s | The same sizes are explained step by step with animation in the DPT popup (reassemble → resize → fuse → 296 → 518), which also gives the channel counts and the fusion order; putting them in the main picture again would be repetition |
| E01-10 | The 5-row comparison table (input / output / scale / reference frame / backbone) and its 4 captions | −8.1 s | All five rows are in the two columns with nothing lost: input = #2–#3, backbone and reference frame = the two model boxes and the small text under them, output = #4–#7, scale = #8–#10; the table would only list them again, and in the “backbone” row the text of two cells would overlap |
| Not cut | E01-07 #15–16, E01-09 #27–28, the Tab. 2 bars in E01-10 | — | They are, respectively, the role of the confidence regularizer, the training recipe for “arbitrary inputs”, and the direct comparison numbers of the two models; cutting them would lose a mechanism or a piece of evidence |

## 4. Render-time details
Details of the rendered shots that go beyond the shot tables above. The numbers from the real model runs are the “measured” ones; on screen they carry the tag “Measured on our test scene (our own test, not a paper experiment)” (`e01_common.py`).

| Shot | What the final shot shows | Time in shot |
|---|---|---|
| 01 | The cameras, depth thumbnails, and point cloud in the 3D view are the VGGT-1B measured output (predicted cameras + depth + camera unprojection, units ÷ s), with a tag stating this. The screen says “Paper Eq. (1)” + “Only (1) and the loss (2) are numbered equations in the paper”; the five terms each have a color ($I$ white, $g$ orange, $D$ pink, $P$ cyan, $T$ gold) shared by the formula, the legend above it, and the picture elements, and when a term lights up the matching element flashes (for $I$ and $D$ arrows are also drawn to the three images). The point cloud is the point head's $P_i$, $T_i$ is a per-pixel 128-dim feature, and the gold line is the white dot position computed by the tracker from it. A popup defines “our test scene” (11.5 s) | 31.5 s |
| 02 | The white dot's cell, 14×14×3 = 588 numbers → flatten by channel → $x=Wp+b$ ($W$ is 1024×588) → 1024; all 1369 cells share one $W$ (a 14×14 conv with stride 14, 603,136 parameters); shape chain + “backbone 24 rounds also at 1024; before the heads concatenated to 2048”. Popup “Where 1024 comes from”: the same 588 numbers into ViT-S/B/L/g with widths 384/768/1024/1536, so 1024 is fixed by the model size; ViT-L = 16 heads × 64; no projection between DINOv2 and the backbone, so the backbone is also 1024. The screen does not say that VGGT “can only” use this width, nor that a wider vector is needed to hold the whole-image information | 35.6 s |
| 03 | A small camera at the left of each row (color = frame number, `CAM_COLORS`) + $g_N$, with the camera token drawing an arc to it; frame 1's small camera carries the coordinate axes + “World origin”; tokens keep their type colors and the corner tag states the two color schemes; the token strip is shifted right by 45 px to make room | 15.0 s |
| 04 | The Tab. 5 bars follow sentence 11 directly (sentence 12 is cut, see §3) | 31.4 s |
| 05 | Output slots in two rows: upper row VGGT's measured 9 numbers on camera 2, lower row scene ground truth ($t/s$, radians); $R(q)$ is computed from the predicted $q$ (rotation 30.9°, $q$ normalized first); one line under the focal-length sentence: “VGGT prediction on our test scene (camera 2): fov_w = 57.55° → $f_x$ = 471.6 px (fov_h = 57.75°)”; the output slot group is named “FoV h, w (rad)”. The depth map is the VGGT measured output (background pixels outside the scene not drawn), with the measured confidence map Σ beside it; the white dot readout is “0.925, GT 0.918” (units ÷ s). The DPT popup (6 sentences) follows VGGT's real structure: the $1369\times2048$ tokens of rounds 4/11/17/23 (= layers 5/12/18/24 of ViT-L in the DPT paper) → back to 37×37 → ×4/×2/×1/÷2 = 148/74/37/19 → from 19 on, each level “add → interpolate to the next size” up to 148, ×2 = 296 → 3×3 conv → interpolate to 518 → 2 channels (depth + confidence; the point head has 4 channels). The 9 camera cells are split into three segments in code order $t$(3)·$q$(4)·$f$(2), then slide together and are boxed as one strip; captions “Concatenated end to end: 3+4+2=9”, “Not 4×3×2=24, but concatenation”, “Paper writes [q,t,f]; code stores [t,q,f]”, with small bars demonstrating the swap. The quaternion popup (8 sentences): input / output → $q=(\sin\frac\theta2\mathbf u,\cos\frac\theta2)$ → camera 2 GT 30.0°, axis (0, −0.966, −0.259), the coordinate axes really rotate about $\mathbf u$ → VGGT measured 30.9° → page 2 “general reasons” table (matrix 9 numbers + orthogonality constraint / Euler angle gimbal lock, with yaw-pitch-roll as the example / a quaternion only needs $\lvert q\rvert=1$), noting that the VGGT paper does not say why it was chosen | 73.0 s |
| 06 | Title “Point Cloud: Two Ways”; split screen “Method A: point head / Regresses the point map”, “Method B: depth head + camera head / Then unprojects depth to 3D with the cameras”; captions, the white-dot comparison line, and the three bar-chart labels all use Method A / B. The intrinsics line reads “VGGT-predicted camera 2 intrinsics: fov_w = 57.55°”. The white dot is unprojected step by step with VGGT's predicted $K,R,t,D$: direction (−0.041, 0.080, 1), multiply by depth 0.925, camera frame (−0.038, 0.074, 0.925), world frame (−0.132, 0.064, 0.972); Method A point head (−0.131, 0.061, 0.971); ground truth (−0.139, 0.065, 0.964). The alignment result is split into “Our test scene: align rotation, translation, scale together” and “After alignment, A and B are nearly identical”, with the picture holding 0.8 s between the two sentences. The bar-chart blocks are labeled “Paper Tab. 3 (ETH3D, 10 images per scene)” (title “Paper Tab. 3 (ETH3D) / Chamfer distance, lower is better”) and “Toy scene of this episode, 3 images” (title “Our test scene: no alignment / Same-pixel error, lower = better”). The popup “How Each Side Computes Error” (6 sentences, illustrative points computed by the definition): Chamfer = prediction→nearest GT, GT→nearest prediction, average each then average both, Umeyama-aligned first, no unit in the paper; our test scene = per-pixel pairing, averaged over 363,918 valid pixels, unit 1 = $s$ = 2.153 m; after the popup retracts, a definition stays under each chart | 42.5 s |
| 07 | One fixed node-figure box at top right; 9 nodes + #11, #12 each have a figure (input images, patch grid, token strips, attention grids [illustration], predicted frustums, depth + confidence, point maps + point cloud, $T_i$ feature block [illustration], white-dot track + zoomed error), with the same box, the same title style, and the corner tag Measured / Illustration. Captions and node-figure titles spell out depth DPT / point DPT / track-feature DPT; the track branch carries a dashed frame “track head = feature DPT + tracker”, and the node figure states that $T_i$ is only handed to the tracker and that the tracks are computed by the tracker from $T_i$. The three DPT boxes are followed by three independent paths in the node figure; notes under boxes, the 48 layers, and the tracking error are in the node figures (≥ 18 px); the tracker box title is 22 px so it stays inside the box; the AA box shape is split into two lines. The right side shows VGGT's measured depth map and point map thumbnails, and the tracking sentence reads “measured on our test scene: error 2.3 px, 1.2 px”; sentence 20 reads “the point head's points' mean distance to camera 1 = 1.001, matching the training normalization convention (the reason is inferred)”, with small text below “363,918 valid pixels; the full image including background is 1.147”. The three boxes in the “each of three DPTs is independent” part are fully lit, without glow (STYLE: glow ≤ 2). Depth maps use “near bright, far dark” via a direct `colorize(d)` (the 05 color bar runs near at top to far at bottom), consistent across 01 / 05 / 07 / 08 / 09 | 62.3 s |
| 08 | Two popups, definition first and numbers after. Popup $\tilde D_i$ (5 sentences): definition (per-pixel distance along the unit ray to the point) + schematic → the whole $\tilde D_2$ map of camera 2 (computed, $1\times518\times518$) → tilde = up-to-scale (two points scale together, pixels stay) → white dot 0.922 = 1.984 m ÷ 2.153 m → hat = known quantity (input / training GT), tilde = network output. The z-depth part first gives the definition “coordinate along the optical axis”, $R_z=\cos$ of the off-axis angle, plus one sentence on the edge pixel 0.7305 × 0.8215 = 0.6001. Popup $m$ (5 sentences): $X^{\text{metric}}_i=m\tilde X_i$ + shape comparison table ($m$ = 1 number) → shared by three views → $L_{\text{scale}}$ makes $m\approx\hat z/\tilde z$ → under the illustrative assumption $m$ = 2.153, the white dot readout rolls into meters → why it is taken out separately. All quantities with a tilde use normalized numbers (0.922 / 0.918, continuing from 05's 0.918). The tag reads “the measured metric scale error is at the end of the next shot” (the next shot does not display $m$, and the measured mean distance of $\tilde X$ is about 1.47, not 1) | 55.6 s |
| 09 | 7 popups following the paper's components, 3 sentences each (P6 has 2), the row being explained glows: P1 DINOv2 ViT-G (image → 14 px grid → 40 layer bars with the first 24 → $F_I$ cube; why this one); P2 ray encoder (ray map → 14×14×3 collected into a column of 588 → 37×37 → conv chain 3×3 / residual blocks 768, 1024 / 1×1 → 1536 + LN, layers taken from the weight file); P3 split depth first (camera 2 GT depth → normalized depth + $\hat z_d$ = 2.023 m → log 0.705; 0 if not metric → unshuffle 196 → conv); P4 global quantities ($\hat T/\hat z_p$ is a normalized translation, not a unit vector; each of the four quantities gets a 4-layer MLP → 1536, copied to all 37×37 patches); P5 LN (formula + two 4-number tokens with LN computed before and after, “match the scales” labeled as our reading); P6 add (5 grids added, missing ones × 0, $1536\times37\times37$ after adding); P7 scale token (16 columns of G/F with the cursor walking layer by layer, in G linked to all views → MLP → 0.767 → exp → 2.153). The scale token label “scale token ×1” sits to the right of its cell, and the 16-layer bar and MLP → exp form one block inside P7. The last sentences show bars of the metric scale error for three input settings (72.3% / 71.9% / 9.5%), with below them “With intrinsics + metric pose, shape error rises 0.015 → 0.082 (inputs checked; cause unknown)” and “The toy scene has no object of known size, so images alone give no scale cue (inferred)” | 79.6 s |
| 10 | The first 31 s are two columns: the same three images and the same white dot flow through the two models: input (images only vs optional rays / pose / depth), backbone, output (four quantities once each; the white dot's two copies of the point = redundancy for VGGT vs a point built from four factors), unit (mean distance 1.001 vs meters, our test scene's metric error 72.3% / intrinsics + metric pose 9.5%), what each suits (our reading); after that the Tab. 2(a) bars, 82.0%, and the license tag. End tag “Models: VGGT-1B (non-commercial), MapAnything (CC-BY-NC 4.0); for study use only” | 36.6 s |
