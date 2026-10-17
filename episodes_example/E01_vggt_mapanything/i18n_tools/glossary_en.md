# E01 terminology list (English source terms)

Use these terms consistently when translating the episode's captions. Left column: the English term as written in the shots; right column: the wording to keep on screen (translate it with the same choice in every shot) and what it means. Formulas are never translated. Captions: at most about 12 words per sentence; split a sentence that does not fit. Translate only; add nothing.

| Term (English source) | Meaning / preferred on-screen wording |
|---|---|
| our test scene | a toy scene we built ourselves, not a paper experiment; one fixed translation across the episode |
| Measured on our test scene (our own test, not a paper experiment) | the unified top-right tag for real model outputs on the test scene |
| Our test scene = a test we built (not a paper experiment) | title of the popup in 01 |
| the white dot | the marker tracked through the whole pipeline |
| ground truth (GT) | known true values; the short form GT when space is tight |
| measured | tag on node-graph pictures: real model output |
| Illustration / Illustration (reduced dims) | corner tags: a computed example, not model output |
| Illustrative assumption | used in 08 |
| Paper Eq. (1) | equation number in the paper |
| paper Tab. 3 / Sec. 3.1 | table and section references in the paper |
| Method A / Method B | the two routes in 06 |
| regresses the point map directly | what the point head does |
| unproject / unprojection | depth + camera -> 3D points |
| camera 1 / 2 / 3 | the three cameras |
| view | MapAnything's word for an input image |
| frame attention / global attention (layer) | the paper's names for the two layer types |
| alternating attention | frame and global layers taking turns |
| one round | one frame layer + one global layer |
| head | output head: camera head / depth head / point head / track head |
| depth DPT / point DPT / track-feature DPT | the three DPT blocks |
| confidence Sigma | per-pixel confidence |
| normalized units | 1 = s |
| mean distance s | mean distance of scene points to camera 1 |
| ray depth | distance along the ray |
| z-depth | distance along the optical axis |
| up-to-scale | scale not fixed (the paper's word) |
| metric scale factor | the factor $m$ that scales the scene to meters |
| "$m$: the metric scale factor shared by all views in one forward pass:" / "scales the whole up-to-scale scene from the network into meters at once" | the two-line definition of $m$; 08 and 09 use the same sentence, so both translations must be identical |
| shared by all views | $m$ is common to all views |
| one $m$ per forward pass | a forward pass yields a single $m$ |
| shared across views | same meaning as above, caption wording |
| scale loss | the loss on $m$ |
| unit ray (direction) | ray direction of length 1 |
| pose | camera rotation + translation |
| intrinsics / extrinsics | camera matrices |
| quaternion | rotation as a 4-vector |
| field of view (FoV) | viewing angle |
| valid pixels | pixels whose ray hits the scene |
| per-pixel pairing | prediction and ground truth compared at the same pixel (06) |
| lower is better / higher is better | direction of a metric |
| Umeyama alignment (rotation + translation + scale) | similarity alignment before computing the error |
| the authors' explanation; not tested separately in the paper | tag for attributed claims |
| (inferred) / (our reading) | tags for interpretation |
| the scene script | the code that builds the test scene |
| ray-cast rendering | how the test images are rendered |
| toy scene | the synthetic scene |
| weights: CC-BY-NC 4.0 | model license line |
