"""pre version, Part 4 · Our own test: toy scene + real model output + errors; all numbers measured on our test scene are in this part. The source shots are not forked."""
from pre_cut import PreShot, Seg

LR = "Left: paper Tab. 3   Right: our test scene"     # corner tag: the English of the 06 tag

shot = PreShot("p4_our_test", [
    Seg("s01_task", 0.3, 9.5, caps=[
        (0.3, 2.6, "Part 4: our own test, not from the papers"),
        (2.6, 4.9, "A toy scene we rendered from 3 cameras"),
        (4.9, 7.2, "Ground truth known: cameras, depth, 3D points"),
        (7.2, 9.5, "We ran real VGGT-1B and MapAnything on it")]),
    # 9.5–11.8 ("Error = prediction vs ground truth") is cut; the lines in the floating window stay and appear during the fade
    Seg("s01_task", 11.8, 14.1, caps=[
        (11.8, 14.1, "White dot: one marker point we follow")]),
    Seg("s05_heads", 67.6, 73.0, caps=[
        (67.6, 70.2, "VGGT depth for camera 2 (real output)"),
        (70.2, 73.0, "Second channel: confidence Σ")],
        tags=[("Depth output by VGGT-1B",)]),
    Seg("s06_two_routes", 0.3, 7.1, caps=[
        (0.3, 2.5, "White dot via Method A (point head)"),
        (2.5, 4.9, "Ground-truth point (÷ s), for reference"),
        (4.9, 7.1, "Method B: depth + camera, unproject")],
        tags=[("VGGT-1B output (units ÷ s)",)]),
    Seg("s06_two_routes", 9.3, 13.7, caps=[
        (9.3, 11.5, "Multiply by predicted depth 0.925"),
        (11.5, 13.7, "Then back to camera 1's frame")],
        tags=[("VGGT-1B output (units ÷ s)",)]),
    Seg("s06_two_routes", 18.5, 21.1, caps=[(18.5, 21.1, "Per-pixel error: B lower without alignment")],
        tags=[(LR,)]),
    Seg("s06_two_routes", 36.6, 40.3, caps=[(36.6, 40.3, "After aligning rotation, translation, scale: A ≈ B")],   # 36.6: the "how the error is computed" popup only finishes at 36.45
        tags=[(LR,)]),
    Seg("s07_vggt_full", 53.7, 56.7, caps=[(53.7, 56.7, "Point-head output: mean distance 1.001, not meters")]),
    Seg("s09_ma_inputs", 75.6, 79.6, hold=0.4, caps=[           # last frame held 0.4 s, 2.2 s for each of the two sentences
        (75.6, 77.8, "MapAnything, images only: metric scale off 72%"),
        (77.8, 80.0, "With intrinsics + metric pose: 9.5%")],
        tags=[("MapAnything weights: CC-BY-NC 4.0",)]),
], tags=[("Our own test (not a paper experiment)", None, None)])
