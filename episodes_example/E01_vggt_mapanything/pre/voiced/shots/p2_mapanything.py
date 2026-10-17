"""pre version, Part 2 · How MapAnything works: factored representation -> optional inputs -> scale token (edit table: pre/storyboard_pre.md).
The numbers in 08 are illustrative values computed from the scene GT with the paper's formula, tagged "Illustration".
The "Encodings are added ..." caption of 09 sits at the fusion popup (47.0–52.2), not at 36.6–39.2 (where the popup is about the MLP); duration unchanged."""
from pre_cut import PreShot, Seg

ILL = "Illustration: values computed from scene ground truth"
M_DEF = "m: metric scale factor, scales the scene into meters"   # same sentence as in 08 / 09 of the full English version

shot = PreShot("p2_mapanything", [
    Seg("s08_mapanything", 0.3, 13.2, caps=[
        (0.3, 2.8, "MapAnything changes the output format"),
        (2.8, 5.4, "One point, four factors"),
        (5.4, 8.0, "ray: this pixel's unit direction"),
        (8.0, 10.6, "All rays of an image = the intrinsics"),
        (10.6, 13.2, "ray depth: distance along the ray")],
        tags=[(ILL, 5.4, 13.2)]),
    Seg("s08_mapanything", 15.8, 18.4, caps=[(15.8, 18.4, "Up to scale: relative distances, no units")],
        tags=[(ILL,)]),
    # Cut: 23.6–26.2 ("Along the ray, not z-depth") and 36.8–39.4 ("Only the scale is missing")
    Seg("s08_mapanything", 31.4, 36.8, caps=[
        (31.4, 34.0, "Ray × depth: point in camera 2's frame"),
        (34.0, 36.8, "Pose O, T: into camera 1's frame")],
        tags=[(ILL,)]),
    Seg("s08_mapanything", 39.4, 47.2, caps=[
        (39.4, 42.0, M_DEF),
        (42.0, 44.6, "Shared across views: one m for all"),
        (44.6, 47.2, "Training pushes m ≈ GT distance ÷ output distance")],
        tags=[(ILL,)]),
    Seg("s08_mapanything", 49.8, 52.4, caps=[(49.8, 52.4, "Relative-scale-only data still trains ray, depth, pose")],
        tags=[(ILL,)]),
    Seg("s09_ma_inputs", 0.3, 8.0, title=("MapAnything inputs", "Encoders · fusion · scale token · §3.1"), caps=[
        (0.3, 2.8, "Optional inputs: feed in what you know"),
        (2.8, 5.4, "Intrinsics as rays; depth and pose too"),
        (5.4, 8.0, "Each input has its own encoder → 1536-d")]),
    Seg("s09_ma_inputs", 47.0, 52.2, caps=[
        (47.0, 49.6, "Fusion: add the encodings to the patch tokens"),
        (49.6, 52.2, "Input missing? Just don't add it")],
        tags=[("Fusion formula: as written in the paper",)]),
    Seg("s09_ma_inputs", 54.8, 67.8, caps=[
        (54.8, 57.4, "View 1 gets a fixed embedding"),
        (57.4, 60.0, "Then append one scale token"),
        (60.0, 62.6, "It joins only the global layers"),
        (62.6, 65.2, "In global layers it sees all views"),
        (65.2, 67.8, "MLP, then exp → scale factor m")]),
    Seg("s09_ma_inputs", 70.2, 75.6, caps=[
        (70.2, 73.0, "Training: inputs randomly hidden"),
        (73.0, 75.6, "One model for any input combination")],
        tags=[("Illustration: switches flicker; probabilities are the paper's",)]),
])
