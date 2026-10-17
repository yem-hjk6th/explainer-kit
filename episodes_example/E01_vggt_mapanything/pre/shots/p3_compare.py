"""pre version, Part 3 · Side by side: input / output / internal representation / units / what each is good for + paper Tab. 2(a).
10 uses the fork in pre/src (it hides the white-dot coordinates, 1.001, 72.3% / 9.5%); the internal representation is written in the two column boxes, so captions do not repeat it."""
from pre_cut import PreShot, Seg

shot = PreShot("p3_compare", [
    Seg("s10_compare_pre", 0.3, 14.6, title=("Side by side", "VGGT vs MapAnything"), caps=[
        (0.3, 2.6, "Same images into both models"),
        (2.6, 5.0, "Input: VGGT needs only images"),
        (5.0, 7.4, "MapAnything can also take known geometry"),
        (7.4, 9.8, "VGGT: camera, depth, point map, one each"),
        (9.8, 12.2, "Points twice: P directly, or from D + g"),
        (12.2, 14.6, "MapAnything: four factors")]),
    Seg("s10_compare_pre", 17.0, 21.6, caps=[
        (17.0, 19.2, "VGGT unit: mean distance, not meters"),
        (19.2, 21.6, "MapAnything multiplies by m: meters")]),
    Seg("s10_compare_pre", 24.2, 36.6, caps=[
        (24.2, 26.6, "Our reading: images only, either works"),
        (26.6, 29.0, "Need point tracking? VGGT"),
        (29.0, 31.4, "Calibration, pose, or meters? MapAnything"),
        (31.4, 33.6, "Paper Tab. 2(a): two views, images only"),
        (33.6, 36.6, "VGGT 43.2%, MapAnything 57.5% (higher is better)")],
        tags=[("MapAnything paper Tab. 2(a) (VGGT via depth + camera route)", 31.4, 36.6),
              ("Models: VGGT-1B (non-commercial), MapAnything (CC-BY-NC 4.0)", 34.6, 36.6)]),
])
