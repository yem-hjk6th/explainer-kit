"""pre version, Part 1 · How VGGT works: paper architecture only (edit table: pre/storyboard_pre.md).
No measured numbers appear: 01 / 05 / 06 use the forks in pre/src; the real model-output "frames" stay, with the corner tag pointing to Part 4.
Part 1–2 captions do not mention "the white dot" (the Part 4 test marker); 799 -> 2173 is presented as a neutral example.
Cuts: in 04 the "799 -> 2173" caption (15.2–18.0) and the "63% to frame 2" caption (25.9–28.6) are dropped;
the cut points move to 15.05 / 18.4 / 28.75, so the cross-fade does not show the white-dot highlight (from 15.2), the 799 -> 2173 label (fading out until 18.3) or the 63% bar (fading out until 28.6)."""
from pre_cut import PreShot, Seg

PIC = "Picture: VGGT-1B output on our test images (numbers in Part 4)"

shot = PreShot("p1_vggt", [
    Seg("s01_task_pre", 14.1, 19.1, caps=[
        (14.1, 16.5, "VGGT: one forward pass, images → 3D"),
        (16.5, 19.1, "Eq. (1): N images in, four outputs per image")]),
    Seg("s01_task_pre", 21.5, 31.5, caps=[
        (21.5, 23.9, "g: camera parameters"),
        (23.9, 26.3, "D: depth map"),
        (26.3, 28.7, "P: point map, in camera 1's frame"),
        (28.7, 31.5, "T: features for point tracking")], tags=[(PIC,)]),
    Seg("s02_patchify", 12.45, 15.0, caps=[                     # starts at 12.45, so our test scene's GT does not flash during the fade
        (12.45, 15.0, "DINOv2 (ViT-L): 37×37 = 1369 patch tokens, 1024-d")],
        tags=[("Illustration: numbers in a vector", 12.6, 15.0)]),
    Seg("s03_tokens", 0.3, 15.0, title=("Tokens per image", "VGGT §3.3"), caps=[
        (0.3, 2.8, "Each image gets 5 extra tokens"),
        (2.8, 5.4, "1 camera token: collects camera info"),
        (5.4, 8.0, "4 register tokens: dropped at the end"),
        (8.0, 10.6, "Example token: #799 within its image"),
        (10.6, 13.0, "Image 1 gets its own token set"),
        (13.0, 15.0, "So the network knows the reference camera")],
        tags=[("Illustration: token color = type; small camera color = frame no.",),
              # only the paper's 794 is mentioned: the 2173 caption is cut, so 2168 has no counterpart
              ("Ordered as in the official code (the paper writes 794)", 8.0, 15.0)]),
    Seg("s04_alt_attn", 0.3, 15.05, title=("Alternating attention", "VGGT §3.2"), caps=[
        (0.3, 2.8, "Tokens of the 3 images, one row each"),
        (2.8, 5.4, "Frame attention: each row on its own"),
        (5.4, 7.6, "Shape (3, 1374, 1024)"),
        (7.6, 10.4, "Join the three rows end to end"),
        (10.4, 12.85, "Global attention: all tokens together"),     # ends at 12.85; the next caption is still >= 2.2 s
        (12.85, 15.05, "Shape (1, 4122, 1024)")],
        tags=[("Not to scale: 4 tokens drawn per frame", 0.3, 15.05)]),
    Seg("s04_alt_attn", 18.4, 23.2, caps=[
        (18.4, 20.6, "Frame attention: only the 3 diagonal blocks"),
        (20.6, 23.2, "Global attention fills the whole table")],
        tags=[("Example values", 18.4, 23.2)]),
    Seg("s04_alt_attn", 28.75, 31.4, caps=[
        (28.75, 31.4, "Same parameter count: alternating has the lowest error")],
        tags=[("Paper, Tab. 5", 28.75, 31.4)]),
    Seg("s05_heads_pre", 0.3, 10.2, title=("Output heads", "Camera head · DPT head · VGGT §3.3"), caps=[
        (0.3, 2.8, "After 24 alternating rounds, take the camera token"),
        (2.8, 5.4, "Camera head: 9 numbers per image"),
        (5.4, 7.8, "3 translation + 4 quaternion + 2 field of view"),
        (7.8, 10.2, "Concatenated end to end: 3 + 4 + 2 = 9")],
        tags=[("Slot order as in the official code (the paper writes [q, t, f])", 3.8, 10.2)]),
    Seg("s05_heads_pre", 46.6, 53.8, caps=[
        (46.6, 48.8, "Patch tokens go to the DPT heads"),
        (48.8, 51.2, "DPT reads 4 of the 24 rounds"),
        (51.2, 53.8, "DPT: tokens → per-pixel map")]),
    Seg("s06_two_routes_pre", 0.3, 2.5, title=("Point cloud: two ways", "A: point head · B: depth + camera heads · VGGT §3.1, §4.3"),
        caps=[(0.3, 2.5, "Method A: point head regresses points")]),
    Seg("s06_two_routes_pre", 4.9, 7.1, caps=[(4.9, 7.1, "Method B: depth + camera, then unproject")]),
    Seg("s06_two_routes_pre", 13.7, 19.5, caps=[                # up to 19.5: the paper bar is still growing at 18.5 (the right-hand bar of our test scene is already hidden in the fork)
        (13.7, 16.1, "Unproject all images → one point cloud"),
        (16.1, 19.5, "Paper (ETH3D, Tab. 3): Method B has lower error")],
        tags=[(PIC, 13.7, 16.1), ("Paper, Tab. 3 (ETH3D)", 16.1, 19.5)]),
    Seg("s06_two_routes_pre", 40.3, 42.5, caps=[(40.3, 42.5, "Authors' view: subtasks are easier to learn")]),
    Seg("s07_vggt_full", 45.5, 48.3, caps=[(45.5, 48.3, "Training: ground truth divided by mean distance s")],
        tags=[("Formula: VGGT paper §3.4",)]),
    Seg("s07_vggt_full", 56.9, 62.3, caps=[                     # starts at 56.9
        (56.9, 59.5, "So VGGT's output has no meters"),
        (59.5, 62.3, "Scaled scene + cameras → identical images")],
        tags=[("Illustration",)]),
])
