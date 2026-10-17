"""E01 English pre cut, voice-over script: one source (SEGS below) that generates narration.json, NARRATION_pre_en.md and NOTES_pre_en.html.

Segment time = time in the cut video: the parts are joined end to end (tools/concat.py, no cross-fade between parts).
Part lengths and times inside a part come from storyboard_pre.md (its "new time" column).
The last segment p4.5 is counted 2.5 s longer (a hold after the last sentence). The voiced cut is timed separately by ../voiced/timing.py; its length there is authoritative.
Word cap max_words = floor(min(2.0 x length, 2.2 x (length - 0.5))): 2.0 words/s is the cap chosen for this cut,
2.2 words/s (about 130 words/min) is the estimated speaking rate of a live presenter, and the 0.5 s is "finish speaking 0.5 s before the picture ends".
Words are counted as spoken words (see spoken_words): numbers are spelled out (72% = seventy two percent = 3),
VGGT / MapAnything / DPT / MLP = 2, DINOv2 = 3, hyphens split (per-pixel = 2). This is higher than a count by spaces, which is the safe side.
Each segment is split into sentences; each sentence has one anchor = the time its subtitle / picture element appears in the final video (subtitle time on the review page + part start).
place() puts the sentences: working backwards it finds the latest start of each sentence; start = max(previous end + 0.25 s, min(anchor, latest start)); a last sentence that ends after t1 - 0.5 counts as overrun.
tts = the text for ryan: numbers spelled as English words, a lone m written as a capital M, DINOv2 written as DINO v2.
Run: python -I -X utf8 build_narration.py (no third-party dependencies).
"""
import html
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACE_TALK = 2.2    # words/s, estimated pace of a live presenter
PACE_RYAN = 2.45   # words/s, measured for the ryan voice at the normal setting
CAP = 2.0          # words/s, word cap for this cut
TAIL = 0.5         # seconds left before the picture ends
PAUSE = 0.25       # pause between sentences (seconds)
END_HOLD = 2.5     # hold on the last frame (seconds), added to the last segment only

# order and length of the parts (storyboard_pre.md)
# Cut version: p1 92.15, p2 62.6, p4 41.8, film 238.85 s; times in 0.01 s
PARTS = [("c0", 3.0), ("c1", 2.0), ("p1", 92.15), ("c2", 2.0), ("p2", 62.6),
         ("c3", 2.0), ("p3", 31.3), ("c4", 2.0), ("p4", 41.8)]
VIDEO = 238.85
PART_TITLE = {"c0": "Opening", "c1": "Part 1 · How VGGT works", "c2": "Part 2 · How MapAnything works",
              "c3": "Part 3 · Side by side", "c4": "Part 4 · Our own test"}

# (id, part, t0 in part, t1 in part, on_screen, why, key_points, sentences [(text, anchor in part, or None = right after the previous one)], hint, refs, backup sentences)
# All times inside a segment are relative to the start of its part; the anchor is the start time of the matching subtitle on the review page.
# Backup sentences: used only if the TTS output runs over; they match the main text sentence by sentence (same anchor).
# ---------- Script ----------
# Spoken style: wording and sentence form only (rhetorical questions, colon announcements and imperatives become plain
# statements, meaning unchanged). Where the picture carries a key caption and no sentence would play, the sentence is the
# spoken form of that caption, no new facts.
# Time inside a segment = seconds of this part (0.01 s); anchor = start time of the matching caption in the cut.
SEGS = [
    ("c0", "c0", 0.0, 3.0,
     "Title card: VGGT × MapAnything, four-part outline.",
     "Opens the talk and names the two models.",
     ["Four parts: VGGT, MapAnything, side by side, our own test."],
     [("VGGT and MapAnything.", 0.0)],
     "Opening card: say the two names and move on; no need to read the outline.",
     [], None),
    ("c1", "c1", 0.0, 2.0,
     "Part 1 card: How VGGT works (Wang et al., CVPR 2025).",
     "Part 1 covers only the VGGT paper's architecture.",
     ["Paper architecture only; our own numbers wait until Part 4."],
     [("First, VGGT.", 0.0)],
     "Part 1 covers only the VGGT paper's architecture; all numbers measured on our test scene wait until Part 4.",
     [], None),

    # ---------- Part 1 ----------
    ("p1.1", "p1", 0.0, 5.0,
     "Eq. (1): N images in, four outputs per image, in one forward pass.",
     "Sets the input / output contract that the rest of Part 1 unpacks.",
     ["Eq. (1): N images → (g, D, P, T) for each image."],
     [("One forward pass: images in, four outputs per image.", 0.0)],
     "Eq. (1): N images in, four outputs g, D, P, T per image, in one forward pass.",
     [], None),
    ("p1.2", "p1", 5.0, 15.0,
     "The four outputs: camera g, depth map D, point map P, tracking features T.",
     "These four outputs are what the rest of Part 1 explains.",
     ["P is expressed in camera 1's frame.",
      "The pictures are real VGGT-1B output on our test images; the numbers come in Part 4."],
     [("Camera g, depth D.", 5.0),
      ("Point map P, in camera one's frame.", 9.8),
      ("Tracking features T.", 12.2)],
     "The four outputs; P is in camera 1's coordinate frame. The depth maps and point clouds on screen are real model output; the numbers come in Part 4.",
     [], None),
    ("p1.3", "p1", 15.0, 25.25,
     "Each image → 1369 DINOv2 patch tokens (1024-d), plus 5 extra tokens: 1 camera, 4 register.",
     "Before attention, we need to know what the tokens are.",
     ["DINOv2 (ViT-L): 37 × 37 = 1369 patch tokens, 1024-d.",
      "The camera token collects camera info; the 4 register tokens are dropped at the end."],
     [("Each image becomes DINOv2 patch tokens, plus five extras.", 15.0),
      ("One of the extras is the camera token.", 20.05)],
     "Each image = 1369 patch tokens + 1 camera token + 4 register tokens; the camera token later turns into the camera parameters.",
     [], None),
    ("p1.4", "p1", 25.25, 32.25,
     "Example token #799; image 1 gets its own camera / register tokens → the reference camera.",
     "Explains why P lives in camera 1's frame.",
     ["Image 1 uses its own camera / register tokens; the other images share one set.",
      "That is how the network knows which camera is the reference."],
     [("Just an example token.", 25.25),
      ("Image one gets its own extra tokens, which mark it as the reference.", 27.85)],
     "Frame 1 uses its own set of camera / register tokens, so the network knows which camera is the reference. #799 is just an example token.",
     [], None),
    ("p1.5", "p1", 32.25, 42.35,
     "Tokens of the 3 images as 3 rows; frame attention on each row alone; shape (3, 1374, 1024).",
     "Alternating attention is the core of VGGT; this is its first half.",
     ["Frame attention: tokens attend only within their own image.",
      "Shape (3, 1374, 1024), with 1374 = 1 + 4 + 1369."],
     [("The core of VGGT is alternating attention.", 32.25),
      ("Frame attention keeps each image's row separate.", 34.75),
      ("Then we join the rows.", 39.55)],
     "First half of alternating attention: frame attention runs on each image on its own, shape (3, 1374, 1024); then the three rows are joined end to end.",
     [], None),
    ("p1.6", "p1", 42.35, 47.0,
     "Global attention over one sequence; shape (1, 4122, 1024).",
     "The second half lets tokens from different images interact.",
     ["Same tokens, only reshaped: 3 × 1374 = 4122."],
     [("Global attention sees one long sequence: every token from every image.", 42.35),
      ("Same tokens, just reshaped.", 44.8)],
     "Global attention: the tokens of the three images form one sequence of length 4122; it is only a reshape, the tokens themselves do not change.",
     [], None),
    ("p1.7", "p1", 47.0, 54.45,
     "Attention table: frame = 3 diagonal blocks, global = whole table; paper Tab. 5.",
     "Shows what each attention type can see, and why VGGT alternates.",
     ["Paper Tab. 5: at the same parameter count, alternating has the lowest error."],
     [("Frame attention fills only the diagonal blocks.", 47.0),
      ("Global attention fills the whole table.", 49.2),
      ("In Table 5, alternating has the lowest error at the same parameter count.", 51.8)],
     "Frame attention fills only the diagonal blocks, global attention fills the whole table. Paper Tab. 5: at the same parameter count, alternating attention has the lowest error.",
     [], None),
    ("p1.8", "p1", 54.45, 64.35,
     "After 24 rounds the camera token → camera head → 9 numbers: 3 translation + 4 quaternion + 2 FoV.",
     "Answers how a camera comes out of tokens.",
     ["24 alternating rounds, then read the camera token.",
      "9 = 3 + 4 + 2, concatenated (slot order as in the official code)."],
     [("For the camera, we take the camera token.", 54.45),
      ("The camera head outputs 9 numbers: translation, rotation, field of view.", 56.95)],
     "After 24 rounds of alternating attention, take the camera token; the camera head outputs 9 numbers: translation 3 + quaternion 4 (the rotation) + field of view 2. Slots on screen follow the official code's order.",
     [], None),
    ("p1.9", "p1", 64.35, 71.55,
     "Patch tokens → DPT heads → per-pixel maps; DPT reads 4 of the 24 rounds.",
     "The camera came from one token; dense maps come from the patch tokens.",
     ["DPT reads 4 of the 24 rounds.",
      "Output: a per-pixel map (depth, points)."],
     [("For per-pixel maps, the patch tokens go to DPT heads instead.", 64.35)],
     "Per-pixel maps (depth, point map) go patch tokens → DPT heads; DPT reads 4 of the 24 rounds.",
     [], None),
    ("p1.10", "p1", 71.55, 83.95,
     "Two routes to points: Method A point head; Method B depth + camera, unproject; paper Tab. 3 (ETH3D).",
     "VGGT gives points twice, so which one do we use?",
     ["Paper Tab. 3 (ETH3D): Method B has lower error.",
      "Authors' view: subtasks are easier to learn (not tested separately in the paper)."],
     [("Method A reads the point map.", 71.55),
      ("Method B unprojects depth with the camera.", 73.75),
      ("In Table 3 of the paper, B has lower error.", 78.35)],
     "Two routes: A uses the point map directly, B unprojects depth + camera. Paper Tab. 3 (ETH3D): B has lower error; \"subtasks are easier to learn\" is the authors' view, not tested separately in the paper.",
     [], None),
    ("p1.11", "p1", 83.95, 92.15,
     "Training divides ground truth by its mean distance s → no meters; a scaled scene gives identical images.",
     "This is the gap Part 2 picks up: MapAnything outputs meters.",
     ["VGGT §3.4: ground truth normalized by its mean distance s.",
      "Scale the scene and the cameras together: the images don't change."],
     [("Training divides ground truth by its mean distance,", 83.95),
      ("so the output has no meters.", 86.75),
      ("Scaling the scene and the cameras together gives the same images.", 89.35)],
     "VGGT training divides ground truth by the mean distance s, so the output has no meters; scale the scene and cameras together and the images do not change. This is the gap MapAnything fills in Part 2.",
     [], None),
    ("c2", "c2", 0.0, 2.0,
     "Part 2 card: How MapAnything works (Keetha et al., arXiv 2509.13414).",
     "Part 2 is MapAnything's design, set against VGGT.",
     ["Keetha et al., arXiv 2509.13414."],
     [("Next, MapAnything.", 0.0)],
     "Part 2: MapAnything, the paper this talk is about.",
     [], None),

    # ---------- Part 2 ----------
    ("p2.1", "p2", 0.0, 10.3,
     "One point = four factors; ray = the pixel's unit direction; all rays = the intrinsics.",
     "MapAnything's change starts with the output format.",
     ["All rays of an image = its intrinsics.",
      "Values here are illustrations computed from the scene's ground truth."],
     [("MapAnything changes the output format.", 0.0),
      ("It splits each point into four factors.", 2.5),
      ("First, a unit ray for each pixel.", 5.1),
      ("All the rays together are the intrinsics.", 7.7)],
     "MapAnything splits a 3D point into four factors; the first is each pixel's unit ray, and all the rays of an image together are the intrinsics.",
     [], None),
    ("p2.2", "p2", 10.3, 20.9,
     "Ray depth: distance along the ray, up to scale; ray × depth → point in camera 2's frame; pose O, T → camera 1's frame.",
     "Shows how the factors assemble into a point.",
     ["Depth is measured along the ray, up to scale.",
      "Ray × depth, then the pose, gives the point in camera 1's frame."],
     [("Depth runs along the ray, up to scale.", 10.3),
      ("Ray times depth gives a point.", 15.5),
      ("The pose moves it into camera one's frame.", 18.1)],
     "Depth along the ray, only up to relative scale; ray × depth gives a point in the camera frame, pose O, T moves it into camera 1's frame.",
     [], None),
    ("p2.3", "p2", 20.9, 31.3,
     "m: one metric scale factor shared by all views; training pushes m ≈ GT ÷ output distance.",
     "m is exactly what VGGT's output lacks.",
     ["One m shared by all views → the scene in meters.",
      "Data with only relative scale still trains ray, depth, pose."],
     [("Then m, one number for all views, scales the scene to meters.", 20.9),
      ("Data without meters still trains ray, depth, pose.", 28.7)],
     "m = one metric scale factor shared by the whole scene; data with only relative scale can still train ray, depth and pose.",
     [], None),
    ("p2.4", "p2", 31.3, 44.2,
     "Optional inputs: intrinsics as rays, depth, pose; each has its own encoder → 1536-d; added to the patch tokens.",
     "The same factors can also go in as inputs.",
     ["Each input type has its own encoder (1536-d).",
      "Fusion: add the encodings (formula as in the paper); a missing input is simply not added."],
     [("The factors can also be inputs.", 31.3),
      ("Intrinsics go in as rays.", 33.8),
      ("Each gets encoded and added to the tokens.", 36.4),
      ("A missing input is simply not added.", 41.6)],
     "Optional inputs: intrinsics (as rays), depth and pose can all be fed in; each has its own encoder to 1536-d and is added to the patch tokens; a missing input is not added.",
     [], None),
    ("p2.5", "p2", 44.2, 57.2,
     "View 1 gets a fixed embedding; one scale token is appended, joins only the global layers, sees all views; MLP + exp → m.",
     "Answers where m comes from.",
     ["The scale token joins only the global layers, so it sees all views.",
      "MLP, then exp → m."],
     [("View one gets a fixed embedding.", 44.2),
      ("The scale m comes from one scale token, which joins only the global layers,", 46.8),
      ("so it sees every view.", 52.0),
      ("An MLP and an exponential turn it into m.", 54.6)],
     "Where m comes from: append a scale token that joins only the global layers, so it sees all views; finally MLP + exp gives m. View 1 gets a fixed embedding as the reference.",
     [], None),
    ("p2.6", "p2", 57.2, 62.6,
     "Training: inputs randomly hidden → one model for any input combination.",
     "Explains how one model handles every input mix.",
     ["Probabilities on screen are the paper's; the flickering switches are an illustration."],
     [("Training hides inputs at random, so one model handles any input mix.", 57.2)],
     "Training hides inputs at random, so one model can take any combination of inputs.",
     [], None),
    ("c3", "c3", 0.0, 2.0,
     "Part 3 card: Side by side.",
     "Puts the two designs next to each other.",
     ["Inputs · outputs · internals · when to use."],
     [("Side by side.", 0.0)],
     "Part 3: the two models side by side.",
     [], None),

    # ---------- Part 3 ----------
    ("p3.1", "p3", 0.0, 7.1,
     "Same images into both; VGGT needs only images; MapAnything can also take known geometry.",
     "Inputs are the first difference.",
     ["MapAnything can also take known geometry: intrinsics, depth, pose."],
     [("For input, VGGT takes images only.", 0.0),
      ("MapAnything also takes known geometry.", 4.7)],
     "Input: VGGT needs only images; MapAnything can also take known geometry (intrinsics, depth, pose).",
     [], None),
    ("p3.2", "p3", 7.1, 18.9,
     "Outputs: VGGT camera, depth, point map (points twice); MapAnything four factors; units: mean distance vs meters.",
     "Outputs and units are the second difference.",
     ["VGGT gives points twice: P directly, or from D + g.",
      "VGGT's unit = mean distance; MapAnything multiplies by m → meters."],
     [("VGGT outputs points twice, as the point map or as depth plus camera.", 7.1),
      ("MapAnything outputs four factors.", 11.9),
      ("And only MapAnything gives meters.", 14.3)],
     "Output: VGGT gives points twice (point map, and depth + camera); MapAnything gives four factors. Units: VGGT's is the mean distance; MapAnything multiplies by m to get meters.",
     [], None),
    ("p3.3", "p3", 18.9, 26.1,
     "Our reading: images only → either works; tracking → VGGT; calibration, pose or meters → MapAnything.",
     "Turns the comparison into a choice.",
     ["This is our reading (marked on screen), not a claim from either paper."],
     [("In our reading, with images only, either model works.", 18.9),
      ("For tracking, use VGGT.", 21.3),
      ("With calibration or poses, or to get meters, use MapAnything.", 23.7)],
     "This is our reading (marked on screen): with only images either works; for point tracking pick VGGT; with calibration, poses, or a need for metric results pick MapAnything.",
     [], None),
    ("p3.4", "p3", 26.1, 31.3,
     "MapAnything paper Tab. 2(a), two views, images only: VGGT 43.2%, MapAnything 57.5% (higher is better).",
     "The paper's own head-to-head number.",
     ["Metric τ: share of points whose distance is within 3% of the truth (both normalized); higher is better.",
      "VGGT is evaluated through its depth + camera route."],
     [("In MapAnything's own Table 2, with images only, it scores higher.", 26.1)],
     "Paper Tab. 2(a): two views, images only, τ (share of points within 3% distance error; higher is better) VGGT 43.2%, MapAnything 57.5%. Measured by the MapAnything authors themselves.",
     [], None),
    ("c4", "c4", 0.0, 2.0,
     "Part 4 card: Our own test — a toy scene we built + real model outputs; not a paper experiment.",
     "Separates our measurements from the papers' claims.",
     ["From here on: our own test, not a paper experiment."],
     [("Our own test.", 0.0)],
     "Part 4: our own test, not a paper experiment.",
     [], None),

    # ---------- Part 4 ----------
    ("p4.1", "p4", 0.0, 11.5,
     "Our own test: a toy scene rendered from 3 cameras, ground truth known; real VGGT-1B and MapAnything; white dot = marker point.",
     "Opens Part 4 by saying plainly that this is our own test.",
     ["Ground truth known: cameras, depth, 3D points (3 cameras).",
      "White dot: one marker point we follow."],
     [("This is our own test, not from the papers.", 0.0),
      ("We rendered a toy scene with known ground truth,", 2.3),
      ("and ran the real models on it.", 6.9),
      ("The white dot is one marker point we follow.", 9.2)],
     "Say it clearly in the first sentence: this is our own test. Toy scene, three cameras, full ground truth; the models are the real VGGT-1B and MapAnything. The white dot is the marker point we follow.",
     [], None),
    ("p4.2", "p4", 11.5, 16.9,
     "VGGT-1B depth for camera 2 (real output); second channel: confidence Σ.",
     "First real output: what VGGT's depth looks like.",
     ["Real model output, not an illustration."],
     [("This is VGGT's depth for camera two, with its confidence.", 11.5)],
     "VGGT's real depth output for camera 2; the second channel is the confidence Σ.",
     [], None),
    ("p4.3", "p4", 16.9, 28.1,
     "White dot via Method A (point head) vs ground truth (÷ s); Method B: predicted depth 0.925, back to camera 1's frame.",
     "Runs Part 1's two routes on one real pixel.",
     ["Method B: predicted depth 0.925 at the white dot, then back to camera 1's frame.",
      "Ground truth is divided by s so the units match VGGT's."],
     [("Method A reads the white dot off the point map.", 16.9),
      ("Method B unprojects the depth using the predicted camera.", 21.5)],
     "Follow the white dot through Part 1's two routes: A reads the point map directly; B multiplies the predicted depth 0.925 by K⁻¹[u,v,1] (a direction with z = 1, not a unit ray), then goes back to camera 1's frame. Ground truth is divided by s so the units match VGGT's.",
     [], None),
    ("p4.4", "p4", 28.1, 34.4,
     "Per-pixel error: B lower without alignment; after aligning rotation, translation, scale: A ≈ B. Left: paper Tab. 3, right: our scene.",
     "Puts our result next to the paper's Tab. 3.",
     ["Ours: without alignment B is lower; after rotation + translation + scale alignment A ≈ B.",
      "Left bars = paper Tab. 3, right = our scene: different data and metric, not directly comparable."],
     [("On our scene, without alignment, B has lower error.", 28.1),
      ("After alignment, the two are about the same.", 30.7)],
     "Our test scene: without alignment B has lower error; after rotation + translation + scale alignment A ≈ B. Left is paper Tab. 3, right is our scene; data and metric both differ, so the sizes cannot be compared directly.",
     [], None),
    ("p4.5", "p4", 34.4, 41.8,
     "VGGT point-head mean distance 1.001 (not meters); MapAnything metric scale off 72% (images only) → 9.5% (with intrinsics + metric pose).",
     "Closes the loop on meters from Parts 1 and 2.",
     ["VGGT: mean distance 1.001, normalized units, not meters.",
      "MapAnything: 72% → 9.5% when given intrinsics + metric poses."],
     [("VGGT's points are not in meters.", 34.4),
      ("With images only, MapAnything's scale is 72% off.", 37.4),
      ("With intrinsics and metric poses, it is 9.5% off.", 39.6)],
     "The mean distance of VGGT's points is 1.001, in normalized units, not meters; MapAnything with images only is off by 72% in metric scale, 9.5% with intrinsics + metric pose. In the voiced cut, the film holds at least 2.0 s after the last word.",
     [], None),
]

SPECIAL = {"vggt": 2, "mapanything": 2, "dpt": 2, "mlp": 2, "dinov2": 3, "3d": 2}
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n):
    if n < 21:
        return 1
    if n < 100:
        return 1 if n % 10 == 0 else 2
    if n < 1000:
        return 2 + (int_words(n % 100) if n % 100 else 0)
    return int_words(n // 1000) + 1 + (int_words(n % 1000) if n % 1000 else 0)


def int_spell(n):
    assert 0 <= n < 1000, n
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    return ONES[n // 100] + " hundred" + (" " + int_spell(n % 100) if n % 100 else "")


def token_words(tok):
    t = tok.strip(".,:;?!()\"“”")
    if not t:
        return 0
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?(%?)", t)
    if m:
        return int_words(int(m.group(1))) + (1 + len(m.group(2)) if m.group(2) else 0) + (1 if m.group(3) else 0)
    base = re.sub(r"[’']s$", "", t).lower()
    if base in SPECIAL:
        return SPECIAL[base]
    return 1


def spoken_words(text):
    return sum(token_words(p) for tok in text.split() for p in tok.split("-"))


def to_tts(text):
    s = text.replace("DINOv2", "DINO v2")
    # a lone m is written as the capital letter M: spelled "em" it is heard as "them" ("give em", "That's em")
    s = re.sub(r"\bm\b", "M", s)
    s = s.replace("VGGT", "V G G T")                 # spelled letter by letter: otherwise it is read as "Vi, GGT"
    s = re.sub(r"\bIntrinsics\b", "In-trin-sics", s)  # a sentence-initial "Intrinsics" in the normal spelling is unclear

    def num(mt):
        out = int_spell(int(mt.group(1)))
        if mt.group(2):
            out += " point " + " ".join(ONES[int(d)] for d in mt.group(2))
        return out + (" percent" if mt.group(3) else "")

    s = re.sub(r"(?<![A-Za-z])(\d+)(?:\.(\d+))?(%?)", num, s)   # leave digits after letters (as in v2) alone
    assert not re.search(r"(?<![A-Za-z])\d", s), s
    return s


def place(sents, t0, t1, pace):
    """sents = [(words, anchor or None)]; returns (start time of each sentence, seconds of overrun)."""
    durs = [w / pace for w, _ in sents]
    latest, end = [0.0] * len(sents), t1 - TAIL
    for i in range(len(sents) - 1, -1, -1):
        latest[i] = end - durs[i]
        end = latest[i] - PAUSE
    starts, prev_end = [], None
    for i, (_, a) in enumerate(sents):
        lo = t0 if i == 0 else prev_end + PAUSE
        s = lo if a is None else max(lo, min(a, latest[i]))
        starts.append(s)
        prev_end = s + durs[i]
    return starts, max(0.0, prev_end - (t1 - TAIL))


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def build():
    offs, acc = {}, 0.0
    for name, dur in PARTS:
        offs[name] = acc
        acc += dur
    video = round(acc, 2)
    assert abs(video - VIDEO) < 1e-6, video
    rows, prev_end = [], 0.0
    for k, (sid, part, r0, r1, on_screen, why, keys, sents, hint, refs, alt) in enumerate(SEGS):
        last = k == len(SEGS) - 1
        t0 = round(offs[part] + r0, 2)
        t1 = round(offs[part] + r1 + (END_HOLD if last else 0.0), 2)
        assert abs(t0 - prev_end) < 1e-6, f"{sid}: gap/overlap at {t0} (prev end {prev_end})"
        prev_end = t1
        dur = round(t1 - t0, 2)
        max_words = math.floor(min(CAP * dur, PACE_TALK * (dur - TAIL)) + 1e-9)
        ss = []
        for txt, a in sents:
            anchor = None if a is None else round(offs[part] + a, 2)
            assert anchor is None or t0 <= anchor < t1, (sid, txt)
            ss.append(dict(text=txt, tts=to_tts(txt), words=spoken_words(txt), anchor=anchor))
        assert ss[0]["anchor"] is not None, sid
        if alt:
            assert len(alt) == len(ss), sid
            alt = [dict(text=t, tts=to_tts(t), words=spoken_words(t)) for t in alt]
        sim = {}
        for name, pace in (("talk", PACE_TALK), ("ryan", PACE_RYAN)):
            st, over = place([(s["words"], s["anchor"]) for s in ss], t0, t1, pace)
            lead = max([0.0] + [s["anchor"] - x for s, x in zip(ss, st) if s["anchor"] is not None])
            sim[name] = dict(starts=st, over=over, lead=lead)
        text = " ".join(s["text"] for s in ss)
        rows.append(dict(id=sid, part=part, t0=t0, t1=t1, dur=dur, on_screen=on_screen, why=why,
                         key_points=keys, text=text, sentences=ss, words=sum(s["words"] for s in ss),
                         max_words=max_words, plain=len(text.split()), hint=hint,
                         alt=alt, hold=END_HOLD if last else 0.0, sim=sim))
    assert abs(prev_end - (video + END_HOLD)) < 1e-6, prev_end
    return rows, video


def write_json(rows):
    out = []
    for r in rows:
        d = {k: r[k] for k in ["id", "t0", "t1", "on_screen", "why", "key_points", "text", "words", "max_words"]}
        d["tts"] = " ".join(s["tts"] for s in r["sentences"])
        d["sentences"] = r["sentences"]
        if r["hold"]:
            d["hold_end_s"] = r["hold"]
        if r["alt"]:
            d["alt_sentences"] = r["alt"]
        d["hint"] = r["hint"]
        out.append(d)
    (HERE / "narration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def write_md(rows, video):
    total = rows[-1]["t1"]
    L = ["# E01 English pre cut, voice-over script (read by ryan / by a live presenter)", "",
         f"Generated by `build_narration.py`; do not edit by hand. Final video `..\\..\\render\\E01_pre_en.mp4` (cut version, {video:.2f} s); times = time in the final video (seconds). "
         "The voiced cut is timed separately by `..\\voiced\\timing.json` (idle stretches sped up, overruns held); the length there is authoritative.",
         "Cap = floor(min(2.0 x length, 2.2 x (length - 0.5))); words are counted as spoken words (numbers spelled out, VGGT / MapAnything / DPT / MLP count 2, DINOv2 counts 3, hyphens split); the figure in brackets is the count by spaces.",
         "In the narration column, `@s` = the sentence's anchor: the time its subtitle / picture element appears (subtitle time on the review page + part start).",
         f"Placement rule (for the voice-over; duration estimated as words / pace): work backwards to the latest start of each sentence; start = max(previous end + {PAUSE} s, min(anchor, latest start)); the last sentence must end before t1 - 0.5 s. "
         "\"Max lead\" = the largest number of seconds a sentence has to start before its anchor in order to finish in time; \"overrun\" = seconds that still do not fit under this rule.",
         ""]
    for r in rows:
        if r["part"].startswith("c"):
            L += ["", f"## {PART_TITLE[r['part']]}", "",
                  "| Seg | Video time (s) | Length | On screen | Narration (@ anchor) | Words / cap | ryan 2.45: max lead / overrun | presenter 2.2: max lead / overrun |",
                  "|---|---|---|---|---|---|---|---|"]
        say = " ".join((f"@{s['anchor']:.1f} " if s["anchor"] is not None else "") + s["text"] for s in r["sentences"])
        if r["hold"]:
            say += f" (hold {r['hold']} s at the end)"
        flag = "" if r["words"] <= r["max_words"] else " **OVER**"
        sim = " | ".join(f"{r['sim'][n]['lead']:.1f} / " + (f"**{r['sim'][n]['over']:.1f}**" if r["sim"][n]["over"] > 1e-9 else "0")
                         for n in ("ryan", "talk"))
        L.append(f"| {r['id']} | {r['t0']:.1f}–{r['t1']:.1f} | {r['dur']:.1f} | {r['on_screen']} | {say} | "
                 f"{r['words']} ({r['plain']}) / {r['max_words']}{flag} | {sim} |")
    tw, tp = sum(r["words"] for r in rows), sum(r["plain"] for r in rows)
    over = [r["id"] for r in rows if r["words"] > r["max_words"]]
    talk = tw / PACE_TALK
    L += ["", "## Statistics", "",
          f"- {len(rows)} segments; {sum(len(r['sentences']) for r in rows)} sentences; planned voiced cut about {total:.1f} s (an estimate: video {video:.1f} s + {END_HOLD} s hold at the end; the final length is the total in ../voiced/timing.json).",
          f"- Total spoken words {tw} (by spaces {tp}), sum of caps {sum(r['max_words'] for r in rows)}; at 2.2 words/s about {talk:.0f} s ({talk / total:.0%} of the video), at ryan's 2.45 words/s about {tw / PACE_RYAN:.0f} s.",
          f"- Segments over the word cap: {', '.join(over) if over else 'none'}.",
          "- Segments that overrun when placed sentence by sentence (pauses included): " + "; ".join(
              f"{n} " + (", ".join(f"{r['id']} {r['sim'][k]['over']:.1f} s" for r in rows if r['sim'][k]['over'] > 1e-9) or "None")
              for n, k in (("ryan", "ryan"), ("presenter", "talk"))) + ".",
          "- Segments where ryan has to start >= 0.5 s before the anchor: " + (", ".join(f"{r['id']} {r['sim']['ryan']['lead']:.1f} s" for r in rows if r['sim']['ryan']['lead'] >= 0.5) or "None") + ".",
          "- No new facts: every claim and number in the narration comes from the checked subtitles; the Part 3 advice follows the \"Our reading\" label on screen; paper numbers carry their table number (Tab. 5, Tab. 3, Tab. 2).", "",
          "## TTS text (sentences written differently from the narration)", ""]
    for r in rows:
        for s in r["sentences"]:
            if s["tts"] != s["text"]:
                L.append(f"- {r['id']}: {s['text']} -> `{s['tts']}`")
    L += ["", "## Backup sentences (used only if the TTS output runs over; sentence by sentence, same anchor)", ""]
    for r in rows:
        if r["alt"]:
            L.append(f"- {r['id']} ({sum(a['words'] for a in r['alt'])} words): " + " ".join(a["text"] for a in r["alt"]))
    L.append("")
    (HERE / "NARRATION_pre_en.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


CSS = """
:root { --bg:#ffffff; --ink:#111111; --dim:#555555; --line:#d8d8d8; --acc:#1a5fb4; --hint:#6b5a2e; --warn:#b00020; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI", Arial, "Microsoft YaHei", sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
header h1 { font-size:24px; margin:8px 0 4px; }
header p { color:var(--dim); font-size:15px; margin:2px 0; }
h2 { font-size:20px; margin:28px 0 8px; padding-top:8px; border-top:3px solid var(--acc); color:var(--acc); }
.card { border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; break-inside:avoid; page-break-inside:avoid; }
.top { display:flex; flex-wrap:wrap; gap:6px 14px; align-items:baseline; font-size:15px; color:var(--dim); }
.top .id { font-weight:700; color:var(--acc); font-size:17px; }
.top .time { font-weight:700; color:var(--ink); font-size:20px; font-variant-numeric:tabular-nums; }
.screen { font-size:15px; color:var(--dim); margin:6px 0 8px; }
.say { margin:6px 0 10px; }
.say p { font-size:30px; line-height:1.35; font-weight:600; margin:0 0 6px; overflow-wrap:anywhere; }
.say .at { font-size:14px; font-weight:400; color:var(--dim); font-variant-numeric:tabular-nums; margin-right:8px; }
.alt { font-size:15px; color:var(--dim); margin:2px 0 8px; }
ul.keys { margin:4px 0 6px; padding-left:20px; font-size:15px; }
.hint { font-size:15px; color:var(--hint); margin:4px 0 0; }
.hold { font-weight:700; color:var(--warn); }
@media (max-width:480px) { .say p { font-size:24px; } .top .time { font-size:18px; } }
@media print { main { max-width:none; padding:0; } .say p { font-size:22pt; } h2 { break-after:avoid; } }
"""


def write_html(rows, video):
    e = html.escape
    tw, total = sum(r["words"] for r in rows), rows[-1]["t1"]
    out = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           "<title>E01 pre narration notes</title>",
           # This page has no formulas; the empty stub only lets tools/render_check.py (which waits for MathJax.startup.promise) run, and loads no external file.
           "<script>window.MathJax = { startup: { promise: Promise.resolve() } };</script>",
           f"<style>{CSS}</style></head><body><main>",
           "<header><h1>VGGT × MapAnything · pre narration notes</h1>",
           f"<p>Video: E01_pre_en.mp4 · {mmss(video)} (+{END_HOLD} s hold at the end) · {len(rows)} cards · {tw} spoken words · paced for 2.2 words/s, done 0.5 s before each cut.</p>",
           "<p>Big text = what to say; the small time before each line = when to start it. Small list = the point to land. Brown line = delivery hint, not read aloud.</p></header>"]
    for r in rows:
        if r["part"].startswith("c"):
            out.append(f"<h2>{e(PART_TITLE[r['part']])}</h2>")
        lines = "".join(f'<p><span class="at">{mmss(st)}</span>{e(s["text"])}</p>'
                        for s, st in zip(r["sentences"], r["sim"]["talk"]["starts"]))
        out += ['<section class="card">',
                f'<div class="top"><span class="id">{e(r["id"])}</span>'
                f'<span class="time">{mmss(r["t0"])} – {mmss(r["t1"])}</span>'
                f'<span>{r["dur"]:.1f} s</span><span>{r["words"]} / {r["max_words"]} words</span>'
                + (f'<span class="hold">hold {r["hold"]} s at the end</span>' if r["hold"] else "") + "</div>",
                f'<p class="screen">On screen: {e(r["on_screen"])}</p>',
                f'<div class="say">{lines}</div>']
        if r["alt"]:
            out.append(f'<p class="alt">Shorter backup (if it runs long): {e(" ".join(a["text"] for a in r["alt"]))}</p>')
        out += ['<ul class="keys">' + "".join(f"<li>{e(k)}</li>" for k in r["key_points"]) + "</ul>",
                f'<p class="hint">{e(r["hint"])}</p>', "</section>"]
    out.append("</main></body></html>")
    (HERE / "NOTES_pre_en.html").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    rows, video = build()
    write_json(rows)
    write_md(rows, video)
    write_html(rows, video)
    for r in rows:
        flag = "" if r["words"] <= r["max_words"] else "  <-- OVER WORDS"
        sy, sr = r["sim"]["talk"], r["sim"]["ryan"]
        print(f"{r['id']:6} {r['t0']:6.1f}-{r['t1']:6.1f} {r['dur']:5.1f}s  {r['words']:2}/{r['max_words']:2}  "
              f"ryan lead {sr['lead']:.1f} over {sr['over']:.2f} | talk lead {sy['lead']:.1f} over {sy['over']:.2f}{flag}")
    print("total words", sum(r["words"] for r in rows), "over:", [r["id"] for r in rows if r["words"] > r["max_words"]])
