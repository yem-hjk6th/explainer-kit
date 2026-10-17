# Verification protocol

Every fact, number, shape and formula that appears on screen or in narration is checked by an independent verifier before the episode ships. The verifier is a fresh agent that did not write the content. This file defines what the coordinator hands over (CLAIMS file), when to launch, what to check, how the report looks, and the rules for applying fixes.

Launch prompt template: `templates/verifier_prompt.md`.

## 1. The CLAIMS file

`CLAIMS_to_verify.md` lives in the episode folder. It is written together with the storyboard, one row per on-screen fact, number, shape or formula. If a storyboard line shows it, a row exists for it. Rows are grouped by section (toy scene, paper claims, code facts, model-run results, comparisons).

Columns:

| column | content |
|---|---|
| id | stable id, never reused (`A-3`, `C-12`, `R-5`); shot references go in the claim or a notes column |
| claim | the exact wording or number as it will appear on screen, at displayed precision |
| source | paper section / table / equation number, or code file + line range, or "our script `name.py`" |
| how to check | one line telling the verifier what to recompute or open |
| label | `paper`, `code`, `measured`, `computed`, `inferred` (see below) |

Labels:

- `paper`: stated in the paper text, table or equation.
- `code`: read from the official code or config (the paper may disagree; code wins for "what the model does").
- `measured`: a real model run on our own inputs, from saved outputs.
- `computed`: calculated by our own script from the toy-scene parameters (has asserts).
- `inferred`: our reading, not stated by the source. It must be tagged as such on screen ("our reading").

Generic example (invented numbers):

| id | claim | source | how to check | label |
|---|---|---|---|---|
| A-1 | f = 320 / tan(40 deg) = 381.4 px | scene script, L12 | recompute from W and FoV | computed |
| A-2 | pixel (200, 150) back-projects to ray (-0.105, -0.052, 1) | scene script, L30 | K^-1 [u v 1]^T at full precision, then round | computed |
| A-3 | "depth 1.80 x 0.99 = 1.78" | storyboard shot 06 | multiply the displayed numbers | computed |
| C-1 | backbone has 24 blocks, width 1024 | paper Sec. 3.2; official `model.py` L57-58 | open code, count blocks, read the weight header | paper + code |
| C-2 | head output is 3 + 4 + 2 = 9 numbers | paper Sec. 3.1; `pose_enc.py` L40-55 | read code; order may differ from the paper | code |
| R-1 | mean abs-rel 2.1 % on valid pixels, lower is better | saved `run_a.npz` | recompute from npz, independent GT | measured |
| R-2 | "scale error 70 %" = (pred mean dist / GT mean dist) - 1 | saved npz | recompute; state denominator on screen | measured |
| I-1 | "the model probably learns scale from the loss" | our reading of Sec. 4 loss | check the loss is consistent; label stays | inferred |

After a verification round, add a "corrections" table at the top (see section 6); old wording is struck through, never deleted.

## 2. When to launch a verifier

1. After the storyboard and CLAIMS exist, before any rendering. A wrong number costs least to fix here.
2. After model runs: when `measured` rows exist and the outputs are saved (npz/json). The verifier re-derives from the saved files; it does not re-run the model.
3. After the on-screen text is final (final video frames): round 2 checks that every correction actually reached the screen, not only the CLAIMS or storyboard, and looks at layout.
4. After the narration script exists: narration must say nothing the verified captions do not say, and must be in sync.
5. After a translation pass: terms, numbers and hedges survive translation.

The verifier is always a fresh agent. Use the strongest available model for rounds that judge claims. The coordinator does not verify its own content, and an agent that wrote a fix does not verify that fix.

## 3. Kinds of checks

### 3.1 Numbers
- Recompute every on-screen number from full-precision values with an independent script that does not import the episode's own scene or analysis code. Copy scene parameters by hand from the source file.
- Rounding: round once, at the end. Carry-over errors (0.5385 shown as 0.539 when full precision gives 0.5385 -> 0.538 at three places) are the most common finding; check each displayed value against the full-precision one.
- Arithmetic shown on screen must multiply out at the displayed precision. "1.984 x 0.996 = 1.975" fails (gives 1.976). Fix by showing more digits on both factors, and make the episode's assert use the same displayed strings, with tolerance no looser than half the last digit.
- Different quantities that look similar (a ray direction with z = 1 versus a unit ray; z-depth versus distance along the ray) must not share one number. Check each use.
- Counts: tokens, patches, parameters. Distinguish an index (a position) from a count. Recount from the code or the weight-file header (read the header, do not load the model).
- Sub-pixel points: a value read off a depth map at the nearest pixel centre differs from the exact value. Mark the point as sub-pixel or choose an integer pixel.
- Ranges quoted on screen ("within +-1.6") must contain the actual extremes.

### 3.2 Shapes, layers and structure versus code
- Check layer counts, widths, channel counts, head counts, iteration counts against the official repository at a named commit. Cite file and line range. Code beats the paper's prose; when they disagree, say so on screen ("paper: X; code: Y").
- Check what is shared and what is not: one module iterated several times versus several modules; which tokens attend across images and which stay per image; what is dropped before the head.
- Read the weight shapes (safetensors header) to confirm layer counts and parameter shares.
- A bundled copy of the code inside another repository is not the official one: compare logic (AST level), and cite the official line numbers.
- Check ordering conventions (concatenation order of outputs, token order), and that "paper order" versus "code order" is labelled.
- Check what a figure's tensor really is (notation in the paper versus actual shape in code).
- If a source file could not be obtained, say "could not verify" in the report rather than guessing.

### 3.3 Re-deriving real model runs
- Use only the saved outputs. Verify the files first: keys, dtypes, shapes, no NaN, internal consistency (pose matrices equal the rotation-from-quaternion of the stored quaternion, depth times ray equals points, and so on). State that an npz cannot prove it came from a GPU run.
- Rebuild ground truth with your own script. Reproduce each reported metric to the printed digits; list the number, not just "holds".
- Note the metric definition exactly: pixel set (valid pixels versus full image including background), alignment (none, scale-only, rigid, similarity), normalisation, which head or branch produced the number.
- A comparison that looks unfair (different pixel sets, different normalisation) must be recomputed under one protocol.
- Confidence maps: do not claim "high confidence means low error" unless the rank correlation supports it.
- When a result looks worse than expected (an extra input made things worse), separate "used wrongly" from "real behaviour": check each input convention against the library docs and code, check the output side for evidence the input was read correctly, and list the re-runs that would be needed to find the cause. Do not run them unless asked. The screen wording becomes "observed; cause not determined".

### 3.4 Input and convention mismatches
- Channel order, normalisation (mean/std, range [0,1] versus [0,255]), image size and resize or crop behaviour.
- Coordinate frames: camera-from-world versus world-from-camera, axis convention, which frame is the reference, pixel-grid convention (integer grid versus +0.5 centres).
- Units: metres versus normalised scale; angles in radians versus degrees; field-of-view order (vertical/horizontal) and which one is quoted.
- Quaternion order, and whether the quaternion is normalised before use (an unnormalised one gives a wrong angle if computed as 2 arccos(w)).
- A scale-free model prints scene-normalised values, not metres. If the screen shows metric values in an output slot, label them as ground truth or show the normalised values.

### 3.5 Tags: illustrative versus real
- Every visual is one of: `illustrative` (invented numbers, schematic shapes), `ground truth` (our scene), `measured` (model output), `paper` (reproduced from a paper). Check that each badge matches the row label.
- A paper figure must not carry a "measured on our scene" badge, even briefly during a transition. Keep both badges, or put our numbers in their own labelled box.
- A schematic cross-section that rotates two things into one plane is fine if the screen says so; angles between the two things are then not true.

### 3.6 Comparisons
Every comparison must state: metric, unit, direction (lower or higher is better), what is compared against what, and for a percentage its denominator ("72 % above ground truth", not "72 % error"). Check:
- paper numbers carry table/figure ids and the conditions (number of views, inputs given, who ran it);
- our numbers say "our scene, N images";
- inference from one scene is not generalised to the paper's claim;
- the same metric is defined in one sentence and used the same way everywhere (define terms before first use: error measures, F1, units of distance measures).

### 3.7 Symbols and wording
- No symbol used for two things in one episode (focal length versus field of view; scale versus deviation; index versus multiplier). Variables on screen are defined at first use, with units.
- A sentence that sounds natural can be a new, wrong claim ("has no unit" while the formula divides by metres). Check every sentence added during a fix against its source.
- Mechanism statements (how the module works) are checked against code, not against an analogy.

### 3.8 Narration and translation
- Every spoken statement must appear in, or follow from, the verified captions or CLAIMS. Narration that is wider than the caption (drops a condition, drops "our scene") is "needs change".
- Word counts: recount with an independent counter and check the per-segment limit formula.
- Timing: sample frames at the time each sentence is spoken; the words should not run ahead of the visual by more than about a second. Estimate speech time from syllable density, not only word count; measure the real audio length after generation.
- Pronunciation traps: single letters read as variables, product names with digits.
- Reviewer-facing note cards (hints for the presenter) are checked too.
- After translation: numbers, hedges ("our reading"), units and term choices unchanged.

### 3.9 Layout (from final frames)
- Grab frames from the final video, not from stills (stills are often older than the final render). Use contact sheets only to locate suspects, then inspect the actual frame.
- Look for overlaps, text running past a box or off canvas, labels sitting on connector lines, readouts crowded together, text fading out before it was readable (full-brightness duration under about a second), colour-map direction contradicting the legend (near bright or far bright; two maps in one episode with opposite conventions).
- Check that shown dynamics (fades, highlights) allow time to read the key number.

### 3.10 Did the fix reach the screen
In round 2, for each round-1 correction, find it in the final frame or video, not only in CLAIMS or storyboard. Also grep the storyboard for residual unstruck old values and for stale documentation that no longer matches the screen.

## 4. Verdict vocabulary and report structure

Per claim, one of:
- `holds` (state the recomputed value, or the source location);
- `holds, wording needs change` (the fact is true but the sentence misleads or the precision is wrong);
- `needs change` (give exact replacement wording);
- `wrong` (give the correct value and where it is used);
- `cannot verify` (say what source was missing).

Report layout, in this order:
1. Summary, at most about 300 words: what was recomputed and how, the count of must-fix items, the most serious one or two mechanism issues.
2. Must-fix list: numbered, each with location (shot, row id), old wording, exact new wording, and one-line evidence.
3. Suggestions: optional or minor items, layout nits.
4. Details by CLAIMS section, one line per row: id, verdict, recomputed value or source (file + line, or paper section). Add small extras (an extra caveat, a better line reference).
5. "What an expert might pick on" list: the screen statements most likely to draw a challenge.
6. Round 2 only: table of round-1 fixes with "on screen: yes / partly / no", then new problems introduced by the fixes.
7. Tooling: where the verifier's scripts and extracted frames are kept, and a statement that the episode folder was not modified.

Evidence format: `path/to/file.py L120-134`, `paper Sec. 3.4`, `Table 3 row 2`. Every verdict that is not "holds" cites evidence.

## 5. Rules

1. Strike-through, never silent deletion. A corrected claim keeps the old wording as ~~old~~ followed by the new wording, in CLAIMS and storyboard, with the correction id.
2. A repair is itself a new claim. Whoever wrote the fix must not verify it, and every new sentence or number added during a fix gets its own row.
3. At most two verification rounds per item. Round 1 checks claims; round 2 checks the final on-screen result plus the new rows from round 1. After round 2, fixes are applied exactly as worded and the coordinator spot-checks them with grep and a few recomputes. They are not re-verified; mark them "round 2 fix, not re-verified" in CLAIMS.
4. The verifier's recompute scripts and extracted frames are kept next to the report (a `scripts/` folder). They are the audit trail.
5. The verifier writes no file other than its report and its scripts, and does not modify the episode.
6. The coordinator also spot-checks a few of the verifier's own numbers (pick the ones the screen depends on most), because verifiers can slip too.
7. Overturning an earlier verifier verdict needs a new independent check with evidence, not an assertion.
8. Verification output can include "could not determine"; that is acceptable. Replace a claim with a hedged one ("observed; cause not determined") rather than guess.
9. Do not delete a statement just because it is hard to verify; first ask whether it is still worth saying. Remove it only if it is false; if unsure, flag it for the reviewer.

## 6. Spot-check after fixes (coordinator checklist)

- [ ] Grep the storyboard, CLAIMS and shot scripts for every old value that was corrected; only struck-through occurrences remain.
- [ ] Each correction appears in all places it is used (storyboard line, CLAIMS row, shot script, caption, narration, notes cards).
- [ ] Arithmetic strings on screen come from the same formatted variables the asserts check; asserts are no looser than half the last displayed digit.
- [ ] Two or three of the verifier's recomputed numbers re-derived by hand or by a quick script.
- [ ] Code line citations point at the official repository at the named commit, not at a bundled copy.
- [ ] Badges (illustrative / ground truth / measured / paper) match the row labels in every shot, including transitions.
- [ ] New sentences added by fixes each have a CLAIMS row with source and label.
- [ ] Definitions appear before first use; no symbol is used for two meanings.
- [ ] Final video: frames grabbed at the times of each changed caption (not stills); no overlaps; colour-map directions agree with legends.
- [ ] Narration re-checked against the changed captions; word counts and timing re-run.
- [ ] CLAIMS corrections table updated (id, what changed, where it landed, "not re-verified" where applicable).
- [ ] Documentation (storyboard sections, notes) updated to match the screen.
