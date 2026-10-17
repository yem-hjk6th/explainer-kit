# STORYBOARD: audience, depth, format, story patterns

The storyboard is the contract: shots implement it literally, the verifier checks it, the reviewer reads it. Write it
before any shot script. Full examples: `episodes_example/E02_depth/storyboard.md`,
`episodes_example/E01_vggt_mapanything/storyboard.md` (final versions; while you work, keep corrections visible as
described below).

## Audience → depth

Ask who watches and what they already know; write it at the top of §0. Then set these knobs explicitly:

| Knob | Self-study viewer new to the topic | Talk audience of grad students in the field |
|---|---|---|
| Prerequisites (quaternions, DPT head, inverse depth ...) | explained on screen, in pop-ups that open and close | named, not explained |
| Mechanism | full order: I/O → formula → shapes → numeric example → why | I/O → key formula → why; shapes only where they are the point |
| Numbers | small worked example with real numbers per mechanism | one or two numbers that carry the argument |
| Length | as long as needed (two real 10-shot episodes ran 6–8 min) | hard cap (e.g. ≤ 5 min); cut the trivial first |
| Structure | one line through the paper | separate parts, one theme each (our test / the architecture / the comparison) |
| Our own experiments | introduced carefully, tagged | kept in their own part, tagged |

The same approved material can serve both: make the detailed version first, then a shorter cut that loads the approved
shots (`episodes_example/E01_vggt_mapanything/pre/`). For a 20–60 s micro explainer: one mechanism, one numeric
example, no pop-ups unless a symbol is unavoidable.

## §0 of the storyboard

- Title, audience, target length, language(s), materials used (with section / line numbers).
- **The through-line object**: one input / token / 3D point followed in every shot, with its color.
- **Numbers table**: every number that will appear, computed by `shots/<ep>_numbers.py` (or `<ep>_common.py`) from the
  scene definition, copied at full precision then formatted; never typed by hand.
- **Notation**: every symbol, its definition sentence (reused verbatim in every shot), letters reserved.
- **Computed vs illustrative** policy and the corner tags you will use (e.g. ground truth / our model run / fake
  prediction built from ground truth / read from code / derivation / interpretation / paper Tab. N).
- Real model runs (optional): which shots, what to install (repo + commit, weights + size + license, new env), GPU
  memory cap. Do not install before the user agrees.

## One section per shot

```
### E03-02 Projection: a 3D point lands on a pixel
**S · 18 s**  (definition → formula → numbers → why divide by Z)

| # | Caption | What changes on screen | Computed / illustrative |
|---|---|---|---|
| 1 | <≤ 12 words, one frame width> | <which element appears / moves / lights, where, with which label> | computed |
| 2 | ... | ... | illustrative (tag "Illustration") |

Formulas: $u = f_x X/Z + c_x$ ... (LaTeX, exactly as on screen; symbols defined)
Why: one or two sentences giving the design reason the viewer should leave with.
Sources: paper §3.1 eq. (2); our script `shots/e03_numbers.py`; official code `file.py` L120–131.
```

Rules for the table: one row = one caption = one visual change; 2–3 s per row, so shot length ≈ rows × 2.5–3.5 s;
the caption text is final wording (shots copy it verbatim into a `CAPS` list); every number in a row also appears in
CLAIMS. Mark rows changed after verification with `[verify fix n]` and keep the old text struck through.

Close the storyboard with: §2 dependencies (shared helpers, library elements missing → `elements_needed.md`);
§3 what to cut first if over time (each cut with a reason: "does not affect understanding because ..."); §4 questions
for the reviewer.

## CLAIMS_to_verify.md

One row per fact / number / shape / formula with source, how to check, and a label (measured / computed / inferred /
paper). Format and the verification protocol: `references/VERIFY.md`. Group rows by section (A numbers, B mechanism,
R real-model results, ...) so a verifier can be scoped to a section.

## Story patterns that work (from breaking down two popular explainer videos)

- One caption per visual change, 10–15 characters, 2–3 s; no static slide.
- Geometry first, formula second, lit term by term with "this term is the arrow we just drew".
- Paper cards (year, title, one-line contribution) as transitions and as evidence.
- Dimension-reduction disclaimer whenever a high-dimensional thing is drawn in 2D.
- One object carried through the whole video.
- Credibility from your own experiments: concrete parameter counts, step counts, loss values; the picture is the
  result of a real computation, so the picture *is* the verification.
- Avoid: chapter-by-chapter photoreal scene changes (expensive, decorative, little understanding).
- Typical density in the references: ~0.5 caption per second, a hard cut every 15–20 s, everything else continuous
  transformation.

## Concepts that animate well

Pick what a static slide only shows if the viewer imagines it moving: pinhole projection and dolly zoom, epipolar
lines sweeping through the epipole, triangulation and bundle adjustment converging, RANSAC iterations with a live inlier
count, volume rendering weights along a ray, Gaussian splats converging, attention sharpening across layers, point
tracks with occlusion, the scale ambiguity of monocular depth with a least-squares scale/shift fit, token reshapes
between per-frame and global attention.
