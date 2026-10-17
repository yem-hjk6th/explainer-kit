# STYLE: the checklist for every storyboard and every shot

Sources: what a demanding reviewer who watched every round asked for, the patterns shared by two
popular explainer videos that were broken down frame by frame, and the visual grammar of the library. Tick each item
per shot; if one cannot be met, say why in the storyboard's notes column.

## A. Content

- [ ] Plain language that does not read like generated text. No building or everyday-life analogies: show the
      principle, show the formula.
- [ ] Mechanism order: what goes in / what comes out → the formula → tensor shapes → one small numeric example →
      why it is designed this way.
- [ ] Terms that papers often redefine (frame attention, camera token, DPT head ...) stay in English; the first time,
      add the gloss in the caption language.
- [ ] Every symbol is defined wherever it appears (`$S$` = number of input images, next to it or in the caption),
      every time; never a bare D or f.
- [ ] Every number says what metric it is, compared with what, and whether that is good or bad; percentages name their
      denominator; each number has a source (paper § / table, code line, or our script).
- [ ] Computed vs illustrative: anything that can be computed is computed (projection, attention weights, reshaped
      shapes, back-projection) and asserted in `@shot.check`; purely illustrative pictures carry a corner tag
      ("Illustration"); a 2D drawing of a high-dimensional thing says "dimension-reduced sketch".

## B. Rhythm

- [ ] One caption per visual change: ≤ 12 words (`caption_len`) and no line wider than the frame, 2–3 s each, one
      caption on screen at a time. The renderer warns about both; the width is the real limit.
- [ ] No static slide: during every caption something appears, moves, or lights up.
- [ ] Picture first, formula second: the geometry / tensor picture is explained, then the formula appears at the top
      and lights up term by term (`Formula` segments + `state` / `part_alpha`), each term pointed back to its part of
      the picture (`part_box` + a line or the same color), with a caption like "this term is the ray we just drew".
- [ ] Paper cards as transitions and evidence: year, title, one-line contribution (text card; a first-page image only
      if you have the right to use it).
- [ ] One object followed through the whole episode (the same input images, the same token, the same 3D point), with a
      fixed color.
- [ ] No scene-per-chapter material changes (photoreal paper, metal, darkroom): highest cost, least understanding.

## C. Visual grammar (library defaults)

- 1080×1440, 30 fps, dark background with vignette and fine grain. Content in the central ~900×760 area; title / formula
  band at y 70–280; captions from y 1265 down. Pop-ups never enter the caption band (y ≥ 1250).
- Colors carry meaning and stay fixed for the episode:

  | Color | Used for |
  |---|---|
  | gold `GOLD` | the object being asked about: ray, 3D point, camera token |
  | cyan `CYAN` | results, projections, patch tokens, network outputs |
  | orange `ORANGE` | geometry that connects things (baseline, epipole), the current highlight box |
  | violet `VIOLET` | register tokens, auxiliary quantities |
  | white / grey | cameras, grids, module outlines |

- Glow only for the protagonist: at most 2 glowing elements (`cv.g`) per frame; the rest on `cv.s`.
- Motion: smoothstep (`ss`) easing; appear in 0.3–0.6 s; stagger elements by 0.2–0.5 s; reshapes move tokens in index
  order (`stagger`).
- Text: symbols with sub/superscripts always through `$...$` / `Formula` (UI fonts lack subscript glyphs). Labels
  26–34 px, formulas 44–84 px, captions 42 px.
- Formula color: warm gold (`style.MATH`) by default; the term being discussed glows once (`bump`), not permanently.

## D. Before delivery

- [ ] `check.json` all pass; every "computed" item in the storyboard has a matching check (including the displayed
      string).
- [ ] Contact sheet reviewed: captions not over the picture, labels inside the frame, tags present.
- [ ] Every number / shape in the storyboard is in `CLAIMS_to_verify.md`, and the verifier's must-fix items are applied.

## E. Reviewer feedback that became rules (apply to every episode)

- [ ] Every term and symbol on screen is explained on the spot. Assume the viewer does not know it (quaternion, DPT
      head, a hatted depth symbol, a scale factor, "why 1024 dimensions"). Use a pop-up side window with a small
      animation, then close it, so the main line is not interrupted.
- [ ] "Several numbers form one vector": draw the concatenation and its order (3 + 4 + 2 = 9 slots), or viewers read
      it as multiplication.
- [ ] Comparisons: which metric, unit, lower or higher is better; methods get explicit names ("method A: regress the
      point map directly" / "method B: depth + camera, then back-project"), never coined labels like "route 1".
- [ ] Formula and picture correspond: each term shares a color or a link line with its part of the picture and lights
      up with it; core formulas carry the paper's equation number.
- [ ] Pipeline diagrams: every node that lights up gets its own small picture, in one consistent style, none missing.
- [ ] Small text never crowds or overlaps; if it does not fit, move it outside the box with a leader line.
- [ ] Besides reproducing the paper's numbers, give an *explanatory* comparison: for the same scene, what each method
      needs as input, what it outputs, how its internal representation differs, when each fits.
- [ ] Our own tests (toy scene + real model inference) are introduced the first time they appear: this is our test,
      not the paper's experiment; where the ground truth comes from; what is compared. Afterwards one fixed phrase and
      one fixed corner tag.
- [ ] Every concrete number on screen states where it comes from ("0.93 = the depth head's prediction at the marked
      pixel").
- [ ] Show the symbol only; do not add small print about notation differences. Explain differences in meaning only.
- [ ] Do not summarize a concept as "a number": say what it does and its scope (a scale factor shared by all views
      that rescales the whole scene to meters). Use the same definition sentence in every shot.
- [ ] Delete transition sentences that carry no information.
- Kept because the reviewer explicitly liked them: shape switching with a small numeric example; dissecting one object
  term by term with side-by-side comparison; showing what a token sequence is made of; a clear overall order.
- Narration / voice-over: do not read the captions; per visual say what it shows, why, and 1–2 key points;
  3Blue1Brown-like conversational tone; word budget per segment (`references/NARRATION.md`).
- Review page: round 1 has a note box per caption line; from round 2 on, one general box + "what changed" per shot.

When your own reviewer gives feedback, add their general points as new items in `episodes/STYLE_local.md` in the same
form, and read both files at the start of every episode.
