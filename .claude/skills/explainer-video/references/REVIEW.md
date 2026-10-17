# REVIEW: rounds with a human reviewer

The reviewer's time is the bottleneck of the whole pipeline. Make every round cheap for them: one page, videos that
play without sound, a note box where they need it, and only the changes to look at from round 2 on.

## Layout

```
<EP>/review.html                      redirect to the newest round (`review.html?list` lists all rounds)
<EP>/review/round1/review.html        generated
<EP>/review/round1/feedback.md        the reviewer's notes, saved verbatim by you
<EP>/review/round2/changes.md         what round 2 changed (you write it before generating the page)
<EP>/review/round2/review.html, feedback.md ...
```

`python -I -B -X utf8 tools/review_page.py <EP> --round N` (refuses without `--round` and tells you the next number;
never overwrites a page it did not write; those are moved to `<EP>/_moved/`).

- **Round 1**: per shot the video, the storyboard section, numbered caption lines, and a note box per line.
- **Round 2+**: one general note box at the top; per shot only the video, "what changed this round" (from
  `changes.md`) and the previous round's notes folded (from `round{N-1}/feedback.md`). A strip of shot ids at the top,
  changed ones highlighted.
- "Copy all notes" exports text whose first line is `<ID> round N review notes`; drafts are kept per round in the
  browser. Videos never autoplay and need no sound (reviewers often watch over a remote-desktop link).
- Shot mp4s are overwritten on every re-render: an old round's page plays the *current* video.
- The page shows the source-language render (`<stem>.mp4`, `meta.json`, `check.json`). For a translated render
  (`<stem>_<code>.mp4`), call `concat.shot_status(ep_dir, shot, lang)` with the language code in a small wrapper.

## Each round, in order

1. Write `review/roundN/changes.md`: `## <ID>-NN` per changed shot with 2–5 bullet points; text before the first
   heading goes to the top of the page; shots without a section show "not changed this round".
2. Re-render changed shots → `tools/concat.py <EP>` → `tools/review_page.py <EP> --round N` →
   optionally `tools/render_check.py <EP>/review/roundN/review.html`.
3. Hand over: the page path, the video path, 3–6 lines on what changed and what you want them to look at.
4. When notes come back: save them as `review/roundN/feedback.md`. Lines that start with a shot id (`E03-02: ...` or a
   caption-level id `E03-02-4: ...`) are attached to that shot on the next page; other lines count as general notes.
5. Answer every note in `STATUS.md` ("note → what I changed / why not"). New explanations or numbers → new CLAIMS
   rows → verifier on those rows only. General preferences → `episodes/STYLE_local.md`.

## Reading feedback well

- The reviewer writes fast and colloquially; restate each note before acting (a misread name, e.g. a typo for a
  model component, is common). Ask through one short message if two readings are plausible.
- "I don't understand X" means: X needs an on-screen definition or a pop-up, not a longer caption.
- Praise is information too: record what was liked (it goes into STYLE §E "kept") and do not "improve" those shots.
- If a note conflicts with a verified fact, keep the fact and explain; do not re-introduce an error to satisfy a note.
- Length grows with each round; when it matters, propose cuts with a reason each ("does not affect understanding
  because ...") instead of cutting silently.
