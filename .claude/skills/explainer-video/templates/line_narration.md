# Line {LINE}: narration script + speaker notes for {VIDEO}

Read `{PROJECT_ROOT}/_prompts/COMMON.md` and `references/NARRATION.md`. Your folder: `{NARRATION_DIR}` (write only
here). This line writes text only: no rendering, no GPU, no sub-agents. Context ≤ 200k.

## The reviewer's request (verbatim)
"{quote}"
Voice: {speaker / "the presenter reads it live"}. Speaking rate: {measured words/s, or "unknown → 2.0 words/s budget"}.

## Materials (read-only)
- The finished video `{VIDEO}` and its cut table (which original shot, which seconds, where in the final video).
- Verified content: `CLAIMS_to_verify.md`, the on-screen captions, verifier reports.
- {Other context, e.g. how the talk is framed}

## Write
1. Segments = continuous stretches about one thing (≈ 5–15 s; each title card is one segment), with t0 / t1 in the
   final video (0.1 s).
2. Per segment: what the visual shows, why we show it, 1–2 key points; conversational, short sentences, "we", one
   takeaway; only the one number that matters. Words ≤ 2.0 × seconds, speech ends 0.5 s before the segment ends.
3. Sentences with anchors (when the thing mentioned appears), a `tts` column (numbers / symbols spelled out),
   alternative short sentences for tight segments, `hold_end` if the end needs room.
4. No new facts; paper numbers name their table; the first mention of our own test says so.
Deliver `narration.json`, `NARRATION_<lang>.md` (table + word counts vs limits), `NOTES_<lang>.html` (one card per
segment; self-contained; check it renders). Message the coordinator (≤ 8 lines + paths) and stop. A verifier checks
facts, sync, word counts and style.
