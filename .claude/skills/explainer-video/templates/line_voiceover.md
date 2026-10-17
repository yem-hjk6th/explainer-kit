# Line {LINE}: TTS voice-over for {VIDEO}

Read `{PROJECT_ROOT}/_prompts/COMMON.md` and `references/TTS.md`; the worked example is
`episodes_example/E01_vggt_mapanything/pre/voiced/` (read its script headers). Your folder:
`{VOICED_DIR}` (write only here; copy the cut shells you need into it, never edit approved shells or the master).

## Decided by the reviewer
Voice {speaker}, pace instruction {normal}, narration `{NARRATION_JSON}` (verified), end hold {s} s.

## Compute (overrides COMMON)
GPU memory fraction {f} (check `nvidia-smi` first; release when done); CPU {MASK}, ≤ {P} render processes.

## Steps
1. Generate {K} candidates per sentence from the `tts` column (`gen_voiced.py`).
2. Pick with whisper medium: key words all present, no pause > 0.9 s, shortest exact match (`pick.py`); regenerate
   failures, re-spell in `tts` if a word fails twice.
3. Time (`timing.py` → `timing.json`): the piecewise-linear map of TTS.md §3 (picture sped up ≤ 2.5× or slowed
   ≥ 0.6× between anchors, then a frame hold), or its simple variant if the picture's rhythm must not change; end
   hold as decided.
4. Voiced variant: times mapped through the knots, captions = spoken sentences (≥ 1.0 s each, no overlap), everything
   else unchanged; render and concat silent (`render_all.ps1`).
5. Mix: sample-accurate placement, two-pass loudnorm −16 LUFS / −2.0 dBTP, AAC; `.srt` (`mix.py`).
6. Self-check (`check_voiced.py`) → a short `CHECK.md` in your folder: whole-track and per-sentence transcription vs
   script, silence table, midpoint sheet, hold sheet, length, loudness, what needs a human ear.
Message the coordinator (≤ 8 lines + paths) and stop. A verifier listens via transcription before the reviewer.
