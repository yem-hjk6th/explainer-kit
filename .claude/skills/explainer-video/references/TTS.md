# TTS: local voice-over from a verified narration script

Input: `narration.json` written and checked per `references/NARRATION.md` (segments with time ranges, sentences with
anchors, a `tts` spelling column, optional alternative sentences, `hold_end`). Output: a video whose audio is the
voice-over, whose on-screen captions are the spoken sentences, and a self-check report.

Worked example (all scripts): `episodes_example/E01_vggt_mapanything/pre/voiced/` — read the docstrings of
`timing.py`, `shots/voiced.py` and `check_voiced.py` first.
Engine used: Qwen3-TTS `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (local, GPU, ~5 GB at bf16), built-in speakers.
Checking: faster-whisper on CPU (int8).

## 0. Pick a voice (before the whole episode)

`tools/tts/gen_samples.py` renders the same 3 passages with several speakers × 2 pace instructions;
`tools/tts/post.py` transcribes them back and writes `samples.html` (players + words/s + how much longer than the
picture the speech is). Let the reviewer pick by ear. Typical speaking rates of the built-in voices are 2–3 words/s,
so reading every on-screen caption aloud usually takes several times the picture time; write the narration per segment
(NARRATION.md), not per caption.

## 1. Generate (`gen_voiced.py`, tts env, GPU)

- One generation per *sentence*, never per paragraph; several candidates per sentence (3 by default): sampling is
  stochastic and the duration of the same text can differ a lot between takes.
- Pace instruction (`INSTRUCT` in `gen_voiced.py`): the worked example uses `Speak at a natural, steady pace, like a
  lecturer explaining to graduate students. No laughter, no filler sounds.` A faster alternative is `Speak quickly and
  fluently, like a confident lecturer. No laughter, no filler sounds.` The instruction changes the speaking rate, so it
  changes sentence durations and therefore the timing in `timing.py`: pick one before generating the whole episode,
  and regenerate and re-time if you change it later.
- Feed the `tts` column, not the display text: numbers spelled as words ("one thousand three hundred seventy-four";
  digits were misread badly), symbols spelled as they should sound, acronyms as letters if they are said as letters.
- A word the voice keeps mumbling gets a phonetic spelling in `tts` only (e.g. hyphenated syllables like
  `In-trin-sics`); the display text and captions stay correct.
- Trim leading / trailing silence (`librosa.effects.trim(top_db=35)` + 30 ms margin), normalize to -18 dBFS RMS with
  a -1 dBFS peak cap. Cap GPU memory (`torch.cuda.set_per_process_memory_fraction`), release it when done.

## 2. Pick (`pick.py`, CPU)

Transcribe every candidate with whisper **medium** (small mishears technical words — "rows" as "roast", "depth" as
"death" — so it cannot be the judge). Accept a candidate only if every key word (numbers, terms, single-letter
symbols) is present, no pause inside the sentence exceeds 0.9 s, and the rate is plausible (0.8–5 words/s). Take the
shortest candidate whose transcript matches exactly (WER 0), else the shortest with WER ≤ 0.12. Sentences with no
acceptable candidate go to `regen.txt` → regenerate those only (`gen_voiced.py --only <ids>`), changing the `tts`
spelling if the same word fails twice. To override the rule for a few sentences (e.g. after listening for a rising end),
pass `--plan <file.json> --only <ids>` with a list of `{"sid", "proposed_k"}`; the chosen candidate must still pass. `voicecheck.py` holds the normalization (lowercase, punctuation off, numbers to
words, joined single letters, term variants) and the key-word comparison.

## 3. Lay out in time (`timing.py`)

The example uses a piecewise-linear time map: every sentence starts when its anchor picture appears, and the picture
between two anchors is sped up or slowed down uniformly so the speech fits. The voice itself is never sped up.

- Between two adjacent anchors the cut has P seconds of picture. The voice needs V = sentence duration + a gap
  (`GAP` 0.3 s; `GAP_SEG` 0.5 s when the next sentence starts a new segment; for the last sentence `END_TAIL` 2.3 s,
  used instead of the script's `hold_end`).
- V ≤ P: speed the stretch up, voiced length L = max(V, P / `MAX_SPEED`) with `MAX_SPEED` = 2.5. Whatever the cap
  leaves over is silence after the sentence (`silence_after_s`).
- V > P: slow the stretch down, rate ≥ `MIN_SLOW` = 0.6. If 0.6 is not enough, play at 0.6 and freeze one frame for the
  rest. The hold point is a frame of the cut shot (on its frame grid), never inside a cross-fade (± 0.125 s around a
  segment boundary of a cut shell) or a title card's fade-out (its last 0.35 s): the still frame (mean frame difference
  ≤ `STILL` 0.03, measured on the cut render without the caption band) closest to `HOLD_LEAD` 0.15 s before the next
  anchor, else the frame with the least motion. The last sentence holds on the film's final frame.
- `CONTENT_END` (per sentence): when the picture a sentence talks about ends before the next anchor (the next picture
  cuts in early), the picture reaches that time exactly when the sentence ends; the rest plays at ≤ 2.5×.
- `ANCHOR_OVERRIDE` (per sentence): move one anchor without editing the narration script, e.g. so a sentence starts
  when the boxes it names appear. Anchors must stay increasing.
- Every L is rounded to whole frames, so every shot starts on the frame grid.
- Anchor times from the narration script are rounded; shift them by each shot's actual start before laying out.
- Output `timing.json`: per interval the mode (normal / speed / slow / slow+hold), rate, hold seconds, silence; per
  shot a list of knots `[(voiced s, cut s)]` (a stretch where the cut time stays constant is a hold); per sentence its
  start / end. The voiced shells, `mix.py` and `check_voiced.py` all read it.

Simple variant (no speed changes): sentence start = max(its anchor, end of the previous sentence + 0.25 s); if the
speech is too long, first swap in the segment's alternative short sentence, then **hold** the segment's last frame for
the missing time; duration = original + holds. Use it when the picture's rhythm must not change.

## 4. Render the voiced variant (`shots/voiced.py`, `render_all.ps1`)

Copy the cut shells into the voiced folder (never edit the approved ones). The variant maps every frame's time through
the shot's knots, replaces the on-screen captions with the spoken sentences (display text, one per sentence, shown for
the sentence's time, at least 1.0 s, never overlapping the next), and keeps titles / tags / pictures. Its checks
assert: knots monotone and spanning both the voiced and the cut shot, one caption per spoken sentence, no caption
overlap, picture rate 0 (hold) or within 0.6–2.5×. Concat to a silent video.

## 5. Mix (`mix.py`)

- Place each selected wav at its `t_start` by sample index into one track with numpy (ffmpeg `amix` scales the volume
  down by the number of inputs; `adelay` + `amix` is the alternative if you must). Fade out the last 20 ms of each
  sentence (a hard cut can click). Digital silence between sentences; no music, no gate.
- Two-pass `loudnorm` to I = -16 LUFS, TP = -2.0 dBTP, 48 kHz. Use -2.0 rather than -1.5: AAC encoding raised the true
  peak by about 0.1 dB. Measure the final file with `ebur128` and report I / LRA / TP.
- Mux with the silent video (video stream copy, AAC 192k); write an `.srt` of the spoken sentences with the on-screen
  caption times (read from each shot's `meta.json`).

## 6. Self-check (`check_voiced.py`) → a short report

- Transcribe the final audio track with whisper medium; compare the words inside each sentence's window with the
  script (key words all present, WER); then transcribe each sentence's slice of the track alone and compare again
  (`check_whisper.json`). List accepted harmless variants (homophones, liaison).
- Silence table (`silence_table.json`, ffmpeg `silencedetect`): per segment, whether there is sound and the longest
  silence; long gaps mean the time map left too much picture without speech.
- A contact sheet of each segment's midpoint with the spoken sentence printed under it: the thing said is on screen.
- A sheet of frames before / during / after each hold: no half-faded frame frozen.
- Total length, number of holds and their seconds, loudness numbers, GPU peak, anything you could not judge (natural
  prosody needs a human ear: whisper adding "!" to several sentences hints at an over-excited delivery).
- Then a fresh verifier listens via transcription and checks timing; the reviewer listens last.

## Lessons

- Spelling for TTS is a script-level fix: write it into the `tts` column, keep the display text untouched.
- Short contractions can be heard as other words in context ("give em" heard as "give them"): rewrite the sentence,
  do not fight it with spelling.
- The same acronym may be read differently across sentences; if it matters, spell it the same way in every `tts` line.
- Do not set `HF_HUB_OFFLINE=1` for the TTS model: its loader queries the hub API and fails offline.
- Respell one sentence without editing the script: `gen_voiced.py --tts <sentence id>=<new spelling>` (recorded in
  `gen_raw.json`).
- Keep everything idempotent: changing one sentence = regenerate that sentence, re-pick, re-time its segment,
  re-render the affected shot(s), re-mix (minutes, not hours).
