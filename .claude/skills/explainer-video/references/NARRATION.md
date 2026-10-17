# Writing and checking the voice-over script

A narration script is written per finished video, after the captions are verified. It is read by the presenter live, or by a TTS voice, while the video plays. This guide covers how to write it, the data format, the deliverables, and how to check it.

## 1. Narration is not the captions read aloud

The captions already carry every label, number and formula on screen. The voice adds what the screen cannot: what we are looking at, why we are looking at it, and what to take away.

Split the finished video into visual segments. A segment is a continuous stretch of the video about one thing, usually 5 to 15 s. Title cards and the opening card each count as a segment. For each segment, say three things only:

1. What this picture shows (one clause).
2. Why we show it now (link to the previous segment or to the main line of the talk).
3. One or two key points. One takeaway per segment, never a tour of every element.

If a sentence only repeats a caption, cut it. If a segment needs more than two key points, the segment is too long; split the visual first.

## 2. Style

Use the register of a whiteboard explainer: conversational English.

- Pose the question, then answer it: "How do we get a camera? Take the camera token."
- Point at the screen: "Watch the blue ray." "Follow the white dot."
- Short sentences. First person plural: "we", "our".
- No hype words, no "amazing", no "simply". Plain connectives are fine ("Here is the core", "The catch:").
- Do not read every number. Name the one that matters, and let the screen carry the rest.
- Use the same terms as the screen (the same method labels, the same names for outputs). Do not rename things in speech.
- Write numbers the way a person says them, but keep a separate `tts` text for the engine (section 5).
- Title cards get one short sentence ("First, the pinhole model.").

## 3. Word budget

Per segment of `L` seconds:

```
max_words = floor( min( 2.0 * L ,  2.2 * (L - 0.5) ) )
```

- 2.0 words per second is the upper bound. It corresponds to about 130 words per minute of live presentation speech (about 2.2 words/s) with room for pauses and breaths.
- The speech must finish 0.5 s before the segment ends, so the next visual starts in silence. The second term of the formula enforces that.
- A segment that ends with a held last frame (extra seconds appended to the video) uses its lengthened `L`.

This fits TTS too: neural TTS voices at normal speed run faster than a person teaching (a typical local voice measured around 2.4 to 2.5 words/s). A script that fits the live-presentation budget therefore fits a TTS render with slack. If the presenter has a recording, measure the real rate (words divided by speaking time, pauses included) and replace 2.0 / 2.2 with it.

Count spoken words, not whitespace tokens: expand digits ("72%" is three words), count acronyms by how they are read (an acronym spelled letter by letter counts as several), split hyphenated words. This overcounts slightly, which is the safe direction. Put the counting rule in the build script and use it everywhere.

Syllable density matters at the edges. A segment full of long technical words can overrun even when the word count passes. Flag segments above about 80% of the available time and give them a backup (section 5).

## 4. No new facts

- Every number and every claim in the voice-over must already exist in the verified claims list or in the verified captions. If you cannot point to the source line, do not say it.
- Record the source of each segment (caption ids, claim ids) in a `refs` field. The verifier uses it.
- A number taken from a paper names its table or figure: "Table 3 of the paper". Say whose experiment it is when the screen mixes sources ("the paper's own Table 2").
- The first mention of the project's own test or toy scene states that it is our own test, not the paper's experiment. Repeat the label whenever paper numbers and own numbers share a screen ("on our scene, unaligned, ...").
- Do not state inferences as results. If the screen labels something "our reading", the voice says "our reading".
- Do not say "equal", "all", "always", "better" unless the source says exactly that, with the same conditions (inputs, number of views, metric direction).

## 5. Timing data: anchors, tts, backups, hold

The voice-over is laid out sentence by sentence, so each sentence needs to know when its subject appears.

- **anchor**: the video time (seconds) at which the thing the sentence talks about appears on screen. Take it from the caption timing plus the segment start. A sentence is spoken at or after its anchor, never before the object is visible.
- **Placement rule** (back to front, so the last sentence finishes in time): for each sentence, latest start = (segment end - 0.5 s) minus the time needed for it and everything after it; start = max(previous sentence end + pause, min(anchor, latest start)). A pause of about 0.25 s between sentences is realistic. If a sentence has to start earlier than its anchor to finish on time, record how early ("max lead"); more than about 0.5 s means the segment is tight. If it cannot finish even then, record the overrun.
- **tts**: a second text column for the engine. Spell out digits ("Table three", "nine point five percent"), symbols and lone letters ("em" for a variable called m, otherwise the engine may say "meters"), and awkward acronyms ("DINO v2"). Keep `text` readable for the human; use `tts` only for synthesis. Listen to every changed word once.
- **alt_sentences** (optional): shorter backup sentences, one-to-one with the main ones and sharing the same anchors. Use them only if a real TTS render overruns. Write them for the tight segments only.
- **hold_end** (optional, on the last segment): extra seconds the final frame is held after the video ends, so a closing sentence can finish. Add it to that segment's length before computing the budget.

## 6. narration.json schema

A list of segments in video order. Times are seconds on the finished video's clock, to 0.1 s.

| field | meaning |
|---|---|
| `id` | segment id, e.g. `c1` for cards, `p2.3` for part 2 segment 3 |
| `t0`, `t1` | start and end time in the finished video |
| `on_screen` | one line: what the picture shows |
| `why` | one line: why we show it here |
| `key_points` | 1 to 2 short strings (small type on the card) |
| `text` | full spoken text of the segment |
| `words` | spoken-word count of `text` |
| `max_words` | the budget from section 3 |
| `tts` | `text` rewritten for the engine |
| `sentences` | list of `{text, tts, words, anchor}` |
| `hint` | optional one-line note for the presenter, not spoken (can be in the presenter's own language) |
| `refs` | source ids for the verifier |
| `alt_sentences` | optional backup sentences |
| `hold_end_s` | optional, last segment only |

Minimal neutral example (invented content):

```json
[
  {
    "id": "p1.2", "t0": 5.0, "t1": 15.0,
    "on_screen": "A pinhole camera: a 3D point projects through the hole onto the image plane.",
    "why": "The projection is what every later step inverts.",
    "key_points": ["Image point = K times the 3D point, divided by depth.",
                   "Dividing by depth is where the distance information is lost."],
    "text": "Watch the point pass through the hole. It lands on the image plane, but its depth is gone.",
    "words": 19, "max_words": 20,
    "tts": "Watch the point pass through the hole. It lands on the image plane, but its depth is gone.",
    "sentences": [
      {"text": "Watch the point pass through the hole.",
       "tts": "Watch the point pass through the hole.", "words": 7, "anchor": 5.0},
      {"text": "It lands on the image plane, but its depth is gone.",
       "tts": "It lands on the image plane, but its depth is gone.", "words": 12, "anchor": 9.4}
    ],
    "hint": "Optional note for the speaker; not read aloud.",
    "refs": ["CAP-2-1", "CAP-2-2", "CLAIM-07"]
  }
]
```

## 7. Deliverables

Generate all three from one source (a single list of segments in a build script), so a fix is made once and rebuilt. Never edit the generated files by hand.

1. `narration.json`, the schema above. Input to the TTS step.
2. A markdown table version, one table per part. Columns: segment id, time range, length, what is on screen, spoken text with `@anchor` before each sentence, words / limit, source ids. Optionally add columns for the placement simulation (max lead and overrun at the TTS rate and at the presenter's rate). End with statistics: segment and sentence count, total words, estimated speaking time at each rate, segments over budget, tight segments, and a list of where `tts` differs from `text` and which backups exist.
3. A speaker-notes HTML for use while the video plays. One card per segment:
   - time range and segment id in a header line;
   - one-line description of the picture;
   - the spoken text in large type (about 28 to 30 px, shrinking a little on narrow screens);
   - the key points in small type;
   - the optional hint line in a muted style, marked as not spoken;
   - a visible marker when a segment has a held end frame or is over budget.
   The file must be self-contained: inline CSS, no external scripts, fonts or images. It has to be readable on a tablet and in print, so use a single column, high contrast, and a page-break-inside: avoid rule on cards. Check it renders without errors at desktop and phone widths.

## 8. What the verifier checks

An independent reviewer (not the author) gets the narration, the verified claims, the captions and the rendered video, and checks:

1. **Facts.** Every sentence against its sources: numbers, directions ("lower is better"), conditions (number of views, aligned or not, which model), and whose experiment it is. Verdict per segment: holds, needs a change, wrong.
2. **Picture sync.** Sample frames at the anchor times and in the middle of each segment. The thing being mentioned must be on screen when the sentence starts. Continuous speech tends to run ahead of captions that change every 2 to 3 s, so check the later sentences of each segment especially. Also check that the voice does not describe something that has already been replaced.
3. **Word counts.** Recount independently, compare with `words` for every segment, recompute `max_words`, total the words, and estimate time at both speaking rates. Add a syllable-density estimate for the densest segments and for very short cards (TTS leading silence eats into a 1.5 s window).
4. **Style.** Reads like speech, not like captions; no stacked jargon; terms match the screen; consistent spelling of repeated names; no repeated sentence between a card and the next segment unless it is intentional. Note any closing line that ends on a negative point.
5. **Speaker notes.** Card count and times match the JSON, no segment wrongly marked over budget, the large type is readable, and every presenter hint is correct (hints are easy to get wrong because nobody reads them aloud and nobody checks them).

The verifier returns a must-fix list with the original text, the replacement, and the reason, plus optional suggestions. The author applies the fixes in the single source and rebuilds.

## 9. Common mistakes found in review

- **Speech ahead of the picture.** A sentence mentions an object that appears 2 to 3 s later. Fix by attaching an anchor to each sentence and padding with silence, or by rewording so the first words match what is already visible.
- **Overstated paper results.** Dropping the conditions of a table (two views, images only, the authors' own evaluation) turns a narrow result into a general one. Name the table and the condition.
- **Mixed sources on one screen.** When a paper's number and the project's own number are both visible, the sentence must say which one it is talking about ("on our scene").
- **Imprecise wording.** "Equal parameters" reads as equal values, when the source says "same parameter count". "Its own tokens" was wrong when only a subset of tokens was separate. Reuse the screen's wording.
- **Hints and notes that mislead.** Presenter hints named the wrong later segment, or called a non-unit direction a unit ray. Check hints with the same rigor as spoken text.
- **Budget that passes by words and fails by sound.** Long technical words make a segment longer than its word count suggests. Keep backups for the densest segments and measure the real TTS durations before locking.
- **Single letters and acronyms in TTS.** A bare variable name or a mixed-case acronym is mispronounced. Give the engine a spelled-out `tts` text.
- **Closing on the weak result.** If the last segment ends on the negative number and the screen shows a positive follow-up, either fit both or ask the presenter to choose.
- **Duplicate statement.** A title card and the following segment both say "this is our own test" within a few seconds. Acceptable when the label is a requirement; otherwise trim one.
- **Hand-edited outputs.** Editing the generated table or HTML instead of the source makes the three deliverables drift apart. Edit the source list, rebuild all three.
- **Facts not in the claims list.** An aside that is true but unverified. Either add it to the claims and verify it first, or leave it out.
