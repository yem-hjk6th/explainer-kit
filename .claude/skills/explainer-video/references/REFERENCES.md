# REFERENCES: studying reference videos and source material

## Reference videos (style only)

1. Get the video from a source you have the right to watch and download (a public link with `yt-dlp`, your own
   recordings, material shared with you for this purpose). Put each in its own folder as `video.mp4`. Reference
   footage never goes into your videos and is not redistributed.
2. `python -I -B -X utf8 tools/analyze_explainer.py <dir> --lang <spoken language, e.g. en; default: auto> --prompt "<domain terms, comma separated>"`
   (CPU; needs faster-whisper + rapidocr-onnxruntime). Writes `<dir>/analysis/`:
   - `asr.json` — narration with timestamps (whisper medium; the prompt biases spelling of domain terms);
   - `ocr.json` — on-screen text changes sampled every 0.5 s (captions, labels);
   - `cuts.json` — hard cuts (frame-difference peaks) and motion per second;
   - `sheet_NN.jpg` — timestamped contact sheets every 3 s.
3. Write a breakdown page for yourself / the user:
   - a table per video: time range → what is explained → what the picture does;
   - caption density (sentences, characters per sentence, characters per second), number of hard cuts;
   - the patterns worth copying, and those not worth the cost;
   - a fact check of the video's claims against the original papers (it shows where popular explainers simplify:
     attribution, history, "X solved Y" narratives).
4. Turn the patterns into storyboard rules (they are already in `STORYBOARD.md` §Story patterns and `STYLE.md` §B).

## Source material (papers, lecture notes)

- Convert once to text (any PDF-to-markdown tool), then write a digest: `## §section [lines X–Y]` + 2–3 sentence
  summary + key claims. Read the digest, then only the line ranges you need. Cite by section / table / equation and
  keep the line numbers in CLAIMS so the verifier can jump there.
- A digest is for navigation only: quote numbers from the full text, never from the digest (a digest once swapped two
  datasets' tables and nobody noticed for days).
- Your own earlier explanations (lecture notes, summaries) are sources too, but anything marked "inferred" there must
  be checked against the paper or the official code before it goes on screen.
- Official code: clone at a fixed commit into `third_party/` (git-ignored), record the commit in CLAIMS; shot checks
  that read the code should skip with a recorded note when the clone is absent (see
  `episodes_example/E01_vggt_mapanything/shots/s02_patchify.py`).
