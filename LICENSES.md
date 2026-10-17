# Licenses and third-party material

## This repository

The code and documents in this repository are under the MIT License (`LICENSE`).

The MIT License covers only what this repository's contributors wrote. It does **not** cover the third-party material
below (model weights, model code, papers and their figures, fonts, reference videos), even where the example episodes
use, describe or depend on it: each keeps its own license, listed here as a pointer, not as legal advice.

## Papers discussed in the example episodes

The example episodes explain published papers (VGGT, MapAnything, Depth Pro, Depth Anything 3). The repository holds
only our own scripts and text about them: paper titles, numbers quoted with their table / section, and our own
drawings. Paper PDFs, page images and figures are **not** included; if you add a paper's page or figure to a video
(for example in a paper card), its copyright and license (often the arXiv license chosen by the authors) apply.

## Not included, and what applies when you fetch them

| Material | Where it comes from | Terms (check the model card / repo at download time; they change) |
|---|---|---|
| VGGT-1B weights | official Hugging Face repo `facebook/VGGT-1B`; code `github.com/facebookresearch/vggt` | the original checkpoint is for non-commercial use; a separate commercial checkpoint exists |
| MapAnything weights | `facebook/map-anything`; code `github.com/facebookresearch/map-anything` | weights CC BY-NC 4.0; an Apache-2.0 variant is published separately |
| Depth Pro weights | `apple/DepthPro`; code `github.com/apple/ml-depth-pro` | Apple's model license on the model card (research terms) |
| Depth Anything 3 weights | `depth-anything/DA3-LARGE-1.1`, `depth-anything/DA3METRIC-LARGE`; code `github.com/ByteDance-Seed/Depth-Anything-3` | per-model license on each card; the large / metric checkpoints used here are non-commercial |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` via the `qwen-tts` package | see the model card |
| Whisper models (faster-whisper) | downloaded on first use | MIT (models and faster-whisper) |
| RapidOCR models | bundled with `rapidocr-onnxruntime` | Apache-2.0 |
| Fonts | your operating system (Segoe UI / Cambria on Windows, Helvetica / Times on macOS, DejaVu on Linux; optional fonts for a translated render) | fonts are **not** redistributed; `config/animkit.json` points at fonts installed on your machine. Formulas use the Computer Modern fonts shipped inside matplotlib |
| Reference videos | whatever you choose to study | only use sources you have the right to watch and download; learn the style, do not put their footage in your videos or redistribute it |

The example episodes show real model outputs only after you run the model scripts yourself; videos made with
non-commercial weights inherit those terms (study / teaching use).

No scraping tools for social-media platforms are included.

## Python packages

numpy, OpenCV (`opencv-python`), Pillow, matplotlib, imageio-ffmpeg (bundles an ffmpeg binary, LGPL/GPL build), faster-whisper,
rapidocr-onnxruntime, Playwright, torch, qwen-tts, soundfile, librosa: each under its own license, installed by you with pip.
