# Automatic Wall Crack Detection, Measurement and Damage Assessment

UCSC313L Image Processing project. Classical image processing only — no deep learning.

## What makes this build distinct from a standard 4-algorithm comparison

1. **Lighting-aware adaptive pipeline.** `preprocessing.py` classifies each image's
   lighting (dark / bright / shadow-uneven / normal) from histogram statistics and
   automatically picks CLAHE parameters for it; `segmentation.py` then automatically
   switches between Otsu and adaptive thresholding based on the same label. Most
   student projects use one fixed setting for the whole dataset — this one adapts
   per image, which is exactly what your faculty's "different lighting conditions"
   requirement is testing for.
2. **A 5th "bonus" Fusion algorithm** (`algorithms/fusion.py`) — a weighted
   combination of Sobel + Prewitt + Laplacian + Canny. The 4 mandatory algorithms
   are still compared independently and honestly; Fusion is an additional research
   angle ("does combining classical operators beat any single one?") that you
   measure with your own ground truth, not something you're allowed to just assert.
3. **Confidence-scored detection** instead of a flat yes/no — combines edge density
   and shape elongation (cracks are long & thin; noise blobs are small & round).
4. **Semi-automatic calibration** — auto-detects a colored reference marker for
   pixel-to-cm conversion, with manual fallback, instead of requiring you to click
   corners on every one of 200+ images.
5. **A real evaluation harness** (`evaluation/comparison.py`) that only reports
   numbers it actually computed against your hand-made ground-truth masks — it
   refuses to output anything if no ground truth exists yet, so you can't
   accidentally present fabricated results.

## Setup

```bash
pip install -r requirements.txt
# Linux only, if tkinter isn't already present:
sudo apt install python3-tk
```

## Running things

**Single image, full pipeline, all intermediate outputs saved:**
```bash
python main.py dataset/raw/YOUR_IMAGE.jpg canny
```
(swap `canny` for `sobel`, `prewitt`, `laplacian`, or `fusion`)

**GUI:**
```bash
python gui/app.py
```

**Batch evaluation across your dataset (after you've made ground-truth masks):**
```bash
python evaluation/comparison.py
```
This writes `reports/comparison_table.csv`, `reports/per_image_results.csv`, and
bar-chart PNGs to `reports/graphs/`.

## What's built vs. what's still on you

Built and tested (verified end-to-end on a synthetic sample image, included at
`dataset/raw/TEST_001.jpg` — check `outputs/final/TEST_001.png` to see a working
annotated result right now):
- Full preprocessing → 4 algorithms → fusion → segmentation → morphology →
  skeletonization → length/width/area/damage measurement → annotated output
- Calibration (auto + manual fallback)
- GUI with Home / Processing Stages / Measurements / Comparison tabs
- Evaluation harness + graph generation

Still needed from you (per your faculty's actual requirements — this cannot be
faked or shortcut):
- **The real 200-250 campus photos** (`dataset_log_template.csv` shows the format
  to log them in — location, crack type, lighting, etc.)
- **40-50 hand-drawn ground-truth masks** for the evaluation subset — paint
  crack pixels white, background black, save as PNG in `dataset/ground_truth/`
  with the same filename as the source image
- Running `evaluation/comparison.py` on your real data to get real
  precision/recall/F1 numbers for your report — the current CSVs will only ever
  contain genuine results, never placeholders

## Known limitations (be upfront about these in your report)

- Faint hairline cracks in low-contrast lighting may be missed
- Shadows can be misread as cracks — the adaptive segmentation reduces but
  doesn't eliminate this
- Physical (cm) measurements are only valid if a reference marker was captured
  in-plane with the crack; without one, only pixel measurements are reported
- This is a detection/measurement prototype, not a certified structural safety
  inspection tool
