# Tamil Palm-Leaf OCR — Issues & Improvements TODO

> Generated from a full run + analysis of `ocr_results/` (26 pages, 804 characters, 27 lines)
> and source review of all 6 Python modules.

---

## 🔴 Critical Issues (Recognition is effectively broken)

### 1. Confidence scores are catastrophically low
- **Every single character** has confidence < 0.50 (max observed: 0.484)
- Mean confidence: **0.147** — the model is essentially guessing randomly
- 64.2% of predictions have confidence < 0.15
- 85.3% of predictions have confidence < 0.20
- The softmax-over-decision-function trick in `classify_crop()` produces flat distributions — evidence that HOG+SVM features computed on isolated character crops from manuscript images look nothing like the training data

### 2. Extreme class collapse — 3 classes dominate 67% of output
- `cha` (ச): 30.7% of all predictions
- `zha` (ழ): 27.9% of all predictions
- `tha` (த): 8.5%
- Only 19 of 20 trained classes ever appear in output (1 class never predicted)
- Real Tamil text should have a much more uniform distribution — this is the model defaulting to a few dominant classes when uncertain

### 3. Output text is nonsensical
- Every page produces gibberish strings like `ணவுஊஊசஊழழழஊதய்நூஊழழஊழஊழஊசழழநூஊ`
- No recognizable Tamil words appear in any of the 26 pages
- Even basic high-frequency Tamil characters (like க, ப, ம) are severely under-predicted or absent

### 4. Line segmentation almost completely fails — each "line" contains 4-12 actual rows
- 25 of 26 pages detect exactly **1 text line** (only 125.jpg detects 2)
- Each detected "line" is **128-415px tall**, packaging 4-12 actual manuscript rows into a single segment
  - e.g. `130.jpg`: 1 detected line, height=372px, contains ~11 real rows but only finds 34 "characters"
  - e.g. `149.jpg`: 1 detected line, height=415px, contains ~12 real rows but only finds 36 "characters"
  - e.g. `125.jpg L0`: height=128px, contains ~4 real rows (L1 at 25px is the only correctly-sized segment)
- Real manuscript pages have 5-8+ lines of text; the system should detect 130-180+ total lines across 26 pages, not 27
- **Root cause**: the horizontal projection threshold (`max(h_proj.max() * 0.02, 5)`) is far too low — ink is dense enough that inter-line gaps never drop below this threshold, so the entire text block merges into one band
- **Downstream cascade**: when character segmentation receives a 370px-tall strip containing 10+ rows, connected components span multiple rows vertically, producing bounding boxes that merge characters from different lines. This makes classification impossible regardless of model quality

### 4a. Character merge cascade destroys most letters — 783 components → 9 boxes
- **Verified on page 132**: source image has ~300+ clearly visible Tamil characters across 8 dense rows, but the output contains only **18 "characters"** (9 giant merged boxes split in half by overlap)
- Root cause chain:
  1. Line segmentation produces a **373px-tall strip** containing all 8 rows
  2. `segment_characters()` correctly finds **783 valid connected components** (the actual character strokes)
  3. The **merge logic (line 246-261)** only checks horizontal overlap (`overlap_x > 0.3 * min_width`) with **no Y-separation check**
  4. Characters from row 1, row 2, row 3... all have overlapping X ranges, so they cascade-merge: **774 merges → 783 components collapse to 9 mega-boxes**
  5. Each final box is a **column strip** spanning 2-25% of page width and the full 373px height — essentially slicing the page into 9 vertical columns
  6. These column-strip crops (e.g., 166×346px, 802×373px) are resized to 64×64 for classification, squashing dozens of characters into an unrecognizable blob
- This is the **primary cause of visible letter loss** — not preprocessing, not speck removal (which only removes 1.9% of ink), but the merge step obliterating 99% of spatial structure
- The 18 "recognized" characters for page 132 (`ஊழதஊசசழசதஊஊயழழஊதரழ`) are classifications of 9 giant column images — each "character" is actually 30-80 real characters mashed together

### 5. sklearn version mismatch warning
- Model trained with scikit-learn 1.7.2, loaded with 1.6.1
- `InconsistentVersionWarning` — may silently produce wrong predictions
- The pickled `LinearSVC` internal state may not be compatible across versions

---

## 🟠 Segmentation Issues

### 6. Character segmentation produces wildly inconsistent box sizes
- Bounding boxes range from tiny fragments (7×19 px) to enormous multi-character blobs (329×122 px)
- The morphological close with a 3×3 kernel is too small to properly merge broken strokes but also can't prevent over-merging of adjacent characters
- No vertical overlap/splitting logic — when connected components span multiple characters (common in Tamil with vowel signs), they're treated as single characters

### 7. Punch-hole removal is inconsistent
- Page 125 (binarized) finds 4 holes in saved results but finds 0 when re-run
- The `remove_punch_holes()` in `tamil_preprocess.py` uses a different algorithm from the one in `palmleaf_pipeline.py` (`detect_and_clear_holes`) — there are **3 different implementations** of hole removal across the codebase (`tamil_preprocess.py`, `palmleaf_pipeline.py`, `_holefix.py`)
- No single canonical hole removal — each file has slightly different parameters

### 8. Speck removal threshold is fixed
- `_remove_specks(binary_img, min_area=25)` uses a hardcoded area threshold
- Doesn't adapt to image resolution or DPI — 25px at 300 DPI vs 600 DPI catches very different noise
- Some legitimate small diacritical marks (puḷḷi dot) could be removed

---

## 🟡 Code Quality Issues

### 9. Duplicated code across modules
- `to_grayscale_luminosity()` is defined in **3 places**: `tamil_preprocess.py`, `tamil_dataset.py`, `palmleaf_pipeline.py`
- `compute_psnr()` is defined in **2 places**: `tamil_preprocess.py`, `palmleaf_pipeline.py`
- `binarize_sauvola()` is defined in **2 places**: `tamil_preprocess.py`, `palmleaf_pipeline.py`
- `binarize_otsu()` is defined in **2 places**: `tamil_preprocess.py`, `palmleaf_pipeline.py`
- Any fix to one copy won't propagate to the others

### 10. `palmleaf_pipeline.py` is entirely unused by the main pipeline
- `recognize_manuscript.py` imports from `tamil_preprocess.py`, not from `palmleaf_pipeline.py`
- `palmleaf_pipeline.py` has a more sophisticated pipeline (leaf-bbox crop, border artifact clearing, better hole detection) but none of it is used
- Dead code that confuses which implementation is canonical

### 11. `_holefix.py` is a standalone orphan
- Third copy of hole detection logic, not imported anywhere
- Slightly different default `dilate_pad=4` vs `dilate_pad=16` in `palmleaf_pipeline.py`

### 12. No error handling or logging
- Silent `except Exception` in batch processing swallows all errors
- No structured logging — only print statements
- No progress bars or timing info for batch runs

### 13. requirements.txt has no version pins
- All 6 dependencies (`opencv-python`, `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `scipy`) are unpinned
- The sklearn version mismatch issue (item 5) is a direct consequence
- No `pyproject.toml` or `setup.py` for proper project packaging

### 14. The confidence metric is misleading
- `classify_crop()` applies softmax to SVM decision function scores to get "confidence"
- SVM decision functions are not calibrated probabilities — this metric is not meaningful
- Should use `sklearn.calibration.CalibratedClassifierCV` for proper probability estimates

---

## 🔵 Dataset & Model Issues

### 15. Training data is too small and narrow
- Only **2,550 images** across 20 classes (~127 per class)
- The 20 character classes cover a tiny fraction of the Tamil abugida (247 basic graphemes + combinations)
- Missing extremely common characters: க (ka), ப (pa is present but under-predicted), ல (la), ர (ra), etc.
- No ligatures, vowel signs, or combined consonant-vowel forms

### 16. Domain gap between training crops and manuscript segmentation
- Training data: clean, pre-cropped character images at 224×224
- Inference data: noisy connected-component crops from degraded manuscripts
- HOG features are sensitive to this domain shift — models trained on clean crops fail on noisy real segmentation

### 17. CLASS_TO_TAMIL mapping is unverified
- The docstring explicitly warns: "*HAVE A TAMIL READER VERIFY THIS MAP*"
- Several entries are noted as ambiguous (e.g., "ee" could be ஈ or ஏ, "oo" could be ஊ or ஓ)
- Recognition results are meaningless until the label map is verified

### 18. Model file is a pickled binary
- `tamil_char_svm.pkl` is an opaque pickle — security risk, not portable, version-fragile
- Should export to ONNX or use `skl2onnx` for stable deployment
- No model versioning or metadata beyond what's in `results.json`

---

## 🟢 Improvements Roadmap

### Phase 1 — Fix what's broken (immediate)

- [ ] **Pin all dependencies** in requirements.txt with exact versions
- [ ] **Retrain the SVM** on the current scikit-learn version to eliminate the version warning
- [ ] **Fix line segmentation** — the horizontal projection threshold is too low; try adaptive valley detection or Gaussian smoothing of the projection profile before thresholding
- [ ] **Add a minimum confidence threshold** — default `--min-confidence 0.0` means every garbage prediction is included; set a reasonable default (e.g., 0.15) or add "uncertain" markers
- [ ] **Consolidate duplicate code** — create a single `utils.py` with `to_grayscale_luminosity()`, `compute_psnr()`, `binarize_sauvola()`, etc., and import from there
- [ ] **Remove or integrate dead files** — decide whether `palmleaf_pipeline.py` or `tamil_preprocess.py` is canonical; remove `_holefix.py` or merge its logic
- [ ] **Have the CLASS_TO_TAMIL map reviewed** by a Tamil-literate person

### Phase 2 — Improve the model (short-term)

- [ ] **Expand character classes** — the 20-class model can't handle real Tamil text; need at least 60-80 classes covering common consonants, vowels, and vowel-sign forms
- [ ] **Collect/annotate more training data** — augment the THPLMD dataset with crops from the NALADIYAR / THIRIKADUGAM / THOLKAPPIYAM manuscripts already in the `tamil ml dataset/` folder
- [ ] **Add data augmentation** — elastic deformation, noise injection, stroke erosion/dilation to bridge the domain gap between clean training crops and real manuscript segments
- [ ] **Replace HOG+SVM with a CNN** — the `train_baseline.py` already has a CNN branch (`--cnn`); use it. The 3-layer CNN with BatchNorm + augmentation should significantly outperform HOG+SVM on degraded inputs
- [ ] **Calibrate model outputs** — use `CalibratedClassifierCV` or CNN softmax for meaningful confidence scores
- [ ] **Run the leakage audit** — `train_baseline.py --audit` exists but hasn't been run on the deployed model; confirm the 97.1% test accuracy is real

### Phase 3 — Upgrade the pipeline (medium-term)

- [ ] **Replace connected-component character segmentation** with a proper scene text detector (CRAFT, DBNet) or a sequence model (CRNN + CTC) that avoids explicit segmentation entirely
- [ ] **Implement word/line-level recognition** — bypass character segmentation; use a CRNN or TrOCR that reads entire line images
- [ ] **Integrate the palmleaf_pipeline.py improvements** — leaf-bbox crop, border artifact clearing, better hole detection using grayscale silhouette
- [ ] **Add evaluation metrics** — CER/WER computation, per-page confidence histograms, segmentation accuracy vs. ground truth counts
- [ ] **Add a post-processing stage** — Tamil lexicon lookup, n-gram language model, or byte-level LM (ByT5) to correct recognition errors
- [ ] **Build a validation/correction UI** — Label Studio or eScriptorium integration for human-in-the-loop verification

### Phase 4 — Production readiness (long-term)

- [ ] **Implement the full architecture from the plan** — DBNet/CRAFT detector → PARSeq/TrOCR recognizer → LM post-processor (as outlined in `Tamil-OCR-HTR-Plan.md`)
- [ ] **Fine-tune from IndicSTR12 PARSeq weights** for Tamil scene text recognition as a pretrained starting point
- [ ] **Add synthetic training data generation** — render Tamil text on palm-leaf textures with realistic degradation
- [ ] **Export model to ONNX** for portable, version-stable deployment
- [ ] **Add proper logging** (Python `logging` module), CLI progress bars (`tqdm`), and structured JSON output
- [ ] **Add unit tests** for each pipeline stage (preprocessing, segmentation, classification)
- [ ] **Set up CI/CD** — automated testing, model benchmarking, regression checks on the held-out test set
- [ ] **Package as a pip-installable module** with `pyproject.toml`

---

## Summary Statistics from Current Run

| Metric | Value |
|--------|-------|
| Pages processed | 26 |
| Lines detected | 27 (should be ~130-180+) |
| Characters recognized | 804 |
| Mean confidence | 0.147 |
| Max confidence | 0.484 |
| Predictions > 50% conf | **0** |
| Classes actually predicted | 19 / 20 |
| Top 3 classes coverage | 67.0% |
| Recognizable Tamil words | **0** |

---

## File Inventory

| File | Status | Notes |
|------|--------|-------|
| `recognize_manuscript.py` | **Main pipeline** | Works but produces nonsense output |
| `tamil_preprocess.py` | Used by main | Preprocessing (denoise → binarize → crop → holes) |
| `train_baseline.py` | Standalone | HOG+SVM and CNN training; CNN unused |
| `tamil_dataset.py` | Used by training | Dataset loader with good quirk handling |
| `palmleaf_pipeline.py` | **Dead code** | Better pipeline but not integrated |
| `_holefix.py` | **Dead code** | Orphan hole-fix, not imported |
| `tamil_char_svm.pkl` | Model file | Version-mismatched, low accuracy on real data |
| `results.json` | Metrics | 97.1% test acc (on clean crops only) |
| `Tamil-OCR-HTR-Plan.md` | Project plan | Comprehensive roadmap, mostly unimplemented |
