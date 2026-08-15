# Upgrading Classifier & Semantic NLP Tokenizer

We have perfected the physical segmentation (extracting clean character crops from the palm leaves). Now, we must fix the logical recognition and linguistic post-processing. This plan covers the two major next steps: upgrading the 20-class classifier to a Deep Learning model supporting the full Tamil alphabet, and building a dictionary-based word segmenter to handle the space-less text.

## User Review Required

> [!IMPORTANT]
> **Dataset Availability:** The current `_char_repo/Dataset` only has 20 classes. To train a CNN on all 247+ Tamil characters, we need a complete dataset. Do you have access to a full printed/handwritten Tamil character dataset (e.g., IWF, HP Labs, or a custom one), or should we generate a synthetic one from Tamil fonts to bootstrap the model?

> [!IMPORTANT]
> **Deep Learning Framework:** I propose using **PyTorch** for the CNN as it's the industry standard for custom computer vision research. Let me know if you strongly prefer TensorFlow/Keras.

> [!NOTE]
> **Tamil Dictionary Source:** The NLP segmenter will require a list of valid Tamil words to know where to place spaces. I propose downloading a basic Tamil wordlist (e.g., from Tamil Wiktionary or a similar open-source corpus) to drive the dictionary-based tokenization.

## Proposed Changes

### Deep Learning Classifier (P1)
We will replace the HOG+SVM pipeline with a Convolutional Neural Network (CNN) capable of learning the intricate features of all 247+ Tamil graphemes.

#### [NEW] `tamil_cnn_model.py`
- Define a PyTorch CNN architecture (e.g., a lightweight ResNet-18 or a custom 4-layer CNN optimized for 64x64 grayscale character crops).
- Include standard dataset loading, augmentation (rotations, elastic transforms to simulate handwriting), and training loops.

#### [MODIFY] `tamil_dataset.py`
- Update `CLASS_TO_TAMIL` mapping to support all 247+ classes.
- Refactor the dataset loader to feed PyTorch `DataLoader` objects.

#### [MODIFY] `recognize_manuscript.py`
- Replace `load_model()` and `classify_crop()` to load the PyTorch `.pth` model and run CNN forward passes instead of the HOG feature extraction and SVM inference.

---

### NLP Semantic Word Segmenter (P2)
Because palm-leaf manuscripts are written in *scriptura continua* (without spaces), we must computationally infer word boundaries.

#### [NEW] `tamil_nlp_segmenter.py`
- Implement a **Viterbi Decoder** or **MaxMatch** algorithm.
- Load a Tamil dictionary (Trie data structure for fast lookups).
- Function: Takes a continuous string of Tamil characters (e.g., `அம்மாவந்தாள்`) and calculates the highest probability word splits (e.g., `அம்மா வந்தாள்`).
- Integrate this as the final step in `recognize_manuscript.py` before saving `_text.txt`.

## Verification Plan

### Automated Tests
- Train the CNN and monitor loss/accuracy. Target > 95% validation accuracy on the full character set.
- Run unit tests on `tamil_nlp_segmenter.py` using known continuous Tamil strings to verify it splits them into valid words.

### Manual Verification
- Run `recognize_manuscript.py` on `128.jpg`.
- Manually inspect the `128_text.txt` to verify that the output now contains recognizable Tamil words separated by spaces, rather than a continuous string of gibberish.
