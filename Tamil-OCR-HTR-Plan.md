# Plan: Tamil Palm-Leaf & Stone-Inscription OCR/HTR System

*A build plan for recognising and converting Tamil palm-leaf manuscript images and stone/copper-plate inscription images into searchable, lexicon-linked text.*

Grounded in two things: the CICT Digital Library architecture (HTR-VT + TrOCR, the CICT-PLM-GT ground-truth corpus, five active research tracks) and the actual published Tamil-OCR research landscape (THPLMD, IndicSTR12, Tamil-Brahmi binarisation work). Structured like the detector → recogniser → cleanup pipeline you already built for MangaTranslate — same shape, harder domain shift — with an eye toward an open-source deployment.

---

## 0. The one decision that shapes everything: scope palm-leaf first

Palm leaf and stone inscriptions wear one label but are two different projects with different failure modes:

- **Palm-leaf manuscripts (ஓலைச் சுவடி)** — stylus-incised / ink cursive on a curved, fibrous, degradable surface. Failures are cracking, staining, insect damage, faded strokes, connected letterforms. Mostly post-medieval to early-modern Tamil, so the script is close to what you already read — a paleographer does *not* have to validate every single label.
- **Stone inscriptions & copper-plates (கல்வெட்டு / செப்பேடு)** — *carved*, not written, spanning ~2,000 years: Tamil-Brahmi (~3rd c. BCE), Vaṭṭeḻuttu, and medieval Chola-era Tamil (~9th–13th c.) all differ meaningfully from modern Tamil Unicode. Failures are erosion, lighting-dependent groove visibility, and — critically — you need an **expert-validated glyph-to-Unicode mapping per era** before a model even has correct labels to learn from. That mapping is a research question, not an engineering task.

**Recommendation: build palm-leaf first, treat stone as Phase 2.** More surviving material, closer to modern script, and an actual existing dataset trail to build on. Everything below is palm-leaf-first with an epigraphy extension in §5.

Success metric throughout: **Character Error Rate (CER) / Word Error Rate (WER)** on a held-out, expert-reviewed test set, plus downstream retrieval (can a scholar find the right verse), plus honest handling of the unreadable.

---

## 1. Why this is hard (design constraints)

**Script complexity.** Tamil is an abugida: base consonants, vowel signs (uyirmei), special glyphs (aytham ஃ, puḷḷi). One user-level "character" is often several visual components. Historic material adds Grantha letters (ஷ, ஜ, etc.) for Sanskrit-origin words. Model grapheme clusters and normalise to canonical Unicode — not naive codepoints.

**Paleographic drift.** Letterforms shift by century and scribe. CICT's own notes document the ந-for-ன gradient, ள/ழ confusion, ந்ந doubling, ற-for-ல word-final substitution. Stone spans far more (Tamil-Brahmi → Vaṭṭeḻuttu → medieval). Carry scribe/era metadata; expect domain adaptation.

**Degradation is two different things.** On palm leaf it's *noise* (recoverable with restoration). On stone, past a threshold, it's *missing information* — the character is physically gone. Build "unreadable / uncertain" as a first-class model output, not a bug to eliminate.

**No word spacing.** Classical Tamil often runs words together with minimal punctuation, so segmentation is a modelling problem, not a whitespace split.

**Low-resource labels.** Verified transcriptions are scarce and expensive. The largest published palm-leaf ground-truth sets are tiny (THPLMD ≈ 271 samples). The whole plan is built around label efficiency: synthetic data, transfer learning, self-supervision, active learning.

---

## 2. What's already been tried (don't reinvent it)

Prior work is real but small and fragmented — mostly isolated-character classifiers, not full detect-then-recognise pipelines.

**Palm leaf.** A CNN cursive-character classifier reported ~94% accuracy vs ~88% for a plain ResNet baseline on a small in-house set. The most usable *public* dataset is **THPLMD** — raw Nikon captures, Otsu-binarised, ~271 deteriorated samples across three classical texts (Naladiyar, Tholkappiyam, Thirikadugam) — tiny by DL standards. A 2024 paper built a larger set from the *Agasthiyar Vaithiya Kaviyam* manuscript — ~1,500 passages across 502 pages at 300–600 dpi — pairing a segmentation stage with a recognition stage; that's the closest existing thing to the full pipeline you want. CICT's own engine reports ~4,037 training lines across six critical editions and CER ≈ 1.6 (vs Tesseract ≈ 3.2).

**Stone.** Binarisation is the recurring bottleneck: a 2024 Tamil-Brahmi ("Tamiḻi") paper built a custom multi-level filter reaching ~92% binarisation accuracy vs ~74% for prior methods, because stone's uneven surface and similar fore/background contrast defeat standard thresholding. Recognition work skews to small ANN/CNN classifiers mapping 9th–12th c. inscribed characters to modern Unicode on hand-built sets, and explicitly names the **absence of an annotated dataset spanning 3rd c. BCE → 12th c. CE** as the open problem — again, the mapping, not the model.

**Modern Tamil as a transfer base.** **IndicSTR12** is the one dataset here with real scale — 27,000+ word images across 12 Indian languages including Tamil (≥1,000 word images/language), with realistic blur/occlusion/perspective, benchmarking PARSeq, CRNN and STARNet, with **PARSeq consistently winning**. It won't know ancient letterforms, but it's a legitimate pretrained starting point for the recognition head's visual features before fine-tuning on your crops (the IndicSTR12 authors themselves found intra-Indic cross-lingual transfer useful).

---

## 3. Data strategy — this is the actual project

Model code is not the bottleneck. Access and labels are. Most of the timeline lives here.

**Source real images (institutional contact, not scraping).**
- Palm leaf: **French Institute of Pondicherry** (~8,500 palm-leaf manuscripts, UNESCO-registered), **Tamil University Thanjavur** (Dept. of Epigraphy & Archaeology), the **Tamil Digital Library**, and — if you can partner — **CICT** (~865 bundles / ~100,000 leaves, six critical editions, the GT corpus).
- Stone: **Tamil Nadu State Dept. of Archaeology**, **ASI Epigraphy branch**, and the published *South Indian Inscriptions* volumes.
- Reality check: this material is mostly *not* bulk-downloadable. Access is a relationship, and it's the true constraint — see §9.

**Anchor the benchmark on verified ground truth.** If you get CICT access, freeze **CICT-PLM-GT-001 → GT-025** (~250 verified Tirukkural couplets, Tamil + Grantha, 12 documented paleographic features) as the untouched gold test set, and use the six critical editions as canonical reference text. If not, use THPLMD / the Agasthiyar set as your seed and build your own expert-reviewed held-out split.

**Four ways to multiply scarce labels:**
1. **Forced text–image alignment.** Where a folio's underlying work is known (a Tirukkural or Naladiyar page), auto-align the canonical edition text to line images to mint training pairs cheaply. Highest-leverage tactic when you have both folios and editions.
2. **Synthetic bootstrap data.** Render Tamil/Grantha Unicode in varied manuscript-style fonts onto textured palm-leaf / stone backgrounds, then apply elastic distortion, crack/erosion overlays, ink-bleed and lighting variation — the SynthText/MJSynth trick that IndicSTR12 replicated for Indic. This is what lets you pretrain a recogniser *before* you have thousands of real crops. Do not expect to skip it.
3. **Self-supervised pretraining.** Use the large pool of *unlabelled* leaves to pretrain the vision encoder (masked-image / span-mask, matching HTR-VT's regularisation) so only the decoder needs scarce labels.
4. **Active learning + human-in-the-loop.** Ship a correction UI (Label Studio with a Tamil font stack, or eScriptorium); route lowest-confidence lines to scholars first; feed corrections back. Every verified page becomes new ground truth.

**Formats & governance.** Store PAGE-XML / ALTO (layout + baselines + text) plus JSONL (TrOCR/VLM) — matching CICT's own export formats so tooling interoperates. Track per-specimen metadata (script, era, scribe, institution, condition, verification status, licence — CICT specimens are CC BY-NC 4.0). **Split train/val/test by specimen *and* scribe** to prevent leakage; hold out whole texts to test generalisation.

---

## 4. Architecture — detector → recogniser → cleanup

Same three-stage shape as MangaTranslate (RT-DETR → PaddleOCR → cleanup), retargeted. Keep a **modular pipeline as the dependable backbone** and run an **end-to-end VLM as a challenger** behind the same benchmark.

### Stage 1 — Detection / layout
Localise text lines (and words where spacing allows) on the folio or stone photo. Prefer a fine-tuned scene-text detector (**DBNet**, **CRAFT**) or a light **DETR/YOLO** variant over classical projection-profile segmentation, because leaf curvature and stone erosion break the clean-horizontal-line assumption. For dense folios, a palm-leaf layout model (PALM-LAY) handles columns, marginalia, title cells. Precede this with **palm-leaf-specific restoration/binarisation** (Res-UNet / PLM-Res-U-Net rather than generic Otsu) plus deskew/dewarp.

### Stage 2 — Recognition (the actual STR)
Per line/word crop → Tamil Unicode sequence.
- **Baseline:** **CRNN** (CNN backbone + BiLSTM + CTC). Cheap, well understood, trains on a free Colab/Kaggle GPU, gives a first checkpoint and a real CER number fast.
- **Stretch:** **PARSeq** or a **TrOCR / HTR-VT** style ViT-encoder + transformer-decoder, **warm-started from IndicSTR12 PARSeq weights** rather than from scratch — transfer helps with general Tamil glyph geometry even though the target script differs. HTR-VT's span-mask regularisation is well suited to the low-data regime; TrOCR gives a language-aware decoder. These can be ensembled (this is the CICT stack).

### Stage 3 — Post-processing / correction
A Tamil language model or n-gram lexicon corrects recognition noise. Two forms:
- **Lexicon link** to a classical dictionary (CICT's 6,758-entry lexicon / Madras Tamil Lexicon) — adds scholarly value *and* acts as a morphological constraint that catches errors.
- **LLM re-ranking** (fine-tuned **ByT5** — byte-level, ideal for Tamil's combining characters — or a small Llama-class model); the literature reports ~50–60% CER reduction on top of base HTR.
- This stage matters *more* for inscriptions: stone text is highly formulaic (regnal year, king's name, donor, land-grant terms repeat constantly), so a domain LM constrained to that vocabulary fixes many erosion-induced errors a generic model can't.

### Challenger — end-to-end VLM OCR
Multimodal models skip explicit segmentation and read a whole folio in one pass; strong for low-resource historical scripts. Fine-tune/evaluate open-weight **olmOCR-2-7B**, **PaddleOCR-VL** (100+ languages, lightweight), **DeepSeek-OCR** (resolution modes), **Qwen3-VL**, history-specific **CHURRO**. Likely outcome: VLM wins on layout-free reading and context, the modular pipeline wins on precise character fidelity and *calibrated per-character confidence* (essential for scholarly trust). Production may **route** (VLM for messy pages, modular for GT-quality work) or **fuse** (VLM drafts, modular verifies). Decide empirically; never ship VLM output as certified text without verification.

---

## 5. Stone inscriptions & copper-plates (Phase 2 extension)

Reuse the recognition core; replace imaging and data.

**Imaging is the differentiator** — flat photos fail on eroded 3D grooves:
- **RTI (Reflectance Transformation Imaging)** / Polynomial Texture Mapping — multi-angle capture, then virtual relighting to reveal incision depth invisible under fixed light.
- **Raking-light photography** — the cheap version of the same idea; flag it to whoever captures images, it can matter more than any model choice.
- **Multispectral (UV→NIR) + PCA** for faded strokes; **photogrammetry/3D** for curved stone, unwrapped to a depth/normal map the recogniser can read.
- Traditional **estampage (inked squeeze)** images are an already-common Tamil-epigraphy data source.

**Script coverage.** Build era/script classifiers and per-script recognisers — no single model spans Tamil-Brahmi to medieval Tamil. Ground-truth validity is the hard part: **Iravatham Mahadevan's Tamil-Brahmi concordances** are the standard reference, but you want a qualified paleographer/epigraphist validating the glyph→Unicode mapping, not just citing it. No architecture fixes wrong labels.

---

## 6. Phased roadmap (≈ a semester of work for a small team)

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Scope + literature/data audit | 1–2 | Palm-leaf-first locked; inventory of data you can *actually* access; eval harness (CER/WER) + specimen/scribe-safe split defined |
| 1. Data acquisition + annotation tooling | 2–3 | Label Studio (Tamil font stack) pipeline; first ~300–500 annotated line crops; PAGE-XML/JSONL schema |
| 2. Synthetic data + preprocessing | 2 | Synthetic pretraining set; restoration/binarisation + deskew/dewarp pipeline |
| 3. Detection model | 3–4 | Line/word detector (DBNet/CRAFT/DETR) fine-tuned on your crops |
| 4. Recognition baseline | 3–4 | CRNN+CTC: synthetic pretrain → real fine-tune; first measured CER/WER |
| 5. Recognition v2 + post-processing LM | 2–3 | PARSeq/TrOCR fine-tune from IndicSTR12 weights + Tamil LM / lexicon correction; VLM challenger benchmarked |
| 6. Evaluation | 1–2 | Expert-reviewed held-out test set; compare vs Tesseract / Google Vision baseline; per-script, per-era, per-degradation CER |
| 7. Deployment | ongoing | Web tool: upload → predicted text + confidence + correction loop; open-source on GitHub |
| 8. Epigraphy (parallel, after Phase 4 proves the core) | — | RTI/raking-light capture; era/script classifier; seed inscription corpus; transfer the stack |

Everything after Phase 4 is measured improvement against that first baseline number.

---

## 7. Evaluation

- **Primary:** CER + WER on a frozen, expert-reviewed test split, reported **per script (Tamil vs Grantha), per era, per degradation level** — aggregates hide where the model fails.
- **Benchmark discipline:** never fine-tune on the held-out set (freeze GT-001→025 if using CICT).
- **Confidence calibration:** trustworthy per-character confidence so low-confidence output auto-routes to human review, and genuinely-gone characters are marked *unreadable*, not hallucinated.
- **Paleographic stress tests:** targeted sets for known confusions (ள/ழ, ந/ன, ற/ல word-final, aytham dissolution).
- **Baselines to beat:** Tesseract Tamil and a commercial API (Google Vision) — cheap sanity checks that your effort actually pays off.
- **Human agreement:** compare against inter-scholar transcription variance; aim for expert-parity on clean material, flag-for-review on damaged.

---

## 8. Tooling & infrastructure

- **Frameworks:** PyTorch; Hugging Face Transformers (TrOCR/PARSeq/ByT5/VLMs); Kraken/eScriptorium or Calamari for HTR plumbing + PAGE-XML.
- **Labelling:** Label Studio (Tamil font stack) or eScriptorium (manuscript-native, PAGE-XML).
- **Compute:** a free/low-cost Colab/Kaggle GPU trains the CRNN baseline; VLM fine-tuning wants a bigger GPU. Synthetic generation is CPU-cheap. Compute is *not* your constraint — access is.
- **Tracking & repro:** Weights & Biases or MLflow; DVC for dataset versioning; pin seeds/splits/checkpoints; log CER per run.
- **Serving:** containerised inference API; IIIF image API for delivery (matches CICT's direction); export JSONL + PAGE-XML. FOSS-friendly stack given an open-source release.

---

## 9. What will actually break this if ignored

- **Access, not compute, is the real constraint.** You can train a CRNN on a free GPU tier; you cannot manufacture 8,500 scans out of nothing. Start the institutional outreach (IFP, Tamil University, CICT) in week 1 — it has the longest lead time.
- **Dataset size.** ~271 samples is the published baseline. Synthetic augmentation is mandatory, not optional, to reach usable accuracy.
- **Ground-truth validity for old scripts.** No architecture fixes wrong labels. Near Tamil-Brahmi / Vaṭṭeḻuttu you need a paleographer in the loop (Mahadevan as reference, an expert for validation).
- **Erosion is missing information, not noise.** Past a threshold no model recovers a vanished character. "Unreadable/uncertain" is a legitimate output.
- **Overfitting to Tirukkural / one scribe.** Split by specimen and scribe; hold out whole texts; report per-era CER.
- **Chasing SOTA VLMs.** The frontier moves every quarter. Keep the modular backbone stable; treat VLMs as swappable challengers behind one benchmark.

---

## 10. Team

Minimum: an ML engineer (detection + STR + fine-tuning), a data engineer (imaging, alignment, labelling tooling), and — non-negotiable — at least one **Tamil paleography scholar** for ground truth and hard readings. For epigraphy add a heritage-imaging specialist (RTI/multispectral/photogrammetry) and an epigraphist. Institutional partnerships (CICT, university Tamil departments, ASI Epigraphy) substitute for some hiring.

---

## 11. Recommended first move (this month)

1. **Send the outreach emails now** — IFP, Tamil University Thanjavur, CICT — longest lead time.
2. Stand up the CER/WER eval harness and a specimen/scribe-safe split.
3. Generate a synthetic Tamil palm-leaf set and pretrain a **CRNN+CTC** baseline; fine-tune on THPLMD / whatever real crops you have; get a first CER number.
4. Deploy the Label Studio correction UI so every scholar hour produces new labels from day one.

---

## Appendix A — Segmentation-first build order (near-term coding track)

Reprioritisation: since recognition is already handled, the leverage is turning **degraded, unlabelled leaves into clean, correctly-ordered line/character crops** — no labels needed. This mirrors the two validated papers (THPLMD; the Heritage Science segmentation paper). Start here.

**Phase 0 — Environment & data sanity.** Download THPLMD (Mendeley `doi:10.17632/xz9rx5wfc5.1` — 262 raw + 199 Otsu-binarised images from Naladiyar, Tholkappiyam, Thirikadugam). Stack: Python + OpenCV + numpy + scikit-image + scipy. Build a `show_pair(raw, gt)` visualiser — it becomes your reusable eval harness. *(Implemented in the starter script.)*

**Phase 1 — Denoising.** Grayscale via luminosity (`0.21R + 0.72G + 0.07B`, not default cvtColor). Compare `GaussianBlur` vs `fastNlMeansDenoisingColored` vs `+medianBlur`, scored with a `compute_psnr()` function (replicate Table 3). Quantitative "did preprocessing help" signal before segmentation. *(Implemented.)*

**Phase 2 — Binarisation.** **Go straight to Sauvola, skip Otsu** — the Heritage Science paper shows Sauvola beating Otsu/Triangle/Niblack on real degraded leaves. `skimage.filters.threshold_sauvola(gray, window_size=25, k=0.2)`; sweep `window_size`/`k`. Sanity-check against THPLMD's Otsu GT. *(Implemented, with a sweep helper.)*

**Phase 3 — Cropping & punch-hole removal.** Content-crop to the outermost contour (+~5px). Punch holes contaminate segmentation as fake character blobs — remove early: sort contours by area, classify holes (aspect ratio ≈1, area > threshold) vs edges (elongated), build a mask, XOR against the binary. *(Implemented.)*

**Phase 4 — Line segmentation (the hard part).** Curvature, not noise, is the challenge — lines droop/arch, so naive HPP misassigns middle characters. **Trisect into 3 horizontal thirds first**, then per third: Horizontal Projection Profile (`np.sum(binary==0, axis=1)`), smooth (`gaussian_filter1d`), `scipy.signal.find_peaks`, split at minima, filter by black/white "0/1 ratio" to drop empty slices. Trisection + ratio filtering is what takes seg accuracy from ~82% to ~98%. Save each line numbered in order.

**Phase 5 — Character segmentation.** `cv2.findContours` per line; filter tiny noise by area; bounding rects → IoU/NMS to merge overlaps; sort left-to-right by x. Save crops as `{manuscript}_{trisection}_{line}_{char}.png` — ordering matters for later sequence reconstruction.

**Phase 6 — Unlabelled corpus + cheap labelling.** Resize crops to 32×32, embed (raw pixels or a small pretrained CNN), cluster (KMeans/HDBSCAN). Label whole clusters at once (show 20, type the character once) instead of one-by-one. Bootstrap: hand-label ~500–1000 chars (Maheswari's 125 writable classes = 12 vowels + 18 consonants + 216 compounds collapsed), train a first-pass CNN, auto-label the rest with active-learning correction on low-confidence.

**Phase 7 — Segmentation eval harness.** `segmentation_accuracy()` = valid segmented crops vs known character count (replicate Eq. 19). Tune Phases 3–5 against a number, not eyeballing.

**Start today:** run Phases 1–3 as one script against THPLMD raw images; compare your Sauvola output to their Otsu GT for a fast feedback loop before touching line/character logic. The starter script `tamil_preprocess.py` does exactly this.

**Data sources (segmentation track):** THPLMD (Mendeley, above); Malayalam HMPLMD and other regional palm-leaf sets (benchmark architecture/augmentation); Maheswari et al.'s set (Agasthiyar Vaithiya Kaviyam 1500, Ramayanam, Thiruvilayadal — not public, "available on reasonable request", email them). Institutions: Tamil Digital Library, U.Ve. Swaminatha Iyer Library (THPLMD source), Government Oriental Manuscripts Library Chennai, French Institute of Pondicherry, Roja Muthiah Research Library, Tamil Nadu Archives. DIY capture: flatbed scanner 300–600 dpi (more consistent for OCR than DSLR).

---

## Appendix B — Existing tools, datasets & the hackathon framing

Arun's concept (#4, "Classical Literature & Palm-Leaf Manuscript Digitizer") reframes this as a **three-part product**, not just OCR: **digitise → translate → summarise**, aligned to the Tamil Virtual Academy's preservation mission. That reframing plus the resource links below change the build from "train models from scratch" to "**assemble and fine-tune existing components.**"

### Full product pipeline (hackathon-aligned)

```
image → [preprocess/segment] → [detect+recognise] → [normalise+correct]
      → transliterate/translate (modern Tamil / English) → semantic summary → search index
```

Stages 1–3 are the OCR core (Appendix A + §4). Arun's differentiator is the **NLP tail**: transliteration of archaic → modern Tamil, translation to English, and semantic summaries so researchers can index and search un-digitised manuscripts. Judges reward the tail because it's what makes the archive *usable*.

### Ready-made components to leverage (don't rebuild)

- **`ocr_tamil` (gnana70, MIT)** — `pip install ocr_tamil`. CRAFT text detection + **PARSeq** recognition; **Tamil >95% / English >98%** on natural scenes, 10–40% faster than EasyOCR/Tesseract, bilingual, HF Space + Colab. Handwritten support is **experimental** (demoed on a Bharathiyar poem). *Use as the recognition/detection baseline; fine-tune PARSeq on palm-leaf crops. Note its stated gaps — no paragraph/reading-order/skew/dewarp for documents — which is exactly what Appendix A's preprocessing supplies.* This validates the CRAFT+PARSeq stack the earlier plan proposed.
- **`GnanaPrasath/ocr_tamil` / `sabaridsnfuji/Tamil_Offline_Handwritten_OCR` (HF)** — pretrained checkpoints to start from for printed and offline-handwritten Tamil.
- **`tamil-tokenizer` (kactlabs, Apache-2.0)** — `pip install tamil-tokenizer`. Word/sentence/character/**syllable/grapheme** tokenisation, **Unicode NFC normalisation**, Tamil-digit standardisation (௦–௯→0–9), zero-width cleanup, script/validation analysis. *Use in the normalisation stage (§4 stage 3) and to build the lexicon-link keys.*
- **`tamil-morph-tokenizer` (Indic-AI-Experiments)** — morphological segmentation; useful for the search index and morphology-aware post-correction (Tamil is agglutinative, so morpheme-level indexing beats whole-word).
- **`SadhanaParameswaran/Character-Recognition-from-Tamil-Palm-Leaves`** — a small but directly-on-point reference: 21 palm-leaf character classes, character-extraction notebooks (pdf→jpg, resize, segment), 8 classifier models, a Streamlit UI. Good scaffolding to read before writing your own segmentation/classification.

### Dataset catalogue (superset of §3)

| Dataset | What | Use |
|---|---|---|
| **THPLMD** (Mendeley `xz9rx5wfc5`; PMC10864864; ScienceDirect S2352340924000738) | 262 raw + 199 Otsu-binarised palm-leaf images, 3 classical texts | Preprocessing/binarisation train+test |
| **Mendeley `b7vhz7z83k`** | Companion Tamil palm-leaf / Data-in-Brief set | Extra real degraded samples |
| **uTHCD** (Kaggle, faizalhajamohideen) | Unconstrained Tamil Handwritten Char DB (~156 classes, offline+online) | Pretrain the char/recognition head |
| **HP Labs `hpl-tamil-iso-char`** (lipitk) | 156-class isolated Tamil handwritten chars | Classic handwritten pretraining/benchmark |
| **CICT-PLM-GT-001→025** | 250 verified Tirukkural couplets, Tamil+Grantha, 12 paleographic features | Gold benchmark (freeze) |
| **IFP Pondicherry** (digitalcollections.ifpindia.org, item-set 307452) | Large digitised palm-leaf manuscript collection | Real unlabelled corpus (self-supervision, alignment) |
| **UTSC Toronto Tamil** (tamil.digital.utsc) | Digitised Tamil manuscript items | Additional real material |

### Optional accessibility layer (voice)

The links include **Whisper-Tamil-large-v2** (vasista22) and **IITM ASR** Tamil models. Not part of OCR, but they enable a "read-aloud / voice-search" feature over the digitised text — pairs naturally with CICT's existing Tamil audio lexicon and strengthens the Tamil Virtual Academy accessibility story for a demo.

### Reference papers

- **Nature Heritage Science `s40494-024-01438-4`** — the palm-leaf segmentation paper Appendix A is built on (Sauvola, trisection + HPP, punch-hole removal, ~98% seg accuracy). *(Full text saved locally; ~98k chars, not re-quoted here.)*
- **Nature Scientific Reports `s41598-026-36330-7`** (2026) — recent Tamil palm-leaf recognition method; review for current SOTA numbers before finalising the recognition choice. *(Full text saved locally.)*

### What this means for the plan

Recognition is a **solved-enough, off-the-shelf** problem (`ocr_tamil` + PARSeq fine-tune). Your real, judge-winning work is: (1) the **preprocessing/segmentation** that makes degraded leaves readable (Appendix A), (2) **domain fine-tuning** on palm-leaf crops, and (3) the **NLP tail** — transliterate → translate → summarise → searchable index. Scope the hackathon MVP as: `ocr_tamil` on a cleaned THPLMD leaf → `tamil-tokenizer` normalise → a translation+summary call → a searchable page. Everything harder (Grantha, stone inscriptions, scribe attribution) is a stretch goal.

---

## Sources & notes

- **CICT Digital Library of Tamil Palm-Leaf Manuscripts** — [digitalarchives.cict.in](https://www.digitalarchives.cict.in/) (HTR-VT + TrOCR stack, GT corpus CICT-PLM-GT-001→025, five research tracks, six-stage roadmap, CER figures).
- **Public datasets:** [THPLMD](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10864864/) (~271 samples); IndicSTR12 (27k+ word images, PARSeq/CRNN/STARNet); *Agasthiyar Vaithiya Kaviyam* set (~1,500 passages / 502 pages, 2024). Related Tamil palm-leaf work: [SASTRA](https://knowledgeconnect.sastra.edu/theses/206/), [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3406209).
- **Methods:** HTR-VT (Li et al., *Pattern Recognition* 2025); TrOCR (Li et al., AAAI 2023); PARSeq / CRNN / DBNet / CRAFT; ByT5 post-correction; Res-UNet / PLM-Res-U-Net restoration; PALM-LAY layout; VLM-OCR — [olmOCR-2](https://arxiv.org/pdf/2510.19817), PaddleOCR-VL, DeepSeek-OCR, CHURRO.
- **Epigraphy reference:** Iravatham Mahadevan, Tamil-Brahmi concordances; *South Indian Inscriptions*.
- **Off-the-shelf tools:** [ocr_tamil](https://github.com/gnana70/tamil_ocr) (MIT; CRAFT+PARSeq), [GnanaPrasath/ocr_tamil](https://huggingface.co/GnanaPrasath/ocr_tamil) + [sabaridsnfuji handwritten OCR](https://huggingface.co/sabaridsnfuji/Tamil_Offline_Handwritten_OCR) (HF), [tamil-tokenizer](https://github.com/kactlabs/tamil-tokenizer) (Apache-2.0), [tamil-morph-tokenizer](https://github.com/Indic-AI-Experiments/tamil-morph-tokenizer), [Sadhana palm-leaf repo](https://github.com/SadhanaParameswaran/Character-Recognition-from-Tamil-Palm-Leaves).
- **Datasets:** [uTHCD (Kaggle)](https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database), [HP Labs isolated chars](https://lipitk.sourceforge.net/datasets/tamilchardata.htm), [THPLMD data article](https://www.sciencedirect.com/science/article/pii/S2352340924000738) / [Mendeley b7vhz7z83k](https://data.mendeley.com/datasets/b7vhz7z83k/1), [IFP Pondicherry manuscripts](https://digitalcollections.ifpindia.org/s/manuscripts/item-set/307452), [UTSC Toronto Tamil](https://tamil.digital.utsc.utoronto.ca/61220/utsc35338).
- **Papers:** [Nature Heritage Science 2024](https://www.nature.com/articles/s40494-024-01438-4) (segmentation), [Nature Scientific Reports 2026](https://www.nature.com/articles/s41598-026-36330-7).
- **Optional voice layer:** [Whisper-Tamil-large-v2](https://huggingface.co/vasista22/whisper-tamil-large-v2), [IITM ASR](https://asr.iitm.ac.in/models).
- The two claude.ai share links could not be fetched (client-rendered → empty); the second link's research content was supplied directly and is folded in above.
- CER/accuracy figures attributed to CICT and prior papers are as reported and should be reproduced on your own held-out split before being relied upon.
