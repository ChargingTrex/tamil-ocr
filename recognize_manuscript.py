#!/usr/bin/env python3
"""
recognize_manuscript.py
=======================
Use the trained HOG+SVM character classifier to recognize Tamil characters
from palm-leaf manuscript page images (raw OR binarized).

Preprocessing:
  - RAW images   -> palmleaf_pipeline.py (denoise -> Sauvola -> crop -> hole removal)
  - BINARIZED    -> direct load + light cleanup

Pipeline per page:
  1. Preprocess (auto-detects raw vs binarized)
  2. Segment text lines via horizontal projection profile
  3. Segment characters from each line via connected components
  4. Classify each character crop with HOG + LinearSVC
  5. Output recognised Tamil text + annotated visualization

Usage:
    # Raw originals (uses palmleaf_pipeline.py)
    python recognize_manuscript.py --indir "./tamil ml dataset /NALADIYAR ORIGINAL" --outdir ./ocr_results

    # Pre-binarized (skips preprocessing)
    python recognize_manuscript.py --indir "./tamil ml dataset /NALADIYAR BINARIZED" --outdir ./ocr_results

    # Single image
    python recognize_manuscript.py --image "./tamil ml dataset /NALADIYAR ORIGINAL/125.jpg"

    # Retrain model first
    python recognize_manuscript.py --indir "..." --retrain --data ./_char_repo/Dataset
"""

import argparse
import os
import glob
import pickle
import json
import numpy as np
import cv2

from tamil_preprocess import (
    content_crop,
    remove_punch_holes,
)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
from palmleaf_pipeline import (
    process_image,
    to_grayscale_luminosity,
)
from train_baseline import hog_feat
from tamil_dataset import (
    load_dataset, stratified_split, CLASS_TO_TAMIL,
    pad_to_square,
)


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "tamil_char_svm.pkl")


def load_model(path=MODEL_PATH):
    """Load the saved HOG+SVM model dict."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[model] Loaded {path}  ({len(bundle['classes'])} classes, "
          f"img_size={bundle['img_size']})")
    return bundle


def train_and_save(data_root, path=MODEL_PATH, img_size=64):
    """Retrain HOG+SVM from the character dataset and save the bundle."""
    from sklearn.svm import LinearSVC
    X, y, names = load_dataset(data_root, size=img_size)
    tr, va, te = stratified_split(y)
    F = np.stack([hog_feat(x) for x in X])
    clf = LinearSVC(C=1.0, max_iter=3000).fit(F[tr], y[tr])

    from sklearn.metrics import accuracy_score
    print(f"[retrain] train={accuracy_score(y[tr], clf.predict(F[tr])):.3f} "
          f"val={accuracy_score(y[va], clf.predict(F[va])):.3f} "
          f"TEST={accuracy_score(y[te], clf.predict(F[te])):.3f}")

    bundle = {
        "clf": clf,
        "classes": names,
        "tamil": {n: CLASS_TO_TAMIL.get(n, "?") for n in names},
        "img_size": img_size,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[retrain] Saved model to {path}")
    return bundle


# ---------------------------------------------------------------------------
# Preprocessing — auto-detect raw vs binarized
# ---------------------------------------------------------------------------

def is_binarized(gray_img):
    """Detect if an image is already binarized (bimodal with peaks near 0/255)."""
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).ravel()
    dark_frac = hist[:32].sum() / hist.sum()
    light_frac = hist[224:].sum() / hist.sum()
    return (dark_frac + light_frac) > 0.85


def preprocess_page(img_path, outdir, sauvola_window=51, sauvola_k=0.2):
    """Preprocess a manuscript page for OCR.

    Auto-detects whether the input is raw or pre-binarized:
      - RAW:       runs tamil_preprocess.py full pipeline
                   (denoise -> grayscale -> Sauvola -> crop -> hole removal)
      - BINARIZED: loads directly, applies light cleanup only

    Returns (cleaned_binary, info_dict).
    """
    bgr = load_image(img_path)
    gray = to_grayscale_luminosity(bgr)

    if is_binarized(gray):
        # --- Pre-binarized path: skip heavy preprocessing ---
        print(f"    [preprocess] Detected PRE-BINARIZED input")
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        cropped = content_crop(binary, pad=5)
        cleaned, n_holes = remove_punch_holes(cropped)
        cleaned = _remove_specks(cleaned, min_area=25)

        return cleaned, {
            "pre_binarized": True,
            "psnr": float("inf"),
            "holes_removed": n_holes,
        }
    else:
        # --- Raw path: full palmleaf_pipeline.py pipeline ---
        print(f"    [preprocess] Detected RAW input -> running palmleaf_pipeline.py")
        result = process_image(img_path, sauvola_window=sauvola_window, sauvola_k=sauvola_k, crop_padding=5)
        cleaned = result["cleaned"]
        cleaned = _remove_specks(cleaned, min_area=25)

        # Save intermediate stages for inspection
        stem = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(outdir, f"{stem}_1_gray.png"), result["raw_gray"])
        cv2.imwrite(os.path.join(outdir, f"{stem}_2_sauvola.png"), result["binary"])
        cv2.imwrite(os.path.join(outdir, f"{stem}_3_cleaned.png"), cleaned)

        return cleaned, {
            "pre_binarized": False,
            "psnr": result["psnr"],
            "holes_removed": len(result["holes_found"]),
        }


def _remove_specks(binary_img, min_area=25):
    """Remove small noise specks from a binary image (ink=0, bg=255)."""
    inv = 255 - binary_img
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    mask = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return 255 - mask


# ---------------------------------------------------------------------------
# Line segmentation (horizontal projection)
# ---------------------------------------------------------------------------

def segment_lines(binary_img, min_line_height=15, merge_gap=8):
    """Segment text lines using horizontal projection profile.
    binary_img: ink=0, bg=255.
    Returns list of (y_start, y_end) tuples."""
    ink = (binary_img < 128).astype(np.int32)
    h_proj = ink.sum(axis=1)

    # Adaptive threshold: rows with significant ink (increased to 10% to ignore noise)
    threshold = max(h_proj.max() * 0.10, 5)

    in_line = False
    lines = []
    y_start = 0

    for y, count in enumerate(h_proj):
        if not in_line and count > threshold:
            in_line = True
            y_start = y
        elif in_line and count <= threshold:
            in_line = False
            if y - y_start >= min_line_height:
                lines.append((y_start, y))

    if in_line and len(h_proj) - y_start >= min_line_height:
        lines.append((y_start, len(h_proj)))

    # Merge lines that are very close (broken by thin horizontal gaps)
    merged = []
    for start, end in lines:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


# ---------------------------------------------------------------------------
# Character segmentation (connected components within a line)
# ---------------------------------------------------------------------------

def segment_characters(line_img, min_char_area=50, max_char_area_frac=0.4):
    """Segment individual characters from a single text line image.
    line_img: ink=0, bg=255.
    Returns list of (x, y, w, h) bounding boxes sorted left-to-right."""
    H, W = line_img.shape
    max_area = max_char_area_frac * H * W

    ink = 255 - line_img
    _, ink_bin = cv2.threshold(ink, 128, 255, cv2.THRESH_BINARY)

    # Light morphological close to merge broken strokes within a character
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_bin = cv2.morphologyEx(ink_bin, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_bin, 8)

    boxes = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_char_area:
            continue
        if area > max_area:
            continue
        if w < 4 or h < 4:
            continue
        boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    # Merge heavily overlapping boxes (fragments of the same character)
    merged = []
    for box in boxes:
        if merged:
            px, py, pw, ph = merged[-1]
            bx, by, bw, bh = box
            overlap_x = min(px + pw, bx + bw) - max(px, bx)
            overlap_y = min(py + ph, by + bh) - max(py, by)
            # Must overlap in X and at least slightly in Y to be the same character
            if overlap_x > 0.3 * min(pw, bw) and overlap_y > 0:
                nx = min(px, bx)
                ny = min(py, by)
                nw = max(px + pw, bx + bw) - nx
                nh = max(py + ph, by + bh) - ny
                merged[-1] = (nx, ny, nw, nh)
                continue
        merged.append(box)

    return merged


# ---------------------------------------------------------------------------
# Character classification
# ---------------------------------------------------------------------------

def classify_crop(crop_gray, bundle):
    """Classify a single character crop using the trained HOG+SVM.
    crop_gray: uint8 grayscale character image (any size).
    Returns (class_name, tamil_char, confidence)."""
    clf = bundle["clf"]
    classes = bundle["classes"]
    tamil = bundle["tamil"]
    size = bundle["img_size"]

    # Preprocess: pad to square, resize, normalize to [0,1]
    sq = pad_to_square(crop_gray, pad_value=255)
    resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    normed = resized.astype(np.float32) / 255.0

    feat = hog_feat(normed).reshape(1, -1)
    pred_idx = clf.predict(feat)[0]

    # Decision function scores for confidence
    scores = clf.decision_function(feat)[0]
    exp_scores = np.exp(scores - scores.max())
    probs = exp_scores / exp_scores.sum()
    confidence = probs[pred_idx]

    name = classes[pred_idx]
    return name, tamil.get(name, "?"), float(confidence)


# ---------------------------------------------------------------------------
# Full page recognition
# ---------------------------------------------------------------------------

def recognize_page(img_path, bundle, outdir, min_confidence=0.0,
                   sauvola_window=51, sauvola_k=0.2):
    """Run full OCR on a manuscript page image.

    Steps:
      1. tamil_preprocess.py preprocessing (for raw) or direct load (binarized)
      2. Line segmentation via horizontal projection
      3. Character segmentation per line via connected components
      4. HOG+SVM classification per character

    Returns a dict with lines, characters, and annotated image.
    """
    print(f"\n  Processing: {os.path.basename(img_path)}")

    # Step 1: Preprocess
    cleaned, info = preprocess_page(img_path, outdir, sauvola_window, sauvola_k)
    psnr = info.get("psnr", 0)
    holes = info.get("holes_removed", 0)
    is_pre = info.get("pre_binarized", False)
    print(f"    [info] PSNR={psnr:.1f} dB, holes={holes}, "
          f"size={cleaned.shape[1]}x{cleaned.shape[0]}")

    # Step 2: Line segmentation
    lines = segment_lines(cleaned)
    print(f"    [segment] Found {len(lines)} text lines")

    # Visualization
    vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    colors = [
        (46, 204, 113),   # green
        (52, 152, 219),   # blue
        (231, 76, 60),    # red
        (241, 196, 15),   # yellow
        (155, 89, 182),   # purple
        (230, 126, 34),   # orange
        (26, 188, 156),   # teal
        (192, 57, 43),    # dark red
    ]

    page_result = {
        "file": os.path.basename(img_path),
        "pre_binarized": is_pre,
        "psnr": round(psnr, 2) if psnr != float("inf") else "inf",
        "holes_removed": holes,
        "image_size": [int(cleaned.shape[1]), int(cleaned.shape[0])],
        "lines": [],
    }

    for li, (y0, y1) in enumerate(lines):
        color = colors[li % len(colors)]
        line_img = cleaned[y0:y1, :]

        # Draw line boundary
        cv2.rectangle(vis, (0, y0), (vis.shape[1] - 1, y1), color, 2)
        cv2.putText(vis, f"L{li}", (5, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Step 3: Character segmentation
        char_boxes = segment_characters(line_img)
        line_text = []
        line_chars = []

        for (cx, cy, cw, ch) in char_boxes:
            crop = line_img[cy:cy + ch, cx:cx + cw]

            # Step 4: Classification
            name, tamil_char, conf = classify_crop(crop, bundle)

            if conf >= min_confidence:
                line_text.append(tamil_char)
                line_chars.append({
                    "bbox": [int(cx), int(y0 + cy), int(cw), int(ch)],
                    "class": name,
                    "tamil": tamil_char,
                    "confidence": round(conf, 3),
                })

                # Draw character box on visualization
                cv2.rectangle(vis,
                              (cx, y0 + cy),
                              (cx + cw, y0 + cy + ch),
                              color, 1)

        tamil_line = "".join(line_text)
        page_result["lines"].append({
            "line_idx": li,
            "y_range": [int(y0), int(y1)],
            "text": tamil_line,
            "n_chars": len(line_chars),
            "characters": line_chars,
        })
        print(f"    Line {li}: {len(line_chars):3d} chars -> {tamil_line}")

    page_result["annotated_image"] = vis
    return page_result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_folder(indir, outdir, bundle, min_confidence=0.0,
                   sauvola_window=51, sauvola_k=0.2):
    """Process all manuscript images in a folder."""
    os.makedirs(outdir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(indir, ext))
        paths += glob.glob(os.path.join(indir, ext.upper()))
    paths = sorted(set(paths))

    if not paths:
        print(f"No images found in {indir}")
        return

    print(f"\n{'='*60}")
    print(f"Tamil Palm-Leaf OCR")
    print(f"  Input : {indir} ({len(paths)} images)")
    print(f"  Output: {outdir}")
    print(f"  Preprocessing: palmleaf_pipeline.py (auto raw/binarized)")
    print(f"{'='*60}")

    all_results = []

    for img_path in paths:
        try:
            result = recognize_page(img_path, bundle, outdir, min_confidence,
                                    sauvola_window, sauvola_k)

            stem = os.path.splitext(os.path.basename(img_path))[0]

            # Save annotated visualization
            vis_path = os.path.join(outdir, f"{stem}_annotated.png")
            cv2.imwrite(vis_path, result["annotated_image"])

            # Save text output
            txt_path = os.path.join(outdir, f"{stem}_text.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for line in result["lines"]:
                    f.write(line["text"] + "\n")

            # Collect JSON result (without the image array)
            result_json = {k: v for k, v in result.items()
                           if k != "annotated_image"}
            all_results.append(result_json)

            total_chars = sum(l["n_chars"] for l in result["lines"])
            print(f"    -> Saved: {stem}_annotated.png ({total_chars} chars)")

        except Exception as ex:
            print(f"  ! {os.path.basename(img_path)}: {ex}")

    # Save combined JSON results
    json_path = os.path.join(outdir, "recognition_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Summary
    total_pages = len(all_results)
    total_lines = sum(len(r["lines"]) for r in all_results)
    total_chars = sum(
        sum(l["n_chars"] for l in r["lines"]) for r in all_results
    )
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Pages processed : {total_pages}")
    print(f"  Lines found     : {total_lines}")
    print(f"  Chars recognized: {total_chars}")
    print(f"  Output directory: {outdir}")
    print(f"  Results JSON    : {json_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Recognize Tamil characters from palm-leaf manuscripts.\n"
                    "Uses palmleaf_pipeline.py for raw images, direct load for binarized.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", help="Single image to process")
    ap.add_argument("--indir", help="Folder of manuscript images")
    ap.add_argument("--outdir", default="./ocr_results",
                    help="Output directory (default: ./ocr_results)")
    ap.add_argument("--model", default=MODEL_PATH,
                    help="Path to saved model pkl")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain the model before inference")
    ap.add_argument("--data", default="./_char_repo/Dataset",
                    help="Character dataset root (for --retrain)")
    ap.add_argument("--sauvola-window", type=int, default=51,
                    help="Sauvola window size (default: 51)")
    ap.add_argument("--sauvola-k", type=float, default=0.2,
                    help="Sauvola k parameter (default: 0.2)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="Min confidence to include a character (0-1)")
    args = ap.parse_args()

    if not args.image and not args.indir:
        ap.error("Provide --image or --indir")

    # Load or retrain model
    if args.retrain:
        bundle = train_and_save(args.data, args.model)
    else:
        bundle = load_model(args.model)

    if args.image:
        os.makedirs(args.outdir, exist_ok=True)
        result = recognize_page(args.image, bundle, args.outdir,
                                args.min_confidence,
                                args.sauvola_window, args.sauvola_k)

        stem = os.path.splitext(os.path.basename(args.image))[0]
        vis_path = os.path.join(args.outdir, f"{stem}_annotated.png")
        cv2.imwrite(vis_path, result["annotated_image"])

        txt_path = os.path.join(args.outdir, f"{stem}_text.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in result["lines"]:
                f.write(line["text"] + "\n")

        total = sum(l["n_chars"] for l in result["lines"])
        print(f"\nSaved: {vis_path} ({total} characters)")
        print(f"       {txt_path}")

    elif args.indir:
        process_folder(args.indir, args.outdir, bundle, args.min_confidence,
                       args.sauvola_window, args.sauvola_k)


if __name__ == "__main__":
    main()
