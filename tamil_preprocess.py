#!/usr/bin/env python3
"""
tamil_preprocess.py
===================
Phase 0-3 starter pipeline for Tamil palm-leaf manuscript OCR/HTR.

Turns a raw, degraded palm-leaf image into a clean binarized image ready for
line/character segmentation -- no labels required. Mirrors the validated
approach in the THPLMD paper and the Heritage Science segmentation paper.

Pipeline:  raw -> luminosity grayscale -> denoise -> Sauvola binarize
                -> content-crop -> punch-hole removal

Dataset:   THPLMD, Mendeley  doi:10.17632/xz9rx5wfc5.1
           (262 raw + 199 Otsu-binarized images)

Deps:      pip install opencv-python numpy scikit-image scipy matplotlib

Usage:
    # visualize a raw/ground-truth pair
    python tamil_preprocess.py show --raw path/to/raw.jpg --gt path/to/gt.png

    # run the full preprocess on one image, write outputs next to it
    python tamil_preprocess.py run --raw path/to/leaf.jpg --outdir ./out

    # batch a folder of raw images
    python tamil_preprocess.py batch --indir ./THPLMD/raw --outdir ./out

    # compare denoisers by PSNR on one image
    python tamil_preprocess.py psnr --raw path/to/leaf.jpg

    # sweep Sauvola window_size / k and save a montage
    python tamil_preprocess.py sweep --raw path/to/leaf.jpg --outdir ./out
"""

import argparse
import os
import glob
import cv2
import numpy as np
from skimage.filters import threshold_sauvola, threshold_niblack

# ----------------------------------------------------------------------------
# Phase 0 - loading / visualizing
# ----------------------------------------------------------------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_grayscale_luminosity(bgr):
    """Grayscale via the luminosity weights used in the reference papers
    (0.21 R + 0.72 G + 0.07 B), not OpenCV's default BT.601 cvtColor."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    gray = 0.21 * r + 0.72 * g + 0.07 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def show_pair(raw_path, gt_path=None):
    """Phase 0 eval harness: eyeball raw vs ground truth (or raw vs binarized)."""
    import matplotlib.pyplot as plt
    raw = cv2.cvtColor(load_image(raw_path), cv2.COLOR_BGR2RGB)
    if gt_path and os.path.exists(gt_path):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        title2 = "ground truth"
    else:
        gt = binarize_sauvola(to_grayscale_luminosity(load_image(raw_path)))
        title2 = "sauvola (no GT given)"
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].imshow(raw); ax[0].set_title("raw"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title(title2); ax[1].axis("off")
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------
# Phase 1 - denoising + PSNR
# ----------------------------------------------------------------------------

def compute_psnr(original, processed):
    """Peak Signal-to-Noise Ratio between two same-shape uint8 images."""
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def denoise(bgr, method="nlm"):
    """method: 'gaussian' | 'nlm' | 'nlm_median'  (nlm+median matches the paper's best)."""
    if method == "gaussian":
        return cv2.GaussianBlur(bgr, (5, 5), 0)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    if method == "nlm_median":
        out = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
        gray = to_grayscale_luminosity(out)
        return cv2.medianBlur(gray, 3)  # returns single-channel
    raise ValueError(f"unknown denoise method: {method}")


def compare_denoisers(raw_path):
    """Phase 1: log PSNR per denoiser vs the raw grayscale as reference."""
    bgr = load_image(raw_path)
    ref = to_grayscale_luminosity(bgr)
    rows = []
    for m in ("gaussian", "nlm", "nlm_median"):
        out = denoise(bgr, m)
        out_gray = out if out.ndim == 2 else to_grayscale_luminosity(out)
        rows.append((m, compute_psnr(ref, out_gray)))
    print(f"\nPSNR vs raw grayscale for {os.path.basename(raw_path)}:")
    for m, p in rows:
        print(f"  {m:12s}  {p:6.2f} dB")
    return rows


# ----------------------------------------------------------------------------
# Phase 2 - binarization (Sauvola preferred)
# ----------------------------------------------------------------------------

def binarize_sauvola(gray, window_size=25, k=0.2):
    """Adaptive local thresholding. Returns a uint8 binary image where
    ink = 0 (black) and background = 255 (white)."""
    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = gray > thresh            # True = background
    return (binary.astype(np.uint8)) * 255


def binarize_otsu(gray):
    """THPLMD's method -- kept only for sanity-checking against their GT."""
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def sweep_sauvola(raw_path, outdir, windows=(15, 25, 35), ks=(0.1, 0.2, 0.3)):
    """Phase 2: montage over window_size x k so you can pick values visually."""
    os.makedirs(outdir, exist_ok=True)
    gray = to_grayscale_luminosity(denoise(load_image(raw_path), "nlm"))
    tiles = []
    for w in windows:
        row = []
        for k in ks:
            b = binarize_sauvola(gray, window_size=w, k=k)
            labelled = cv2.putText(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR),
                                   f"w={w} k={k}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            row.append(labelled)
        tiles.append(cv2.hconcat(row))
    montage = cv2.vconcat(tiles)
    out = os.path.join(outdir, "sauvola_sweep.png")
    cv2.imwrite(out, montage)
    print(f"wrote {out}")
    return out


# ----------------------------------------------------------------------------
# Phase 3 - content crop + punch-hole removal
# ----------------------------------------------------------------------------

def _ink_mask(binary):
    """Return a mask where ink pixels are 255 (invert the binary)."""
    return cv2.bitwise_not(binary)


def content_crop(binary, pad=5):
    """Crop to the outermost content bounding box with a small padding."""
    ink = _ink_mask(binary)
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return binary
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    H, W = binary.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return binary[y0:y1, x0:x1]


def remove_punch_holes(binary, area_frac=0.002, ar_lo=0.6, ar_hi=1.7):
    """Palm-leaf binding holes get picked up as fake character blobs.
    Detect large, roughly-square filled contours and erase them.

    area_frac : minimum blob area as a fraction of the image, to count as a hole
    ar_lo/hi  : aspect-ratio band (near 1.0 == round/square hole)
    """
    ink = _ink_mask(binary)
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape
    min_area = area_frac * H * W
    mask = np.zeros_like(binary)         # white where we will erase
    holes = 0
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            break                        # contours are sorted, rest are smaller
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h) if h else 0
        extent = area / float(w * h) if w * h else 0   # filledness
        if ar_lo <= ar <= ar_hi and extent > 0.55:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            holes += 1
    # XOR the hole regions back to background (white)
    cleaned = binary.copy()
    cleaned[mask == 255] = 255
    return cleaned, holes


# ----------------------------------------------------------------------------
# Phase 7 - segmentation accuracy stub (fill in once you have counts)
# ----------------------------------------------------------------------------

def segmentation_accuracy(valid_crops, known_char_count):
    """Eq.19-style metric: how many expected characters did we recover.
    Returns a value in [0, 1]; >1 means over-segmentation (noise blobs)."""
    if known_char_count <= 0:
        raise ValueError("known_char_count must be > 0")
    return valid_crops / float(known_char_count)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def preprocess_one(raw_path, outdir, window_size=25, k=0.2, save=True):
    """Full Phase 1-3 chain on a single image. Returns the cleaned binary."""
    os.makedirs(outdir, exist_ok=True)
    bgr = load_image(raw_path)
    den = denoise(bgr, "nlm")
    gray = to_grayscale_luminosity(den)
    binary = binarize_sauvola(gray, window_size=window_size, k=k)
    cropped = content_crop(binary)
    cleaned, holes = remove_punch_holes(cropped)

    stem = os.path.splitext(os.path.basename(raw_path))[0]
    if save:
        cv2.imwrite(os.path.join(outdir, f"{stem}_1_gray.png"), gray)
        cv2.imwrite(os.path.join(outdir, f"{stem}_2_sauvola.png"), binary)
        cv2.imwrite(os.path.join(outdir, f"{stem}_3_cleaned.png"), cleaned)
    print(f"{stem}: removed {holes} punch-hole blob(s) -> "
          f"{stem}_3_cleaned.png ({cleaned.shape[1]}x{cleaned.shape[0]})")
    return cleaned


def batch(indir, outdir, window_size=25, k=0.2):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(indir, e))
        paths += glob.glob(os.path.join(indir, e.upper()))
    if not paths:
        print(f"No images found in {indir}")
        return
    print(f"Processing {len(paths)} image(s)...")
    for p in sorted(paths):
        try:
            preprocess_one(p, outdir, window_size, k)
        except Exception as ex:                       # noqa: BLE001
            print(f"  ! {os.path.basename(p)}: {ex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="visualize raw vs ground truth")
    s.add_argument("--raw", required=True)
    s.add_argument("--gt", default=None)

    s = sub.add_parser("run", help="preprocess one image")
    s.add_argument("--raw", required=True)
    s.add_argument("--outdir", default="./out")
    s.add_argument("--window", type=int, default=25)
    s.add_argument("--k", type=float, default=0.2)

    s = sub.add_parser("batch", help="preprocess a folder")
    s.add_argument("--indir", required=True)
    s.add_argument("--outdir", default="./out")
    s.add_argument("--window", type=int, default=25)
    s.add_argument("--k", type=float, default=0.2)

    s = sub.add_parser("psnr", help="compare denoisers by PSNR")
    s.add_argument("--raw", required=True)

    s = sub.add_parser("sweep", help="sweep Sauvola window/k into a montage")
    s.add_argument("--raw", required=True)
    s.add_argument("--outdir", default="./out")

    a = ap.parse_args()
    if a.cmd == "show":
        show_pair(a.raw, a.gt)
    elif a.cmd == "run":
        preprocess_one(a.raw, a.outdir, a.window, a.k)
    elif a.cmd == "batch":
        batch(a.indir, a.outdir, a.window, a.k)
    elif a.cmd == "psnr":
        compare_denoisers(a.raw)
    elif a.cmd == "sweep":
        sweep_sauvola(a.raw, a.outdir)


if __name__ == "__main__":
    main()
