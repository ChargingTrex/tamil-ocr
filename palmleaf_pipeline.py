"""
palmleaf_pipeline.py

Phase 1-3 of the Tamil palm-leaf manuscript OCR pipeline:
  1. Denoising      (Gaussian blur -> Fast Non-Local Means -> Median filter)
  2. Binarization    (Sauvola adaptive thresholding)
  3. Cropping        (auto-crop to content) + punch-hole removal

Usage:
    from palmleaf_pipeline import process_image
    result = process_image("raw_leaf.jpg")
    # result is a dict with every intermediate stage, so you can inspect
    # and tune each step individually.

Run directly for a demo on a synthetic test image:
    python3 palmleaf_pipeline.py

CHANGELOG (this revision):
  * Rewrote punch-hole removal. The previous version searched for dark ink
    BLOBS, but a punch hole photographed over a white sheet is BRIGHT -- in
    the binary it's background, not a blob, so the old code only ever caught
    the dark text/Sauvola ring that happened to encircle a hole (which
    worked for one hole and missed the other on the test image). The new
    detect_and_clear_holes() finds holes correctly as "a compact, round
    region enclosed by the leaf" using the leaf silhouette, on the grayscale
    image, and clears a disc wide enough to also remove the Sauvola edge-ring.
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


# ---------------------------------------------------------------------------
# Phase 1: Denoising
# ---------------------------------------------------------------------------

def to_grayscale_luminosity(bgr_img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale using the luminosity formula
    (0.21 R + 0.72 G + 0.07 B), matching the formula used in the
    Heritage Science palm-leaf OCR paper rather than OpenCV's default."""
    b, g, r = cv2.split(bgr_img.astype(np.float64))
    gray = 0.21 * r + 0.72 * g + 0.07 * b
    return gray.astype(np.uint8)


def denoise(bgr_img: np.ndarray) -> np.ndarray:
    """Light denoising to preserve stroke sharpness (avoids 'low-res' look).
    Returns a denoised grayscale image."""
    gray = to_grayscale_luminosity(bgr_img)
    # A single median blur is enough to remove salt-and-pepper leaf noise 
    # while perfectly preserving the sharp, high-resolution edges of the text.
    denoised = cv2.medianBlur(gray, 3)
    return denoised


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute PSNR between two grayscale images of the same shape."""
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ---------------------------------------------------------------------------
# Phase 2: Binarization
# ---------------------------------------------------------------------------

def binarize_sauvola(gray_img: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola adaptive thresholding. Returns a binary image where
    text/foreground = 0 (black) and background = 255 (white)."""
    thresh = threshold_sauvola(gray_img, window_size=window_size, k=k)
    binary = (gray_img > thresh).astype(np.uint8) * 255
    return binary


def binarize_otsu(gray_img: np.ndarray) -> np.ndarray:
    """Otsu's method, for comparison purposes."""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def clear_border_artifacts(binary_img: np.ndarray, margin: int = 4) -> np.ndarray:
    """Force a thin margin at the image edges to background (white).

    Sauvola's local window gets clipped at a hard crop boundary, producing a
    blotchy dark frame that can merge with real edge text into one blob and
    break downstream contour steps. Clearing a small margin is a cheap fix.
    """
    cleaned = binary_img.copy()
    cleaned[:margin, :] = 255
    cleaned[-margin:, :] = 255
    cleaned[:, :margin] = 255
    cleaned[:, -margin:] = 255
    return cleaned


# ---------------------------------------------------------------------------
# Phase 3a: Content-cropping
# ---------------------------------------------------------------------------

def find_leaf_bbox(gray_img: np.ndarray, padding: int = 5) -> tuple:
    """Find the leaf's bounding box using the GRAYSCALE (pre-Sauvola) image.

    Must run on grayscale/denoised pixels, not the Sauvola output: Sauvola is
    a local threshold that erases the global leaf-vs-sheet contrast, whereas a
    global Otsu still sees the leaf as one darker blob against a lighter sheet.
    """
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, gray_img.shape[1], gray_img.shape[0])

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(gray_img.shape[1], x + w + padding)
    y1 = min(gray_img.shape[0], y + h + padding)
    return (x0, y0, x1, y1)


def crop_to_content(img: np.ndarray, bbox: tuple) -> np.ndarray:
    """Crop any image (grayscale or binary) to a given bounding box."""
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Phase 3b: Punch-hole removal  (rewritten)
# ---------------------------------------------------------------------------

def detect_and_clear_holes(binary_img: np.ndarray, gray_img: np.ndarray,
                           min_radius: int = 6, max_radius: int = 60,
                           min_circularity: float = 0.55, border_margin: int = 6,
                           dilate_pad: int = 16):
    """Remove punch holes, framed correctly as: *a compact, roughly-circular
    region enclosed by the leaf that is not itself leaf or ink.*

    Why not the old "find a dark near-square blob" approach: a binding hole
    photographed over a white sheet reads BRIGHT, so in the binary it is
    background, not a blob. The old detector could therefore only catch the
    dark text/Sauvola ring encircling a hole -- fragile, and it missed holes
    whose ring wasn't a clean closed contour.

    Method:
      1. Build the solid leaf silhouette (largest grayscale contour, filled).
      2. Hole candidates = silhouette AND NOT leaf_mask. This isolates
         interior gaps (bright show-through, or dark shadow -- either way
         "not leaf") while excluding the white sheet border (outside the
         silhouette) and dark ink (part of the leaf foreground).
      3. Keep candidates that are round enough and within a radius band.
      4. Clear a disc of radius r + dilate_pad in the binary. The Sauvola
         edge-ring around a bright hole is ~half the Sauvola window wide and
         does NOT scale with hole size, so pass dilate_pad ~= window//2 + a
         few px to guarantee the ring is erased too.

    Radius/area thresholds assume holes are larger than a single glyph stroke
    and smaller than max_radius; RE-TUNE per scan resolution and always
    visually sanity-check a few pages before trusting on a full batch.

    Returns (cleaned_binary, [(cx, cy, r), ...]).
    """
    H, W = gray_img.shape
    _, leaf = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return binary_img.copy(), []
    silhouette = np.zeros((H, W), np.uint8)
    cv2.drawContours(silhouette, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

    enclosed = cv2.bitwise_and(silhouette, cv2.bitwise_not(leaf))
    inner = cv2.erode(silhouette, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_margin * 2 + 1, border_margin * 2 + 1)))
    enclosed = cv2.bitwise_and(enclosed, inner)
    enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(enclosed, 8)
    holes, hole_mask = [], np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < np.pi * (min_radius ** 2):
            continue
        r = 0.5 * (w + h) / 2.0
        if not (min_radius <= r <= max_radius):
            continue
        circularity = area / (np.pi * r * r)          # ~1.0 for a filled disc
        if circularity < min_circularity:
            continue
        cx, cy = cent[i]
        cv2.circle(hole_mask, (int(cx), int(cy)), int(r + dilate_pad), 255, -1)
        holes.append((int(cx), int(cy), int(r)))

    cleaned = binary_img.copy()
    cleaned[hole_mask == 255] = 255
    return cleaned, holes


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_image(path: str, sauvola_window: int = 25, sauvola_k: float = 0.2,
                   crop_padding: int = 5) -> dict:
    """Run the full Phase 1-3 pipeline on an image file and return every
    intermediate result, so you can inspect/tune each stage."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    raw_gray = to_grayscale_luminosity(bgr)
    denoised = denoise(bgr)

    # Crop FIRST using grayscale leaf-vs-sheet contrast, then binarize.
    bbox = find_leaf_bbox(denoised, padding=crop_padding)
    denoised_cropped = crop_to_content(denoised, bbox)

    binary_raw = binarize_sauvola(denoised_cropped, window_size=sauvola_window, k=sauvola_k)
    binary = clear_border_artifacts(binary_raw, margin=max(4, sauvola_window // 2 + 3))

    # Hole removal needs the GRAYSCALE crop (holes are bright there) plus a
    # clear-pad wide enough to swallow the Sauvola ring (~window/2).
    cleaned, holes = detect_and_clear_holes(
        binary, denoised_cropped, dilate_pad=sauvola_window // 2 + 4)

    psnr = compute_psnr(raw_gray, denoised)

    return {
        "bgr": bgr,
        "raw_gray": raw_gray,
        "denoised": denoised,
        "bbox": bbox,
        "cropped_gray": denoised_cropped,
        "binary": binary,
        "cleaned": cleaned,
        "holes_found": holes,
        "psnr": psnr,
    }


# ---------------------------------------------------------------------------
# Demo on a synthetic test image
# ---------------------------------------------------------------------------

def _make_synthetic_leaf(path: str = "synthetic_leaf.png"):
    """Fake palm-leaf: tan strip on a white sheet, fibrous texture, wavy dark
    'text' rows, two bright punch holes, and salt-and-pepper noise."""
    h, w = 400, 1000
    rng = np.random.default_rng(42)
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    leaf_x0, leaf_y0, leaf_x1, leaf_y1 = 60, 100, w - 60, h - 100
    img[leaf_y0:leaf_y1, leaf_x0:leaf_x1] = (60, 130, 175)

    for _ in range(150):
        y = rng.integers(leaf_y0, leaf_y1)
        x0 = rng.integers(leaf_x0, leaf_x1 - 50)
        length = rng.integers(20, 60)
        shade = rng.integers(-20, 20)
        color = tuple(int(np.clip(c + shade, 0, 255)) for c in (60, 130, 175))
        cv2.line(img, (x0, y), (min(x0 + length, leaf_x1), y), color, 1)

    for row in range(5):
        base_y = leaf_y0 + 30 + row * 35
        x = leaf_x0 + 30
        while x < leaf_x1 - 40:
            wobble = int(5 * np.sin(x / 15))
            cv2.circle(img, (x, base_y + wobble), rng.integers(2, 5), (20, 20, 20), -1)
            x += rng.integers(6, 14)

    cv2.circle(img, (leaf_x0 + 80, (leaf_y0 + leaf_y1) // 2), 12, (235, 235, 235), -1)
    cv2.circle(img, (leaf_x1 - 80, (leaf_y0 + leaf_y1) // 2), 12, (235, 235, 235), -1)

    noise_mask = rng.random((h, w)) < 0.01
    img[noise_mask] = rng.integers(0, 255, size=(noise_mask.sum(), 3))

    cv2.imwrite(path, img)
    return path


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("No real manuscript supplied -- generating a synthetic test leaf...")
    test_path = _make_synthetic_leaf("synthetic_leaf.png")

    result = process_image(test_path)

    print(f"PSNR (raw vs denoised): {result['psnr']:.2f} dB")
    print(f"Punch holes detected: {len(result['holes_found'])} -> {result['holes_found']}")
    print(f"Crop bounding box: {result['bbox']}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    stages = [
        ("1. Raw (BGR->RGB)", cv2.cvtColor(result["bgr"], cv2.COLOR_BGR2RGB), False),
        ("2. Grayscale", result["raw_gray"], True),
        ("3. Denoised", result["denoised"], True),
        ("4. Cropped (grayscale)", result["cropped_gray"], True),
        ("5. Sauvola Binary", result["binary"], True),
        ("6. Punch-holes removed", result["cleaned"], True),
    ]
    for ax, (title, im, is_gray) in zip(axes.flat, stages):
        ax.imshow(im, cmap="gray" if is_gray else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    out_path = "pipeline_demo.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved visualization to {out_path}")
