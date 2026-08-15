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
    """Three-stage denoising: Gaussian blur -> Fast-NLM -> median filter.
    Returns a denoised grayscale image."""
    # 1. Light Gaussian blur to smooth fibrous leaf texture
    blurred = cv2.GaussianBlur(bgr_img, (5, 5), 0)

    # 2. Fast Non-Local Means denoising (color version, then grayscale)
    nlm = cv2.fastNlMeansDenoisingColored(blurred, None, h=10, hColor=10,
                                           templateWindowSize=7, searchWindowSize=21)

    # 3. Convert to grayscale (luminosity method)
    gray = to_grayscale_luminosity(nlm)

    # 4. Median filter to remove salt-and-pepper noise
    denoised = cv2.medianBlur(gray, 3)

    return denoised


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute PSNR between two grayscale images of the same shape."""
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float("inf")
    max_intensity = 255.0
    return 20 * np.log10(max_intensity / np.sqrt(mse))


# ---------------------------------------------------------------------------
# Phase 2: Binarization
# ---------------------------------------------------------------------------

def binarize_sauvola(gray_img: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola adaptive thresholding. Returns a binary image where
    text/foreground = 0 (black) and background = 255 (white),
    matching typical document-image convention."""
    thresh = threshold_sauvola(gray_img, window_size=window_size, k=k)
    binary = (gray_img > thresh).astype(np.uint8) * 255  # True(bg)->255
    return binary


def binarize_otsu(gray_img: np.ndarray) -> np.ndarray:
    """Otsu's method, for comparison purposes."""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def clear_border_artifacts(binary_img: np.ndarray, margin: int = 4) -> np.ndarray:
    """Force a thin margin at the image edges to background (white).

    Sauvola's local window gets clipped right at a hard crop boundary,
    which commonly produces a blotchy dark frame artifact that has
    nothing to do with real content. Left alone, this frame can visually
    (and topologically, for contour-finding) merge with real text near
    the edges into one giant connected blob -- breaking downstream
    contour-based steps like punch-hole removal or line segmentation.
    Clearing a small margin after binarization is a standard, cheap fix.
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

    IMPORTANT: this must run on grayscale/denoised pixels, not on the
    Sauvola-binarized output. Sauvola is a *local* adaptive threshold, so
    it erases the global brightness contrast between the leaf and the
    white sheet it was photographed on -- there is no 'leaf border' left
    to find contours from once you've already binarized per-character.
    A simple global Otsu threshold on the grayscale image, by contrast,
    still sees the leaf as one darker blob against a lighter sheet.
    """
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Clean up small speckle noise so we get one solid leaf blob
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
# Phase 3b: Punch-hole removal
# ---------------------------------------------------------------------------

def remove_punch_holes(binary_img: np.ndarray, num_holes: int = 2,
                        aspect_tolerance: float = 0.3, min_area: int = 300,
                        max_area_fraction: float = 0.4) -> np.ndarray:
    """Locate punch-hole blobs (near-square contours above a minimum area)
    and edge artifacts (highly elongated contours), then mask them out --
    following the algorithm described in Maheswari et al. (2024), with two
    important fixes versus a literal reading of the paper's steps:

    1. We mask the ACTUAL CONTOUR SHAPE (cv2.drawContours, filled), not its
       bounding rectangle. A thin frame-shaped contour (e.g. the border of
       a hard crop edge picked up by Sauvola) has a bounding rectangle that
       spans nearly the whole image even though the contour itself is a
       thin loop -- masking the bounding rect would wipe out real content
       around it.
    2. We skip any contour whose bounding box covers more than
       `max_area_fraction` of the image. That's almost certainly the crop
       boundary itself, not a genuine punch hole or a small edge artifact
       worth removing.

    IMPORTANT LIMITATION: the "aspect ratio close to 1" heuristic used to
    identify holes cannot, by itself, distinguish a punch hole from any
    sufficiently round/blob-shaped glyph (Tamil has several vowel signs
    and characters with circular or loop-like forms). The `min_area`
    threshold is doing real work here -- it assumes punch holes are
    reliably larger than a single character stroke, which holds at
    typical manuscript photo resolutions but WILL need re-tuning per
    manuscript/scan resolution. Always visually sanity-check the output
    on a few pages before trusting this on a full batch.
    """
    img_area = binary_img.shape[0] * binary_img.shape[1]
    inv = cv2.bitwise_not(binary_img)
    # RETR_EXTERNAL (not RETR_LIST) is important here: Sauvola often draws
    # a thin ring outline around a hole rather than a solid fill, which
    # would otherwise produce two nested contours (inner+outer boundary)
    # for the SAME hole -- double-counting it against num_holes and
    # causing the loop to stop before reaching genuinely distinct holes.
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    holes_contours = []
    edge_contours = []
    holes_found = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if (w * h) > max_area_fraction * img_area:
            continue  # crop boundary artifact, not a real defect -- skip
        aspect = w / float(h) if h > 0 else 0

        if abs(aspect - 1.0) <= aspect_tolerance:
            holes_contours.append(c)
            holes_found += 1
        else:
            edge_contours.append(c)

        if holes_found >= num_holes:
            break

    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, holes_contours + edge_contours, -1, 255, thickness=-1)

    # Where mask is set, flip the binary image to white (background)
    result = binary_img.copy()
    result[mask == 255] = 255

    holes_bboxes = [cv2.boundingRect(c) for c in holes_contours]
    return result, holes_bboxes


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

    # Crop FIRST, using the grayscale leaf-vs-sheet contrast, then binarize
    # the already-cropped region. This order matters -- see find_leaf_bbox().
    bbox = find_leaf_bbox(denoised, padding=crop_padding)
    denoised_cropped = crop_to_content(denoised, bbox)

    binary_raw = binarize_sauvola(denoised_cropped, window_size=sauvola_window, k=sauvola_k)
    # Margin should be roughly half the Sauvola window size, since that's
    # how far the boundary-clipping artifact tends to bleed inward.
    binary = clear_border_artifacts(binary_raw, margin=max(4, sauvola_window // 2 + 3))
    cleaned, holes = remove_punch_holes(binary)

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
# Demo on a synthetic test image (runs if no real manuscript is available)
# ---------------------------------------------------------------------------

def _make_synthetic_leaf(path: str = "synthetic_leaf.png"):
    """Generate a fake palm-leaf-like image: a tan leaf strip photographed
    on a white sheet (matching real acquisition setup), with fibrous
    horizontal texture, wavy dark 'text' rows, two punch holes, and
    salt-and-pepper noise -- just enough to exercise every stage of the
    pipeline before you have real scans."""
    h, w = 400, 1000
    rng = np.random.default_rng(42)

    # White sheet background (as in real acquisition: leaf placed on white paper)
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    # Leaf region: a horizontal strip, tan/brown, inset from the edges
    leaf_x0, leaf_y0, leaf_x1, leaf_y1 = 60, 100, w - 60, h - 100
    img[leaf_y0:leaf_y1, leaf_x0:leaf_x1] = (60, 130, 175)  # BGR: brownish tan

    # Fibrous horizontal streaks, confined to the leaf region
    for _ in range(150):
        y = rng.integers(leaf_y0, leaf_y1)
        x0 = rng.integers(leaf_x0, leaf_x1 - 50)
        length = rng.integers(20, 60)
        shade = rng.integers(-20, 20)
        color = tuple(int(np.clip(c + shade, 0, 255)) for c in (60, 130, 175))
        cv2.line(img, (x0, y), (min(x0 + length, leaf_x1), y), color, 1)

    # Simulated text rows: wavy dark strokes, confined to the leaf region
    for row in range(5):
        base_y = leaf_y0 + 30 + row * 35
        x = leaf_x0 + 30
        while x < leaf_x1 - 40:
            wobble = int(5 * np.sin(x / 15))
            cv2.circle(img, (x, base_y + wobble), rng.integers(2, 5), (20, 20, 20), -1)
            x += rng.integers(6, 14)

    # Two punch holes (near-circular), inside the leaf
    cv2.circle(img, (leaf_x0 + 80, (leaf_y0 + leaf_y1) // 2), 12, (235, 235, 235), -1)
    cv2.circle(img, (leaf_x1 - 80, (leaf_y0 + leaf_y1) // 2), 12, (235, 235, 235), -1)

    # Salt-and-pepper noise across the whole image
    noise_mask = rng.random((h, w)) < 0.01
    img[noise_mask] = rng.integers(0, 255, size=(noise_mask.sum(), 3))

    cv2.imwrite(path, img)
    return path


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("No real manuscript supplied -- generating a synthetic test leaf...")
    test_path = _make_synthetic_leaf("/home/claude/synthetic_leaf.png")

    result = process_image(test_path)

    print(f"PSNR (raw vs denoised): {result['psnr']:.2f} dB")
    print(f"Punch holes detected: {len(result['holes_found'])}")
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
    out_path = "/home/claude/pipeline_demo.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved visualization to {out_path}")
