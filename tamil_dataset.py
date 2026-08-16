"""
tamil_dataset.py
================
Loader + preprocessing for the Tamil palm-leaf CHARACTER dataset
(SadhanaParameswaran/Character-Recognition-from-Tamil-Palm-Leaves).

Handles three real quirks found by profiling the actual repo:

  1. HIDDEN 20th CLASS. `Dataset/vee/` contains a nested `vu/` subdirectory
     with 85 images of a visually distinct character. A naive
     `glob('*/*.jpg')` silently DROPS them; a naive `rglob` silently merges
     them INTO `vee`, poisoning that class. We treat `vu` as its own class.
     (The repo README claims 21 classes; there are 19 top-level dirs + `vu`
     = 20 actual character classes.)

  2. MIXED EXTENSIONS. Most classes are .jpg, but `nuu`, `y` and `zha` are
     .png. Globbing only *.jpg silently drops 3 entire classes.

  3. MIXED SIZES. Most crops are 224x224 but a minority are arbitrary
     (e.g. 90x204, 65x103), so everything must be resized to a fixed shape.

The images are raw colour crops off the leaf (tan background, dark incised
strokes) -- NOT pre-binarised -- so the preprocessing here mirrors the
palm-leaf pipeline: luminosity grayscale -> CLAHE contrast -> optional
Sauvola binarisation -> pad-to-square resize.

Usage:
    from tamil_dataset import load_dataset, CLASS_TO_TAMIL
    X, y, classes = load_dataset("/path/to/Dataset", size=64, binarize=False)
"""

import os
import glob
import numpy as np
import cv2

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")

# ---------------------------------------------------------------------------
# Folder name -> Tamil Unicode.
#
# WARNING: these are TENTATIVE romanisation guesses based on the folder names
# and a visual look at the crops. Several are genuinely ambiguous from the
# romanisation alone (e.g. "ee" could be ஈ or ஏ; "nna"/"nnna" could be
# ண/ன/ஞ; "oo" could be ஊ or ஓ). Ligature/vowel-sign forms on palm leaf make
# this worse. HAVE A TAMIL READER VERIFY THIS MAP before you report accuracy
# as "character recognition" -- the model is only ever as correct as its
# labels. Training works fine on the folder names alone; this map is for
# human-readable output only.
# ---------------------------------------------------------------------------
CLASS_TO_TAMIL = {
    "ai":   "ஐ",
    "cha":  "ச",
    "ee":   "ஈ",
    "la":   "ல",
    "ma":   "ம",
    "moo":  "மூ",
    "nna":  "ண",
    "nnna": "ன",
    "nu":   "நு",
    "nuu":  "நூ",
    "oo":   "ஊ",
    "pa":   "ப",
    "ra":   "ர",
    "tha":  "த",
    "va":   "வ",
    "vee":  "வே",
    "vu":   "வு",   # the hidden nested class
    "y":    "ய்",
    "ya":   "ய",
    "zha":  "ழ",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_classes(root: str) -> dict:
    """Map class_name -> [file paths], correctly separating the nested
    `vee/vu` class instead of dropping or merging it.

    Returns an ordered dict-like (plain dict, insertion-ordered) so class
    indices are stable across runs.
    """
    root = os.path.abspath(root)
    classes = {}

    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d):
            continue

        # files directly inside this class dir
        direct = []
        for ext in IMG_EXTS:
            direct += glob.glob(os.path.join(d, ext))
        direct = sorted(set(direct))
        if direct:
            classes[entry] = direct

        # any nested subdirectory is its own class (the vee/vu case)
        for sub in sorted(os.listdir(d)):
            sd = os.path.join(d, sub)
            if not os.path.isdir(sd):
                continue
            nested = []
            for ext in IMG_EXTS:
                nested += glob.glob(os.path.join(sd, ext))
            nested = sorted(set(nested))
            if nested:
                if sub in classes:
                    raise ValueError(f"nested class name collides: {sub}")
                classes[sub] = nested

    return classes


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def to_grayscale_luminosity(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    return np.clip(0.21 * r + 0.72 * g + 0.07 * b, 0, 255).astype(np.uint8)


def pad_to_square(img, pad_value=None):
    """Pad (not stretch) to a square so aspect ratio is preserved -- important
    because several Tamil characters differ mainly in width/height ratio."""
    h, w = img.shape[:2]
    if h == w:
        return img
    if pad_value is None:
        # pad with the border median = leaf background, not black
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        pad_value = int(np.median(border))
    s = max(h, w)
    out = np.full((s, s), pad_value, img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def preprocess(path, size=64, binarize=False, clahe=True):
    """Load one crop -> fixed-size float32 grayscale in [0,1].
    Returns None if the file is unreadable (caller should skip)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = to_grayscale_luminosity(bgr)

    if clahe:
        # local contrast boost: palm-leaf crops are low-contrast and unevenly lit
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if binarize:
        try:
            from skimage.filters import threshold_sauvola
            t = threshold_sauvola(gray, window_size=25, k=0.2)
            gray = ((gray > t).astype(np.uint8)) * 255
        except ImportError:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 10)

    gray = pad_to_square(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(root, size=64, binarize=False, clahe=True, verbose=True):
    """Returns (X, y, class_names).
    X: (N, size, size) float32 in [0,1];  y: (N,) int64 class indices."""
    classes = discover_classes(root)
    names = list(classes.keys())
    X, y, skipped = [], [], 0
    for ci, name in enumerate(names):
        for p in classes[name]:
            im = preprocess(p, size=size, binarize=binarize, clahe=clahe)
            if im is None:
                skipped += 1
                continue
            X.append(im)
            y.append(ci)
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, np.int64)
    if verbose:
        print(f"Loaded {len(X)} images / {len(names)} classes "
              f"(skipped {skipped} unreadable)")
        for ci, n in enumerate(names):
            print(f"  {ci:2d} {n:5s} {CLASS_TO_TAMIL.get(n,'?'):3s} n={(y==ci).sum()}")
    return X, y, names


def stratified_split(y, seed=0, val_frac=0.15, test_frac=0.15):
    """Per-class stratified index split. Small classes stay represented in
    every split, which matters here: class sizes range ~85 to ~235."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te = max(1, int(round(test_frac * n)))
        n_va = max(1, int(round(val_frac * n)))
        te += list(idx[:n_te])
        va += list(idx[n_te:n_te + n_va])
        tr += list(idx[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "Dataset"
    X, y, names = load_dataset(root, size=64)
    tr, va, te = stratified_split(y)
    print(f"\nsplit: train={len(tr)} val={len(va)} test={len(te)}")
    print("X:", X.shape, X.dtype, "range", float(X.min()), float(X.max()))
