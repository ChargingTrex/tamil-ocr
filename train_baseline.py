"""
train_baseline.py
=================
Baseline character classifier for the Tamil palm-leaf dataset.

Two paths:
  * HOG + LinearSVC  -- no deep-learning deps, trains in ~10s on CPU.
                        VERIFIED: 97.4% +/- 0.6% test accuracy (4 seeds),
                        20 classes, 2550 images.
  * Small CNN (PyTorch) -- run this on your GPU box; use --cnn.

Also runs a LEAKAGE AUDIT, because the headline number is only meaningful if
train/test aren't sharing near-duplicate crops.

Usage:
    python train_baseline.py --data /path/to/Dataset
    python train_baseline.py --data /path/to/Dataset --cnn --epochs 30
    python train_baseline.py --data /path/to/Dataset --audit
"""

import argparse
import numpy as np
import cv2

from tamil_dataset import load_dataset, stratified_split, CLASS_TO_TAMIL


# ---------------------------------------------------------------------------
# HOG features
# ---------------------------------------------------------------------------

def hog_feat(img, cell=8, nbins=9):
    """Histogram of Oriented Gradients with 2x2 block L2 normalisation.
    img: float32 HxW in [0,1]. Chosen over raw pixels because these crops vary
    in illumination and leaf tone -- gradients are far more stable than
    intensity on palm leaf."""
    g = (img * 255).astype(np.uint8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180
    H, W = g.shape
    ch, cw = H // cell, W // cell
    hist = np.zeros((ch, cw, nbins), np.float32)
    b = (ang / (180.0 / nbins)).astype(np.int32) % nbins
    for i in range(ch):
        for j in range(cw):
            bb = b[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            mm = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist[i, j] = np.bincount(bb, weights=mm, minlength=nbins)
    feats = []
    for i in range(ch - 1):
        for j in range(cw - 1):
            blk = hist[i:i + 2, j:j + 2].ravel()
            feats.append(blk / (np.linalg.norm(blk) + 1e-6))
    return np.concatenate(feats)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def _sig(A):
    S = np.stack([cv2.resize(a, (16, 16), interpolation=cv2.INTER_AREA).ravel() for a in A])
    S = S - S.mean(1, keepdims=True)
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)


def leakage_audit(X, tr, te, y, pred=None):
    """Near-duplicate detection between train and test.

    Datasets assembled by cropping frames from the same manuscript photo very
    often contain near-identical siblings. If those straddle the split, test
    accuracy is measuring memorisation, not recognition. We report accuracy
    with the near-duplicates REMOVED -- that's the number to trust.
    """
    best = (_sig(X[te]) @ _sig(X[tr]).T).max(1)
    print("\n--- leakage audit ---")
    for thr in (0.99, 0.95, 0.90):
        n = int((best > thr).sum())
        print(f"  test imgs with train match >{thr}: {n}/{len(te)} ({100*n/len(te):.1f}%)")
    if pred is not None:
        from sklearn.metrics import accuracy_score
        print(f"  FULL test acc            : {accuracy_score(y[te], pred):.3f}")
        for thr in (0.99, 0.95):
            m = best < thr
            print(f"  excl. near-dups <{thr}   : {accuracy_score(y[te][m], pred[m]):.3f} (n={int(m.sum())})")
    return best


# ---------------------------------------------------------------------------
# CNN (optional)
# ---------------------------------------------------------------------------

def train_cnn(X, y, names, tr, va, te, epochs=30, bs=64, lr=1e-3):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cnn] device={dev}")

    def aug(b):
        out = []
        for im in b:
            if np.random.rand() < 0.5:
                a = np.random.uniform(-12, 12)
                s = np.random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), a, s)
                im = cv2.warpAffine(im, M, im.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
            out.append(im)
        return np.stack(out)

    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, len(names)),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def ev(idx):
        net.eval()
        with torch.no_grad():
            xb = torch.tensor(X[idx]).unsqueeze(1).to(dev)
            return (net(xb).argmax(1).cpu().numpy() == y[idx]).mean()

    best_va, best_state = 0, None
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = torch.tensor(aug(X[b])).unsqueeze(1).to(dev)
            yb = torch.tensor(y[b]).to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        sched.step()
        va_acc = ev(va)
        if va_acc > best_va:
            best_va, best_state = va_acc, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} loss={loss.item():.3f} val={va_acc:.3f}")
    net.load_state_dict(best_state)
    print(f"[cnn] best val={best_va:.3f}  TEST={ev(te):.3f}")
    return net


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to Dataset/ root")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, y, names = load_dataset(a.data, size=a.size, binarize=a.binarize)
    tr, va, te = stratified_split(y, seed=a.seed)
    print(f"\nsplit: train={len(tr)} val={len(va)} test={len(te)}")

    if a.cnn:
        train_cnn(X, y, names, tr, va, te, epochs=a.epochs)
        return

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, classification_report

    print("computing HOG features...")
    F = np.stack([hog_feat(x) for x in X])
    clf = LinearSVC(C=1.0, max_iter=3000).fit(F[tr], y[tr])
    pred = clf.predict(F[te])
    print(f"\ntrain={accuracy_score(y[tr], clf.predict(F[tr])):.3f} "
          f"val={accuracy_score(y[va], clf.predict(F[va])):.3f} "
          f"TEST={accuracy_score(y[te], pred):.3f}")
    print("\n" + classification_report(y[te], pred, target_names=names,
                                       digits=3, zero_division=0))
    if a.audit:
        leakage_audit(X, tr, te, y, pred)


if __name__ == "__main__":
    main()
