import numpy as np, cv2

def detect_and_clear_holes(binary_img, gray_img, min_radius=6, max_radius=60,
                           min_circularity=0.55, border_margin=6, dilate_pad=4):
    """Remove punch holes framed as: a compact, roughly-circular region that
    is *enclosed by the leaf* but is not itself leaf/ink.

    Method: build the solid leaf silhouette (largest contour, filled), then
    hole candidates = filled_leaf AND NOT leaf_mask. That isolates interior
    gaps (punch holes show through bright; shadowed holes read dark -- either
    way they're 'not leaf' inside the silhouette), while excluding the white
    sheet border (outside the silhouette) and dark ink (part of the leaf
    foreground). Each candidate is filtered by area + circularity, then a
    slightly-dilated disc is cleared in the binary (also erasing the Sauvola
    edge-ring around the hole).

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
    # drop a thin band just inside the silhouette edge (avoids rim slivers)
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
        circ = area / (np.pi * r * r)                  # filled-circle-ness, ~1 for a disc
        if circ < min_circularity:
            continue
        cx, cy = cent[i]
        cv2.circle(hole_mask, (int(cx), int(cy)), int(r + dilate_pad), 255, -1)
        holes.append((int(cx), int(cy), int(r)))

    cleaned = binary_img.copy()
    cleaned[hole_mask == 255] = 255
    return cleaned, holes
