"""
length.py
----------
Estimates crack length by skeletonizing the cleaned binary mask
(reducing the crack blob to a 1-pixel-wide centerline) and counting
skeleton pixels, corrected for diagonal steps.

Why skeletonize first: a crack mask is a blob with some width; simply
counting all white pixels would measure AREA, not LENGTH. Reducing to
a centerline lets us walk along the crack and sum true path distance.
"""

import numpy as np
from skimage.morphology import skeletonize


def get_skeleton(binary_mask):
    """Returns a boolean skeleton array from a 0/255 binary mask."""
    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    return skeleton


def measure_length_pixels(skeleton):
    """
    Sums path length along the skeleton. Straight (4-connected) steps
    count as 1.0, diagonal (8-connected) steps count as sqrt(2), which
    is a standard correction so diagonal cracks aren't undercounted.
    """
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return 0.0

    coords = set(zip(xs.tolist(), ys.tolist()))
    visited_edges = set()
    total_length = 0.0

    for (x, y) in coords:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]  # only "look back/left" to avoid double count
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (nx, ny) in coords:
                edge = tuple(sorted([(x, y), (nx, ny)]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    step = np.sqrt(dx ** 2 + dy ** 2)
                    total_length += step

    # fallback for very sparse/disjoint skeletons: at least count pixels
    return max(total_length, len(coords) * 1.0) if total_length == 0 else total_length


def measure_length(binary_mask, cm_per_pixel=None):
    """
    Returns a dict with pixel length and, if calibration is available,
    the estimated physical length in cm.
    """
    skeleton = get_skeleton(binary_mask)
    length_px = measure_length_pixels(skeleton)

    result = {"length_px": round(length_px, 2), "length_cm": None}
    if cm_per_pixel:
        result["length_cm"] = round(length_px * cm_per_pixel, 2)
    return result, skeleton
