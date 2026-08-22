"""
morphology.py
--------------
Cleans up the raw binary crack mask:
  - Closing (dilation then erosion): bridges small gaps in a crack
    line that edge detection may have broken.
  - Connected-component area filter: removes small noise specks
    (isolated blobs too small to plausibly be a real crack).

NOTE ON OPENING: a naive implementation would run MORPH_OPEN before
closing to "remove noise first." That is deliberately NOT done here.
Canny and Otsu-thresholded output are already close to 1-pixel-wide
lines; eroding a 1px-wide line with even a 3x3 kernel deletes it
completely before dilation can bring it back, wiping out entire real
cracks. Sobel/Prewitt/Laplacian magnitude maps (which are typically a
few pixels wide before thresholding) survive opening better, but for
consistency across all five algorithms we rely on closing (safe for
thin lines) plus component-size filtering (which removes small noise
blobs just as effectively as opening, without erasing thin cracks).

Kernel sizes are intentionally small (3x3) because cracks are thin —
a large kernel would erase real crack pixels along with noise.
"""

import cv2
import numpy as np


def clean_mask(binary_mask, min_area=8):
    kernel = np.ones((3, 3), np.uint8)

    # bridge small gaps in the crack line (safe for thin lines, unlike opening)
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # remove tiny isolated components (area-based noise filter) —
    # anything smaller than min_area is very unlikely to be a real crack
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned
