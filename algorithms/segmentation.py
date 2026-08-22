"""
segmentation.py
----------------
Converts a grayscale edge-response map into a binary crack mask
(white = crack, black = background).

UNIQUE FEATURE — Lighting-aware thresholding choice:
Otsu's method assumes a roughly bimodal, evenly-lit histogram, so it
works well for daylight/normal images but breaks down on
shadow/uneven images (it tends to either miss faint cracks in shadow
or misclassify the shadow edge itself as crack). Adaptive (local)
thresholding handles uneven lighting much better. Rather than picking
one method for the whole dataset, this module selects automatically
based on the lighting label produced in preprocessing.py — this is
the same design idea as the CLAHE preset selection, applied at the
segmentation stage, and is worth a paragraph in your report's
"Proposed System" section as it's not something a typical
Sobel/Prewitt/Laplacian/Canny comparison project does.
"""

import cv2


def segment(edge_map, lighting_label="normal"):
    """
    Binarizes an edge-response image into a crack mask.
    Returns (binary_mask, method_used).
    """
    if lighting_label in ("shadow_uneven",):
        method = "adaptive"
        mask = cv2.adaptiveThreshold(
            edge_map, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21, C=-5
        )
    else:
        method = "otsu"
        _, mask = cv2.threshold(edge_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, method
