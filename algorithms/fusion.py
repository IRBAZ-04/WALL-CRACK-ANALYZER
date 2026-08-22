"""
fusion.py — Weighted Multi-Operator Fusion (the project's novelty algorithm)

WHY THIS EXISTS:
The faculty asked for 4 algorithms, so Sobel/Prewitt/Laplacian/Canny
remain the mandatory core comparison. This module adds a genuinely
useful 5th "bonus" method that most student projects on this topic do
NOT include, which is what will make yours stand out in review:

    Fusion = w1*Sobel + w2*Prewitt + w3*Laplacian + w4*Canny(as 0/255 mask)

Different operators fail in different ways: Sobel/Prewitt smear on
diagonal cracks, Laplacian over-triggers on noise, Canny can break a
crack into disconnected segments under harsh shadows. Combining them
with weights lets strong responses from any operator contribute,
often producing a MORE continuous, LESS noisy crack mask than any
single operator alone. This gives you a real, defensible research
angle: "does a weighted fusion outperform each individual classical
operator?" — exactly the kind of comparative-study depth a reviewer
wants to see, without turning this into a deep-learning project.

The weights below are a sensible default; the evaluation stage in
this project will let you show, with your OWN ground truth data,
whether fusion actually improves F1-score — you must not simply
assert it does without measuring it.
"""

import cv2
import numpy as np
import time


DEFAULT_WEIGHTS = {"sobel": 0.3, "prewitt": 0.2, "laplacian": 0.2, "canny": 0.3}


def detect(gray_image, sobel_map, prewitt_map, laplacian_map, canny_map, weights=None):
    """
    Combines the four already-computed edge maps into one fused map.
    Passing in the maps (rather than recomputing) avoids duplicate
    work and keeps timing measurements for each individual algorithm
    honest/unaffected.
    """
    start = time.time()
    w = weights or DEFAULT_WEIGHTS

    # normalize canny (binary 0/255) onto the same continuous scale as the others
    canny_norm = canny_map.astype(np.float32)

    fused = (
        w["sobel"] * sobel_map.astype(np.float32)
        + w["prewitt"] * prewitt_map.astype(np.float32)
        + w["laplacian"] * laplacian_map.astype(np.float32)
        + w["canny"] * canny_norm
    )
    fused = np.uint8(255 * fused / (np.max(fused) + 1e-6))

    elapsed = time.time() - start
    return fused, elapsed
