"""
laplacian.py — Laplacian edge detection.

Concept: Unlike Sobel/Prewitt (first-derivative, directional),
Laplacian is a second-derivative operator — it responds to regions
where the rate of intensity change itself changes, regardless of
direction. This makes it good at catching thin cracks in all
orientations at once, but it is also the most noise-sensitive of the
four, so we apply a light blur first.
"""

import cv2
import numpy as np
import time


def detect(gray_image):
    start = time.time()

    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap = np.uint8(255 * np.abs(lap) / (np.max(np.abs(lap)) + 1e-6))

    elapsed = time.time() - start
    return lap, elapsed
