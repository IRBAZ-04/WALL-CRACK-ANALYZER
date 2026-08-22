"""
canny.py — Canny edge detection.

Concept: A multi-stage algorithm (gradient computation -> non-maximum
suppression -> double thresholding -> edge tracking by hysteresis)
that produces thin, well-connected edges. It usually gives the
cleanest crack outlines of the four, but needs good threshold values.

UNIQUE FEATURE — Auto-thresholding via Otsu's value:
Instead of hardcoding Canny's low/high thresholds (which is what most
student implementations do, and which then fails on your night /
strong-sunlight images), we derive the thresholds from the image's
own Otsu threshold value. This makes Canny adapt per-image instead of
using one fixed setting for your entire 200+ image dataset.
"""

import cv2
import time


def detect(gray_image):
    start = time.time()

    # Use Otsu's threshold to derive adaptive Canny bounds instead of
    # fixed magic numbers -- makes this robust across your lighting-varied dataset.
    otsu_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = 0.5 * otsu_thresh
    upper = otsu_thresh

    edges = cv2.Canny(gray_image, lower, upper)

    elapsed = time.time() - start
    return edges, elapsed
