"""
sobel.py — Sobel edge detection.

Concept: Sobel convolves the image with two 3x3 kernels (one for
horizontal gradient Gx, one for vertical gradient Gy). The gradient
magnitude sqrt(Gx^2 + Gy^2) is high wherever intensity changes
sharply — i.e. at edges, including crack boundaries.
"""

import cv2
import numpy as np
import time


def detect(gray_image):
    start = time.time()

    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-6))

    elapsed = time.time() - start
    return magnitude, elapsed
