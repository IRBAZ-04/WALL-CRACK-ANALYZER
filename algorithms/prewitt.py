"""
prewitt.py — Prewitt edge detection.

Concept: Similar to Sobel, but uses simpler uniform-weight 3x3
kernels instead of Sobel's weighted-center kernels. This makes it
slightly faster but generally a bit more noise-sensitive than Sobel.
OpenCV has no built-in Prewitt, so we implement the kernels manually
with cv2.filter2D — good talking point for the report (shows you
understand the math, not just calling a library function).
"""

import cv2
import numpy as np
import time

KERNEL_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

KERNEL_Y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)


def detect(gray_image):
    start = time.time()

    gx = cv2.filter2D(gray_image.astype(np.float32), -1, KERNEL_X)
    gy = cv2.filter2D(gray_image.astype(np.float32), -1, KERNEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-6))

    elapsed = time.time() - start
    return magnitude, elapsed
