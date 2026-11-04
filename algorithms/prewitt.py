"""
algorithms/prewitt.py
---------------------
Prewitt edge detection implemented from scratch in pure NumPy.

Mathematical Formulation:
Uses uniform-weight directional 3x3 difference operators:
    Gx = [[ -1, 0, 1 ],
          [ -1, 0, 1 ],
          [ -1, 0, 1 ]]

    Gy = [[ -1, -1, -1 ],
          [  0,  0,  0 ],
          [  1,  1,  1 ]]

Gradient magnitude:
    G = sqrt(Gx^2 + Gy^2)
Normalized to [0, 255] uint8.
"""

import time
import numpy as np
from .convolution import convolve2d

PREWITT_X = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
], dtype=np.float64)

PREWITT_Y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float64)


def detect(gray_image: np.ndarray) -> tuple:
    """
    Computes Prewitt gradient magnitude image in pure NumPy.
    Returns (magnitude_map, elapsed_seconds).
    """
    start = time.time()

    gx = convolve2d(gray_image, PREWITT_X, padding="reflect")
    gy = convolve2d(gray_image, PREWITT_Y, padding="reflect")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = np.uint8(255.0 * (magnitude / max_val))
    else:
        magnitude = np.zeros_like(gray_image, dtype=np.uint8)

    elapsed = time.time() - start
    return magnitude, elapsed
