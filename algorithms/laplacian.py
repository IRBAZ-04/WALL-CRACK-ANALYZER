"""
algorithms/laplacian.py
-----------------------
Laplacian (Laplacian of Gaussian - LoG) edge detection implemented
from scratch in pure NumPy.

Mathematical Formulation:
2nd-order spatial isotropic differential operator:
    ∇²I = ∂²I/∂x² + ∂²I/∂y²

Approximated on a discrete grid using the 8-neighbor kernel:
    K = [[ 1,  1,  1 ],
         [ 1, -8,  1 ],
         [ 1,  1,  1 ]]

Pre-smoothed with Gaussian kernel G(x, y, σ=1.0) to stabilize against high-frequency noise.
"""

import time
import numpy as np
from .convolution import convolve2d, create_gaussian_kernel

LAPLACIAN_KERNEL = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -8.0,  1.0],
    [1.0,  1.0,  1.0]
], dtype=np.float64)


def detect(gray_image: np.ndarray) -> tuple:
    """
    Computes Laplacian of Gaussian (LoG) response in pure NumPy.
    Returns (response_map, elapsed_seconds).
    """
    start = time.time()

    # Pre-smoothing via Gaussian filter
    gauss_k = create_gaussian_kernel(size=3, sigma=1.0)
    blurred = convolve2d(gray_image, gauss_k, padding="reflect")

    # 2nd derivative discrete Laplacian convolution
    lap = convolve2d(blurred, LAPLACIAN_KERNEL, padding="reflect")

    # Absolute magnitude response
    abs_lap = np.abs(lap)
    max_val = np.max(abs_lap)
    if max_val > 0:
        magnitude = np.uint8(255.0 * (abs_lap / max_val))
    else:
        magnitude = np.zeros_like(gray_image, dtype=np.uint8)

    elapsed = time.time() - start
    return magnitude, elapsed
