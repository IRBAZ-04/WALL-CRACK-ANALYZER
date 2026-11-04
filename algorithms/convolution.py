"""
algorithms/convolution.py
-------------------------
High-performance 2D spatial convolution engine and kernel generators
implemented in 100% pure NumPy math without any external computer vision libraries.
"""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = "reflect") -> np.ndarray:
    """
    Computes 2D spatial convolution of a single-channel 2D float image with a 2D kernel.
    Uses NumPy sliding-window vectorization for high speed.

    Args:
        image: 2D array of shape (H, W).
        kernel: 2D array of shape (kh, kw), typically odd dimensions (e.g. 3x3, 5x5).
        padding: Padding mode supported by np.pad ('reflect', 'edge', 'constant').

    Returns:
        2D float64 array of same shape (H, W) as input.
    """
    image_f = np.asarray(image, dtype=np.float64)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad image to maintain exact input dimensions
    padded = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode=padding)

    # Use sliding window view for vectorized convolution
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, window_shape=(kh, kw))

    # Kernel flipping is standard for mathematical convolution
    flipped_kernel = np.flipud(np.fliplr(kernel))

    # Vectorized sum of element-wise products across window dimensions
    output = np.einsum("ijkl,kl->ij", windows, flipped_kernel)
    return output


def create_gaussian_kernel(size: int = 3, sigma: float = 0.8) -> np.ndarray:
    """
    Generates a 2D discrete Gaussian smoothing kernel from first principles:
        G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    Normalized so sum of all kernel weights equals 1.0.
    """
    k = size // 2
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g /= (2.0 * np.pi * sigma ** 2)
    return g / np.sum(g)
