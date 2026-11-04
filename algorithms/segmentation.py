"""
algorithms/segmentation.py
--------------------------
Converts grayscale edge response maps into binary crack masks (255 = crack, 0 = wall).
100% pure NumPy implementation with ZERO OpenCV dependencies.

Includes:
  1. Otsu's Global Optimal Thresholding (Maximizing Between-Class Variance)
  2. Local Adaptive Mean Window Thresholding (for Uneven Illumination & Shadows)
"""

import numpy as np
from .convolution import convolve2d


def compute_otsu_threshold(image: np.ndarray) -> int:
    """
    Computes Nobuyuki Otsu's optimal binarization threshold from first principles:
        Maximizes inter-class variance:
            σ_B²(t) = ω0(t) * ω1(t) * [μ0(t) - μ1(t)]²
                    = [μ_total * ω0(t) - μ(t)]² / [ω0(t) * (1 - ω0(t))]
    """
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = float(image.size)
    if total_pixels == 0:
        return 128

    prob = hist / total_pixels

    # Cumulative class probabilities (weights)
    omega0 = np.cumsum(prob)
    omega1 = 1.0 - omega0

    # Cumulative class means
    bin_centers = np.arange(256)
    mu_cum = np.cumsum(prob * bin_centers)
    mu_total = mu_cum[-1]

    # Valid indices where both classes have non-zero probability
    valid = (omega0 > 0) & (omega1 > 0)
    sigma_b_squared = np.zeros(256, dtype=np.float64)

    numerator = (mu_total * omega0[valid] - mu_cum[valid]) ** 2
    denominator = omega0[valid] * omega1[valid]
    sigma_b_squared[valid] = numerator / denominator

    max_val = np.max(sigma_b_squared)
    if max_val > 0:
        max_indices = np.where(sigma_b_squared == max_val)[0]
        best_thresh = int(np.mean(max_indices))
    else:
        best_thresh = 128
    return best_thresh


def adaptive_threshold(image: np.ndarray, block_size: int = 21, c: float = -5.0) -> np.ndarray:
    """
    Computes local adaptive thresholding:
        T(x, y) = Local_Mean(x, y) - C
        Output(x, y) = 255 if Image(x, y) > T(x, y) else 0
    """
    if block_size % 2 == 0:
        block_size += 1

    kernel = np.ones((block_size, block_size), dtype=np.float64) / float(block_size * block_size)
    local_mean = convolve2d(image, kernel, padding="reflect")

    mask = np.where(image > (local_mean - c), 255, 0).astype(np.uint8)
    return mask


def segment(edge_map: np.ndarray, lighting_label: str = "normal") -> tuple:
    """
    Binarizes an edge-response image into a crack mask.
    Automatically chooses Adaptive vs Otsu based on lighting condition.
    Returns (binary_mask, method_used).
    """
    # If already a binary edge map (e.g., Canny output with 0 and 255)
    unique_vals = np.unique(edge_map)
    if len(unique_vals) <= 2 and (255 in unique_vals or 1 in unique_vals):
        return (edge_map > 0).astype(np.uint8) * 255, "canny_direct"

    if lighting_label in ("shadow_uneven",):
        method = "adaptive"
        mask = adaptive_threshold(edge_map, block_size=21, c=-5.0)
    else:
        method = "otsu"
        thresh_val = compute_otsu_threshold(edge_map)
        # Strictly greater than threshold so background 0 is never included
        mask = np.where(edge_map > thresh_val, 255, 0).astype(np.uint8)

    return mask, method
