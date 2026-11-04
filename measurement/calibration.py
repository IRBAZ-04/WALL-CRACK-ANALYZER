"""
algorithms/measurement/calibration.py
-------------------------------------
Handles pixel-to-centimeter conversion using a known-size reference marker.
100% pure NumPy implementation with ZERO OpenCV dependencies.
"""

import numpy as np
from algorithms.morphology import connected_components_labeling, morphological_closing


def rgb_to_hsv(rgb: np.ndarray) -> tuple:
    """
    Converts RGB image (0..255) to HSV color space using standard formulas:
        H in [0, 360), S in [0, 1], V in [0, 1]
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask = delta > 1e-6

    # When red is max
    r_idx = mask & (cmax == r)
    h[r_idx] = 60.0 * (((g[r_idx] - b[r_idx]) / delta[r_idx]) % 6.0)

    # When green is max
    g_idx = mask & (cmax == g)
    h[g_idx] = 60.0 * (((b[g_idx] - r[g_idx]) / delta[g_idx]) + 2.0)

    # When blue is max
    b_idx = mask & (cmax == b)
    h[b_idx] = 60.0 * (((r[b_idx] - g[b_idx]) / delta[b_idx]) + 4.0)

    # Saturation
    s = np.zeros_like(cmax)
    s[cmax > 0] = delta[cmax > 0] / cmax[cmax > 0]

    # Value
    v = cmax
    return h, s, v


def auto_detect_marker(image_rgb: np.ndarray, h_range=(10, 40), s_min=0.5, v_min=0.5) -> float:
    """
    Detects bright orange/yellow reference marker using HSV color filtering
    and connected component bounding box analysis.
    Returns the pixel length of the marker, or None if not found.
    """
    if len(image_rgb.shape) != 3:
        return None

    h, s, v = rgb_to_hsv(image_rgb)
    marker_mask = (h >= h_range[0]) & (h <= h_range[1]) & (s >= s_min) & (v >= v_min)
    bin_mask = (marker_mask.astype(np.uint8)) * 255

    # Filter small noise
    closed = morphological_closing(bin_mask, iterations=1)
    num_labels, labels, stats = connected_components_labeling(closed)

    if num_labels == 0:
        return None

    # Find largest component
    best_lbl = max(stats.keys(), key=lambda k: stats[k]["area"])
    if stats[best_lbl]["area"] < 200:
        return None

    bbox = stats[best_lbl]["bbox"]  # (min_r, min_c, max_r, max_c)
    h_box = bbox[2] - bbox[0]
    w_box = bbox[3] - bbox[1]
    pixel_length = float(max(h_box, w_box))
    return pixel_length


def manual_calibration(pixel_length: float, real_world_cm: float = 10.0) -> float:
    """Computes cm-per-pixel ratio from marker length."""
    if pixel_length is None or pixel_length <= 0:
        return None
    return real_world_cm / float(pixel_length)


def get_cm_per_pixel(image_rgb: np.ndarray, real_world_cm: float = 10.0) -> tuple:
    """
    Convenience wrapper: tries auto-detection of marker, returns (cm_per_pixel, marker_px).
    """
    pixel_length = auto_detect_marker(image_rgb)
    if pixel_length is None:
        return None, None
    cm_per_px = manual_calibration(pixel_length, real_world_cm)
    return cm_per_px, pixel_length
