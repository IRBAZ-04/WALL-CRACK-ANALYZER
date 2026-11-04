"""
algorithms/measurement/width.py
-------------------------------
Estimates crack width using Euclidean Distance Transform (EDT), calculates
crack orientation angle via Principal Component Analysis (PCA covariance matrix),
and assesses structural risk level.
100% pure NumPy / SciPy implementation with ZERO OpenCV dependencies.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_orientation_and_type(ys: np.ndarray, xs: np.ndarray) -> tuple:
    """
    Computes primary crack orientation angle from scratch using Principal Component Analysis (PCA).
    Calculates covariance matrix of crack pixel coordinates and solves for eigenvectors:
        Cov = (1 / (N - 1)) * (X - μ)^T (X - μ)
    """
    if len(xs) < 5:
        return 0.0, "Undetermined"

    pts = np.column_stack((xs, ys)).astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # Covariance matrix 2x2
    cov = np.cov(centered, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort in descending order of variance
    idx = np.argsort(eigenvalues)[::-1]
    primary_vector = eigenvectors[:, idx[0]]  # [vx, vy]

    # Angle in radians and degrees
    angle_rad = np.arctan2(abs(primary_vector[1]), abs(primary_vector[0]))
    angle_deg = round(float(np.degrees(angle_rad)), 1)

    if angle_deg >= 70.0:
        crack_type = "Vertical (Load-Bearing Shear)"
    elif angle_deg <= 20.0:
        crack_type = "Horizontal (Thermal/Settlement Joint)"
    else:
        crack_type = "Diagonal (Structural Shear)"

    return angle_deg, crack_type


def compute_severity_rating(max_width_mm: float, damage_pct: float) -> tuple:
    if max_width_mm is None:
        max_width_mm = 0.0

    if max_width_mm < 1.0 and damage_pct < 1.5:
        return "LOW (Cosmetic Surface Hairline)", "#22c55e"
    elif max_width_mm <= 3.0 and damage_pct <= 4.0:
        return "MEDIUM (Minor Monitoring Required)", "#38bdf8"
    elif max_width_mm <= 6.0 and damage_pct <= 8.0:
        return "HIGH (Active Maintenance Required)", "#f59e0b"
    else:
        return "CRITICAL (Immediate Structural Review)", "#ef4444"


def measure_width(binary_mask: np.ndarray, skeleton: np.ndarray, cm_per_pixel: float = None, damage_pct: float = 0.0) -> dict:
    """
    Measures crack width at every skeleton pixel using exact Euclidean Distance Transform.
    """
    # Euclidean distance transform from background
    dist_transform = distance_transform_edt(binary_mask > 0)

    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return {
            "min_width_px": 0.0, "max_width_px": 0.0, "avg_width_px": 0.0,
            "min_width_cm": None, "max_width_cm": None, "avg_width_cm": None,
            "orientation_deg": 0.0, "orientation_type": "None",
            "severity_level": "NEGLIGIBLE (No Crack)", "severity_color": "#22c55e"
        }

    widths_px = dist_transform[ys, xs] * 2.0  # radius -> full width

    angle_deg, orientation_type = compute_orientation_and_type(ys, xs)

    min_w_px = round(float(np.min(widths_px)), 2)
    max_w_px = round(float(np.max(widths_px)), 2)
    avg_w_px = round(float(np.mean(widths_px)), 2)

    result = {
        "min_width_px": min_w_px,
        "max_width_px": max_w_px,
        "avg_width_px": avg_w_px,
        "min_width_cm": None,
        "max_width_cm": None,
        "avg_width_cm": None,
        "orientation_deg": angle_deg,
        "orientation_type": orientation_type,
    }

    max_width_mm = None
    if cm_per_pixel:
        result["min_width_cm"] = round(min_w_px * cm_per_pixel, 2)
        result["max_width_cm"] = round(max_w_px * cm_per_pixel, 2)
        result["avg_width_cm"] = round(avg_w_px * cm_per_pixel, 2)
        max_width_mm = result["max_width_cm"] * 10.0

    severity_label, severity_color = compute_severity_rating(max_width_mm, damage_pct)
    result["severity_level"] = severity_label
    result["severity_color"] = severity_color

    return result
