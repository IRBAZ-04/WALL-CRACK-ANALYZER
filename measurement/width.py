"""
width.py
---------
Estimates crack width using the Euclidean distance transform, calculates
crack orientation angle via Image Moments / PCA, and assesses structural risk level.
"""

import cv2
import numpy as np


def compute_orientation_and_type(ys, xs):
    if len(xs) < 5:
        return 0.0, "Undetermined"

    # Compute orientation using Image Central Moments / Principal Axes
    pts = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

    # Primary direction vector
    v = eigenvectors[0]
    angle_rad = np.arctan2(abs(v[1]), abs(v[0]))
    angle_deg = round(float(np.degrees(angle_rad)), 1)

    if angle_deg >= 70:
        crack_type = "Vertical (Load-Bearing Shear)"
    elif angle_deg <= 20:
        crack_type = "Horizontal (Thermal/Settlement Joint)"
    else:
        crack_type = "Diagonal (Structural Shear)"

    return angle_deg, crack_type


def compute_severity_rating(max_width_mm, damage_pct):
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


def measure_width(binary_mask, skeleton, cm_per_pixel=None, damage_pct=0.0):
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return {
            "min_width_px": 0, "max_width_px": 0, "avg_width_px": 0,
            "min_width_cm": None, "max_width_cm": None, "avg_width_cm": None,
            "orientation_deg": 0.0, "orientation_type": "None",
            "severity_level": "NEGLIGIBLE (No Crack)", "severity_color": "#22c55e"
        }

    widths_px = dist_transform[ys, xs] * 2  # radius -> full width

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
