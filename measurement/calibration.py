"""
calibration.py
---------------
Handles pixel-to-centimeter conversion using a known-size reference
marker placed near the crack during photography.

IMPORTANT (per project requirements): a photograph alone gives no
physical scale. All physical measurements (cm) are only valid if a
reference object of known real-world size was captured in the SAME
photo, at roughly the SAME distance/plane as the crack. If no
reference is available, this module returns None for cm values and
the caller must report pixel measurements only, clearly labeled as
such — never invent a physical size.

UNIQUE FEATURE — Semi-automatic marker detection:
Rather than requiring you to manually click both ends of the
reference marker for every one of 200+ images, this module can
auto-detect a marker if you use a distinctly colored reference object
(we recommend a flat bright-orange or bright-green card/sticker of
known width, e.g. 10 cm). If auto-detection fails (marker missing,
occluded, wrong color), the GUI falls back to manual two-click
calibration. Both paths are provided.
"""

import cv2
import numpy as np


def auto_detect_marker(image_bgr, hsv_lower=(5, 150, 150), hsv_upper=(20, 255, 255)):
    """
    Attempts to auto-detect a bright-orange reference marker via HSV
    color thresholding and returns its pixel width (the longer side
    of its bounding box), or None if not found confidently.

    hsv_lower/upper default to an orange range; adjust if you use a
    different colored marker.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:  # too small to be a trustworthy marker
        return None

    rect = cv2.minAreaRect(largest)
    (w, h) = rect[1]
    pixel_length = max(w, h)
    return pixel_length


def manual_calibration(pixel_length, real_world_cm):
    """User clicked two points a known real-world distance apart."""
    if pixel_length <= 0:
        return None
    return real_world_cm / pixel_length  # returns cm-per-pixel


def get_cm_per_pixel(image_bgr, real_world_cm=10.0):
    """
    Convenience wrapper: tries auto-detection first, falls back to
    None (meaning: caller should prompt for manual calibration or
    report in pixels only).
    """
    pixel_length = auto_detect_marker(image_bgr)
    if pixel_length is None:
        return None, None
    cm_per_px = manual_calibration(pixel_length, real_world_cm)
    return cm_per_px, pixel_length
