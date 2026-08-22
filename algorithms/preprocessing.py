"""
preprocessing.py
-----------------
Stage 1 of the pipeline: turns a raw campus photo into a clean,
consistently-lit grayscale image ready for edge detection.

UNIQUE FEATURE — Lighting-Aware Adaptive Enhancement:
Instead of applying the same fixed CLAHE settings to every image
(which is what most student projects do), this module first measures
how bright/dark/shadowed the image is using histogram statistics,
classifies it into a lighting category, and then picks CLAHE
parameters suited to that category. This directly targets your
faculty requirement of "different lighting conditions" and gives you
a genuine, explainable novelty point for the report.
"""

import cv2
import numpy as np


def resize_image(image, target_width=800):
    """Resize keeping aspect ratio so all images are processed at a
    consistent scale (keeps pixel-based measurements comparable)."""
    h, w = image.shape[:2]
    scale = target_width / w
    new_size = (target_width, int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def classify_lighting(gray):
    """
    Classifies the image into one of: 'dark', 'bright', 'shadow_uneven',
    'normal' using simple, explainable statistics:
      - mean brightness  -> overall dark/bright
      - std deviation     -> low std + patches of very different
                              local means => uneven/shadow
    Returns a string label used to pick enhancement parameters.
    """
    mean_val = np.mean(gray)
    std_val = np.std(gray)

    # split into a 4x4 grid and look at how much local means vary —
    # a strong sign of shadow / uneven illumination
    h, w = gray.shape
    grid_means = []
    for i in range(4):
        for j in range(4):
            block = gray[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            grid_means.append(np.mean(block))
    grid_std = np.std(grid_means)

    if grid_std > 35:
        return "shadow_uneven"
    elif mean_val < 80:
        return "dark"
    elif mean_val > 190:
        return "bright"
    else:
        return "normal"


# CLAHE parameter presets tuned per lighting category.
# clipLimit: how aggressively contrast is boosted
# tileGridSize: smaller tiles = more local/aggressive correction (good for shadows)
LIGHTING_PRESETS = {
    "dark":          {"clipLimit": 3.5, "tileGridSize": (8, 8)},
    "bright":        {"clipLimit": 1.5, "tileGridSize": (8, 8)},
    "shadow_uneven": {"clipLimit": 2.5, "tileGridSize": (4, 4)},  # smaller tiles fight local shadows
    "normal":        {"clipLimit": 2.0, "tileGridSize": (8, 8)},
}


def enhance_contrast(gray, lighting_label=None):
    """Applies CLAHE using parameters chosen for the detected lighting
    condition. Returns (enhanced_image, lighting_label_used)."""
    if lighting_label is None:
        lighting_label = classify_lighting(gray)
    params = LIGHTING_PRESETS[lighting_label]
    clahe = cv2.createCLAHE(clipLimit=params["clipLimit"], tileGridSize=params["tileGridSize"])
    enhanced = clahe.apply(gray)
    return enhanced, lighting_label


def reduce_noise(gray):
    """Gaussian blur to suppress high-frequency noise before edge
    detection, without erasing thin crack lines (small 3x3 kernel)."""
    return cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)


def preprocess_pipeline(image_bgr):
    """
    Runs the full preprocessing chain and returns a dict of every
    intermediate result (needed for the "save intermediate outputs"
    requirement) plus the lighting label detected.
    """
    resized = resize_image(image_bgr)
    gray = to_grayscale(resized)
    lighting_label = classify_lighting(gray)
    enhanced, _ = enhance_contrast(gray, lighting_label)
    denoised = reduce_noise(enhanced)

    return {
        "resized": resized,
        "grayscale": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "lighting_label": lighting_label,
    }
