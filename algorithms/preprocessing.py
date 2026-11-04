"""
algorithms/preprocessing.py
----------------------------
Stage 1: Raw image normalization, grayscale conversion, lighting-aware
adaptive contrast enhancement (from-scratch CLAHE), and Gaussian denoising.
100% pure NumPy implementation with ZERO OpenCV dependencies.
"""

import numpy as np
from PIL import Image
from .convolution import convolve2d, create_gaussian_kernel


def resize_image(image: np.ndarray, target_width: int = 800) -> np.ndarray:
    """
    Resizes image maintaining aspect ratio using bilinear interpolation.
    """
    h, w = image.shape[:2]
    scale = target_width / float(w)
    target_height = int(round(h * scale))

    # Fast Pillow-backed bilinear resizing for arrays
    pil_img = Image.fromarray(image.astype(np.uint8))
    resized_pil = pil_img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return np.array(resized_pil)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts RGB/BGR image to grayscale using ITU-R BT.601 standard weights:
        Y = 0.299 * R + 0.587 * G + 0.114 * B
    """
    if len(image.shape) == 2:
        return image.astype(np.uint8)

    channels = image.shape[2]
    if channels == 4:
        image = image[:, :, :3]

    # Assume standard RGB order from PIL
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def classify_lighting(gray: np.ndarray) -> str:
    """
    Classifies image lighting based on statistical distribution of pixel intensities:
      - Global mean brightness (dark vs bright)
      - Spatial grid standard deviation (shadowed / uneven illumination)
    """
    mean_val = float(np.mean(gray))

    # 4x4 spatial block grid variance to detect shadows
    h, w = gray.shape
    grid_means = []
    for i in range(4):
        for j in range(4):
            block = gray[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            grid_means.append(float(np.mean(block)))
    grid_std = float(np.std(grid_means))

    if grid_std > 35:
        return "shadow_uneven"
    elif mean_val < 80:
        return "dark"
    elif mean_val > 190:
        return "bright"
    else:
        return "normal"


LIGHTING_PRESETS = {
    "dark":          {"clip_limit": 3.5, "grid_size": (8, 8)},
    "bright":        {"clip_limit": 1.5, "grid_size": (8, 8)},
    "shadow_uneven": {"clip_limit": 2.5, "grid_size": (4, 4)},
    "normal":        {"clip_limit": 2.0, "grid_size": (8, 8)},
}


def clahe_from_scratch(gray: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Textbook Contrast-Limited Adaptive Histogram Equalization (CLAHE) from scratch:
      1. Divides image into contextual tiles (grid_rows x grid_cols).
      2. Computes histogram per tile and clips at clip_limit * (num_pixels / 256).
      3. Redistributes clipped excess pixels uniformly across all bins.
      4. Computes Cumulative Distribution Function (CDF) transformation table.
      5. Bilinearly interpolates grayscale mappings between tile center points to prevent tile boundary artifacts.
    """
    gray_clean = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    h, w = gray_clean.shape
    num_tiles_y, num_tiles_x = grid_size
    tile_h = h // num_tiles_y
    tile_w = w // num_tiles_x

    # Compute clipped CDF mapping table for each tile
    mappings = np.zeros((num_tiles_y, num_tiles_x, 256), dtype=np.float32)

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            y_start = ty * tile_h
            y_end = (ty + 1) * tile_h if ty < num_tiles_y - 1 else h
            x_start = tx * tile_w
            x_end = (tx + 1) * tile_w if tx < num_tiles_x - 1 else w

            tile = gray_clean[y_start:y_end, x_start:x_end]
            num_pixels = tile.size

            # Histogram
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))

            # Clip limit threshold
            actual_clip = int(round(clip_limit * (num_pixels / 256.0)))
            if actual_clip > 0:
                excess = np.maximum(hist - actual_clip, 0)
                total_excess = int(np.sum(excess))
                hist = np.minimum(hist, actual_clip)

                # Uniform redistribution of excess
                hist += total_excess // 256
                hist[:total_excess % 256] += 1

            # Cumulative Distribution Function (CDF)
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
            denom = num_pixels - cdf_min
            if denom > 0:
                lut = np.round((cdf - cdf_min) * 255.0 / denom)
            else:
                lut = cdf * (255.0 / max(num_pixels, 1))

            mappings[ty, tx] = np.clip(lut, 0, 255)

    # Bilinear interpolation between tile centers
    center_y = np.array([(ty + 0.5) * tile_h for ty in range(num_tiles_y)])
    center_x = np.array([(tx + 0.5) * tile_w for tx in range(num_tiles_x)])

    y_indices = np.arange(h)
    x_indices = np.arange(w)

    # Find tile indices for interpolation
    ty0 = np.clip(np.searchsorted(center_y, y_indices) - 1, 0, num_tiles_y - 2).astype(np.intp)
    ty1 = (ty0 + 1).astype(np.intp)
    tx0 = np.clip(np.searchsorted(center_x, x_indices) - 1, 0, num_tiles_x - 2).astype(np.intp)
    tx1 = (tx0 + 1).astype(np.intp)

    ay = np.clip((y_indices - center_y[ty0]) / (center_y[ty1] - center_y[ty0] + 1e-6), 0.0, 1.0).astype(np.float32)[:, None]
    ax = np.clip((x_indices - center_x[tx0]) / (center_x[tx1] - center_x[tx0] + 1e-6), 0.0, 1.0).astype(np.float32)[None, :]

    # Lookup values for the 4 contextual tile neighbors
    g = gray_clean.astype(np.intp)
    ty0_arr = ty0[:, None]
    ty1_arr = ty1[:, None]
    tx0_arr = tx0[None, :]
    tx1_arr = tx1[None, :]

    v00 = mappings[ty0_arr, tx0_arr, g]
    v01 = mappings[ty0_arr, tx1_arr, g]
    v10 = mappings[ty1_arr, tx0_arr, g]
    v11 = mappings[ty1_arr, tx1_arr, g]

    w00 = (1.0 - ay) * (1.0 - ax)
    w01 = (1.0 - ay) * ax
    w10 = ay * (1.0 - ax)
    w11 = ay * ax

    out = w00 * v00
    out += w01 * v01
    out += w10 * v10
    out += w11 * v11

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def enhance_contrast(gray: np.ndarray, lighting_label: str = None) -> tuple:
    """
    Applies lighting-aware adaptive contrast enhancement (from-scratch CLAHE).
    Returns (enhanced_image, lighting_label_used).
    """
    if lighting_label is None:
        lighting_label = classify_lighting(gray)
    params = LIGHTING_PRESETS[lighting_label]
    enhanced = clahe_from_scratch(
        gray, clip_limit=params["clip_limit"], grid_size=params["grid_size"]
    )
    return enhanced, lighting_label


def reduce_noise(gray: np.ndarray) -> np.ndarray:
    """
    Suppresses high-frequency sensor noise via 2D spatial Gaussian convolution
    without blunting thin fracture lines.
    """
    kernel = create_gaussian_kernel(size=3, sigma=0.8)
    smoothed = convolve2d(gray, kernel, padding="reflect")
    return np.clip(np.round(smoothed), 0, 255).astype(np.uint8)


def preprocess_pipeline(image_rgb: np.ndarray) -> dict:
    """
    Complete Stage 1 preprocessing chain in pure mathematical NumPy:
      Input RGB -> Resized -> Grayscale -> CLAHE Enhanced -> Gaussian Denoised.
    """
    resized = resize_image(image_rgb)
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
