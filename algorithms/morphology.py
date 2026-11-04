"""
algorithms/morphology.py
------------------------
Binary morphological operations and connected-component area filtering
implemented 100% from scratch in pure NumPy.

Includes:
  1. Mathematical Dilation (Minkowski Sum / Neighborhood Maximum)
  2. Mathematical Erosion  (Minkowski Difference / Neighborhood Minimum)
  3. Morphological Closing (Dilation -> Erosion to bridge micro-gaps)
  4. 8-Connected Component Labeling & Area-Based Noise Suppression
"""

import numpy as np
from collections import deque
from numpy.lib.stride_tricks import sliding_window_view


def binary_dilation(binary_mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Mathematical dilation with 3x3 square structuring element:
        (A ⊕ B)(x, y) = max { A(x - i, y - j) | (i, j) ∈ B }
    """
    res = (binary_mask > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = np.pad(res, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        windows = sliding_window_view(padded, window_shape=(3, 3))
        res = windows.max(axis=(-1, -2))
    return (res * 255).astype(np.uint8)


def binary_erosion(binary_mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Mathematical erosion with 3x3 square structuring element:
        (A ⊖ B)(x, y) = min { A(x + i, y + j) | (i, j) ∈ B }
    """
    res = (binary_mask > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = np.pad(res, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        windows = sliding_window_view(padded, window_shape=(3, 3))
        res = windows.min(axis=(-1, -2))
    return (res * 255).astype(np.uint8)


def morphological_closing(binary_mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Morphological closing: Dilation followed by Erosion.
    Bridges small discontinuous micro-fractures without expanding overall perimeter.
    """
    dilated = binary_dilation(binary_mask, iterations=iterations)
    closed = binary_erosion(dilated, iterations=iterations)
    return closed


def connected_components_labeling(binary_mask: np.ndarray) -> tuple:
    """
    8-Connectivity Connected Component Labeling from scratch via Breadth-First Search (BFS).
    Returns (num_labels, labels_array, stats_dict)
    stats_dict: {label_id: {'area': count, 'bbox': (min_r, min_c, max_r, max_c)}}
    """
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    stats = {}

    bin_img = binary_mask > 0
    visited = np.zeros((h, w), dtype=bool)

    # 8-connectivity neighbor offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    # Iterate over foreground pixels
    ys, xs = np.where(bin_img)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        current_label += 1
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        labels[start_y, start_x] = current_label

        area_count = 0
        min_r, max_r = start_y, start_y
        min_c, max_c = start_x, start_x

        while queue:
            cy, cx = queue.popleft()
            area_count += 1
            if cy < min_r: min_r = cy
            if cy > max_r: max_r = cy
            if cx < min_c: min_c = cx
            if cx > max_c: max_c = cx

            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if bin_img[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        labels[ny, nx] = current_label
                        queue.append((ny, nx))

        stats[current_label] = {
            "area": area_count,
            "bbox": (min_r, min_c, max_r, max_c),
        }

    return current_label, labels, stats


def clean_mask(binary_mask: np.ndarray, min_area: int = 15, gray_image: np.ndarray = None) -> np.ndarray:
    """
    Cleans raw binary crack mask:
      1. Morphological closing (2 iterations) to bridge fractured crack segments into continuous paths.
      2. Connected component filtering using geometric elongation (extent) and contrast to eliminate
         isolated noise/plaster specks while preserving the complete crack trajectory.
    """
    closed = morphological_closing(binary_mask, iterations=2)
    num_labels, labels, stats = connected_components_labeling(closed)

    cleaned = np.zeros_like(closed, dtype=np.uint8)
    wall_mean = float(np.mean(gray_image)) if gray_image is not None else 200.0

    for lbl, data in stats.items():
        area = data["area"]
        min_r, min_c, max_r, max_c = data["bbox"]
        extent = max(max_r - min_r, max_c - min_c)

        if gray_image is not None:
            mask_c = (labels == lbl)
            min_val = int(np.min(gray_image[mask_c]))
            # A true crack is an elongated fracture or large connected body entering a shadowed fissure
            is_crack = (
                (extent >= 25 and min_val < 0.88 * wall_mean)
                or (area >= 60 and min_val < 0.90 * wall_mean)
                or (extent >= 60)
                or (area >= 180)
            )
        else:
            is_crack = (extent >= 20 and area >= min_area) or (area >= 50)

        if is_crack:
            cleaned[labels == lbl] = 255

    return cleaned
