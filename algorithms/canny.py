"""
algorithms/canny.py
-------------------
Complete 5-stage Canny Edge Detector implemented 100% from scratch in pure NumPy:
  Stage 1: Gaussian Smoothing Filter
  Stage 2: Spatial Gradient Magnitude & Direction (Sobel Gx, Gy, arctan2)
  Stage 3: Non-Maximum Suppression (NMS) with 4-Sector Angle Discretization
  Stage 4: Adaptive Double Thresholding (Strong / Weak Edge Partitioning via Otsu)
  Stage 5: Edge Tracking by Hysteresis (8-Connectivity Connected Component Search)
"""

import time
import numpy as np
from collections import deque
from .convolution import convolve2d, create_gaussian_kernel
from .segmentation import compute_otsu_threshold

SOBEL_X = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0]
], dtype=np.float64)


def non_maximum_suppression(magnitude: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """
    Stage 3: Thins broad edge ridges by suppressing non-peak values along the gradient vector.
    Discretizes gradient directions into 4 principal angular sectors:
      - 0°   (East-West horizontal): compare with left and right neighbors
      - 45°  (North-East diagonal) : compare with top-right and bottom-left
      - 90°  (North-South vertical): compare with top and bottom neighbors
      - 135° (North-West diagonal) : compare with top-left and bottom-right
    """
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)

    # Convert negative angles into positive range [0, 180)
    angle = (angle_deg % 180.0).astype(np.float32)

    # Sector masks
    sec0 = ((angle >= 0.0) & (angle < 22.5)) | ((angle >= 157.5) & (angle < 180.0))
    sec1 = (angle >= 22.5) & (angle < 67.5)
    sec2 = (angle >= 67.5) & (angle < 112.5)
    sec3 = (angle >= 112.5) & (angle < 157.5)

    pad_mag = np.pad(magnitude, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    # Neighbor shifts (r, c) inside 1-padded array corresponds to r+1, c+1
    # Sector 0: (r, c-1) and (r, c+1)
    q0 = pad_mag[1:-1, 0:-2]
    r0 = pad_mag[1:-1, 2:]

    # Sector 1: (r-1, c+1) and (r+1, c-1)
    q1 = pad_mag[0:-2, 2:]
    r1 = pad_mag[2:, 0:-2]

    # Sector 2: (r-1, c) and (r+1, c)
    q2 = pad_mag[0:-2, 1:-1]
    r2 = pad_mag[2:, 1:-1]

    # Sector 3: (r-1, c-1) and (r+1, c+1)
    q3 = pad_mag[0:-2, 0:-2]
    r3 = pad_mag[2:, 2:]

    # Peak condition
    keep0 = sec0 & (magnitude >= q0) & (magnitude >= r0)
    keep1 = sec1 & (magnitude >= q1) & (magnitude >= r1)
    keep2 = sec2 & (magnitude >= q2) & (magnitude >= r2)
    keep3 = sec3 & (magnitude >= q3) & (magnitude >= r3)

    peak_mask = keep0 | keep1 | keep2 | keep3
    nms[peak_mask] = magnitude[peak_mask]
    return nms


def hysteresis_tracking(nms: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
    """
    Stage 4 & 5: Double Thresholding and Hysteresis Edge Tracking.
    Identifies strong edges and weak edges. Promotes weak edges to strong if and only if
    they are connected to strong edges in an 8-neighborhood.
    """
    STRONG = 255
    WEAK = 75

    h, w = nms.shape
    res = np.zeros((h, w), dtype=np.uint8)

    strong_r, strong_c = np.where(nms >= high_thresh)
    weak_r, weak_c = np.where((nms >= low_thresh) & (nms < high_thresh))

    res[strong_r, strong_c] = STRONG
    res[weak_r, weak_c] = WEAK

    # BFS flood tracking from strong edge pixels
    queue = deque(zip(strong_r, strong_c))
    visited = np.zeros((h, w), dtype=bool)
    visited[strong_r, strong_c] = True

    # 8-neighborhood offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if not visited[nr, nc] and res[nr, nc] == WEAK:
                    res[nr, nc] = STRONG
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # All unpromoted weak edges are suppressed
    res[res != STRONG] = 0
    return res


def detect(gray_image: np.ndarray) -> tuple:
    """
    Full 5-stage Canny Edge Detection in pure NumPy.
    Returns (edges_uint8, elapsed_seconds).
    """
    start = time.time()

    # Stage 1: Gaussian filter smoothing
    gauss_k = create_gaussian_kernel(size=5, sigma=1.2)
    smoothed = convolve2d(gray_image, gauss_k, padding="reflect")

    # Stage 2: Gradient magnitude & angle
    gx = convolve2d(smoothed, SOBEL_X, padding="reflect")
    gy = convolve2d(smoothed, SOBEL_Y, padding="reflect")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # Gradient direction in degrees [0, 360)
    angle_deg = np.rad2deg(np.arctan2(gy, gx))

    # Stage 3: Non-Maximum Suppression on gradient magnitude
    nms = non_maximum_suppression(magnitude, angle_deg)

    # Stage 4: Adaptive thresholds based on gradient statistics
    # Suppresses ambient surface texture noise while capturing continuous crack fractures
    non_zeros = nms[nms > 8]
    if len(non_zeros) > 0:
        med = float(np.median(non_zeros))
        high_thresh = max(med * 1.8, 55.0)
        low_thresh = 0.35 * high_thresh
    else:
        high_thresh = 60.0
        low_thresh = 25.0

    # Stage 5: Hysteresis Tracking
    edges = hysteresis_tracking(nms, low_thresh, high_thresh)

    elapsed = time.time() - start
    return edges, elapsed
