"""
dataset/generate_synthetic_dataset.py
--------------------------------------
Generates a representative suite of campus wall crack images and ground-truth masks
across diverse lighting conditions (daylight, noon, evening, shadow) and crack types
(hairline, deep, vertical, horizontal, diagonal, multiple, no-crack) along with a
populated dataset_log.csv.
"""

import os
import cv2
import numpy as np
import pandas as pd

RAW_DIR = "dataset/raw"
GT_DIR = "dataset/ground_truth"
CSV_PATH = "dataset_log_template.csv"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)


def draw_synthetic_wall(width=800, height=600, lighting="daylight"):
    # Base wall texture
    if lighting == "daylight":
        base_color = 190
        noise_level = 15
    elif lighting == "noon_sunlight":
        base_color = 230
        noise_level = 10
    elif lighting == "low_light_evening":
        base_color = 70
        noise_level = 25
    elif lighting == "shadow_uneven":
        base_color = 160
        noise_level = 20
    else:
        base_color = 180
        noise_level = 15

    wall = np.full((height, width), base_color, dtype=np.uint8)
    noise = np.random.normal(0, noise_level, (height, width)).astype(np.float32)
    wall = np.clip(wall + noise, 0, 255).astype(np.uint8)

    # Add lighting gradient / uneven illumination if specified
    if lighting == "shadow_uneven":
        gradient = np.linspace(0.3, 1.2, width)
        wall = np.clip(wall * gradient, 0, 255).astype(np.uint8)
    elif lighting == "noon_sunlight":
        X, Y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        vignette = 1 - 0.3 * (X ** 2 + Y ** 2)
        wall = np.clip(wall * vignette, 0, 255).astype(np.uint8)

    wall_bgr = cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)
    return wall_bgr


def draw_crack_pattern(shape, crack_type="vertical"):
    height, width = shape[:2]
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    points = []

    if crack_type == "vertical":
        x_center = width // 2
        for y in range(80, height - 80, 10):
            x = int(x_center + np.sin(y / 30.0) * 25 + np.random.randint(-5, 6))
            points.append((x, y))
    elif crack_type == "horizontal":
        y_center = height // 2
        for x in range(80, width - 80, 10):
            y = int(y_center + np.cos(x / 30.0) * 20 + np.random.randint(-4, 5))
            points.append((x, y))
    elif crack_type == "diagonal":
        for t in range(100, min(width, height) - 100, 10):
            x = int(t + np.sin(t / 20.0) * 15 + np.random.randint(-3, 4))
            y = int(t + np.cos(t / 25.0) * 15 + np.random.randint(-3, 4))
            points.append((x, y))
    elif crack_type == "hairline":
        x_center = width // 3
        for y in range(120, height - 120, 8):
            x = int(x_center + np.sin(y / 20.0) * 12 + np.random.randint(-2, 3))
            points.append((x, y))
    elif crack_type == "deep_wide":
        x_center = width * 2 // 3
        for y in range(60, height - 60, 12):
            x = int(x_center + np.sin(y / 40.0) * 35 + np.random.randint(-8, 9))
            points.append((x, y))
    elif crack_type == "multiple":
        # Main vertical crack + side branch
        for y in range(70, height - 70, 10):
            x = int(width // 2 + np.sin(y / 25.0) * 20)
            points.append((x, y))
        # branch
        branch_pts = []
        for step in range(0, 150, 10):
            branch_pts.append((width // 2 + step, height // 2 + step // 2))
        if len(branch_pts) > 1:
            pts_arr = np.array(branch_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(gt_mask, [pts_arr], False, 255, 3)

    if len(points) > 1:
        pts_arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        thickness = 2 if crack_type == "hairline" else (8 if crack_type == "deep_wide" else 4)
        cv2.polylines(gt_mask, [pts_arr], False, 255, thickness)

    return gt_mask


def generate_dataset_samples():
    sample_configs = [
        ("CAMPUS_ACADEMIC_001", "Academic Block A", "deep_wide", "daylight", "outdoor", "1.5m"),
        ("CAMPUS_HOSTEL_002", "Hostel Block B", "hairline", "shadow_uneven", "outdoor", "2.0m"),
        ("CAMPUS_STAIR_003", "Main Staircase Wall", "vertical", "low_light_evening", "indoor", "1.2m"),
        ("CAMPUS_PARKING_004", "Basement Parking Pillar", "horizontal", "noon_sunlight", "indoor", "1.8m"),
        ("CAMPUS_CORRIDOR_005", "Science Wing Corridor", "diagonal", "daylight", "indoor", "1.0m"),
        ("CAMPUS_COMPOUND_006", "Outer Boundary Wall", "multiple", "shadow_uneven", "outdoor", "2.5m"),
        ("CAMPUS_CLEAN_007", "Library Reading Room", "no_crack", "daylight", "indoor", "1.5m"),
    ]

    log_rows = []

    for img_id, loc, c_type, lighting, ind_out, dist in sample_configs:
        wall_bgr = draw_synthetic_wall(800, 600, lighting)

        if c_type != "no_crack":
            gt_mask = draw_crack_pattern(wall_bgr.shape, c_type)
            # Overlay crack onto wall in darker tone
            crack_indices = gt_mask > 0
            wall_bgr[crack_indices] = (wall_bgr[crack_indices].astype(np.float32) * 0.35).astype(np.uint8)
        else:
            gt_mask = np.zeros((600, 800), dtype=np.uint8)

        # Draw a 10 cm reference marker in bottom right (bright orange)
        cv2.rectangle(wall_bgr, (650, 520), (750, 540), (0, 140, 255), -1)  # BGR orange
        cv2.putText(wall_bgr, "10 cm", (670, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        raw_path = os.path.join(RAW_DIR, f"{img_id}.jpg")
        gt_path = os.path.join(GT_DIR, f"{img_id}.png")

        cv2.imwrite(raw_path, wall_bgr)
        cv2.imwrite(gt_path, gt_mask)

        log_rows.append({
            "image_id": img_id,
            "location": loc,
            "has_crack": "YES" if c_type != "no_crack" else "NO",
            "crack_type": c_type,
            "lighting_condition": lighting,
            "indoor_outdoor": ind_out,
            "distance": dist,
            "notes": "Representative campus wall dataset sample with 10cm reference marker"
        })

    df = pd.DataFrame(log_rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated {len(sample_configs)} dataset samples and updated {CSV_PATH}.")


if __name__ == "__main__":
    generate_dataset_samples()
