"""
dataset/generate_synthetic_dataset.py
--------------------------------------
Generates a representative suite of campus wall crack images and ground-truth masks
across diverse lighting conditions (daylight, noon, evening, shadow) and crack types
(hairline, deep, vertical, horizontal, diagonal, multiple, no-crack) along with a
populated dataset_log.csv.

Pure NumPy + Pillow implementation — zero OpenCV dependency.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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

    # Convert to 3-channel RGB array
    wall_rgb = np.stack([wall, wall, wall], axis=-1)
    return wall_rgb


def draw_crack_pattern(shape, crack_type="vertical"):
    height, width = shape[:2]
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    points = []
    branch_pts = []

    if crack_type == "vertical":
        for y in range(50, height - 50, 10):
            x = int(width // 2 + np.sin(y / 30.0) * 15 + np.random.randint(-3, 4))
            points.append((x, y))
    elif crack_type == "horizontal":
        for x in range(50, width - 50, 10):
            y = int(height // 2 + np.cos(x / 40.0) * 12 + np.random.randint(-2, 3))
            points.append((x, y))
    elif crack_type == "diagonal":
        for step in range(60, min(height, width) - 60, 10):
            x = int(step + np.random.randint(-2, 3))
            y = int(step * 0.7 + np.random.randint(-2, 3))
            points.append((x, y))
    elif crack_type == "hairline":
        for y in range(100, height - 100, 8):
            x = int(width // 3 + np.sin(y / 20.0) * 8 + np.random.randint(-1, 2))
            points.append((x, y))
    elif crack_type == "deep_wide":
        for y in range(40, height - 40, 10):
            x = int(width * 0.6 + np.cos(y / 35.0) * 25 + np.random.randint(-4, 5))
            points.append((x, y))
    elif crack_type == "multiple":
        for y in range(70, height - 70, 10):
            x = int(width // 2 + np.sin(y / 25.0) * 20)
            points.append((x, y))
        for step in range(0, 150, 10):
            branch_pts.append((width // 2 + step, height // 2 + step // 2))

    thickness = 2 if crack_type == "hairline" else (8 if crack_type == "deep_wide" else 4)

    if len(branch_pts) > 1:
        draw.line(branch_pts, fill=255, width=3)
    if len(points) > 1:
        draw.line(points, fill=255, width=thickness)

    return np.array(mask_img, dtype=np.uint8)


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
        wall_rgb = draw_synthetic_wall(800, 600, lighting)

        if c_type != "no_crack":
            gt_mask = draw_crack_pattern(wall_rgb.shape, c_type)
            crack_indices = gt_mask > 0
            wall_rgb[crack_indices] = (wall_rgb[crack_indices].astype(np.float32) * 0.35).astype(np.uint8)
        else:
            gt_mask = np.zeros((600, 800), dtype=np.uint8)

        # Draw 10 cm reference marker in bottom right (bright orange RGB: 255, 140, 0)
        img_pil = Image.fromarray(wall_rgb)
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([650, 520, 750, 540], fill=(255, 140, 0))
        draw.text((670, 502), "10 cm", fill=(255, 255, 255))

        raw_path = os.path.join(RAW_DIR, f"{img_id}.jpg")
        gt_path = os.path.join(GT_DIR, f"{img_id}.png")

        img_pil.save(raw_path, quality=95)
        Image.fromarray(gt_mask).save(gt_path)

        log_rows.append({
            "image_id": img_id,
            "location": loc,
            "crack_type": c_type,
            "lighting": lighting,
            "indoor_outdoor": ind_out,
            "capture_distance": dist,
            "camera_device": "Campus Survey Camera",
            "reference_marker_cm": 10.0,
            "raw_image_path": raw_path,
            "ground_truth_path": gt_path,
            "date_logged": "2024-10-25"
        })

    df = pd.DataFrame(log_rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated {len(sample_configs)} synthetic wall dataset samples with ground truth masks.")
    print(f"Logged dataset records to {CSV_PATH}")


if __name__ == "__main__":
    generate_dataset_samples()
