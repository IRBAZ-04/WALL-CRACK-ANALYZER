"""
source.py
---------
Core pipeline interface and algorithms re-exporter for Wall Crack Analyzer.
Provides pure classical mathematical image processing without OpenCV algorithm dependencies.
"""
import os
import sys
import time
import numpy as np
from PIL import Image

# Re-export key modules and functions
import main
from algorithms import preprocessing, sobel, prewitt, laplacian, canny, fusion, segmentation, morphology
from measurement import calibration, length, width, area

run_pipeline = main.run_pipeline
generate_width_heatmap = main.generate_width_heatmap
annotate_final = main.annotate_final
crack_confidence_score = main.crack_confidence_score
save = main.save
OUTPUT_DIR = main.OUTPUT_DIR


def analyze_image_classical(image_path: str, chosen_algo: str = "canny", cm_per_pixel: float = None):
    """
    Pure classical image processing analysis pipeline (no AI bypass).
    Returns all 8 intermediate images (as NumPy arrays) and metrics dictionary.
    """
    start_time = time.time()
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at path: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)

    # 1. Preprocessing
    pre = preprocessing.preprocess_pipeline(image_rgb)
    resized, gray, enhanced, denoised = pre["resized"], pre["grayscale"], pre["enhanced"], pre["denoised"]
    lighting = pre["lighting_label"]

    # 2. Edge Detection Operators
    edge_maps = {}
    timings = {}
    edge_maps["sobel"], timings["sobel"] = sobel.detect(denoised)
    edge_maps["prewitt"], timings["prewitt"] = prewitt.detect(denoised)
    edge_maps["laplacian"], timings["laplacian"] = laplacian.detect(denoised)
    edge_maps["canny"], timings["canny"] = canny.detect(denoised)
    edge_maps["fusion"], timings["fusion"] = fusion.detect(
        denoised, edge_maps["sobel"], edge_maps["prewitt"], edge_maps["laplacian"], edge_maps["canny"]
    )

    algo_key = chosen_algo.lower().strip()
    if algo_key not in edge_maps:
        algo_key = "canny"
    chosen_map = edge_maps[algo_key]

    # 3. Segmentation & Morphology
    binary_mask, seg_method = segmentation.segment(chosen_map, lighting)
    cleaned_mask = morphology.clean_mask(binary_mask, gray_image=denoised)

    # 4. Calibration & Measurement
    if cm_per_pixel is None or cm_per_pixel <= 0:
        calib_px = None
    else:
        calib_px = float(cm_per_pixel)

    length_result, skeleton = length.measure_length(cleaned_mask, calib_px)
    skeleton_img = (skeleton.astype(np.uint8)) * 255

    area_result = area.measure_area(cleaned_mask, cm_per_pixel=calib_px)
    width_result = width.measure_width(cleaned_mask, skeleton, calib_px, damage_pct=area_result["damage_percentage"])
    confidence = crack_confidence_score(cleaned_mask, chosen_map)

    # 5. Width Heatmap & Annotated Overlay
    heatmap_img = generate_width_heatmap(resized, cleaned_mask)
    
    crack_found = area_result["crack_area_px"] > 0
    total_time = round(time.time() - start_time, 3)

    measurements = {
        "crack_detected": crack_found,
        "lighting_label": lighting,
        "segmentation_method": seg_method,
        "calibrated": calib_px is not None,
        "cm_per_pixel": calib_px,
        "length": length_result,
        "width": width_result,
        "area": area_result,
        "confidence": confidence,
        "runtime_sec": total_time,
        "operator_timings": timings,
    }

    annotated = annotate_final(resized, cleaned_mask, measurements, algo_key, skeleton=skeleton)

    images = {
        "original": resized,
        "enhanced": enhanced,
        "edge_map": chosen_map,
        "cleaned_mask": cleaned_mask,
        "skeleton": skeleton_img,
        "width_heatmap": heatmap_img,
        "fusion_map": edge_maps["fusion"],
        "annotated_overlay": annotated,
    }

    return images, measurements
