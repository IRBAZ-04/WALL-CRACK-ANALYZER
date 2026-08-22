"""
main.py
--------
Orchestrates the complete classical pipeline for ONE image:

  Input -> Preprocess -> [Sobel|Prewitt|Laplacian|Canny|Fusion]
        -> Segment -> Morphology -> Skeletonize -> Measure
        -> Damage % -> Orientation & Severity Triage -> Heatmap -> Annotated Output

Saves every intermediate stage to outputs/<stage>/<image_id>.png
(faculty requirement: intermediate outputs must be generated and saved).

Usage:
    python main.py path/to/image.jpg
"""

import os
import sys
import cv2
import numpy as np

from algorithms import preprocessing, sobel, prewitt, laplacian, canny, fusion, segmentation, morphology
from measurement import calibration, length, width, area

OUTPUT_DIR = "outputs"


def save(stage, image_id, img):
    folder = os.path.join(OUTPUT_DIR, stage)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, f"{image_id}.png"), img)


def crack_confidence_score(binary_mask, edge_map):
    total_px = binary_mask.shape[0] * binary_mask.shape[1]
    crack_px = np.count_nonzero(binary_mask)
    density_score = min(crack_px / (total_px * 0.02), 1.0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    elongation_score = 0.0
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, a = stats[largest_idx]
        aspect = max(w, h) / max(min(w, h), 1)
        elongation_score = min(aspect / 8.0, 1.0)

    confidence = round(50 * density_score + 50 * elongation_score, 1)
    return confidence


def generate_width_heatmap(original_bgr, binary_mask):
    """Generates a pseudocolor heatmap indicating local crack width intensity."""
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    norm_dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm_dist, cv2.COLORMAP_JET)

    blended = original_bgr.copy()
    mask_idx = binary_mask > 0
    blended[mask_idx] = cv2.addWeighted(original_bgr[mask_idx], 0.3, heatmap[mask_idx], 0.7, 0)
    return blended


def annotate_final(original_bgr, binary_mask, measurements, algo_name):
    overlay = original_bgr.copy()
    overlay[binary_mask > 0] = [0, 0, 255]  # red highlight
    blended = cv2.addWeighted(original_bgr, 0.65, overlay, 0.35, 0)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 255), 1)

    w_m = measurements["width"]
    lines = [
        f"Algorithm: {algo_name.upper()}",
        f"Length: {measurements['length']['length_px']} px"
        + (f" ({measurements['length']['length_cm']} cm)" if measurements['length']['length_cm'] else ""),
        f"Avg width: {w_m['avg_width_px']} px"
        + (f" ({w_m['avg_width_cm']} cm)" if w_m['avg_width_cm'] else ""),
        f"Orientation: {w_m['orientation_type']}",
        f"Severity: {w_m['severity_level']}",
        f"Damage: {measurements['area']['damage_percentage']}%",
        f"Confidence: {measurements['confidence']}%",
    ]
    y0 = 25
    for i, line in enumerate(lines):
        y = y0 + i * 20
        cv2.putText(blended, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def run_pipeline(image_path, chosen_algo="canny", reference_cm=10.0, save_outputs=True):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pre = preprocessing.preprocess_pipeline(image_bgr)
    resized, gray, enhanced, denoised = pre["resized"], pre["grayscale"], pre["enhanced"], pre["denoised"]
    lighting = pre["lighting_label"]

    edge_maps = {}
    timings = {}
    edge_maps["sobel"], timings["sobel"] = sobel.detect(denoised)
    edge_maps["prewitt"], timings["prewitt"] = prewitt.detect(denoised)
    edge_maps["laplacian"], timings["laplacian"] = laplacian.detect(denoised)
    edge_maps["canny"], timings["canny"] = canny.detect(denoised)
    edge_maps["fusion"], timings["fusion"] = fusion.detect(
        denoised, edge_maps["sobel"], edge_maps["prewitt"], edge_maps["laplacian"], edge_maps["canny"]
    )

    if save_outputs:
        save("grayscale", image_id, gray)
        save("enhanced", image_id, enhanced)
        for name in ["sobel", "prewitt", "laplacian", "canny", "fusion"]:
            save(name, image_id, edge_maps[name])

    chosen_map = edge_maps[chosen_algo]
    binary_mask, seg_method = segmentation.segment(chosen_map, lighting)
    cleaned_mask = morphology.clean_mask(binary_mask)

    if save_outputs:
        save("segmentation", image_id, binary_mask)
        save("morphology", image_id, cleaned_mask)

    length_result, skeleton = length.measure_length(cleaned_mask)
    skeleton_img = (skeleton.astype(np.uint8)) * 255
    if save_outputs:
        save("skeleton", image_id, skeleton_img)

    cm_per_pixel, marker_px = calibration.get_cm_per_pixel(resized, real_world_cm=reference_cm)

    length_result, skeleton = length.measure_length(cleaned_mask, cm_per_pixel)
    area_result = area.measure_area(cleaned_mask, cm_per_pixel=cm_per_pixel)
    width_result = width.measure_width(cleaned_mask, skeleton, cm_per_pixel, damage_pct=area_result["damage_percentage"])
    confidence = crack_confidence_score(cleaned_mask, chosen_map)

    heatmap_img = generate_width_heatmap(resized, cleaned_mask)
    if save_outputs:
        save("heatmap", image_id, heatmap_img)

    measurements = {
        "length": length_result,
        "width": width_result,
        "area": area_result,
        "confidence": confidence,
        "lighting_detected": lighting,
        "segmentation_method": seg_method,
        "calibrated": cm_per_pixel is not None,
        "timings_sec": timings,
    }

    final_img = annotate_final(resized, cleaned_mask, measurements, chosen_algo)
    if save_outputs:
        save("final", image_id, final_img)

    return final_img, measurements


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [algorithm]")
        sys.exit(1)

    img_path = sys.argv[1]
    algo = sys.argv[2] if len(sys.argv) > 2 else "canny"
    _, m = run_pipeline(img_path, chosen_algo=algo)

    print(f"\nLighting detected : {m['lighting_detected']}")
    print(f"Segmentation used : {m['segmentation_method']}")
    print(f"Calibrated        : {m['calibrated']}")
    print(f"Confidence        : {m['confidence']}%")
    print(f"Severity Level    : {m['width']['severity_level']}")
    print(f"Orientation       : {m['width']['orientation_type']} ({m['width']['orientation_deg']} deg)")
    print(f"Length            : {m['length']}")
    print(f"Width             : {m['width']}")
    print(f"Area/Damage       : {m['area']}")
