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
from ai import get_classifier

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


def annotate_normal(original_bgr, ai_res):
    """Draws a clean status badge when AI classifies wall as normal (sound)."""
    annotated = original_bgr.copy()
    h, w = annotated.shape[:2]
    banner = annotated.copy()
    cv2.rectangle(banner, (10, 10), (min(w - 10, 500), 80), (20, 20, 20), -1)
    cv2.rectangle(banner, (10, 10), (min(w - 10, 500), 80), (34, 197, 94), 2)
    blended = cv2.addWeighted(annotated, 0.4, banner, 0.6, 0)

    cv2.putText(blended, "AI: NO CRACK DETECTED", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2, cv2.LINE_AA)
    conf = ai_res.get("confidence", 0.0) if ai_res else 0.0
    cv2.putText(blended, f"Confidence: {conf}% | Sound Wall (Pipeline Bypassed)", (22, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return blended


def annotate_final(original_bgr, binary_mask, measurements, algo_name):
    overlay = original_bgr.copy()
    overlay[binary_mask > 0] = [0, 0, 255]  # red highlight
    blended = cv2.addWeighted(original_bgr, 0.65, overlay, 0.35, 0)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 255), 1)

    w_m = measurements["width"]
    lines = []
    if measurements.get("ai") and measurements["ai"].get("model_loaded"):
        ai_res = measurements["ai"]
        lines.append(f"AI: {ai_res['status']} ({ai_res['confidence']}%)")

    lines.extend([
        f"Algorithm: {algo_name.upper()}",
        f"Length: {measurements['length']['length_px']} px"
        + (f" ({measurements['length']['length_cm']} cm)" if measurements['length']['length_cm'] else ""),
        f"Avg width: {w_m['avg_width_px']} px"
        + (f" ({w_m['avg_width_cm']} cm)" if w_m['avg_width_cm'] else ""),
        f"Orientation: {w_m['orientation_type']}",
        f"Severity: {w_m['severity_level']}",
        f"Damage: {measurements['area']['damage_percentage']}%",
        f"Classical Conf: {measurements['confidence']}%",
    ])
    y0 = 25
    for i, line in enumerate(lines):
        y = y0 + i * 20
        cv2.putText(blended, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def run_pipeline(image_path, chosen_algo="canny", reference_cm=10.0, save_outputs=True, run_ai=True, skip_if_normal=True):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Stage 0: AI MobileNetV2 Screening
    ai_res = None
    if run_ai:
        try:
            clf = get_classifier()
            ai_res = clf.predict(image_bgr)
        except Exception as exc:
            ai_res = {
                "status": "AI_ERROR",
                "label": "ERROR",
                "is_crack": True,
                "confidence": 0.0,
                "raw_score": 0.0,
                "model_loaded": False,
                "error": str(exc),
            }

    # If AI classifies as NORMAL, bypass expensive classical edge & measurement processing
    if ai_res is not None and not ai_res["is_crack"] and skip_if_normal:
        pre = preprocessing.preprocess_pipeline(image_bgr)
        resized = pre["resized"]
        lighting = pre["lighting_label"]
        final_img = annotate_normal(resized, ai_res)

        if save_outputs:
            save("final", image_id, final_img)

        measurements = {
            "ai": ai_res,
            "crack_detected": False,
            "length": {"length_px": 0, "length_cm": 0.0},
            "width": {
                "min_width_px": 0,
                "max_width_px": 0,
                "avg_width_px": 0,
                "min_width_cm": 0.0,
                "max_width_cm": 0.0,
                "avg_width_cm": 0.0,
                "orientation_type": "N/A",
                "orientation_deg": 0.0,
                "severity_level": "NONE",
            },
            "area": {
                "crack_area_px": 0,
                "wall_area_px": int(resized.shape[0] * resized.shape[1]),
                "damage_percentage": 0.0,
                "crack_area_cm2": 0.0,
                "wall_area_cm2": None,
            },
            "confidence": 0.0,  # Classical confidence score remains 0
            "lighting_detected": lighting,
            "segmentation_method": "N/A (AI screen: Normal)",
            "calibrated": False,
            "timings_sec": {},
        }
        return final_img, measurements

    # Existing classical pipeline execution
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
        "ai": ai_res,
        "crack_detected": True,
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

    if m.get("ai"):
        ai_m = m["ai"]
        print(f"\n--- AI Classification (MobileNetV2) ---")
        print(f"Status            : {ai_m['status']}")
        print(f"Confidence        : {ai_m['confidence']}% (raw score: {ai_m['raw_score']})")
        if not m.get("crack_detected", True):
            print(f"Result            : Sound wall detected. Detailed crack quantification pipeline skipped.")
            sys.exit(0)

    print(f"\n--- Classical Image Processing Analysis ---")
    print(f"Lighting detected : {m['lighting_detected']}")
    print(f"Segmentation used : {m['segmentation_method']}")
    print(f"Calibrated        : {m['calibrated']}")
    print(f"Classical Conf    : {m['confidence']}%")
    print(f"Severity Level    : {m['width']['severity_level']}")
    print(f"Orientation       : {m['width']['orientation_type']} ({m['width']['orientation_deg']} deg)")
    print(f"Length            : {m['length']}")
    print(f"Width             : {m['width']}")
    print(f"Area/Damage       : {m['area']}")

