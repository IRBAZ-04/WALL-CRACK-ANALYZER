"""
main.py
--------
Orchestrates the complete from-scratch mathematical pipeline for ONE image:

  Input -> Preprocess -> [Sobel|Prewitt|Laplacian|Canny|Fusion]
        -> Segment -> Morphology -> Skeletonize -> Measure
        -> Damage % -> Orientation & Severity Triage -> Heatmap -> Annotated Output

Saves every intermediate stage to outputs/<stage>/<image_id>.png
100% pure mathematical image processing in NumPy / Pillow with ZERO OpenCV dependencies.

Usage:
    python main.py path/to/image.jpg [algorithm]
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

from algorithms import preprocessing, sobel, prewitt, laplacian, canny, fusion, segmentation, morphology
from measurement import calibration, length, width, area
from ai import get_classifier

OUTPUT_DIR = "outputs"


def save(stage: str, image_id: str, img: np.ndarray) -> None:
    folder = os.path.join(OUTPUT_DIR, stage)
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{image_id}.png")

    if len(img.shape) == 2:
        pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(img.astype(np.uint8), mode="RGB")
    pil_img.save(out_path)


def crack_confidence_score(binary_mask: np.ndarray, edge_map: np.ndarray) -> float:
    """
    Computes geometric crack confidence score from density and aspect ratio elongation
    using pure mathematical connected-component analysis.
    """
    total_px = binary_mask.shape[0] * binary_mask.shape[1]
    crack_px = np.count_nonzero(binary_mask)
    density_score = min(crack_px / (total_px * 0.02 + 1e-6), 1.0)

    num_labels, labels, stats = morphology.connected_components_labeling(binary_mask)
    elongation_score = 0.0
    if num_labels > 0:
        largest_lbl = max(stats.keys(), key=lambda k: stats[k]["area"])
        bbox = stats[largest_lbl]["bbox"]
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        aspect = max(w, h) / max(min(w, h), 1)
        elongation_score = min(aspect / 8.0, 1.0)

    confidence = round(50.0 * density_score + 50.0 * elongation_score, 1)
    return confidence


def generate_width_heatmap(original_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """
    Generates pseudocolor width intensity heatmap using exact Euclidean Distance Transform
    and Matplotlib jet colormap without OpenCV.
    """
    dist = distance_transform_edt(binary_mask > 0)
    max_d = np.max(dist)
    if max_d > 0:
        norm_dist = dist / max_d
    else:
        norm_dist = np.zeros_like(dist)

    rgba = cm.jet(norm_dist)[:, :, :3]
    heatmap_rgb = (rgba * 255.0).astype(np.uint8)

    blended = original_rgb.copy()
    mask_idx = binary_mask > 0
    blended[mask_idx] = (0.3 * original_rgb[mask_idx] + 0.7 * heatmap_rgb[mask_idx]).astype(np.uint8)
    return blended


def annotate_normal(original_rgb: np.ndarray, ai_res: dict) -> np.ndarray:
    """Draws a clean status badge when AI screens wall as sound (normal)."""
    pil_img = Image.fromarray(original_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    banner_w = min(pil_img.width - 20, 520)
    draw.rectangle([10, 10, 10 + banner_w, 75], fill=(15, 25, 20), outline=(34, 197, 94), width=2)
    draw.text((22, 18), "AI: NO CRACK DETECTED", fill=(34, 197, 94))
    conf = ai_res.get("confidence", 0.0) if ai_res else 0.0
    draw.text((22, 45), f"Confidence: {conf}% | Sound Wall (Pipeline Bypassed)", fill=(220, 220, 220))

    return np.array(pil_img)


def annotate_final(original_rgb: np.ndarray, binary_mask: np.ndarray, measurements: dict, algo_name: str, skeleton: np.ndarray = None) -> np.ndarray:
    """Draws vivid crack highlighting, clean continuous centerline, boundary contours, and measurement summary HUD using Pillow."""
    overlay = original_rgb.copy()
    crack_idx = binary_mask > 0

    if np.any(crack_idx):
        # 1. Semi-transparent red highlight over the crack fissure body
        overlay[crack_idx] = (0.45 * original_rgb[crack_idx] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)

        # 2. Crisp boundary contour in vivid red
        from scipy.ndimage import binary_erosion, binary_dilation
        eroded = binary_erosion(crack_idx, structure=np.ones((3, 3), dtype=bool))
        boundary = crack_idx & (~eroded)
        overlay[boundary] = [255, 50, 50]

        # 3. Complete, continuous crack line along the skeleton (electric gold / yellow)
        if skeleton is None:
            from skimage.morphology import skeletonize
            skel_map = skeletonize(crack_idx)
        else:
            skel_map = skeleton > 0

        # Dilate by 1 px to produce a sharp, continuous 2-3 px high-visibility crack line
        skel_thick = binary_dilation(skel_map, structure=np.ones((3, 3), dtype=bool), iterations=1)
        overlay[skel_thick] = [255, 230, 0]

    pil_img = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(pil_img)

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

    box_h = 16 + len(lines) * 20
    draw.rectangle([10, 10, 480, box_h], fill=(13, 17, 23), outline=(56, 189, 248), width=1)
    for i, line in enumerate(lines):
        draw.text((18, 15 + i * 20), line, fill=(240, 246, 252))

    return np.array(pil_img)


def run_pipeline(image_path: str, chosen_algo: str = "canny", reference_cm: float = 10.0, save_outputs: bool = True, run_ai: bool = True, skip_if_normal: bool = True) -> tuple:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at path: {image_path}")

    image_id = os.path.splitext(os.path.basename(image_path))[0]
    try:
        pil_img = Image.open(image_path).convert("RGB")
        image_rgb = np.array(pil_img)
    except Exception as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc

    # Stage 0: AI MobileNetV2 Screening
    ai_res = None
    if run_ai:
        try:
            clf = get_classifier()
            ai_res = clf.predict(image_rgb)
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

    # If AI classifies as NORMAL, bypass expensive classical processing
    if ai_res is not None and not ai_res["is_crack"] and skip_if_normal:
        pre = preprocessing.preprocess_pipeline(image_rgb)
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
            "confidence": 0.0,
            "lighting_detected": lighting,
            "segmentation_method": "N/A (AI screen: Normal)",
            "calibrated": False,
            "timings_sec": {},
        }
        return final_img, measurements

    # From-scratch mathematical preprocessing
    pre = preprocessing.preprocess_pipeline(image_rgb)
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
    cleaned_mask = morphology.clean_mask(binary_mask, gray_image=denoised)

    if save_outputs:
        save("segmentation", image_id, binary_mask)
        save("morphology", image_id, cleaned_mask)

    cm_per_pixel, marker_px = calibration.get_cm_per_pixel(resized, real_world_cm=reference_cm)

    length_result, skeleton = length.measure_length(cleaned_mask, cm_per_pixel)
    skeleton_img = (skeleton.astype(np.uint8)) * 255
    if save_outputs:
        save("skeleton", image_id, skeleton_img)

    area_result = area.measure_area(cleaned_mask, cm_per_pixel=cm_per_pixel)
    width_result = width.measure_width(cleaned_mask, skeleton, cm_per_pixel, damage_pct=area_result["damage_percentage"])
    confidence = crack_confidence_score(cleaned_mask, chosen_map)

    heatmap_img = generate_width_heatmap(resized, cleaned_mask)
    if save_outputs:
        save("heatmap", image_id, heatmap_img)

    crack_found = area_result["crack_area_px"] > 0
    measurements = {
        "ai": ai_res,
        "crack_detected": crack_found,
        "length": length_result,
        "width": width_result,
        "area": area_result,
        "confidence": confidence,
        "lighting_detected": lighting,
        "segmentation_method": seg_method,
        "calibrated": cm_per_pixel is not None,
        "timings_sec": timings,
    }

    final_img = annotate_final(resized, cleaned_mask, measurements, chosen_algo, skeleton=skeleton)
    if save_outputs:
        save("final", image_id, final_img)

    return final_img, measurements


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 60)
        print("  AUTOMATIC WALL CRACK ANALYZER (Zero OpenCV Algorithms)")
        print("=" * 60)
        print("No image path specified. Launching Desktop GUI...")
        print("CLI usage: python main.py <image_path> [algorithm] [--no-ai]")
        print("=" * 60 + "\n")
        import subprocess
        gui_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "app_pyside.py")
        ret = subprocess.call([sys.executable, gui_script])
        sys.exit(ret)

    img_path = sys.argv[1]
    algo = "canny"
    run_ai = True
    for arg in sys.argv[2:]:
        if arg.lower() in ["--no-ai", "--no_ai", "-noai", "noai"]:
            run_ai = False
        elif not arg.startswith("-"):
            algo = arg.lower()

    _, m = run_pipeline(img_path, chosen_algo=algo, run_ai=run_ai)

    if m.get("ai"):
        ai_m = m["ai"]
        print(f"\n--- AI Classification (MobileNetV2) ---")
        print(f"Status            : {ai_m['status']}")
        print(f"Confidence        : {ai_m['confidence']}% (raw score: {ai_m['raw_score']})")
        if not m.get("crack_detected", True):
            print(f"Result            : Sound wall detected. Detailed crack quantification pipeline skipped.")
            sys.exit(0)

    print(f"\n--- From-Scratch Mathematical Pipeline Analysis ---")
    print(f"Lighting detected : {m['lighting_detected']}")
    print(f"Segmentation used : {m['segmentation_method']}")
    print(f"Calibrated        : {m['calibrated']}")
    print(f"Classical Conf    : {m['confidence']}%")
    print(f"Severity Level    : {m['width']['severity_level']}")
    print(f"Orientation       : {m['width']['orientation_type']} ({m['width']['orientation_deg']} deg)")
    print(f"Length            : {m['length']}")
    print(f"Width             : {m['width']}")
    print(f"Area/Damage       : {m['area']}")
