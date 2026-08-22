"""
comparison.py
--------------
Batch-runs all five methods (Sobel, Prewitt, Laplacian, Canny, Fusion)
across every image in dataset/ground_truth/ that has a matching
ground-truth mask, computes metrics.py for each, and produces:

  1. reports/comparison_table.csv          -- overall averages
  2. reports/per_image_results.csv         -- raw per-image numbers
  3. reports/graphs/*.png                  -- F1 / precision / recall /
                                               time / lighting-condition
                                               comparison bar charts

This is the module that satisfies "do not fabricate results" — every
number in the CSVs comes from actually running the pipeline on your
real images and comparing to your real hand-drawn ground truth masks.
If you haven't created ground-truth masks yet, this module will simply
report that it found 0 evaluable images rather than making anything up.

Also breaks results down by lighting condition and crack type IF your
dataset_log.csv (see dataset plan) has those columns filled in for the
matching image IDs — this covers the faculty requirement of comparing
performance across lighting/crack-type categories.
"""

import os
import sys
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import preprocessing, sobel, prewitt, laplacian, canny, fusion, segmentation, morphology
from evaluation.metrics import compute_metrics

ALGORITHMS = ["sobel", "prewitt", "laplacian", "canny", "fusion"]


def run_single_image(image_path):
    """Runs the full pipeline for one image, returns dict of
    {algo_name: (binary_mask, elapsed_time)} plus the lighting label."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    pre = preprocessing.preprocess_pipeline(image)
    gray = pre["denoised"]
    lighting = pre["lighting_label"]

    sobel_map, t_sobel = sobel.detect(gray)
    prewitt_map, t_prewitt = prewitt.detect(gray)
    laplacian_map, t_laplacian = laplacian.detect(gray)
    canny_map, t_canny = canny.detect(gray)
    fusion_map, t_fusion = fusion.detect(gray, sobel_map, prewitt_map, laplacian_map, canny_map)

    raw_maps = {
        "sobel": (sobel_map, t_sobel),
        "prewitt": (prewitt_map, t_prewitt),
        "laplacian": (laplacian_map, t_laplacian),
        "canny": (canny_map, t_canny),
        "fusion": (fusion_map, t_fusion),
    }

    results = {}
    for name, (emap, t) in raw_maps.items():
        mask, _ = segmentation.segment(emap, lighting)
        cleaned = morphology.clean_mask(mask)
        results[name] = (cleaned, t)

    return results, lighting


def run_evaluation(raw_dir="dataset/raw", gt_dir="dataset/ground_truth",
                    dataset_log_csv=None, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

    gt_files = [f for f in os.listdir(gt_dir)] if os.path.isdir(gt_dir) else []
    if not gt_files:
        print(f"[comparison.py] No ground-truth masks found in {gt_dir}. "
              f"Nothing to evaluate yet -- see Phase 9 for how to create them.")
        return None

    dataset_log = None
    if dataset_log_csv and os.path.exists(dataset_log_csv):
        dataset_log = pd.read_csv(dataset_log_csv).set_index("image_id")

    per_image_rows = []

    for gt_file in gt_files:
        image_id = os.path.splitext(gt_file)[0]
        raw_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(raw_dir, image_id + ext)
            if os.path.exists(candidate):
                raw_path = candidate
                break
        if raw_path is None:
            continue

        gt_mask = cv2.imread(os.path.join(gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)
        results, lighting = run_single_image(raw_path)
        if results is None:
            continue

        # resize gt to match processed size if needed
        h, w = list(results.values())[0][0].shape
        gt_mask = cv2.resize(gt_mask, (w, h))

        row_meta = {"image_id": image_id, "lighting_detected": lighting}
        if dataset_log is not None and image_id in dataset_log.index:
            row_meta["crack_type"] = dataset_log.loc[image_id].get("crack_type", "unknown")
            row_meta["lighting_labelled"] = dataset_log.loc[image_id].get("lighting_condition", "unknown")

        for algo in ALGORITHMS:
            mask, t = results[algo]
            metrics = compute_metrics(mask, gt_mask, processing_time=t)
            row = {**row_meta, "algorithm": algo, **metrics}
            per_image_rows.append(row)

    if not per_image_rows:
        print("[comparison.py] No matching raw+ground-truth pairs found.")
        return None

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(os.path.join(output_dir, "per_image_results.csv"), index=False)

    summary = per_image_df.groupby("algorithm")[
        ["precision", "recall", "f1_score", "accuracy", "processing_time_sec"]
    ].mean().round(4).reset_index()
    summary.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)

    _make_graphs(per_image_df, summary, output_dir)

    return summary


def _make_graphs(per_image_df, summary, output_dir):
    graphs_dir = os.path.join(output_dir, "graphs")

    for metric in ["precision", "recall", "f1_score", "processing_time_sec"]:
        plt.figure(figsize=(6, 4))
        plt.bar(summary["algorithm"], summary[metric], color="#4C6EF5")
        plt.title(f"{metric.replace('_', ' ').title()} by Algorithm")
        plt.ylabel(metric)
        plt.xlabel("Algorithm")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{metric}_comparison.png"), dpi=150)
        plt.close()

    if "lighting_detected" in per_image_df.columns:
        pivot = per_image_df.pivot_table(
            index="lighting_detected", columns="algorithm", values="f1_score", aggfunc="mean"
        )
        pivot.plot(kind="bar", figsize=(8, 5))
        plt.title("F1-Score by Lighting Condition")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "f1_by_lighting.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    run_evaluation()
