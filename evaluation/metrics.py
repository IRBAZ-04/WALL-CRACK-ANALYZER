"""
metrics.py
-----------
Compares a predicted crack mask against a hand-annotated ground-truth
mask, pixel by pixel, and computes standard detection metrics.

White (255) = crack, Black (0) = background, in both masks.

TP = predicted crack AND actually crack
TN = predicted background AND actually background
FP = predicted crack BUT actually background   (false alarm)
FN = predicted background BUT actually crack   (missed crack)

Precision = TP / (TP + FP)   -> of predicted cracks, how many were real
Recall    = TP / (TP + FN)   -> of real cracks, how many were found
F1        = 2 * P * R / (P + R)
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
"""

import numpy as np


def compute_confusion(pred_mask, gt_mask):
    pred = (pred_mask > 0)
    gt = (gt_mask > 0)

    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    return tp, tn, fp, fn


def compute_metrics(pred_mask, gt_mask, processing_time=None):
    tp, tn, fp, fn = compute_confusion(pred_mask, gt_mask)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "processing_time_sec": round(processing_time, 5) if processing_time is not None else None,
    }
