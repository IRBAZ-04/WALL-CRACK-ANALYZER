"""
evaluation/test_ai_integration.py
---------------------------------
Automated test suite verifying the MobileNetV2 AI classifier integration:
  1. Normal wall image -> 'NO CRACK DETECTED'
  2. Cracked wall image -> 'CRACK DETECTED'
  3. Invalid image input -> handled gracefully with clear error
  4. Missing model file -> handled gracefully with fallback
  5. Existing classical pipeline still works identically
  6. Model single-load and singleton reuse
"""

import os
import sys
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import main as pipeline
from ai import CrackClassifier, get_classifier


def test_ai_normal_image():
    print("\n--- Test 1: Normal Wall Image ---")
    img_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "CAMPUS_CLEAN_007.jpg")
    assert os.path.exists(img_path), f"Sample image missing: {img_path}"

    clf = get_classifier()
    res = clf.predict(img_path)
    print(f"Prediction: {res}")

    assert res["status"] == "NO CRACK DETECTED", f"Expected NO CRACK DETECTED, got {res['status']}"
    assert res["label"] == "NORMAL", f"Expected NORMAL, got {res['label']}"
    assert res["is_crack"] is False, "Expected is_crack == False"
    assert res["confidence"] > 50.0, f"Expected confidence > 50%, got {res['confidence']}"

    # Verify pipeline skips classical processing for normal image
    final_img, measurements = pipeline.run_pipeline(img_path, run_ai=True, skip_if_normal=True)
    assert measurements["crack_detected"] is False, "Expected crack_detected == False in measurements"
    assert measurements["ai"]["status"] == "NO CRACK DETECTED"
    assert measurements["length"]["length_px"] == 0
    print("Test 1 Passed: Normal wall image correctly identified as sound; pipeline skipped.")


def test_ai_cracked_image():
    print("\n--- Test 2: Cracked Wall Image ---")
    img_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "CAMPUS_ACADEMIC_001.jpg")
    assert os.path.exists(img_path), f"Sample image missing: {img_path}"

    clf = get_classifier()
    res = clf.predict(img_path)
    print(f"Prediction: {res}")

    assert res["status"] == "CRACK DETECTED", f"Expected CRACK DETECTED, got {res['status']}"
    assert res["label"] == "CRACK", f"Expected CRACK, got {res['label']}"
    assert res["is_crack"] is True, "Expected is_crack == True"
    assert res["confidence"] > 50.0, f"Expected confidence > 50%, got {res['confidence']}"

    # Verify pipeline runs full classical processing for cracked image
    final_img, measurements = pipeline.run_pipeline(img_path, run_ai=True)
    assert measurements["crack_detected"] is True, "Expected crack_detected == True"
    assert measurements["ai"]["status"] == "CRACK DETECTED"
    assert measurements["length"]["length_px"] > 0, "Expected non-zero crack length"
    assert measurements["confidence"] > 0, "Expected classical confidence score"
    print("Test 2 Passed: Cracked wall detected, full classical analysis executed.")


def test_invalid_image_input():
    print("\n--- Test 3: Invalid Image Input ---")
    clf = get_classifier()

    # 3a. Non-existent path
    fake_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "NON_EXISTENT_12345.jpg")
    res_fake = clf.predict(fake_path)
    print(f"Non-existent path result: {res_fake}")
    assert res_fake["error"] is not None, "Expected error for missing file"
    assert res_fake["is_crack"] is True, "Expected fallback is_crack=True on error"

    # 3b. Empty array
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = clf.predict(empty_arr)
    print(f"Empty array result: {res_empty}")
    assert res_empty["error"] is not None, "Expected error for empty array"

    print("Test 3 Passed: Invalid inputs handled gracefully without uncaught exceptions.")


def test_missing_model_file():
    print("\n--- Test 4: Missing Model File (Graceful Fallback) ---")
    bogus_model_path = os.path.join(PROJECT_ROOT, "models", "missing_model.keras")
    clf_missing = CrackClassifier(model_path=bogus_model_path)
    print(f"Missing model status: is_ready={clf_missing.is_ready()}, error={clf_missing.error}")

    assert clf_missing.is_ready() is False, "Expected is_ready == False"
    assert clf_missing.error is not None, "Expected descriptive error"

    sample_img = os.path.join(PROJECT_ROOT, "dataset", "raw", "CAMPUS_ACADEMIC_001.jpg")
    res = clf_missing.predict(sample_img)
    print(f"Missing model prediction: {res}")
    assert res["model_loaded"] is False, "Expected model_loaded == False"
    assert res["status"] == "AI_UNAVAILABLE"
    assert res["is_crack"] is True, "Expected fallback to True so classical pipeline continues"

    print("Test 4 Passed: Missing model fails gracefully and falls back safely.")


def test_classical_pipeline_integrity():
    print("\n--- Test 5: Classical Pipeline Direct Execution (run_ai=False) ---")
    img_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "CAMPUS_ACADEMIC_001.jpg")
    final_img, measurements = pipeline.run_pipeline(img_path, run_ai=False)

    assert measurements.get("ai") is None, "Expected no AI entry when run_ai=False"
    assert measurements["length"]["length_px"] > 0
    assert measurements["width"]["avg_width_px"] > 0
    assert "canny" in measurements["timings_sec"] or "sobel" in measurements["timings_sec"]
    assert "confidence" in measurements
    print("Test 5 Passed: Classical pipeline operates independently and untouched.")


def test_singleton_reuse():
    print("\n--- Test 6: Model Loaded Once & Reused ---")
    c1 = get_classifier()
    c2 = get_classifier()
    assert c1 is c2, "get_classifier() must return identical singleton instance"
    assert c1.model is c2.model, "Underlying model must be the same instance"
    print("Test 6 Passed: Model loaded once and reused across invocations.")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING AUTOMATED AI INTEGRATION TEST SUITE")
    print("==================================================")
    test_ai_normal_image()
    test_ai_cracked_image()
    test_invalid_image_input()
    test_missing_model_file()
    test_classical_pipeline_integrity()
    test_singleton_reuse()
    print("\n==================================================")
    print("ALL 6 AI INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
