"""
ai/crack_classifier.py
-----------------------
MobileNetV2-based binary classifier for Wall Crack vs Normal screening.

Requirements:
  - Input: 224 x 224 x 3 (RGB)
  - Preprocessing: tf.keras.applications.mobilenet_v2.preprocess_input (no division by 255)
  - Threshold: 0.5 default
  - Single-load reuse (Singleton pattern)
  - Project-relative model path: models/wall_crack_mobilenetv2.keras
  - Robust error handling for missing models and invalid inputs
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Union, Dict, Any

# Suppress verbose TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Project-relative default model path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wall_crack_mobilenetv2.keras")


class CrackClassifier:
    _instance: Optional["CrackClassifier"] = None
    _model = None
    _model_path: Optional[str] = None
    _preprocess_fn = None

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        """
        Initializes the classifier.
        Note: Use get_classifier() for singleton reuse to load once.
        """
        self.model_path = os.path.abspath(model_path or DEFAULT_MODEL_PATH)
        self.threshold = float(threshold)
        self.target_size = (224, 224)
        self.error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the pre-trained Keras model once."""
        if CrackClassifier._model is not None and CrackClassifier._model_path == self.model_path:
            self.model = CrackClassifier._model
            self.error = None
            return

        if not os.path.exists(self.model_path):
            self.model = None
            self.error = f"Model file not found at: {self.model_path}"
            return

        try:
            # Try importing keras first (Keras 3 / TensorFlow 2.16+ default format)
            try:
                import keras
                self.model = keras.models.load_model(self.model_path, compile=False)
            except (Exception, OSError, BaseException):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path, compile=False)

            CrackClassifier._model = self.model
            CrackClassifier._model_path = self.model_path
            self.error = None
        except (Exception, OSError, BaseException) as exc:
            self.model = None
            self.error = f"Failed to load AI model: {str(exc)}"

    def is_ready(self) -> bool:
        """Returns True if the model is loaded and ready for inference."""
        return self.model is not None and self.error is None

    def preprocess_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Prepares the image for MobileNetV2 inference:
          1. Reads image if path provided.
          2. Ensures RGB color ordering.
          3. Resizes to 224 x 224 using training coordinate scaling.
          4. Scales pixels to [-1, 1] using standard MobileNetV2 formulation (x / 127.5 - 1.0).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            try:
                pil_img = Image.open(image_input).convert("RGB")
            except Exception as e:
                raise ValueError(f"Could not decode image file: {image_input}") from e
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty.")
            # Assume RGB uint8 array
            pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        raw_array = np.array(pil_img, dtype=np.float32)

        # Batch dimension: (1, H, W, 3)
        batched = np.expand_dims(raw_array, axis=0)

        # Resize to (224, 224) using exact TensorFlow coordinate logic if available, else PIL
        try:
            import tensorflow as tf
            resized = tf.image.resize(batched, self.target_size).numpy()
        except (Exception, OSError, BaseException):
            resized_pil = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)
            resized = np.expand_dims(np.array(resized_pil, dtype=np.float32), axis=0)

        # Standard MobileNetV2 mathematical scaling: map [0, 255] to [-1.0, 1.0]
        preprocessed = (resized.astype(np.float32) / 127.5) - 1.0
        return preprocessed


    def predict(self, image_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Runs crack-vs-normal classification.
        Returns:
            dict with:
                - status: 'CRACK DETECTED' | 'NO CRACK DETECTED'
                - label: 'CRACK' | 'NORMAL'
                - is_crack: bool
                - confidence: float (percentage 0.0 - 100.0%)
                - raw_score: float (0.0 to 1.0)
                - model_loaded: bool
                - error: Optional[str]
        """
        if not self.is_ready():
            return {
                "status": "AI_UNAVAILABLE",
                "label": "UNKNOWN",
                "is_crack": True,  # Fallback to classical pipeline on missing model
                "confidence": 0.0,
                "raw_score": 0.0,
                "model_loaded": False,
                "error": self.error or "Model is not loaded."
            }

        try:
            preprocessed = self.preprocess_image(image_input)
            raw_pred = self.model.predict(preprocessed, verbose=0)
            score = float(np.squeeze(raw_pred))

            # Threshold evaluation: score >= threshold -> CRACK
            is_crack = score >= self.threshold
            label = "CRACK" if is_crack else "NORMAL"
            status = "CRACK DETECTED" if is_crack else "NO CRACK DETECTED"

            # Compute AI confidence percentage for the predicted class
            if is_crack:
                conf = score * 100.0
            else:
                conf = (1.0 - score) * 100.0

            return {
                "status": status,
                "label": label,
                "is_crack": is_crack,
                "confidence": round(conf, 2),
                "raw_score": round(score, 4),
                "model_loaded": True,
                "error": None
            }
        except Exception as exc:
            return {
                "status": "ERROR",
                "label": "ERROR",
                "is_crack": True,  # Fallback to classical pipeline on inference failure
                "confidence": 0.0,
                "raw_score": 0.0,
                "model_loaded": True,
                "error": str(exc)
            }


_GLOBAL_CLASSIFIER: Optional[CrackClassifier] = None


def get_classifier(model_path: Optional[str] = None, threshold: float = 0.5) -> CrackClassifier:
    """
    Returns the singleton instance of CrackClassifier to ensure the model
    is loaded once and reused across invocations.
    """
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        _GLOBAL_CLASSIFIER = CrackClassifier(model_path=model_path, threshold=threshold)
    return _GLOBAL_CLASSIFIER
