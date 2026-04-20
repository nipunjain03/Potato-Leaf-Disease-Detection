"""
Prediction API for the potato leaf disease classifier.
Returns disease label and confidence (softmax max value).
Uses the same preprocessing as training (resize + model-specific preprocess).
"""

import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, IMG_SIZE, MODEL_DIR, BASE_MODEL_NAME
from classifier.model import get_preprocess_input, build_classifier

# Default classifier weights path
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_tl_best.weights.h5")

_model = None


def load_classifier_model(path: str = None):
    """Load the trained classifier once; reuse for multiple predictions."""
    global _model
    if _model is not None:
        return _model
    path = path or CLASSIFIER_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Classifier not found at {path}. Train the classifier first (train_classifier.py)."
        )
    # Support both full-model files and weights-only checkpoints.
    keras_mod = __import__("tensorflow", fromlist=["keras"]).keras
    if path.lower().endswith(".weights.h5"):
        _model = build_classifier()
        _model.load_weights(path)
    else:
        # Inference-only load: skip optimizer deserialization to avoid version mismatch.
        _model = keras_mod.models.load_model(path, compile=False)
    return _model


def predict_image(image_path: str, model_path: str = None):
    """
    Run classifier on a single image.
    Returns:
        label: str (one of Early_Blight, Healthy, Late_Blight)
        confidence: float (softmax max value, 0–1)
    """
    model = load_classifier_model(model_path)
    preprocess_fn = get_preprocess_input(BASE_MODEL_NAME)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_fn(img)

    pred = model.predict(img, verbose=0)
    class_idx = int(np.argmax(pred[0]))
    confidence = float(np.max(pred[0]))
    return CLASS_NAMES[class_idx], confidence


def predict_image_with_probs(image_path: str, model_path: str = None):
    """
    Run classifier on a single image and return full class probabilities.
    Returns:
        label: str
        confidence: float
        probs: dict[class_name -> probability]
    """
    model = load_classifier_model(model_path)
    preprocess_fn = get_preprocess_input(BASE_MODEL_NAME)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_fn(img)

    pred = model.predict(img, verbose=0)[0]
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    probs = {CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[class_idx], confidence, probs


if __name__ == "__main__":
    # Simple CLI to test a single image from the terminal
    import argparse

    p = argparse.ArgumentParser(description="Predict potato leaf disease from an image using the trained classifier.")
    p.add_argument("image_path", help="Path to the image file (.jpg/.png)")
    p.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Optional model/weights path; defaults to model/classifier_tl_best.weights.h5",
    )
    args = p.parse_args()

    label, conf = predict_image(args.image_path, model_path=args.model_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.4f}")
