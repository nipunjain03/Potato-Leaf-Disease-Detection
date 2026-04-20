"""
Image Analysis Module: Potato leaf disease classification using transfer learning.
Uses a pretrained CNN (EfficientNet/MobileNet/ResNet) for Early Blight, Late Blight, Healthy.
Output: predicted label + confidence (softmax).
"""

from .model import build_classifier, get_preprocess_input
from .predict import predict_image, load_classifier_model

__all__ = [
    "build_classifier",
    "get_preprocess_input",
    "predict_image",
    "load_classifier_model",
]
