"""
Transfer-learning classifier for potato leaf disease.
Architecture: Pretrained CNN (EfficientNetB0 by default) + global pooling + dense head.
Design: Frozen base for stability on limited data; optional fine-tuning.
"""

import tensorflow as tf
from tensorflow import keras

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, IMG_SIZE, BASE_MODEL_NAME


def get_base_model(name: str, input_shape: tuple):
    """Return the pretrained base model (no top)."""
    if name == "EfficientNetB0":
        base = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif name == "MobileNetV2":
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif name == "ResNet50":
        base = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {name}")
    return base


def get_preprocess_input(name: str):
    """Return the correct preprocessing function for the base model."""
    if name == "EfficientNetB0":
        return keras.applications.efficientnet.preprocess_input
    if name == "MobileNetV2":
        return keras.applications.mobilenet_v2.preprocess_input
    if name == "ResNet50":
        return keras.applications.resnet50.preprocess_input
    return None


def build_classifier(
    base_name: str = BASE_MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    freeze_base: bool = True,
):
    """
    Build classifier: pretrained base + GlobalAveragePooling2D + Dense(softmax).
    Inputs are expected to be preprocessed with the corresponding keras.applications
    `preprocess_input` function **outside** the model (in the data pipeline or predict step).
    freeze_base=True keeps pretrained weights fixed for better generalization on small data.
    """
    input_shape = (img_size, img_size, 3)
    base = get_base_model(base_name, input_shape)
    if freeze_base:
        base.trainable = False

    inputs = keras.Input(shape=input_shape)
    # Assume data is already preprocessed (e.g. EfficientNetB0's preprocess_input)
    x = base(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model
