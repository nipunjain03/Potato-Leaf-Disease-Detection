"""
Evaluate the transfer-learning classifier on the test set and show accuracy/metrics.
Usage:
    (venv) python -m src.classifier.evaluate_classifier
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_DIR, MODEL_DIR, CLASS_NAMES, IMG_SIZE, BATCH_SIZE, BASE_MODEL_NAME
from classifier.model import get_preprocess_input, build_classifier

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for headless/terminal use
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# Use ImageDataGenerator via keras.preprocessing to avoid direct import of tensorflow.keras.preprocessing
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix heatmap and optionally save to file."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        annot_kws={"size": 12},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix – Potato Leaf Disease Classifier", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def get_test_generator():
    preprocess_fn = get_preprocess_input(BASE_MODEL_NAME)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return test_gen


def main():
    model_path = os.path.join(MODEL_DIR, "classifier_tl_best.weights.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Classifier not found at {model_path}. Train it first with `python -m src.classifier.train_classifier`."
        )

    print(f"Loading model from {model_path} ...")
    # Evaluate from weights-only checkpoint for cross-version stability.
    model = build_classifier()
    model.load_weights(model_path)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    print("Preparing test data ...")
    test_gen = get_test_generator()

    print("Evaluating on test set ...")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Loss:     {loss:.4f}\n")

    print("Generating predictions for detailed metrics ...")
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix (raw):")
    print(cm)

    # Visual plot and save
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_classes, CLASS_NAMES, save_path=cm_path)


if __name__ == "__main__":
    main()

