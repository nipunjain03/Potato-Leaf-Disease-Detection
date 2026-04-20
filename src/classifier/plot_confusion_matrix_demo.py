"""
Generate a demo confusion matrix image (no trained model required).
Useful to preview the visualization or when test data is unavailable.

Usage:
    python src/classifier/plot_confusion_matrix_demo.py
"""

import os
import sys

# Resolve paths without importing full package (avoids TensorFlow)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
MODEL_DIR = os.path.join(_project_root, "model")
CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Demo confusion matrix (plausible counts for Early_Blight, Healthy, Late_Blight)
DEMO_CM = np.array([
    [45,  3,  2],   # Early_Blight: 45 correct, 3 → Healthy, 2 → Late_Blight
    [ 2, 52,  1],   # Healthy
    [ 1,  2, 48],   # Late_Blight
])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "confusion_matrix.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        DEMO_CM,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        annot_kws={"size": 12},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix – Potato Leaf Disease Classifier\n(demo – run evaluate_classifier for real results)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Demo confusion matrix saved to: {os.path.abspath(save_path)}")
    print("Open this file to view the image.")


if __name__ == "__main__":
    main()
