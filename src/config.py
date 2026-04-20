"""
Central configuration for the Potato Leaf Disease Detection and Advisory System.
All paths, class labels, and hyperparameters are defined here for consistency.
"""

import os

# ---------------------------------------------------------------------------
# Project paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # RAG documents, KG, etc.

# Dataset splits
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# ---------------------------------------------------------------------------
# Classification (Image Analysis Module)
# ---------------------------------------------------------------------------
# Class labels must match folder names in dataset (Early_Blight, Healthy, Late_Blight)
CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 224
BATCH_SIZE = 64  # GPU can handle larger batches; faster classifier training
# Transfer learning: one of "EfficientNetB0", "MobileNetV2", "ResNet50"
BASE_MODEL_NAME = "EfficientNetB0"

# RAG
# ---------------------------------------------------------------------------
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
KNOWLEDGE_GRAPH_PATH = os.path.join(DATA_DIR, "knowledge_graph.json")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"  # or "mistral", "phi", etc. — user can change

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
for _dir in (MODEL_DIR, DATA_DIR, DOCUMENTS_DIR, VECTOR_STORE_PATH):
    os.makedirs(_dir, exist_ok=True)
