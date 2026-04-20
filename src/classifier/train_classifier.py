"""
Train the transfer-learning classifier on potato leaf images.
Saves best weights to model/classifier_tl_best.weights.h5.

Usage:
    python -m src.classifier.train_classifier
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_DIR, VAL_DIR, MODEL_DIR,
    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, BASE_MODEL_NAME, NUM_CLASSES,
)
from classifier.model import build_classifier, get_preprocess_input

import numpy as np
import tensorflow as tf
from tensorflow import keras

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator


CLASSIFIER_EPOCHS = 20  # Adjust based on convergence


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    preprocess_fn = get_preprocess_input(BASE_MODEL_NAME)
    train_datagen = ImageDataGenerator(
        rescale=None,
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print(f"Train samples: {train_gen.samples}, Val samples: {val_gen.samples}")

    model = build_classifier()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_weights_path = os.path.join(MODEL_DIR, "classifier_tl_best.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]

    print(f"Training for up to {CLASSIFIER_EPOCHS} epochs ...")
    model.fit(train_gen, epochs=CLASSIFIER_EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)

    # If checkpoint captured a better epoch, ensure those weights are used.
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
    print(f"Best classifier weights saved to {best_weights_path}")


if __name__ == "__main__":
    train()
