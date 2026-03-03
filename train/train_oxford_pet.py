"""
Fine-tune MobileNetV2 on Oxford-IIIT-Pet (37-class breed classification).

Uses the project's ImageDataset class for loading and 80/20 split.

Usage:
    python train/train_oxford_pet.py

Outputs:
    models/oxford_pet.keras          Keras model (gitignored)
    demo/test_data/oxford_pet/       20 test images + labels.json
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.image_dataset import ImageDataset

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = "dataset/Oxford-IIIT-Pet"
LABELS_CSV = "dataset/oxford_labels.csv"
MODEL_OUT = "models/oxford_pet.h5"
TEST_OUT = "demo/test_data/oxford_pet"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
N_TEST_SAMPLES = 20

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading Oxford-IIIT-Pet dataset...")
ds = ImageDataset(ROOT, lazy=True, labels_file=LABELS_CSV)
train_ds, test_ds = ds.split(0.8)
print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

classes = sorted(set(ds.labels))
label_to_idx = {c: i for i, c in enumerate(classes)}
n_classes = len(classes)
print(f"  Classes: {n_classes}")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def load_sample(subset, idx):
    img, label = subset[idx]
    img = np.array(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE)),
                   dtype=np.float32)
    img = preprocess_input(img)
    return img, label_to_idx[label]


def make_tf_dataset(subset, shuffle=False):
    indices = list(range(len(subset)))
    if shuffle:
        np.random.shuffle(indices)

    def gen():
        for i in indices:
            yield load_sample(subset, i)

    return (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ── Build model ───────────────────────────────────────────────────────────────
print("Building model...")
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

model = tf.keras.Model(base.input, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training for {EPOCHS} epochs...")
train_tf = make_tf_dataset(train_ds, shuffle=True)
test_tf = make_tf_dataset(test_ds)
model.fit(train_tf, epochs=EPOCHS, validation_data=test_tf)

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)
SAVED_MODEL_DIR = MODEL_OUT.replace('.h5', '_savedmodel')
model.export(SAVED_MODEL_DIR)
print(f"Exported SavedModel -> {SAVED_MODEL_DIR}")
print(f"Saved model -> {MODEL_OUT}")

# ── Save test samples ─────────────────────────────────────────────────────────
os.makedirs(TEST_OUT, exist_ok=True)
metadata = []
for i in range(min(N_TEST_SAMPLES, len(test_ds))):
    img, label = test_ds[i]
    fname = f"{i:03d}.jpg"
    Image.fromarray(img).resize((400, 400)).save(os.path.join(TEST_OUT, fname))
    metadata.append({"file": fname, "label": label})

with open(os.path.join(TEST_OUT, "labels.json"), "w") as f:
    json.dump({"classes": classes, "samples": metadata}, f, indent=2)

print(f"Saved {N_TEST_SAMPLES} test samples -> {TEST_OUT}")
print("Done!")
