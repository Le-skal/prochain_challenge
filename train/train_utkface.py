"""
Fine-tune MobileNetV2 on UTKFace (age regression).

Uses the project's ImageDataset class for loading and 80/20 split.
Output is a single float: predicted age in years.

Usage:
    python train/train_utkface.py

Outputs:
    models/utkface.keras             Keras model (gitignored)
    demo/test_data/utkface/          20 test images + labels.json
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
ROOT = "dataset/UTKFace/UTKFace"
LABELS_CSV = "dataset/utk_labels.csv"
MODEL_OUT = "models/utkface.h5"
TEST_OUT = "demo/test_data/utkface"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
N_TEST_SAMPLES = 20
AGE_MAX = 116.0      # normalise labels to [0, 1] during training

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading UTKFace dataset...")
ds = ImageDataset(ROOT, lazy=True, labels_file=LABELS_CSV)
train_ds, test_ds = ds.split(0.8)
print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def load_sample(subset, idx):
    img, age = subset[idx]
    img = np.array(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE)),
                   dtype=np.float32)
    img = preprocess_input(img)
    return img, float(age) / AGE_MAX     # normalise to [0, 1]


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
                tf.TensorSpec(shape=(), dtype=tf.float32),
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
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)   # sigmoid -> [0,1]

model = tf.keras.Model(base.input, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mae"],
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
# Store raw age (not normalised) in the JSON so the demo can display it.
os.makedirs(TEST_OUT, exist_ok=True)
metadata = []
for i in range(min(N_TEST_SAMPLES, len(test_ds))):
    img, age = test_ds[i]
    fname = f"{i:03d}.jpg"
    Image.fromarray(img).resize((400, 400)).save(os.path.join(TEST_OUT, fname))
    metadata.append({"file": fname, "age": int(age)})

with open(os.path.join(TEST_OUT, "labels.json"), "w") as f:
    json.dump({"age_max": AGE_MAX, "samples": metadata}, f, indent=2)

print(f"Saved {N_TEST_SAMPLES} test samples -> {TEST_OUT}")
print("Done!")
