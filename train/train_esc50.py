"""
Fine-tune MobileNetV2 on ESC-50 (50-class environmental sound classification).

Audio pipeline:
    (y, sr)  ->  Resample(22050)  ->  MelSpectrogram(128 bands)
             ->  power_to_db  ->  normalise 0-255  ->  resize 224x224
             ->  3-channel image  ->  MobileNetV2

Uses the project's AudioDataset class for loading and 80/20 split.

Usage:
    python train/train_esc50.py

Outputs:
    models/esc50.keras               Keras model (gitignored)
    demo/test_data/esc50/            20 mel-spectrogram PNGs + labels.json
"""

import json
import os
import sys

import librosa
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.audio_dataset import AudioDataset

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = "dataset/ESC-50-master/audio"
LABELS_CSV = "dataset/esc50_labels.csv"
MODEL_OUT = "models/esc50.h5"
TEST_OUT = "demo/test_data/esc50"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10          # audio models need more epochs
N_TEST_SAMPLES = 20
TARGET_SR = 22050
N_MELS = 128
DURATION = 5.0       # ESC-50 clips are exactly 5 s

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading ESC-50 dataset...")
ds = AudioDataset(ROOT, lazy=True, labels_file=LABELS_CSV)
train_ds, test_ds = ds.split(0.8)
print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

classes = sorted(set(ds.labels))
label_to_idx = {c: i for i, c in enumerate(classes)}
n_classes = len(classes)
print(f"  Classes: {n_classes}")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def audio_to_image(audio_tuple):
    """Convert (waveform, sr) -> (224, 224, 3) float32 array."""
    y, sr = audio_tuple

    # Resample to target sample rate
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Pad or trim to fixed duration
    n_samples = int(DURATION * TARGET_SR)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]

    # Mel spectrogram -> dB -> normalise to [0, 255]
    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_img = (mel_norm * 255).astype(np.uint8)

    # Resize to IMG_SIZE x IMG_SIZE, stack to 3 channels
    pil = Image.fromarray(mel_img).resize((IMG_SIZE, IMG_SIZE))
    rgb = np.stack([np.array(pil)] * 3, axis=-1).astype(np.float32)
    return preprocess_input(rgb)


def load_sample(subset, idx):
    audio, label = subset[idx]
    img = audio_to_image(audio)
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
x = tf.keras.layers.Dropout(0.4)(x)
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

# ── Save test samples (mel spectrogram PNGs) ──────────────────────────────────
# The demo loads these PNGs directly — no audio processing needed in the browser.
os.makedirs(TEST_OUT, exist_ok=True)
metadata = []
for i in range(min(N_TEST_SAMPLES, len(test_ds))):
    audio, label = test_ds[i]

    # Save mel spectrogram as PNG (what the model sees)
    spec_fname = f"{i:03d}_spec.png"
    y, sr = audio
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    n_samples = int(DURATION * TARGET_SR)
    y = y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_img = (mel_norm * 255).astype(np.uint8)
    Image.fromarray(mel_img).resize((224, 224)).save(
        os.path.join(TEST_OUT, spec_fname)
    )

    metadata.append({"file": spec_fname, "label": label})

with open(os.path.join(TEST_OUT, "labels.json"), "w") as f:
    json.dump({"classes": classes, "samples": metadata}, f, indent=2)

print(f"Saved {N_TEST_SAMPLES} test samples -> {TEST_OUT}")
print("Done!")
