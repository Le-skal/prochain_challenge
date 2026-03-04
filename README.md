<div align="center">

# DataKit
### *Python Library for ML Dataset Loading, Batching & Preprocessing*

<p><em>Clean OOP design for image and audio ML pipelines — with live in-browser demo</em></p>

![Status](https://img.shields.io/badge/status-complete-success?style=flat)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-46%20passing-success?style=flat&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat)

<p><em>Built with the tools and technologies:</em></p>

![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat&logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-10.x-blue?style=flat)
![Librosa](https://img.shields.io/badge/Librosa-0.10-green?style=flat)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Sphinx](https://img.shields.io/badge/Sphinx-docs-0a507a?style=flat&logo=sphinx&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-7.x-0A9EDC?style=flat&logo=pytest&logoColor=white)

**Datasets Used:**
![Oxford Pet](https://img.shields.io/badge/Oxford--IIIT--Pet-37%20breeds-orange?style=flat)
![UTKFace](https://img.shields.io/badge/UTKFace-age%20regression-blue?style=flat)
![ESC-50](https://img.shields.io/badge/ESC--50-50%20sounds-green?style=flat)
![BallroomData](https://img.shields.io/badge/BallroomData-10%20genres-purple?style=flat)

---

### 📊 OOP for AI — Final Project 2025-2026
**Raphael MARTIN** | ECE Paris — B3 Data & IA

**[🌐 Live Demo](https://datakit.deepskal.com) · [📚 Documentation](https://le-skal.github.io/datakit/)**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Stats](#key-stats)
- [Library Structure](#library-structure)
  - [Dataset Hierarchy](#dataset-hierarchy)
  - [BatchLoader](#batchloader)
  - [Preprocessing Transforms](#preprocessing-transforms)
- [Design Decisions](#design-decisions)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Datasets](#datasets-1)
  - [BatchLoader](#batchloader-1)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
- [Results](#results)
- [Testing](#testing)
- [Documentation](#documentation)
- [Interactive Demo](#interactive-demo)
- [Project Structure](#project-structure)

---

## Overview

DataKit is a Python library for loading, batching, and preprocessing image and audio datasets, built from scratch with clean object-oriented design as a final project for the *OOP for AI* course.

The library covers:
- A **dataset hierarchy** with abstract base classes, supporting labeled/unlabeled datasets, lazy/eager loading, CSV and folder-hierarchy label formats, and regression/classification tasks
- A **BatchLoader** that wraps any dataset and yields batches via a generator, with optional shuffling and last-batch control
- Eight **preprocessing transforms** (four image, four audio) implemented as callable classes, composable via a `Pipeline`
- Four **MobileNetV2 models** fine-tuned using the DataKit pipeline, exportable to TensorFlow.js for in-browser inference

---

## Key Stats

| | |
|---|---|
| Images | 9,390 (Oxford-IIIT-Pet + UTKFace) |
| Audio clips | 2,698 (ESC-50 + BallroomData) |
| Tests passing | 46 |
| Oxford-IIIT-Pet val accuracy | ~90% |
| UTKFace age estimation MAE | ~6 years |

---

## Library Structure

### Dataset Hierarchy

```
Dataset (ABC)
├── LabeledDataset (ABC)
│   ├── ImageDataset
│   └── AudioDataset
└── UnlabeledDataset (ABC)
    ├── UnlabeledImageDataset
    └── UnlabeledAudioDataset
```

`Dataset`, `LabeledDataset`, and `UnlabeledDataset` are abstract base classes that enforce implementation of `_scan_files()` and `_load_file()` in every concrete subclass. All attributes are private and exposed via `@property`.

Both `ImageDataset` and `AudioDataset` support two storage layouts:

- **CSV mode** (`labels_file` provided): flat folder of files + external CSV mapping `filename → label`. Used for regression (UTKFace ages, Ballroom BPM) and flat classification (ESC-50, Oxford breeds).
- **Folder mode** (`labels_file=None`): one subdirectory per class, label = directory name. Used for BallroomData genre classification.

### BatchLoader

`BatchLoader` wraps any `Dataset` and yields batches via a generator (`yield`), loading data from the dataset only when each batch is consumed. Shuffling is applied to indices only — no data is duplicated. The `drop_last` flag discards the final incomplete batch when needed.

### Preprocessing Transforms

```
Transform (ABC)
├── CenterCrop          (image)
├── RandomCrop          (image)
├── RandomFlip          (image)
├── Padding             (image)
├── MelSpectrogram      (audio: (y,sr) → np.ndarray)
├── AudioRandomCrop     (audio: (y,sr) → (y,sr))
├── Resample            (audio: (y,sr) → (y,sr))
├── PitchShift          (audio: (y,sr) → (y,sr))
└── Pipeline            (any → any, chains transforms sequentially)
```

All transforms are callable classes: hyperparameters are set at construction time and `__call__` receives only the data. `Pipeline` uses variadic `*transforms` and applies them left to right. Note that `MelSpectrogram` changes the data type from `(y, sr)` to a 2D `np.ndarray` — it should always be placed last in an audio pipeline.

---

## Design Decisions

**ABCs prevent incomplete instantiation.** Using `abc.ABC` ensures every concrete class implements the required interface, and makes the hierarchy self-documenting.

**All attributes are private, exposed via `@property`.** No setters are provided — datasets are treated as immutable once constructed.

**`split()` shuffles indices, not data.** `_create_subset()` uses `object.__new__()` to bypass `__init__` and directly copies the relevant slices, avoiding filesystem re-scans and data duplication.

**CSV label auto-casting (int → float → str).** Labels are automatically cast to the most specific numeric type, transparently supporting both regression and classification from the same code path.

**Pillow over OpenCV.** Pillow's `.convert("RGB")` guarantees RGB output regardless of source format, avoiding the BGR-vs-RGB pitfall.

**Audio `_load_file` returns `(np.ndarray, int)`.** Keeping the sample rate alongside the waveform is mandatory for all librosa preprocessing functions.

**Type checks centralised in `src/utils.py`.** Shared utility functions avoid duplication without forcing an artificial inheritance relationship across unrelated classes.

---

## Datasets

| Dataset | Task | Size | Labels |
|---|---|---|---|
| Oxford-IIIT-Pet | Classification | 7,390 images | 37 breeds (CSV) |
| UTKFace | Regression | 23,708 images | Age in years (CSV) |
| ESC-50 | Classification | 2,000 audio clips | 50 sound categories (CSV) |
| BallroomData | Classification | 698 audio clips | 10 dance genres (folder hierarchy) |
| BallroomData/Waltz | Regression | 110 audio clips | BPM (CSV) |

---

## Installation

```bash
git clone https://github.com/le-skal/datakit
cd datakit
pip install -r requirements.txt
```

To build the HTML documentation locally:

```bash
pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs/ docs/_build/html
```

---

## Usage

### Datasets

```python
from src.image_dataset import ImageDataset, UnlabeledImageDataset
from src.audio_dataset import AudioDataset, UnlabeledAudioDataset

# Labeled image dataset — classification, lazy
ds = ImageDataset("dataset/Oxford-IIIT-Pet", lazy=True,
                  labels_file="dataset/oxford_labels.csv")
img, label = ds[0]          # (np.ndarray shape (H,W,3), str)
train, test = ds.split(0.8)

# Labeled image dataset — regression, lazy
ds_age = ImageDataset("dataset/UTKFace/UTKFace", lazy=True,
                      labels_file="dataset/utk_labels.csv")
img, age = ds_age[0]        # (np.ndarray, int)

# Unlabeled image dataset
ds_u = UnlabeledImageDataset("dataset/Oxford-IIIT-Pet", lazy=True)
img = ds_u[0]               # np.ndarray

# Audio dataset — folder hierarchy, classification
ds_audio = AudioDataset("dataset/BallroomData", lazy=True)
(y, sr), genre = ds_audio[0]

# Eager loading (all data loaded into RAM at construction)
ds_eager = ImageDataset("dataset/Oxford-IIIT-Pet", lazy=False,
                        labels_file="dataset/oxford_labels.csv")
```

### BatchLoader

```python
from src.batch_loader import BatchLoader

loader = BatchLoader(ds, batch_size=32, shuffle=True, drop_last=False)
print(len(loader))   # number of batches

for batch in loader:
    # batch is a list of (img, label) tuples
    pass
```

### Preprocessing Pipeline

```python
from src.preprocessing import (
    Pipeline, CenterCrop, RandomFlip, Padding,
    AudioRandomCrop, Resample, MelSpectrogram
)

# Image pipeline
img_pipeline = Pipeline(
    CenterCrop(256, 256),
    RandomFlip(p=0.5),
    Padding(300, 300, color=(128, 128, 128)),
)
processed = img_pipeline(img)   # shape (300, 300, 3)

# Audio pipeline
audio_pipeline = Pipeline(
    AudioRandomCrop(duration=5.0),
    Resample(target_sr=22050),
    MelSpectrogram(n_mels=128),
)
spectrogram = audio_pipeline((y, sr))   # shape (128, T)

# Pipeline applied over a BatchLoader
for batch in loader:
    processed_batch = [img_pipeline(img) for img, label in batch]
```

---

## Results

### Datasets

```
Oxford-IIIT-Pet — 7,390 images, 37 classes, split 80/20 → train=5912 / test=1478
UTKFace         — 23,708 images, age regression
ESC-50          — 2,000 audio clips, 50 classes
BallroomData    — 698 audio clips, 10 genres, split 80/20 → train=558 / test=140
```

### Sample images (Oxford-IIIT-Pet)

![Sample images](img/results/sample_images.png)

### Image preprocessing pipeline

`CenterCrop(256×256) → RandomFlip(p=0.5) → Padding(300×300)`

![Image preprocessing](img/results/sample_preprocessing_images.png)

### Audio preprocessing pipeline

`AudioRandomCrop(3s) → Resample(22050 Hz) → MelSpectrogram(128 bands)`

![Audio preprocessing](img/results/sample_preprocessing_audio.png)

### Fine-tuned models

| Model | Task | Dataset | Val Performance |
|---|---|---|---|
| MobileNetV2 | Breed classification | Oxford-IIIT-Pet | ~90% accuracy |
| MobileNetV2 | Age estimation | UTKFace | MAE ~6 years |
| MobileNetV2 | Sound classification | ESC-50 | ~65% accuracy |
| MobileNetV2 | Genre classification | BallroomData | ~70% accuracy |

---

## Testing

46 tests in `tests/test_all.py`, runnable with:

```bash
pytest tests/ -v
```

No real dataset files are required — all tests use temporary directories with synthetic data generated on the fly.

| Test class | Coverage |
|---|---|
| `TestUtils` | `check_type`, `check_range`, `parse_labels_csv` |
| `TestImageDatasetCSV` | `len`, `__getitem__`, `split`, eager mode, numeric labels, `IndexError` |
| `TestImageDatasetFolder` | `len`, folder-name labels |
| `TestUnlabeledImageDataset` | `len`, `__getitem__` return type |
| `TestAudioDatasetCSV` | `len`, `__getitem__` types, `split` |
| `TestAudioDatasetFolder` | `len`, folder-name labels |
| `TestUnlabeledAudioDataset` | `__getitem__` return type |
| `TestBatchLoader` | `len` with/without `drop_last`, batch sizes, total items, invalid batch size |
| `TestPreprocessing` | Output shapes for all 8 transforms + Pipeline, edge cases |

---

## Documentation

HTML documentation generated from Google-style docstrings using Sphinx + autodoc + napoleon.

**Hosted at: https://le-skal.github.io/datakit/**

---

## Interactive Demo

A live demo is hosted at **https://datakit.deepskal.com** — no backend or server-side computation.

**Preprocessing pipeline:** `CenterCrop`, `RandomFlip`, and `Padding` are reimplemented in JavaScript via the Canvas API. Toggle each transform on/off, adjust parameters in real time, and see before/after output with pixel dimensions.

**In-browser ML inference:** All four MobileNetV2 models run client-side via TensorFlow.js (converted from SavedModel format). Click any held-out test image to run inference and compare the prediction against the ground truth.

---

## Project Structure

```
datakit/
├── src/
│   ├── utils.py              Type-check helpers, CSV parser, image loader
│   ├── dataset.py            Dataset, LabeledDataset, UnlabeledDataset (ABCs)
│   ├── image_dataset.py      ImageDataset, UnlabeledImageDataset
│   ├── audio_dataset.py      AudioDataset, UnlabeledAudioDataset
│   ├── batch_loader.py       BatchLoader
│   └── preprocessing.py      Transform (ABC), image/audio transforms, Pipeline
├── train/
│   ├── train_oxford_pet.py
│   ├── train_utkface.py
│   ├── train_esc50.py
│   ├── train_ballroom.py
│   └── convert_to_tfjs.sh
├── demo/
│   └── index.html            Live demo (preprocessing + in-browser inference)
├── docs/                     Sphinx documentation source
├── tests/
│   └── test_all.py           46 pytest tests
├── main.py                   Showcase script
├── requirements.txt
└── report.md                 Implementation report
```