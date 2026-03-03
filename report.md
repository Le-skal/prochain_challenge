# OOP for AI — Final Project Report

**Author:** Raphael MARTIN
**Date:** 2024

---

## Workload

Solo project — all implementation, design, and testing carried out by Raphael MARTIN.

---

## Project Structure

```
src/
├── utils.py           Type-check helpers, CSV parser, image loader
├── dataset.py         Dataset, LabeledDataset, UnlabeledDataset (ABCs)
├── image_dataset.py   ImageDataset, UnlabeledImageDataset
├── audio_dataset.py   AudioDataset, UnlabeledAudioDataset
├── batch_loader.py    BatchLoader
└── preprocessing.py   Transform (ABC), image/audio transforms, Pipeline
main.py                Showcase script
```

---

## 1. Dataset Hierarchy

### Class hierarchy

```
Dataset (ABC)
├── LabeledDataset (ABC)
│   ├── ImageDataset
│   └── AudioDataset
└── UnlabeledDataset (ABC)
    ├── UnlabeledImageDataset
    └── UnlabeledAudioDataset
```

### Design decisions

**Abstract base classes via `ABC`**
`Dataset`, `LabeledDataset`, and `UnlabeledDataset` are declared as abstract using Python's `abc.ABC`. This prevents accidental instantiation of incomplete classes and enforces that every concrete subclass implements `_scan_files()` and `_load_file()`. It also makes the hierarchy self-documenting: the type of a dataset object immediately communicates whether it carries labels.

**All attributes are private, exposed via `@property`**
Every instance attribute uses a leading underscore (`_root`, `_lazy`, `_file_paths`, `_data`, `_labels`). Read-only `@property` decorators expose them publicly. No setters are provided: datasets are treated as immutable once constructed, which avoids inconsistent state (e.g. changing `_root` without re-scanning files).

**`_file_paths` is always populated, even in eager mode**
In eager mode all data is pre-loaded into `_data`, but `_file_paths` is still kept. This is necessary for `split()` to work correctly — splitting by index requires knowing how many items exist regardless of loading mode.

**`LabeledDataset.__init__` sets `_labels = []` before calling `super().__init__()`**
`super().__init__()` internally calls `_scan_files()`. Only after that can `_load_labels()` be called safely (it relies on `_file_paths` being populated). Initialising `_labels` to `[]` beforehand prevents `AttributeError` if any code path touches `_labels` during the parent constructor.

**`split()` shuffles indices, not data**
The shuffle is performed on a list of integer indices rather than on the data itself. This avoids duplicating data in memory and works identically for both lazy and eager datasets. `_create_subset()` uses `object.__new__()` to bypass `__init__` (which would re-scan the filesystem) and instead directly copies the relevant slices of `_file_paths`, `_data`, and `_labels`.

**CSV label auto-casting (int → float → str)**
CSV values are always strings. Rather than requiring the user to specify a type, `_load_labels()` attempts to cast each label to `int`, then `float`, falling back to `str`. This transparently supports regression (numeric labels) and classification (string labels) from the same code path.

**Two storage layouts for labeled datasets**
Both `ImageDataset` and `AudioDataset` support:
- *CSV mode* (`labels_file` provided): flat folder of files + external CSV mapping filename → label. Used for regression (UTKFace ages, Ballroom BPM) and flat classification (ESC-50 categories, Oxford breeds).
- *Folder mode* (`labels_file=None`): one subdirectory per class, label = directory name. Used for BallroomData genre classification.

**Pillow over OpenCV for image loading**
Pillow's `.convert("RGB")` guarantees RGB output regardless of source format. OpenCV loads images in BGR by default, which would require an extra conversion step and is a common source of subtle bugs.

**Audio `_load_file` returns `(np.ndarray, int)`**
`librosa.load(path, sr=None)` returns `(waveform, sample_rate)`. Keeping the sample rate alongside the waveform is mandatory for all librosa preprocessing functions (resampling, mel spectrogram, etc.), so returning a tuple is the natural representation.

**Type checks centralised in `src/utils.py`**
`check_type()` and `check_range()` are module-level functions rather than private methods on a base class. They are reused across six unrelated classes (`Dataset`, `BatchLoader`, all `Transform` subclasses). A shared utility module avoids duplication without forcing an artificial inheritance relationship.

---

## 2. BatchLoader

### Design decisions

**`__iter__` as a generator**
Using `yield` makes `__iter__` a generator function. Data is loaded from the dataset only when each batch is consumed by the caller — satisfying the "loaded only when needed" requirement. It also means `BatchLoader` is an *iterable* (not an *iterator*): calling `iter(loader)` multiple times returns fresh generators, so a new shuffle is applied at the start of each epoch.

**Index-only internal logic**
The batch is constructed as `[self._dataset[i] for i in batch_indices]`. The dataset's own `__getitem__` handles the lazy/eager distinction transparently — `BatchLoader` never touches files directly.

**`drop_last` handled inside the generator**
When `drop_last=True`, the generator simply `return`s (stops iteration) as soon as a batch smaller than `batch_size` is encountered. This keeps the logic in one place and makes `__len__` consistent: it uses integer division when `drop_last=True` and `math.ceil` otherwise.

---

## 3. Preprocessing

### Class hierarchy

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
└── Pipeline            (any → any)
```

### Design decisions

**Callable classes via `__call__`**
Every transform stores its hyperparameters at construction time and exposes a single `__call__(data)` method. This makes transforms usable as first-class objects: they can be stored in lists, passed as arguments, and composed in a `Pipeline`.

**`Pipeline` uses variadic `*transforms`**
The variadic constructor allows pipelines of arbitrary length without any boilerplate. Each transform is type-checked at construction time so errors are caught early rather than at call time.

**`MelSpectrogram` changes the data type**
`MelSpectrogram` converts `(y, sr)` → `np.ndarray`. This is intentional and documented: a mel spectrogram is a 2D array and is no longer compatible with audio-specific transforms like `Resample`. The user is responsible for placing `MelSpectrogram` last in any audio pipeline.

**`Padding` uses a pre-filled canvas**
Rather than calling `np.pad` (which only supports scalar fill values per channel), `Padding` creates a full canvas with `np.full(..., self._color)` and pastes the original image into the center. This cleanly handles RGB colour fills without per-channel loops.

**`CenterCrop` and `RandomCrop` only crop oversized dimensions**
If an image is smaller than the target in a given dimension, that dimension is left unchanged. This matches the spec ("if specified height and width are greater than the original image, the crop is not performed") and avoids raising errors on small images.

---

## 4. Testing

46 tests are provided in `tests/test_all.py`, runnable with `pytest`:

```
pytest tests/ -v
# 46 passed in ~10s
```

Tests are organised by component and use temporary directories with synthetic
data (small PNG images and WAV files generated on the fly), so no real dataset
files are required to run them.

| Test class | What is covered |
|---|---|
| `TestUtils` | `check_type`, `check_range`, `parse_labels_csv` — correct output and error raising |
| `TestImageDatasetCSV` | `len`, `__getitem__` shape/type, `split` sizes, no overlap, eager mode, numeric label casting, `IndexError` |
| `TestImageDatasetFolder` | `len`, labels equal folder names |
| `TestUnlabeledImageDataset` | `len`, `__getitem__` returns `np.ndarray` |
| `TestAudioDatasetCSV` | `len`, `__getitem__` types, `split` sizes |
| `TestAudioDatasetFolder` | `len`, labels equal folder names |
| `TestUnlabeledAudioDataset` | `__getitem__` returns `(ndarray, int)` |
| `TestBatchLoader` | `len` with/without `drop_last`, batch sizes, total items, `__len__` consistent with iteration, invalid batch size |
| `TestPreprocessing` | Output shapes for all 8 transforms + Pipeline, edge cases (no-crop when smaller, short audio unchanged) |

**Refactoring notes:**
- No duplicate code: image and audio datasets share the same CSV-parsing and label-casting logic via `src/utils.py`.
- No dead code: every method is exercised either by `main.py` or the test suite.
- No large classes: the largest class (`ImageDataset`) has 4 methods totalling ~60 lines.

---

## 5. Documentation

HTML documentation is generated from docstrings using **Sphinx** with the
`autodoc` and `napoleon` extensions. Every public class, method, and function
has a Google-style docstring covering arguments, return values, and exceptions.

To build the docs locally:

```bash
pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs/ docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

---

## 6. Results

### Datasets

```
[Image] Oxford-IIIT-Pet — labeled, classification, lazy
  Size : 7390
  ds[0]: shape=(400, 600, 3)  label='Abyssinian'
  Split 80/20 -> train=5912  test=1478

[Image] UTKFace — labeled, regression (age), lazy
  Size : 23708
  ds[0]: shape=(200, 200, 3)  age=100

[Image] Oxford-IIIT-Pet — unlabeled, lazy
  Size : 7390
  ds[0]: shape=(400, 600, 3)

[Image] Oxford-IIIT-Pet — EAGER
  Size: 7390  (all loaded into memory at construction)
  ds[5]: shape=(351, 500, 3)  label='Abyssinian'

[Audio] ESC-50 — labeled, classification, lazy
  Size : 2000
  ds[0]: duration=5.0s  sr=44100  category='dog'

[Audio] BallroomData — labeled, folder hierarchy, lazy
  Size : 698
  ds[0]: duration=30.1s  genre='ChaChaCha'
  Split 80/20 -> train=558  test=140

[Audio] BallroomData/Waltz — labeled, regression (BPM), lazy
  Size : 110
  ds[0]: duration=31.8s  BPM=86.72
```

### Sample images (Oxford-IIIT-Pet)

![Sample images](img/results/sample_images.png)

### BatchLoader

```
[Image] Sequential, batch_size=64, drop_last=False
  Number of batches : 116  (ceil(7390/64))
  First batch size  : 64

[Image] Random, batch_size=64, drop_last=True
  Number of batches : 115  (7390//64)
  First batch size  : 64

[Audio] Sequential, batch_size=8, drop_last=False
  Number of batches : 250
  First item        : duration=5.0s  cat='dog'

[Audio] Random, batch_size=8, drop_last=True
  Number of batches : 250
```

### Preprocessing pipelines

**Image pipeline:** `CenterCrop(256×256) → RandomFlip(p=0.5) → Padding(300×300)`

![Image preprocessing](img/results/sample_preprocessing_images.png)

```
sample 0: (400, 600, 3) -> (300, 300, 3)  label='Abyssinian'
sample 1: (901, 600, 3) -> (300, 300, 3)  label='Bengal'
sample 2: (500, 375, 3) -> (300, 300, 3)  label='Birman'
sample 3: (275, 183, 3) -> (300, 300, 3)  label='Bombay'
sample 4: (350, 233, 3) -> (300, 300, 3)  label='British_Shorthair'
```

**Audio pipeline:** `AudioRandomCrop(3s) → Resample(22050 Hz) → MelSpectrogram(128 bands)`

![Audio preprocessing](img/results/sample_preprocessing_audio.png)

```
sample 0: dur=5.0s -> spec(128, 130)  cat='dog'
sample 1: dur=5.0s -> spec(128, 130)  cat='footsteps'
sample 2: dur=5.0s -> spec(128, 130)  cat='frog'
sample 3: dur=5.0s -> spec(128, 130)  cat='thunderstorm'
sample 4: dur=5.0s -> spec(128, 130)  cat='breathing'
```

---

## 7. Usage

```python
from src.image_dataset import ImageDataset, UnlabeledImageDataset
from src.audio_dataset import AudioDataset
from src.batch_loader import BatchLoader
from src.preprocessing import Pipeline, CenterCrop, RandomFlip, Padding

# Labeled image dataset (classification, lazy)
ds = ImageDataset("dataset/Oxford-IIIT-Pet", lazy=True,
                  labels_file="dataset/oxford_labels.csv")
img, label = ds[0]          # (np.ndarray, str)
train, test = ds.split(0.8)

# Labeled image dataset (regression, lazy)
ds_age = ImageDataset("dataset/UTKFace/UTKFace", lazy=True,
                      labels_file="dataset/utk_labels.csv")
img, age = ds_age[0]        # (np.ndarray, int)

# Unlabeled image dataset
ds_u = UnlabeledImageDataset("dataset/Oxford-IIIT-Pet", lazy=True)
img = ds_u[0]               # np.ndarray

# Audio dataset (folder hierarchy, classification)
ds_audio = AudioDataset("dataset/BallroomData", lazy=True)
(y, sr), genre = ds_audio[0]

# BatchLoader
loader = BatchLoader(ds, batch_size=32, shuffle=True, drop_last=False)
print(len(loader))          # number of batches
for batch in loader:
    # batch is a list of (img, label) tuples
    pass

# Preprocessing pipeline
pipeline = Pipeline(
    CenterCrop(256, 256),
    RandomFlip(p=0.5),
    Padding(300, 300, color=(0, 0, 0)),
)
processed = pipeline(img)   # (300, 300, 3)

# Pipeline applied over a BatchLoader
for batch in loader:
    processed_batch = [pipeline(img) for img, label in batch]
```
