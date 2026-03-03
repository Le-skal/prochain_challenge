# Claude Checkpoint — OOP for AI (cours)

## Project
Root: `C:/Users/rmartin/Desktop/code agefi/cours/`
Goal: OOP final project — dataset hierarchy, BatchLoader, preprocessing pipeline.
GitHub: https://github.com/le-skal/datakit
Sphinx docs: https://le-skal.github.io/datakit/
Vercel demo: **live** at the Vercel URL (connect repo → auto-deploys on push to main)
Local preview: `cd demo && python -m http.server 8000`

---

## Dataset Download Links

All datasets go in `dataset/` (gitignored). Download, extract, done.

| # | Dataset | Download | Size | Notes |
|---|---|---|---|---|
| 1 | Oxford-IIIT-Pet | https://www.robots.ox.ac.uk/~vgg/data/pets/ → `images.tar.gz` | ~800 MB | Extract to `dataset/Oxford-IIIT-Pet/` |
| 2 | UTKFace | https://www.kaggle.com/datasets/jangedoo/utkface-new | ~3 GB | Extract to `dataset/UTKFace/UTKFace/` |
| 3 | ESC-50 | https://github.com/karolpiczak/ESC-50/archive/master.zip | ~600 MB | Extract to `dataset/ESC-50-master/` |
| 4 | BallroomData | http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz | ~1.2 GB | Extract to `dataset/BallroomData/` |
| 5 | Ballroom Annotations | http://mtg.upf.edu/ismir2004/contest/tempoContest/data2.tar.gz | ~5 MB | Extract to `dataset/BallroomAnnotations/` (needed for BPM labels) |

**After downloading:** run `python main.py` once to generate label CSVs
(`dataset/oxford_labels.csv`, `dataset/utk_labels.csv`, `dataset/esc50_labels.csv`).

---

## Implementation Status

### Core library ✅ (complete)
- `src/utils.py`, `src/dataset.py`, `src/image_dataset.py`, `src/audio_dataset.py`
- `src/batch_loader.py`, `src/preprocessing.py`
- `main.py`, `tests/test_all.py` (46 tests passing), `report.md`
- Sphinx docs on GitHub Pages

### Vercel demo ✅ (deployed + live)
- CSS redesign done — professional dark theme, gradient title, Inter font
- Oxford Pet classifier: **live** (20 test images, top-3 predictions + ground truth)
- UTKFace / ESC-50 / BallroomData: "Coming soon" placeholders in the UI
- JS uses `tf.loadGraphModel()` + `model.execute()` (NOT executeAsync, NOT loadLayersModel)

### Model training

Install deps on training PC: `pip install tensorflow librosa soundfile pillow numpy`
(no tensorflowjs needed on Windows — conversion happens in Colab)

| Model | Script | Status | Output |
|---|---|---|---|
| Oxford-IIIT-Pet (37 breeds) | `python train/train_oxford_pet.py` | ✅ Done | 90% val acc, 8 epochs |
| UTKFace (age regression) | `python train/train_utkface.py` | ✅ Done | MAE ~6 yrs, fine-tuned 10 epochs |
| ESC-50 (50 sounds) | `python train/train_esc50.py` | ✅ Done | audio → mel spec → MobileNetV2 |
| BallroomData (8 genres) | `python train/train_ballroom.py` | ⬜ TODO | audio → mel spec → MobileNetV2 |

Each training script automatically:
1. Trains MobileNetV2 (frozen base + custom head) on 80% of data
2. Saves `models/<name>.h5` (gitignored — too large)
3. Exports `models/<name>_savedmodel/` (gitignored — needed for Colab conversion)
4. Saves 20 test samples to `demo/test_data/<name>/` + `labels.json` (committed ✅)

---

## TF.js Conversion (Colab — do this after each model is trained)

**CRITICAL:** Do NOT use `--input_format=keras` or the CLI converter — both are broken with Keras 3.
**Use SavedModel + Python API instead.**

### Step 1 — Upload to Colab
Upload `models/<name>.h5` (or `models/<name>_savedmodel/` folder zipped) to Google Colab.

### Step 2 — Convert (3 Colab cells)

```python
# Cell 1 — re-export to SavedModel (skip if you already have _savedmodel folder)
import tensorflow as tf
model = tf.keras.models.load_model('mymodel.h5')
model.export('mymodel_savedmodel')
```

```python
# Cell 2 — convert to TF.js
import tensorflowjs as tfjs
tfjs.converters.convert_tf_saved_model('mymodel_savedmodel', 'mymodel_tfjs')
# Use Python API only — CLI has a bug ("Missing output_path" even with correct args)
```

```python
# Cell 3 — download as zip
import shutil
shutil.make_archive('mymodel_tfjs', 'zip', 'mymodel_tfjs')
# Then Files panel → download mymodel_tfjs.zip
```

### Step 3 — Add to repo
Extract zip → copy contents into `demo/models/<modelname>/`
Commit + push → Vercel auto-deploys.

---

## Adding a New Model to the Frontend

When a model is ready, replace its "Coming soon" card in `demo/index.html`.
Follow the Oxford Pet pattern exactly:

1. Add a `<div id="<name>-grid" class="test-grid">` and `<div id="<name>-predict" class="predict-area">` inside the card
2. Add badge `<span class="badge badge-live"><span class="dot"></span>Live</span>`
3. Write `init<Name>()`, `build<Name>Grid()`, `run<Name>(idx)` functions in JS
4. Call `init<Name>()` at the bottom of the script

**UTKFace specifics (regression, not classification):**
- Model output is a single sigmoid value → multiply by `AGE_MAX=116` to get predicted age
- Display as "Predicted age: X yrs" instead of top-3 bars
- Labels in `labels.json` are integer ages

**ESC-50 / BallroomData specifics (audio → mel spec → image):**
- Test samples are already saved as PNG spectrograms (not audio files)
- Load and run exactly like Oxford Pet (same image → MobileNetV2 pipeline)
- Display top-3 class predictions with confidence bars (same as Oxford Pet)

---

## Key Design Decisions
- `split()` shuffles indices; `_create_subset` uses `object.__new__`
- CSV labels auto-cast: int → float → str
- Audio `_load_file` returns `(np.ndarray, int)` tuple
- `LabeledDataset.__init__` sets `_labels = []` before `super().__init__()`
- `.gitignore` uses `/models/` (with leading slash) so `demo/models/` is NOT ignored

## Dependencies
Core: `pillow`, `numpy`, `librosa`, `soundfile`, `matplotlib`, `scipy`
Training: `tensorflow`
Conversion: Google Colab (free) — `tensorflowjs` Python API
