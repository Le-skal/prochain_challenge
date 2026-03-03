# Claude Checkpoint — OOP for AI (cours)

## Project
Root: `C:/Users/rmartin/Desktop/code/cours/`
Goal: OOP final project — dataset hierarchy, BatchLoader, preprocessing pipeline.
GitHub: https://github.com/le-skal/datakit
Sphinx docs: https://le-skal.github.io/datakit/
Vercel demo: not yet deployed (serve locally with `cd demo && python -m http.server 8000`)

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

### Model training
Install: `pip install tensorflow` (training only — no tensorflowjs needed on Windows)

| Model | Script | Status | Notes |
|---|---|---|---|
| Oxford-IIIT-Pet (37 breeds) | `python train/train_oxford_pet.py` | ✅ Trained + converted + working in demo | 90% val acc, 8 epochs |
| UTKFace (age regression) | `python train/train_utkface.py` | ⬜ TODO (other PC) | |
| ESC-50 (50 sounds) | `python train/train_esc50.py` | ⬜ TODO (other PC) | |
| BallroomData (8 genres) | `python train/train_ballroom.py` | ⬜ TODO (other PC) | |

Training scripts now also call `model.export(...)` → saves a `_savedmodel/` folder.

### TF.js conversion (important — Keras 3 quirks)
**Do NOT use `--input_format=keras`** — Keras 3 model topology is incompatible with TF.js.
**Use SavedModel format instead:**

In Google Colab (no WSL/admin needed):
```python
# Cell 1 — export
import tensorflow as tf
model = tf.keras.models.load_model('mymodel.h5')
model.export('mymodel_savedmodel')

# Cell 2 — convert
import tensorflowjs as tfjs
tfjs.converters.convert_tf_saved_model('mymodel_savedmodel', 'mymodel_tfjs')
# (use Python API, not CLI — CLI has a bug with argument parsing)

# Cell 3 — download
import shutil
shutil.make_archive('mymodel_tfjs', 'zip', 'mymodel_tfjs')
```
Extract zip → put contents in `demo/models/<modelname>/`.

In the browser JS: use `tf.loadGraphModel()` + `model.executeAsync()` (NOT loadLayersModel/predict).
TF.js version pinned to `4.22.0` in index.html to match the converter.

### Vercel demo ✅ Working locally
- `demo/index.html` — preprocessing pipeline demo + ML inference section
- Oxford Pet classifier: **live and working** (20 test images, top-3 predictions, ground truth)
- UTKFace / ESC-50 / BallroomData: "Coming soon" placeholders
- **TODO:** deploy to Vercel (connect repo at vercel.com, vercel.json already configured)
- **TODO:** redesign CSS — user wants a more professional look for portfolio

### Next session priorities
1. CSS/design overhaul of `demo/index.html` — remove v1.0 badge, more portfolio-worthy
2. Train UTKFace + ESC-50 + BallroomData on other PC
3. Convert remaining 3 models via Google Colab (use Python API, not CLI)
4. Add model inference sections for all 3 (JS already structured for this)
5. Deploy to Vercel + push everything

---

## Key Design Decisions
- `split()` shuffles indices; `_create_subset` uses `object.__new__`
- CSV labels auto-cast: int → float → str
- Audio `_load_file` returns `(np.ndarray, int)` tuple
- `LabeledDataset.__init__` sets `_labels = []` before `super().__init__()`

## Dependencies
Core: `pillow`, `numpy`, `librosa`, `soundfile`, `matplotlib`, `scipy`
Training: `tensorflow`
Conversion: Google Colab (free) — `tensorflowjs` Python API
