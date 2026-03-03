"""Main showcase script for the OOP for AI library.

Demonstrates:
  1. CSV generation helpers (UTKFace age labels, Ballroom BPM labels)
  2. Dataset hierarchy — labeled/unlabeled, classification/regression,
     lazy/eager
  3. BatchLoader — sequential vs random, keep vs drop last batch
  4. Preprocessing pipelines — applied to dataset samples and
     batch-loader batches
"""

import csv
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.audio_dataset import AudioDataset, UnlabeledAudioDataset
from src.batch_loader import BatchLoader
from src.image_dataset import ImageDataset, UnlabeledImageDataset
from src.preprocessing import (
    AudioRandomCrop,
    CenterCrop,
    MelSpectrogram,
    Padding,
    Pipeline,
    PitchShift,
    RandomFlip,
    Resample,
)

# ---------------------------------------------------------------------------
# 0.  Paths
# ---------------------------------------------------------------------------

ROOT_OXFORD = "dataset/Oxford-IIIT-Pet"
ROOT_UTK = "dataset/UTKFace/UTKFace"
ROOT_ESC50 = "dataset/ESC-50/audio"
ROOT_BALLROOM = "dataset/BallroomData"
ROOT_WALTZ = "dataset/BallroomData/Waltz"
ANN_DIR = "dataset/BallroomAnnotations"

CSV_OXFORD = "dataset/oxford_labels.csv"
CSV_UTK = "dataset/utk_labels.csv"
CSV_ESC50 = "dataset/ESC-50/meta/esc50.csv"
CSV_BPM = "dataset/ballroom_bpm.csv"


# ---------------------------------------------------------------------------
# 1.  CSV generation helpers
# ---------------------------------------------------------------------------

def generate_utk_csv(root: str, out_path: str) -> None:
    """Generate a CSV mapping UTKFace filename -> age (regression label).

    Filenames follow the pattern ``age_gender_race_timestamp.jpg.chip.jpg``.
    """
    if os.path.exists(out_path):
        print(f"[CSV] {out_path} already exists — skipping generation.")
        return
    rows = []
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(".jpg"):
            continue
        try:
            age = int(fname.split("_")[0])
        except ValueError:
            continue
        rows.append((fname, age))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[CSV] Generated {out_path} ({len(rows)} rows).")


def _bpm_from_beats_file(path: str) -> float:
    """Estimate BPM from a .beats file (mean inter-beat interval)."""
    times = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                times.append(float(parts[0]))
    if len(times) < 2:
        return 0.0
    intervals = np.diff(times)
    return float(60.0 / np.mean(intervals))


def generate_ballroom_bpm_csv(
        audio_root: str, ann_dir: str, out_path: str) -> None:
    """Generate a CSV mapping Waltz filename -> BPM (regression label).

    Only covers files in *audio_root* (a single genre folder) that have a
    matching annotation in *ann_dir*.
    """
    if os.path.exists(out_path):
        print(f"[CSV] {out_path} already exists — skipping generation.")
        return
    rows = []
    for fname in sorted(os.listdir(audio_root)):
        if not fname.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(fname)[0]
        beats_path = os.path.join(ann_dir, stem + ".beats")
        if not os.path.exists(beats_path):
            continue
        bpm = _bpm_from_beats_file(beats_path)
        rows.append((fname, round(bpm, 2)))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[CSV] Generated {out_path} ({len(rows)} rows).")


def generate_esc50_flat_csv(
        meta_csv: str, audio_root: str, out_path: str) -> None:
    """Re-format ESC-50 meta CSV to (filename, category) for our parser."""
    if os.path.exists(out_path):
        print(f"[CSV] {out_path} already exists — skipping generation.")
        return
    rows = []
    with open(meta_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = row["filename"]
            if os.path.exists(os.path.join(audio_root, fname)):
                rows.append((fname, row["category"]))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[CSV] Generated {out_path} ({len(rows)} rows).")


CSV_ESC50_FLAT = "dataset/esc50_labels.csv"


# ---------------------------------------------------------------------------
# 2.  Dataset showcase
# ---------------------------------------------------------------------------

def showcase_datasets() -> None:
    print("\n" + "=" * 60)
    print("SECTION 1 — Datasets")
    print("=" * 60)

    # --- 1a. Labeled image dataset, CSV mode, classification, lazy ----------
    print("\n[Image] Oxford-IIIT-Pet — labeled, classification, lazy")
    ds_oxford = ImageDataset(ROOT_OXFORD, lazy=True, labels_file=CSV_OXFORD)
    print(f"  Size : {len(ds_oxford)}")
    img, label = ds_oxford[0]
    print(f"  ds[0]: shape={img.shape}  label='{label}'")
    train, test = ds_oxford.split(0.8)
    print(f"  Split 80/20 -> train={len(train)}  test={len(test)}")

    # --- 1b. Labeled image dataset, CSV mode, regression, lazy --------------
    print("\n[Image] UTKFace — labeled, regression (age), lazy")
    ds_utk = ImageDataset(ROOT_UTK, lazy=True, labels_file=CSV_UTK)
    print(f"  Size : {len(ds_utk)}")
    img_utk, age = ds_utk[0]
    print(f"  ds[0]: shape={img_utk.shape}  age={age}")

    # --- 1c. Unlabeled image dataset -----------------------------------------
    print("\n[Image] Oxford-IIIT-Pet — unlabeled, lazy")
    ds_unlabeled_img = UnlabeledImageDataset(ROOT_OXFORD, lazy=True)
    print(f"  Size : {len(ds_unlabeled_img)}")
    img_u = ds_unlabeled_img[0]
    print(f"  ds[0]: shape={img_u.shape}")

    # --- 1d. Eager image dataset (loads all into RAM) -----------------------
    print("\n[Image] Oxford-IIIT-Pet — first 50 items, EAGER")
    ds_eager = ImageDataset(ROOT_OXFORD, lazy=False, labels_file=CSV_OXFORD)
    print(f"  Size: {len(ds_eager)}  (all loaded into memory at construction)")
    img_e, label_e = ds_eager[5]
    print(f"  ds[5]: shape={img_e.shape}  label='{label_e}'")

    # --- 1e. Labeled audio, CSV mode, classification ------------------------
    print("\n[Audio] ESC-50 — labeled, classification, lazy")
    ds_esc50 = AudioDataset(ROOT_ESC50, lazy=True, labels_file=CSV_ESC50_FLAT)
    print(f"  Size : {len(ds_esc50)}")
    (y, sr), category = ds_esc50[0]
    dur = librosa.get_duration(y=y, sr=sr)
    print(f"  ds[0]: duration={dur:.1f}s  sr={sr}"
          f"  category='{category}'")

    # --- 1f. Labeled audio, folder mode, classification ---------------------
    print("\n[Audio] BallroomData — labeled, folder hierarchy, lazy")
    ds_ballroom = AudioDataset(ROOT_BALLROOM, lazy=True)
    print(f"  Size : {len(ds_ballroom)}")
    (y_b, sr_b), genre = ds_ballroom[0]
    dur_b = librosa.get_duration(y=y_b, sr=sr_b)
    print(f"  ds[0]: duration={dur_b:.1f}s  genre='{genre}'")
    train_b, test_b = ds_ballroom.split(0.8)
    print(f"  Split 80/20 -> train={len(train_b)}  test={len(test_b)}")

    # --- 1g. Labeled audio, CSV mode, regression (BPM) ---------------------
    print("\n[Audio] BallroomData/Waltz — labeled, regression (BPM), lazy")
    ds_bpm = AudioDataset(ROOT_WALTZ, lazy=True, labels_file=CSV_BPM)
    print(f"  Size : {len(ds_bpm)}")
    (y_w, sr_w), bpm = ds_bpm[0]
    dur_w = librosa.get_duration(y=y_w, sr=sr_w)
    print(f"  ds[0]: duration={dur_w:.1f}s  BPM={bpm}")

    # --- 1h. Unlabeled audio ------------------------------------------------
    print("\n[Audio] BallroomData/Waltz — unlabeled, lazy")
    ds_unlabeled_aud = UnlabeledAudioDataset(ROOT_WALTZ, lazy=True)
    print(f"  Size : {len(ds_unlabeled_aud)}")

    # --- Plot a few images --------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Oxford-IIIT-Pet — sample images")
    for ax, idx in zip(axes, [0, 1, 2, 3]):
        img_plot, lbl_plot = ds_oxford[idx * 500]
        ax.imshow(img_plot)
        ax.set_title(lbl_plot, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100)
    print("\n  [Plot] Saved sample_images.png")
    plt.close()

    return ds_oxford, ds_esc50


# ---------------------------------------------------------------------------
# 3.  BatchLoader showcase
# ---------------------------------------------------------------------------

def showcase_batchloader(ds_img: ImageDataset, ds_audio: AudioDataset) -> None:
    print("\n" + "=" * 60)
    print("SECTION 2 — BatchLoader")
    print("=" * 60)

    # --- Image BatchLoader --------------------------------------------------
    print("\n[Image] Sequential, batch_size=64, drop_last=False")
    bl_seq = BatchLoader(ds_img, batch_size=64, shuffle=False, drop_last=False)
    print(f"  Number of batches : {len(bl_seq)}")
    first_batch = next(iter(bl_seq))
    print(f"  First batch size  : {len(first_batch)}")
    print(f"  First item label  : {first_batch[0][1]}")

    print("\n[Image] Random, batch_size=64, drop_last=True")
    bl_rand = BatchLoader(ds_img, batch_size=64, shuffle=True, drop_last=True)
    print(f"  Number of batches : {len(bl_rand)}")
    first_rand = next(iter(bl_rand))
    print(f"  First batch size  : {len(first_rand)}")

    # --- Audio BatchLoader --------------------------------------------------
    print("\n[Audio] Sequential, batch_size=8, drop_last=False")
    ds_esc = AudioDataset(ROOT_ESC50, lazy=True, labels_file=CSV_ESC50_FLAT)
    bl_audio = BatchLoader(
        ds_esc,
        batch_size=8,
        shuffle=False,
        drop_last=False)
    print(f"  Number of batches : {len(bl_audio)}")
    audio_batch = next(iter(bl_audio))
    print(f"  First batch size  : {len(audio_batch)}")
    (y0, sr0), cat0 = audio_batch[0]
    dur0 = librosa.get_duration(y=y0, sr=sr0)
    print(f"  First item        : duration={dur0:.1f}s  cat='{cat0}'")

    print("\n[Audio] Random, batch_size=8, drop_last=True")
    bl_audio_rand = BatchLoader(
        ds_esc,
        batch_size=8,
        shuffle=True,
        drop_last=True)
    print(f"  Number of batches : {len(bl_audio_rand)}")


# ---------------------------------------------------------------------------
# 4.  Preprocessing pipeline showcase
# ---------------------------------------------------------------------------

def showcase_preprocessing(ds_img: ImageDataset,
                           ds_audio: AudioDataset) -> None:
    print("\n" + "=" * 60)
    print("SECTION 3 — Preprocessing Pipelines")
    print("=" * 60)

    # --- Image pipeline: CenterCrop -> RandomFlip -> Padding -----------------
    img_pipeline = Pipeline(
        CenterCrop(height=256, width=256),
        RandomFlip(p=0.5),
        Padding(height=300, width=300, color=(128, 128, 128)),
    )

    print("\n[Image pipeline] CenterCrop(256,256) -> RandomFlip(0.5)"
          " -> Padding(300,300)")
    print("  Applied to 5 dataset samples:")
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for i in range(5):
        raw_img, lbl = ds_img[i * 200]
        processed = img_pipeline(raw_img)
        axes[0, i].imshow(raw_img)
        axes[0, i].set_title(f"raw {raw_img.shape[:2]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(processed)
        axes[1, i].set_title(f"proc {processed.shape[:2]}", fontsize=8)
        axes[1, i].axis("off")
        print(f"    sample {i}: {raw_img.shape} -> "
              f"{processed.shape}  label='{lbl}'")
    fig.suptitle("Image pipeline: CenterCrop -> RandomFlip -> Padding")
    plt.tight_layout()
    plt.savefig("sample_preprocessing_images.png", dpi=100)
    print("  [Plot] Saved sample_preprocessing_images.png")
    plt.close()

    print("\n  Applied through a BatchLoader (first batch):")
    bl = BatchLoader(ds_img, batch_size=4, shuffle=True, drop_last=False)
    for batch in bl:
        for item in batch:
            raw, lbl = item
            out = img_pipeline(raw)
            print(f"    {raw.shape} -> {out.shape}  label='{lbl}'")
        break  # only first batch

    # --- Audio pipeline: AudioRandomCrop -> Resample -> MelSpectrogram -------
    audio_pipeline = Pipeline(
        AudioRandomCrop(duration=3.0),
        Resample(target_sr=22050),
        MelSpectrogram(n_mels=128, n_fft=2048, hop_length=512),
    )

    print("\n[Audio pipeline] AudioRandomCrop(3s) -> Resample(22050)"
          " -> MelSpectrogram(128)")
    print("  Applied to 5 dataset samples:")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i in range(5):
        (y, sr), cat = ds_audio[i * 40]
        spec = audio_pipeline((y, sr))
        librosa.display.specshow(
            librosa.power_to_db(spec, ref=np.max),
            ax=axes[i], sr=22050, hop_length=512,
        )
        axes[i].set_title(f"'{cat}'", fontsize=8)
        axes[i].axis("off")
        dur = librosa.get_duration(y=y, sr=sr)
        print(f"    sample {i}: dur={dur:.1f}s"
              f" -> spec{spec.shape}  cat='{cat}'")
    fig.suptitle("Audio pipeline: RandomCrop -> Resample -> MelSpectrogram")
    plt.tight_layout()
    plt.savefig("sample_preprocessing_audio.png", dpi=100)
    print("  [Plot] Saved sample_preprocessing_audio.png")
    plt.close()

    print("\n  Applied through a BatchLoader (first batch):")
    bl_audio = BatchLoader(
        ds_audio,
        batch_size=4,
        shuffle=True,
        drop_last=False)
    for batch in bl_audio:
        for item in batch:
            (y, sr), cat = item
            spec = audio_pipeline((y, sr))
            dur = librosa.get_duration(y=y, sr=sr)
            print(f"    dur={dur:.1f}s"
                  f" -> spec{spec.shape}  cat='{cat}'")
        break

    # --- Standalone: PitchShift only ----------------------------------------
    print("\n[Audio] PitchShift(+4 semitones) — applied to one sample:")
    (y_raw, sr_raw), cat_raw = ds_audio[0]
    y_shifted, sr_shifted = PitchShift(n_steps=4.0)((y_raw, sr_raw))
    print(f"  Original : {len(y_raw)} samples  sr={sr_raw}")
    print(f"  Shifted  : {len(y_shifted)} samples  sr={sr_shifted}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 0 — generate CSVs if not already present
    print("=" * 60)
    print("STEP 0 — Generating label CSVs")
    print("=" * 60)
    generate_utk_csv(ROOT_UTK, CSV_UTK)
    generate_ballroom_bpm_csv(ROOT_WALTZ, ANN_DIR, CSV_BPM)
    generate_esc50_flat_csv(CSV_ESC50, ROOT_ESC50, CSV_ESC50_FLAT)

    # Step 1 — datasets
    ds_img, ds_audio = showcase_datasets()

    # Step 2 — batch loaders
    showcase_batchloader(ds_img, ds_audio)

    # Step 3 — preprocessing
    showcase_preprocessing(ds_img, ds_audio)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
