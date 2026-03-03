"""Tests for dataset hierarchy, BatchLoader, and preprocessing.

All tests use temporary directories with synthetic data so no real
dataset files are required.
"""

import csv
import math
import os

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from src.audio_dataset import AudioDataset, UnlabeledAudioDataset
from src.batch_loader import BatchLoader
from src.image_dataset import ImageDataset, UnlabeledImageDataset
from src.preprocessing import (
    AudioRandomCrop,
    CenterCrop,
    MelSpectrogram,
    Padding,
    Pipeline,
    RandomCrop,
    RandomFlip,
    Resample,
)
from src.utils import check_range, check_type, parse_labels_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_images(folder, names):
    """Save small RGB PNG images and return their paths."""
    os.makedirs(folder, exist_ok=True)
    for name in names:
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(os.path.join(folder, name))


def _make_wavs(folder, names, duration=1.0, sr=22050):
    """Save short WAV files and return their paths."""
    os.makedirs(folder, exist_ok=True)
    samples = int(duration * sr)
    for name in names:
        y = np.zeros(samples, dtype=np.float32)
        sf.write(os.path.join(folder, name), y, sr)


def _make_csv(path, rows):
    """Write a CSV file from a list of (filename, label) tuples."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_check_type_ok(self):
        check_type("hello", str, "x")  # should not raise

    def test_check_type_raises(self):
        with pytest.raises(TypeError):
            check_type(42, str, "x")

    def test_check_type_tuple_of_types(self):
        check_type(3.14, (int, float), "x")  # should not raise

    def test_check_range_ok(self):
        check_range(0.5, 0.0, 1.0, "x")  # should not raise

    def test_check_range_raises_low(self):
        with pytest.raises(ValueError):
            check_range(-0.1, 0.0, 1.0, "x")

    def test_check_range_raises_high(self):
        with pytest.raises(ValueError):
            check_range(1.1, 0.0, 1.0, "x")

    def test_parse_labels_csv(self, tmp_path):
        csv_path = str(tmp_path / "labels.csv")
        _make_csv(csv_path, [("a.jpg", "cat"), ("b.jpg", "dog")])
        mapping = parse_labels_csv(csv_path)
        assert mapping == {"a.jpg": "cat", "b.jpg": "dog"}


# ---------------------------------------------------------------------------
# ImageDataset — CSV mode
# ---------------------------------------------------------------------------

class TestImageDatasetCSV:
    @pytest.fixture
    def setup(self, tmp_path):
        names = [f"img_{i}.png" for i in range(10)]
        root = str(tmp_path / "images")
        _make_images(root, names)
        csv_path = str(tmp_path / "labels.csv")
        _make_csv(csv_path, [(n, f"class_{i % 3}") for i, n in enumerate(names)])
        return root, csv_path

    def test_len(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        assert len(ds) == 10

    def test_getitem_returns_tuple(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_getitem_image_shape(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        img, _ = ds[0]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3

    def test_getitem_label(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        _, label = ds[0]
        assert isinstance(label, str)

    def test_split_sizes(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        train, test = ds.split(0.8)
        assert len(train) + len(test) == len(ds)

    def test_split_no_overlap(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        train, test = ds.split(0.8)
        train_paths = set(train._file_paths)
        test_paths = set(test._file_paths)
        assert train_paths.isdisjoint(test_paths)

    def test_eager_loads_data(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=False, labels_file=csv_path)
        assert ds._data is not None
        assert len(ds._data) == 10

    def test_numeric_label_regression(self, tmp_path):
        names = [f"img_{i}.png" for i in range(5)]
        root = str(tmp_path / "images")
        _make_images(root, names)
        csv_path = str(tmp_path / "labels.csv")
        _make_csv(csv_path, [(n, str(i * 10)) for i, n in enumerate(names)])
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        _, label = ds[0]
        assert isinstance(label, int)

    def test_index_out_of_range(self, setup):
        root, csv_path = setup
        ds = ImageDataset(root, lazy=True, labels_file=csv_path)
        with pytest.raises(IndexError):
            _ = ds[100]


# ---------------------------------------------------------------------------
# ImageDataset — folder mode
# ---------------------------------------------------------------------------

class TestImageDatasetFolder:
    @pytest.fixture
    def setup(self, tmp_path):
        root = str(tmp_path / "images")
        for cls in ["cats", "dogs"]:
            _make_images(
                os.path.join(root, cls),
                [f"{cls}_{i}.png" for i in range(5)]
            )
        return root

    def test_len(self, setup):
        ds = ImageDataset(setup, lazy=True)
        assert len(ds) == 10

    def test_labels_are_folder_names(self, setup):
        ds = ImageDataset(setup, lazy=True)
        assert set(ds.labels) == {"cats", "dogs"}


# ---------------------------------------------------------------------------
# UnlabeledImageDataset
# ---------------------------------------------------------------------------

class TestUnlabeledImageDataset:
    def test_getitem_returns_array(self, tmp_path):
        root = str(tmp_path / "images")
        _make_images(root, ["a.png", "b.png"])
        ds = UnlabeledImageDataset(root, lazy=True)
        item = ds[0]
        assert isinstance(item, np.ndarray)

    def test_len(self, tmp_path):
        root = str(tmp_path / "images")
        _make_images(root, ["a.png", "b.png", "c.png"])
        ds = UnlabeledImageDataset(root)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# AudioDataset — CSV mode
# ---------------------------------------------------------------------------

class TestAudioDatasetCSV:
    @pytest.fixture
    def setup(self, tmp_path):
        names = [f"audio_{i}.wav" for i in range(6)]
        root = str(tmp_path / "audio")
        _make_wavs(root, names)
        csv_path = str(tmp_path / "labels.csv")
        _make_csv(csv_path, [(n, f"genre_{i % 2}") for i, n in enumerate(names)])
        return root, csv_path

    def test_len(self, setup):
        root, csv_path = setup
        ds = AudioDataset(root, lazy=True, labels_file=csv_path)
        assert len(ds) == 6

    def test_getitem_returns_tuple(self, setup):
        root, csv_path = setup
        ds = AudioDataset(root, lazy=True, labels_file=csv_path)
        (y, sr), label = ds[0]
        assert isinstance(y, np.ndarray)
        assert isinstance(sr, int)
        assert isinstance(label, str)

    def test_split_sizes(self, setup):
        root, csv_path = setup
        ds = AudioDataset(root, lazy=True, labels_file=csv_path)
        train, test = ds.split(0.8)
        assert len(train) + len(test) == len(ds)


# ---------------------------------------------------------------------------
# AudioDataset — folder mode
# ---------------------------------------------------------------------------

class TestAudioDatasetFolder:
    @pytest.fixture
    def setup(self, tmp_path):
        root = str(tmp_path / "audio")
        for genre in ["jazz", "rock"]:
            _make_wavs(
                os.path.join(root, genre),
                [f"{genre}_{i}.wav" for i in range(4)]
            )
        return root

    def test_len(self, setup):
        ds = AudioDataset(setup, lazy=True)
        assert len(ds) == 8

    def test_labels_are_folder_names(self, setup):
        ds = AudioDataset(setup, lazy=True)
        assert set(ds.labels) == {"jazz", "rock"}


# ---------------------------------------------------------------------------
# UnlabeledAudioDataset
# ---------------------------------------------------------------------------

class TestUnlabeledAudioDataset:
    def test_getitem_returns_tuple(self, tmp_path):
        root = str(tmp_path / "audio")
        _make_wavs(root, ["a.wav", "b.wav"])
        ds = UnlabeledAudioDataset(root)
        y, sr = ds[0]
        assert isinstance(y, np.ndarray)
        assert isinstance(sr, int)


# ---------------------------------------------------------------------------
# BatchLoader
# ---------------------------------------------------------------------------

class TestBatchLoader:
    @pytest.fixture
    def ds(self, tmp_path):
        root = str(tmp_path / "images")
        names = [f"img_{i}.png" for i in range(10)]
        _make_images(root, names)
        csv_path = str(tmp_path / "labels.csv")
        _make_csv(csv_path, [(n, "cat") for n in names])
        return ImageDataset(root, lazy=True, labels_file=csv_path)

    def test_len_keep_last(self, ds):
        bl = BatchLoader(ds, batch_size=3, drop_last=False)
        assert len(bl) == math.ceil(10 / 3)

    def test_len_drop_last(self, ds):
        bl = BatchLoader(ds, batch_size=3, drop_last=True)
        assert len(bl) == 10 // 3

    def test_iteration_batch_size(self, ds):
        bl = BatchLoader(ds, batch_size=4, drop_last=False)
        batches = list(bl)
        assert len(batches[0]) == 4

    def test_iteration_drop_last(self, ds):
        bl = BatchLoader(ds, batch_size=3, drop_last=True)
        for batch in bl:
            assert len(batch) == 3

    def test_total_items_keep_last(self, ds):
        bl = BatchLoader(ds, batch_size=3, drop_last=False)
        total = sum(len(b) for b in bl)
        assert total == 10

    def test_len_consistent_with_iteration(self, ds):
        bl = BatchLoader(ds, batch_size=3, drop_last=True)
        assert len(bl) == len(list(bl))

    def test_invalid_batch_size(self, ds):
        with pytest.raises(ValueError):
            BatchLoader(ds, batch_size=0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    IMG = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    AUDIO = (np.zeros(22050, dtype=np.float32), 22050)

    # Image transforms
    def test_center_crop(self):
        out = CenterCrop(64, 64)(self.IMG)
        assert out.shape == (64, 64, 3)

    def test_center_crop_no_crop_when_smaller(self):
        small = np.zeros((50, 50, 3), dtype=np.uint8)
        out = CenterCrop(100, 100)(small)
        assert out.shape == (50, 50, 3)

    def test_random_crop(self):
        out = RandomCrop(64, 64)(self.IMG)
        assert out.shape == (64, 64, 3)

    def test_random_flip_preserves_shape(self):
        out = RandomFlip(p=1.0)(self.IMG)
        assert out.shape == self.IMG.shape

    def test_padding_expands(self):
        small = np.zeros((50, 50, 3), dtype=np.uint8)
        out = Padding(100, 120)(small)
        assert out.shape == (100, 120, 3)

    def test_padding_no_change_when_larger(self):
        out = Padding(50, 50)(self.IMG)
        assert out.shape == self.IMG.shape

    # Audio transforms
    def test_mel_spectrogram_shape(self):
        spec = MelSpectrogram(n_mels=64)(self.AUDIO)
        assert spec.ndim == 2
        assert spec.shape[0] == 64

    def test_audio_random_crop_duration(self):
        long_audio = (np.zeros(110250, dtype=np.float32), 22050)  # 5s
        y_out, sr_out = AudioRandomCrop(duration=2.0)(long_audio)
        assert len(y_out) == 2 * 22050

    def test_audio_random_crop_short_unchanged(self):
        y_out, sr_out = AudioRandomCrop(duration=10.0)(self.AUDIO)
        assert len(y_out) == len(self.AUDIO[0])

    def test_resample(self):
        _, sr_out = Resample(target_sr=16000)(self.AUDIO)
        assert sr_out == 16000

    # Pipeline
    def test_pipeline_image(self):
        pipe = Pipeline(CenterCrop(64, 64), RandomFlip(p=0.0))
        out = pipe(self.IMG)
        assert out.shape == (64, 64, 3)

    def test_pipeline_audio(self):
        pipe = Pipeline(
            AudioRandomCrop(0.5),
            Resample(16000),
            MelSpectrogram(n_mels=32),
        )
        spec = pipe(self.AUDIO)
        assert spec.ndim == 2 and spec.shape[0] == 32

    def test_pipeline_invalid_transform(self):
        with pytest.raises(TypeError):
            Pipeline("not_a_transform")
