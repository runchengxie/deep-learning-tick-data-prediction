"""数据集测试，用合成的 149 列 .npy 模拟官方 FI-2010 布局，无需真实数据。"""

import json

import numpy as np

from src.dataset import K_TO_LABEL_COLUMN, NUM_FEATURES, WINDOW_SIZE, FI2010WindowDataset


def _fake_fi2010(n: int = 500, seed: int = 1):
    rng = np.random.default_rng(seed)
    fake = np.zeros((n, 149), dtype=np.float32)
    fake[:, :NUM_FEATURES] = rng.standard_normal((n, NUM_FEATURES)).astype(np.float32)
    # 标签列 144/145 用 1/2/3 编码（FI-2010 约定），后面会归一化成 0/1/2
    fake[:, 144] = rng.integers(1, 4, n).astype(np.float32)
    fake[:, 145] = rng.integers(1, 4, n).astype(np.float32)
    return fake


def _fake_with_meta(segments, seed: int = 1):
    """Build a (N,149) npy where feature col 0 == global row index, plus meta.json.

    Encoding the row index into column 0 lets tests verify that a window never
    straddles a segment boundary (all its rows must come from one segment).
    """
    n = max(seg["end"] for seg in segments)
    rng = np.random.default_rng(seed)
    fake = np.zeros((n, 149), dtype=np.float32)
    fake[:, 0] = np.arange(n, dtype=np.float32)  # row index marker
    fake[:, 1:NUM_FEATURES] = rng.standard_normal((n, NUM_FEATURES - 1)).astype(np.float32)
    fake[:, 144] = rng.integers(1, 4, n).astype(np.float32)
    fake[:, 145] = rng.integers(1, 4, n).astype(np.float32)
    return fake, {"rows": n, "n_features": NUM_FEATURES, "n_label_cols": 5, "segments": segments}


def test_window_shape_and_label_range(tmp_path):
    path = tmp_path / "fake.npy"
    np.save(path, _fake_fi2010())

    for k in (10, 20):
        ds = FI2010WindowDataset(str(path), k=k, window_size=WINDOW_SIZE, split="train")
        x, y = ds[0]
        assert x.shape == (1, WINDOW_SIZE, NUM_FEATURES)
        assert y in (0, 1, 2)


def test_k_selects_different_label_column(tmp_path):
    path = tmp_path / "fake.npy"
    np.save(path, _fake_fi2010())

    ds10 = FI2010WindowDataset(str(path), k=10, window_size=WINDOW_SIZE, split="train")
    ds20 = FI2010WindowDataset(str(path), k=20, window_size=WINDOW_SIZE, split="train")
    x10, y10 = ds10[0]
    x20, y20 = ds20[0]
    # 两个 k 选的是不同标签列，且取出的标签都在归一化后的 {0,1,2} 内
    assert ds10.label_col == K_TO_LABEL_COLUMN[10]
    assert ds20.label_col == K_TO_LABEL_COLUMN[20]
    assert y10 in (0, 1, 2)
    assert y20 in (0, 1, 2)


def test_train_split_length(tmp_path):
    path = tmp_path / "fake.npy"
    np.save(path, _fake_fi2010(n=500))
    ds = FI2010WindowDataset(str(path), k=10, window_size=WINDOW_SIZE, split="train")
    tr_rows = int(500 * 0.7)
    assert len(ds) == tr_rows - WINDOW_SIZE + 1


def _window_rows(x):
    """The 100 global row indices covered by a window (col 0 encodes them)."""
    return x[0, :, 0].astype(np.int64)


def test_windows_do_not_cross_segment(tmp_path):
    # 3 segments, each 100 rows: cf1[0,100) cf2[100,200) cf3[200,300).
    segments = [
        {"cf": 1, "role": "train", "split": "Training", "start": 0, "end": 100},
        {"cf": 2, "role": "train", "split": "Training", "start": 100, "end": 200},
        {"cf": 3, "role": "train", "split": "Training", "start": 200, "end": 300},
    ]
    data, meta = _fake_with_meta(segments)
    path = tmp_path / "fake.npy"
    meta_path = tmp_path / "meta.json"
    np.save(path, data)
    meta_path.write_text(json.dumps(meta))
    # test_fold=0 -> cf1 is the test fold, so cf2 & cf3 feed training (two segments).
    ds = FI2010WindowDataset(
        str(path), k=10, window_size=WINDOW_SIZE, split="train",
        protocol="standard9", meta_path=str(meta_path), test_fold=0,
    )
    # Every window's rows must all fall within a single 100-row segment block.
    for i in range(len(ds)):
        rows = _window_rows(ds[i][0])
        blocks = set((rows // 100).tolist())
        assert len(blocks) == 1, f"window {i} crossed segment boundary: rows {rows}"


def test_train_val_no_time_interleave(tmp_path):
    # single 2000-row training segment, standard9 with cf2 as the (absent) test fold.
    segments = [
        {"cf": 1, "role": "train", "split": "Training", "start": 0, "end": 2000},
    ]
    data, meta = _fake_with_meta(segments)
    path = tmp_path / "fake.npy"
    meta_path = tmp_path / "meta.json"
    np.save(path, data)
    meta_path.write_text(json.dumps(meta))
    # test_fold=1 means cf2 is test; cf1 (the only segment) is training.
    train_ds = FI2010WindowDataset(
        str(path), k=10, window_size=WINDOW_SIZE, split="train",
        protocol="standard9", meta_path=str(meta_path), test_fold=1,
    )
    val_ds = FI2010WindowDataset(
        str(path), k=10, window_size=WINDOW_SIZE, split="val",
        protocol="standard9", meta_path=str(meta_path), test_fold=1,
    )
    train_max = max(_window_rows(train_ds[i][0]).max() for i in range(len(train_ds)))
    val_min = min(_window_rows(val_ds[i][0]).min() for i in range(len(val_ds)))
    # Validation windows must sit strictly AFTER training windows in time.
    assert train_max <= val_min, f"train max {train_max} > val min {val_min} (leak!)"


def test_light_setup2_uses_cf7_train(tmp_path):
    # cf7 train [0,100); cf7/8/9 test [100,400).
    segments = [
        {"cf": 7, "role": "train", "split": "Training", "start": 0, "end": 100},
        {"cf": 7, "role": "test", "split": "Testing", "start": 100, "end": 200},
        {"cf": 8, "role": "test", "split": "Testing", "start": 200, "end": 300},
        {"cf": 9, "role": "test", "split": "Testing", "start": 300, "end": 400},
    ]
    data, meta = _fake_with_meta(segments)
    path = tmp_path / "fake.npy"
    meta_path = tmp_path / "meta.json"
    np.save(path, data)
    meta_path.write_text(json.dumps(meta))
    train_ds = FI2010WindowDataset(
        str(path), k=10, window_size=WINDOW_SIZE, split="train",
        protocol="light_setup2", meta_path=str(meta_path), test_cf=7,
    )
    test_ds = FI2010WindowDataset(
        str(path), k=10, window_size=WINDOW_SIZE, split="test",
        protocol="light_setup2", meta_path=str(meta_path), test_cf=7,
    )
    # Training windows must come entirely from cf7's Training segment (rows < 100).
    for i in range(len(train_ds)):
        assert _window_rows(train_ds[i][0]).max() < 100
    # Test windows must come from cf7/8/9 Testing segments (rows >= 100).
    for i in range(len(test_ds)):
        assert _window_rows(test_ds[i][0]).min() >= 100
