"""FI-2010 数据集的结构和实验协议测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deeplob.dataset import (
    K_TO_LABEL_COLUMN,
    NUM_FEATURES,
    TOTAL_COLUMNS,
    WINDOW_SIZE,
    FI2010WindowDataset,
)


def _fake_with_meta(segments, seed: int = 1):
    rows = max(segment["end"] for segment in segments)
    rng = np.random.default_rng(seed)
    data = np.zeros((rows, TOTAL_COLUMNS), dtype=np.float32)
    data[:, 0] = np.arange(rows, dtype=np.float32)
    data[:, 1:NUM_FEATURES] = rng.standard_normal(
        (rows, NUM_FEATURES - 1),
        dtype=np.float32,
    )
    labels = np.resize(np.array([1, 2, 3], dtype=np.float32), rows)
    for column in K_TO_LABEL_COLUMN.values():
        data[:, column] = labels
    metadata = {
        "rows": rows,
        "raw_feature_columns": 144,
        "model_feature_columns": NUM_FEATURES,
        "label_columns": 5,
        "segments": segments,
    }
    return data, metadata


def _write_fixture(tmp_path, segments):
    data, metadata = _fake_with_meta(segments)
    data_path = tmp_path / "fi2010.npy"
    meta_path = tmp_path / "fi2010_meta.json"
    np.save(data_path, data)
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return data_path, meta_path


def _rows(sample: np.ndarray) -> np.ndarray:
    return sample[0, :, 0].astype(np.int64)


def test_horizon_columns_match_official_layout():
    assert K_TO_LABEL_COLUMN == {
        10: 144,
        20: 145,
        30: 146,
        50: 147,
        100: 148,
    }


@pytest.mark.parametrize("horizon", K_TO_LABEL_COLUMN)
def test_window_shape_and_label_range(tmp_path, horizon):
    segments = [
        {"cf": 7, "role": "train", "start": 0, "end": 500},
        {"cf": 7, "role": "test", "start": 500, "end": 1_000},
        {"cf": 8, "role": "test", "start": 1_000, "end": 1_500},
        {"cf": 9, "role": "test", "start": 1_500, "end": 2_000},
    ]
    data_path, meta_path = _write_fixture(tmp_path, segments)
    dataset = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        k=horizon,
        split="train",
        protocol="setup2",
    )
    features, label = dataset[0]
    assert features.shape == (1, WINDOW_SIZE, NUM_FEATURES)
    assert label in {0, 1, 2}


def test_setup1_uses_matching_training_and_testing_files(tmp_path):
    segments = [
        {"cf": 1, "role": "train", "start": 0, "end": 500},
        {"cf": 1, "role": "test", "start": 500, "end": 1_000},
        {"cf": 2, "role": "train", "start": 1_000, "end": 1_500},
        {"cf": 2, "role": "test", "start": 1_500, "end": 2_000},
    ]
    data_path, meta_path = _write_fixture(tmp_path, segments)
    train = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        split="train",
        protocol="setup1",
        test_cf=2,
    )
    test = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        split="test",
        protocol="setup1",
        test_cf=2,
    )
    assert _rows(train[0][0]).min() >= 1_000
    assert _rows(train[-1][0]).max() < 1_500
    assert _rows(test[0][0]).min() >= 1_500
    assert _rows(test[-1][0]).max() < 2_000


def test_setup2_uses_cf7_train_and_three_test_files(tmp_path):
    segments = [
        {"cf": 7, "role": "train", "start": 0, "end": 500},
        {"cf": 7, "role": "test", "start": 500, "end": 1_000},
        {"cf": 8, "role": "test", "start": 1_000, "end": 1_500},
        {"cf": 9, "role": "test", "start": 1_500, "end": 2_000},
    ]
    data_path, meta_path = _write_fixture(tmp_path, segments)
    train = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        split="train",
        protocol="setup2",
    )
    test = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        split="test",
        protocol="setup2",
    )
    assert _rows(train[0][0]).min() == 0
    assert _rows(train[-1][0]).max() < 500
    assert len(test) == 3 * (500 - WINDOW_SIZE + 1)

    boundaries = ((500, 1_000), (1_000, 1_500), (1_500, 2_000))
    for index in range(len(test)):
        rows = _rows(test[index][0])
        assert any(start <= rows.min() <= rows.max() < end for start, end in boundaries)


def test_train_and_validation_windows_share_no_source_rows(tmp_path):
    segments = [
        {"cf": 7, "role": "train", "start": 0, "end": 2_000},
        {"cf": 7, "role": "test", "start": 2_000, "end": 2_500},
        {"cf": 8, "role": "test", "start": 2_500, "end": 3_000},
        {"cf": 9, "role": "test", "start": 3_000, "end": 3_500},
    ]
    data_path, meta_path = _write_fixture(tmp_path, segments)
    train = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        split="train",
        protocol="setup2",
    )
    validation = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        split="val",
        protocol="setup2",
    )
    train_last_row = _rows(train[-1][0]).max()
    validation_first_row = _rows(validation[0][0]).min()
    assert train_last_row < validation_first_row


def test_rejects_wrong_array_shape(tmp_path):
    data_path = tmp_path / "bad.npy"
    np.save(data_path, np.zeros((500, 148), dtype=np.float32))
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "rows": 500,
                "segments": [{"cf": 7, "role": "train", "start": 0, "end": 500}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="149"):
        FI2010WindowDataset(str(data_path), str(meta_path), protocol="setup2")


def test_negative_index_and_out_of_range_index(tmp_path):
    segments = [
        {"cf": 7, "role": "train", "start": 0, "end": 500},
        {"cf": 7, "role": "test", "start": 500, "end": 1_000},
        {"cf": 8, "role": "test", "start": 1_000, "end": 1_500},
        {"cf": 9, "role": "test", "start": 1_500, "end": 2_000},
    ]
    data_path, meta_path = _write_fixture(tmp_path, segments)
    dataset = FI2010WindowDataset(
        str(data_path),
        str(meta_path),
        protocol="setup2",
    )
    assert np.array_equal(dataset[-1][0], dataset[len(dataset) - 1][0])
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
