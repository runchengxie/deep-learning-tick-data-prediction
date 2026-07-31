"""FI-2010 转换脚本测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.convert_fi2010 import TOTAL_ROWS, _read_txt, main


def _official_matrix(samples: int = 12) -> np.ndarray:
    matrix = np.zeros((TOTAL_ROWS, samples), dtype=np.float32)
    matrix[:144] = np.arange(144, dtype=np.float32)[:, None]
    labels = np.resize(np.array([1, 2, 3], dtype=np.float32), samples)
    matrix[144:] = labels
    return matrix


def _write_fold(root, *, role: str, fold: int = 1):
    split = "Training" if role == "train" else "Testing"
    prefix = "Train" if role == "train" else "Test"
    directory = root / "NoAuction" / "1.NoAuction_Zscore" / f"NoAuction_Zscore_{split}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_Dst_NoAuction_ZScore_CF_{fold}.txt"
    np.savetxt(path, _official_matrix())
    return path


def test_read_txt_transposes_official_layout(tmp_path):
    path = tmp_path / "sample.txt"
    np.savetxt(path, _official_matrix(samples=9))
    converted = _read_txt(str(path))
    assert converted.shape == (9, 149)
    assert converted.dtype == np.float32
    assert np.array_equal(converted[:, 144], [1, 2, 3] * 3)


def test_read_txt_rejects_invalid_labels(tmp_path):
    path = tmp_path / "sample.txt"
    matrix = _official_matrix()
    matrix[144, 0] = 4
    np.savetxt(path, matrix)
    with pytest.raises(ValueError, match="无效标签"):
        _read_txt(str(path))


def test_main_writes_data_and_segment_metadata(tmp_path):
    _write_fold(tmp_path, role="train")
    _write_fold(tmp_path, role="test")
    output = tmp_path / "FI2010_normalised.npy"
    main(
        [
            "--base-dir",
            str(tmp_path),
            "--folds",
            "1",
            "--out",
            str(output),
        ]
    )
    data = np.load(output)
    metadata = json.loads((tmp_path / "FI2010_normalised_meta.json").read_text(encoding="utf-8"))
    assert data.shape == (24, 149)
    assert metadata["rows"] == 24
    assert [(segment["cf"], segment["role"]) for segment in metadata["segments"]] == [
        (1, "train"),
        (1, "test"),
    ]
