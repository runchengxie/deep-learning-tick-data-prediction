"""分钟分片准备脚本的 NaN 填充逻辑测试。"""

import numpy as np
import pytest

from scripts.prepare_minute_shards import (
    _compute_median_fill,
    _fill_nan_with_medians,
    _pad_to_window,
)


def _matrix(values):
    return np.asarray(values, dtype=np.float64)


def test_compute_median_fill_ignores_nan_and_uses_train_only():
    collected = [
        (
            _matrix([[1.0, 2.0], [3.0, 4.0]]),
            {"trading_date": "2024-01-02"},
        ),
        (
            _matrix([[np.nan, 6.0], [7.0, 8.0]]),
            {"trading_date": "2024-03-01"},
        ),
        (
            _matrix([[100.0, 200.0], [300.0, 400.0]]),
            {"trading_date": "2024-06-01"},
        ),
    ]
    from datetime import date

    fill = _compute_median_fill(
        collected,
        feature_count=2,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 4, 30),
    )
    assert fill.shape == (2,)
    assert fill[0] == pytest.approx(3.0)
    assert fill[1] == pytest.approx(5.0)


def test_fill_nan_with_medians_replaces_nan_per_column():
    matrix = _matrix([[np.nan, 2.0], [3.0, np.nan], [1.0, 4.0]])
    fill = np.asarray([7.0, 8.0])
    filled = _fill_nan_with_medians(matrix, fill)
    assert filled[0, 0] == 7.0
    assert filled[1, 1] == 8.0
    assert np.all(np.isfinite(filled))
    assert filled.dtype == np.float64


def test_pad_to_window_fills_short_window_with_nan_tail():
    matrix = _matrix([[1.0, 2.0], [3.0, 4.0]])
    padded, missing = _pad_to_window(matrix, window_minutes=4)
    assert padded.shape == (4, 2)
    assert missing == 2
    assert np.all(np.isnan(padded[2:]))


def test_pad_to_window_noop_for_full_window():
    matrix = _matrix([[1.0, 2.0], [3.0, 4.0]])
    padded, missing = _pad_to_window(matrix, window_minutes=2)
    assert missing == 0
    assert np.array_equal(padded, matrix)


def test_compute_median_fill_raises_when_train_all_nan():
    collected = [
        (
            _matrix([[np.nan, 1.0], [np.nan, 2.0]]),
            {"trading_date": "2024-01-02"},
        ),
    ]
    from datetime import date

    with pytest.raises(ValueError, match="全部为 NaN"):
        _compute_median_fill(
            collected,
            feature_count=2,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 2, 1),
        )
