"""分钟基线数据管线合成数据测试。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.nextday.labels import NextDayTarget
from ticknet.nextday.minute_baseline import (
    MinuteExtractionReport,
    _aggregate_trailing,
    _feature_columns,
    _stream_l2_modality,
    _tushare_symbol,
    build_samples,
    read_l2_minute_rows,
    read_tushare_minute_rows,
)


def _target(trading_date: date, symbol: str, label_date: date, label: int) -> NextDayTarget:
    return NextDayTarget(
        symbol=symbol,
        trading_date=trading_date,
        label_date=label_date,
        raw_return=0.01,
        target_return=0.005,
        label=label,
    )


def test_feature_columns_exclude_valid() -> None:
    names = ("date", "ticker", "minute", "snapshot__spread_bps", "snapshot__valid")
    assert _feature_columns(names, "snapshot") == ("snapshot__spread_bps",)
    assert _feature_columns(("order__a", "order__valid", "order__b"), "order") == (
        "order__a",
        "order__b",
    )


def test_tushare_symbol_strips_suffix() -> None:
    assert _tushare_symbol("000001.SZ") == "000001"
    assert _tushare_symbol("600519.SH") == "600519"


def test_aggregate_trailing_shape_and_values() -> None:
    matrix = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float64,
    )
    result = _aggregate_trailing(matrix)
    assert result.shape == (2 * 4,)
    expected_mean = matrix.mean(axis=0)
    expected_std = matrix.std(axis=0)
    expected_last = matrix[-1]
    expected_delta = matrix[-1] - matrix[0]
    np.testing.assert_allclose(
        result, np.concatenate((expected_mean, expected_std, expected_last, expected_delta))
    )


def test_aggregate_trailing_rejects_empty() -> None:
    matrix = np.empty((0, 3))
    try:
        _aggregate_trailing(matrix)
    except ValueError:
        pass
    else:
        raise AssertionError("空矩阵应抛出 ValueError")


def test_build_samples_window_and_counts(tmp_path) -> None:
    targets = [
        _target(date(2024, 6, 3), "000001", date(2024, 6, 4), 2),
        _target(date(2024, 6, 4), "000002", date(2024, 6, 5), 0),
    ]
    rows = {
        (20240603, "000001"): [
            (index, np.arange(4, dtype=np.float64) + index) for index in range(100)
        ],
    }
    report = MinuteExtractionReport()
    samples = build_samples(
        rows,
        targets,
        window_minutes=10,
        min_window_minutes=10,
        report=report,
    )
    assert len(samples) == 1
    assert report.written_samples == 1
    assert report.missing_rows == 1
    assert samples[0].features.shape == (4 * 4,)
    np.testing.assert_allclose(samples[0].features[8:12], [99.0, 100.0, 101.0, 102.0])


def test_build_samples_insufficient_window() -> None:
    target = _target(date(2024, 6, 3), "000001", date(2024, 6, 4), 1)
    rows = {(20240603, "000001"): [(index, np.zeros(4)) for index in range(5)]}
    report = MinuteExtractionReport()
    samples = build_samples(rows, [target], window_minutes=60, min_window_minutes=30, report=report)
    assert samples == []
    assert report.insufficient_window == 1


def _write_l2_snapshot(path, rows) -> None:
    table = pa.table(
        {
            "date": pa.array([item[0] for item in rows], type=pa.int64()),
            "ticker": pa.array([item[1] for item in rows], type=pa.string()),
            "minute": pa.array([item[2] for item in rows], type=pa.int16()),
            "snapshot__spread_bps": pa.array([item[3] for item in rows], type=pa.float64()),
            "snapshot__valid": pa.array([1] * len(rows), type=pa.int8()),
        }
    )
    pq.write_table(table, path)


def test_stream_l2_modality_trims_to_keep_minutes(tmp_path) -> None:
    path = tmp_path / "2024.parquet"
    rows = [(20240603, "000001", minute, float(minute)) for minute in range(0, 240)]
    _write_l2_snapshot(path, rows)
    report = MinuteExtractionReport()
    result = _stream_l2_modality(
        path,
        wanted_dates={20240603},
        wanted_symbols={"000001"},
        modality="snapshot",
        keep_minutes=60,
        report=report,
    )
    day_rows = dict(result[(20240603, "000001")])
    assert len(day_rows) == 60
    assert min(day_rows) == 180
    assert max(day_rows) == 239
    assert day_rows[239][0] == 239.0


def test_stream_l2_modality_filters(tmp_path) -> None:
    path = tmp_path / "2024.parquet"
    rows = [
        (20240603, "000001", 10, 1.0),
        (20240603, "000001", 11, 2.0),
        (20240603, "000999", 10, 9.0),
        (20240603, "000002", 10, 3.0),
        (20240602, "000001", 10, 4.0),
    ]
    table = pa.table(
        {
            "date": pa.array([item[0] for item in rows], type=pa.int64()),
            "ticker": pa.array([item[1] for item in rows], type=pa.string()),
            "minute": pa.array([item[2] for item in rows], type=pa.int16()),
            "snapshot__spread_bps": pa.array([item[3] for item in rows], type=pa.float64()),
            "snapshot__valid": pa.array([1] * len(rows), type=pa.int8()),
        }
    )
    pq.write_table(table, path, row_group_size=4)
    report = MinuteExtractionReport()
    result = _stream_l2_modality(
        path,
        wanted_dates={20240603},
        wanted_symbols={"000001"},
        modality="snapshot",
        keep_minutes=100,
        report=report,
    )
    assert set(result) == {(20240603, "000001")}
    day_rows = dict(result[(20240603, "000001")])
    assert sorted(day_rows) == [10, 11]
    assert day_rows[10][0] == 1.0
    assert report.scanned_row_groups == 1
    assert report.skipped_row_groups == 1


def test_read_l2_inner_join_across_modalities(tmp_path) -> None:
    modalities = (
        ("snapshot", "snapshot__spread_bps"),
        ("order", "order__a"),
        ("trade", "trade__a"),
    )
    for modality, feature_name in modalities:
        root = tmp_path / "yearly" / modality
        root.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "date": pa.array([20240603, 20240603], type=pa.int64()),
                "ticker": pa.array(["000001", "000001"], type=pa.string()),
                "minute": pa.array([10, 11], type=pa.int16()),
                feature_name: pa.array([1.0, 2.0], type=pa.float64()),
            }
        )
        pq.write_table(table, root / "2024.parquet")
    target = _target(date(2024, 6, 3), "000001", date(2024, 6, 4), 2)
    report = MinuteExtractionReport()
    rows = read_l2_minute_rows(tmp_path, [target], keep_minutes=100, report=report)
    key = (20240603, "000001")
    assert key in rows
    assert [minute for minute, _row in rows[key]] == [10, 11]
    minute_10 = dict(rows[key])[10]
    assert minute_10.shape == (3,)
    np.testing.assert_allclose(minute_10, [1.0, 1.0, 1.0])


def test_read_l2_drops_key_missing_modality(tmp_path) -> None:
    modalities = (
        ("snapshot", "snapshot__spread_bps"),
        ("order", "order__a"),
        ("trade", "trade__a"),
    )
    rows = {"000001": [10, 11], "000999": [10, 11]}
    for index, (modality, feature_name) in enumerate(modalities):
        root = tmp_path / "yearly" / modality
        root.mkdir(parents=True, exist_ok=True)
        data = [
            (20240603, symbol, minute, float(minute)) for symbol in rows for minute in rows[symbol]
        ]
        if index == 2:
            data = [item for item in data if item[1] != "000999"]
        table = pa.table(
            {
                "date": pa.array([item[0] for item in data], type=pa.int64()),
                "ticker": pa.array([item[1] for item in data], type=pa.string()),
                "minute": pa.array([item[2] for item in data], type=pa.int16()),
                feature_name: pa.array([item[3] for item in data], type=pa.float64()),
            }
        )
        pq.write_table(table, root / "2024.parquet")
    targets = [
        _target(date(2024, 6, 3), "000001", date(2024, 6, 4), 2),
        _target(date(2024, 6, 3), "000999", date(2024, 6, 4), 0),
    ]
    report = MinuteExtractionReport()
    rows_read = read_l2_minute_rows(tmp_path, targets, keep_minutes=100, report=report)
    assert set(rows_read) == {(20240603, "000001")}
    for _minute, row in rows_read[(20240603, "000001")]:
        assert row.shape == (3,)


def test_read_tushare_minute_rows(tmp_path) -> None:
    partition = tmp_path / "trade_date=20240603"
    partition.mkdir(parents=True)
    table = pa.table(
        {
            "ts_code": pa.array(["000001.SZ", "000001.SZ", "000999.SZ"], type=pa.string()),
            "trade_time": pa.array(
                [
                    np.datetime64("2024-06-03T14:30:00", "ns"),
                    np.datetime64("2024-06-03T14:31:00", "ns"),
                    np.datetime64("2024-06-03T14:30:00", "ns"),
                ],
                type=pa.timestamp("ns"),
            ),
            "open": pa.array([10.0, 10.1, 99.0], type=pa.float64()),
            "close": pa.array([10.2, 10.3, 99.0], type=pa.float64()),
            "high": pa.array([10.3, 10.4, 99.0], type=pa.float64()),
            "low": pa.array([9.9, 10.0, 99.0], type=pa.float64()),
            "vol": pa.array([100.0, 200.0, 99.0], type=pa.float64()),
            "amount": pa.array([1000.0, 2000.0, 99.0], type=pa.float64()),
        }
    )
    pq.write_table(table, partition / "part-00000.parquet")
    target = _target(date(2024, 6, 3), "000001", date(2024, 6, 4), 1)
    report = MinuteExtractionReport()
    rows = read_tushare_minute_rows(tmp_path, [target], keep_minutes=100, report=report)
    key = (20240603, "000001")
    assert key in rows
    assert [minute for minute, _row in rows[key]] == [870, 871]
    np.testing.assert_allclose(dict(rows[key])[870], [10.0, 10.2, 10.3, 9.9, 100.0, 1000.0])
