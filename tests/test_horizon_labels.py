"""多周期标签侧车、来源绑定和边界 purge 测试。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.nextday.dataset import NextDayShardDataset
from ticknet.nextday.horizon_labels import (
    HORIZON_RETURN_CONTRACT,
    HorizonTarget,
    build_horizon_targets,
    load_horizon_sidecar,
    prepare_horizon_sidecar,
    write_horizon_sidecar,
)
from ticknet.nextday.io import PreparedSample, write_sharded_dataset
from ticknet.nextday.labels import NextDayTarget
from ticknet.nextday.snapshot_config import DailyPanel
from ticknet.nextday.splits import WalkForwardSplit


def _dates(count: int = 9) -> tuple[date, ...]:
    start = date(2024, 1, 2)
    return tuple(start + timedelta(days=offset) for offset in range(count))


def _panels() -> tuple[DailyPanel, DailyPanel]:
    dates = _dates()
    symbols = tuple(f"S{index}" for index in range(5))
    opens = np.full((len(dates), len(symbols)), 100.0, dtype=np.float64)
    closes = np.column_stack(
        [
            100.0 * (1.0 + 0.01 * symbol_index + 0.001 * np.arange(len(dates)))
            for symbol_index in range(5)
        ]
    )
    return (
        DailyPanel(dates=dates, symbols=symbols, values=opens),
        DailyPanel(dates=dates, symbols=symbols, values=closes),
    )


def test_build_horizon_targets_uses_next_open_and_horizon_close() -> None:
    dates = _dates()
    open_panel, close_panel = _panels()
    benchmark = {day: (100.0, 100.0 * (1 + 0.001 * index)) for index, day in enumerate(dates)}
    sample_symbols = {dates[0]: open_panel.symbols, dates[1]: open_panel.symbols}

    targets = build_horizon_targets(
        sample_symbols,
        open_panel,
        close_panel,
        benchmark,
        horizons=(1, 3, 5),
        min_cross_section=5,
    )

    first_h3 = next(
        row
        for row in targets
        if row.trading_date == dates[0] and row.horizon == 3 and row.symbol == "S4"
    )
    assert first_h3.entry_date == dates[1]
    assert first_h3.return_end_date == dates[3]
    assert first_h3.raw_return == pytest.approx(close_panel.values[3, 4] / 100.0 - 1.0)
    assert first_h3.benchmark_return == pytest.approx(benchmark[dates[3]][1] / 100.0 - 1.0)
    assert first_h3.label == 2
    assert {row.horizon for row in targets} == {1, 3, 5}


def _feature_manifest(tmp_path):
    samples = []
    for input_day, label_day in ((2, 3), (3, 4), (6, 7)):
        target = NextDayTarget(
            symbol="S0",
            trading_date=date(2024, 1, input_day),
            label_date=date(2024, 1, label_day),
            raw_return=0.01,
            target_return=0.01,
            label=2,
        )
        samples.append(
            PreparedSample(
                target=target,
                events=np.ones((4, 40), dtype=np.float32),
                last_event_timestamp=datetime(2024, 1, input_day, 14, 54),
                signal_timestamp=datetime(2024, 1, input_day, 14, 55),
            )
        )
    return write_sharded_dataset(
        samples,
        tmp_path / "features",
        chunks_per_sample=1,
        chunk_size=4,
    )


def test_dataset_selects_sidecar_horizon_and_purges_on_return_end_date(tmp_path) -> None:
    manifest_path = _feature_manifest(tmp_path)
    base = NextDayShardDataset(
        manifest_path,
        date_split=WalkForwardSplit.from_strings(
            train_start="2024-01-02",
            train_end="2024-01-05",
            val_start="2024-01-06",
            val_end="2024-01-08",
            test_start="2024-01-09",
            test_end="2024-01-10",
        ),
        split="train",
    )
    targets = [
        HorizonTarget(
            symbol="S0",
            trading_date=date(2024, 1, 2),
            entry_date=date(2024, 1, 3),
            return_end_date=date(2024, 1, 5),
            horizon=3,
            label=0,
            raw_return=-0.02,
            benchmark_return=0.0,
            target_return=-0.02,
        ),
        HorizonTarget(
            symbol="S0",
            trading_date=date(2024, 1, 3),
            entry_date=date(2024, 1, 4),
            return_end_date=date(2024, 1, 6),
            horizon=3,
            label=2,
            raw_return=0.02,
            benchmark_return=0.0,
            target_return=0.02,
        ),
    ]
    sidecar_path = write_horizon_sidecar(
        targets,
        tmp_path / "labels",
        source_dataset_fingerprint=base.dataset_fingerprint,
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=2,
    )
    del base
    dataset = NextDayShardDataset(
        manifest_path,
        date_split=WalkForwardSplit.from_strings(
            train_start="2024-01-02",
            train_end="2024-01-05",
            val_start="2024-01-06",
            val_end="2024-01-08",
            test_start="2024-01-09",
            test_end="2024-01-10",
        ),
        split="train",
        target_sidecar_path=sidecar_path,
        target_horizon=3,
        verify_checksums=True,
    )

    assert len(dataset) == 1
    assert dataset.purged_samples == 1
    assert dataset.missing_target_samples == 1
    assert dataset.return_end_dates == [date(2024, 1, 5)]
    assert dataset.target_returns.tolist() == pytest.approx([-0.02])
    assert dataset.target_return_contract == HORIZON_RETURN_CONTRACT


def test_sidecar_is_bound_to_exact_feature_fingerprint(tmp_path) -> None:
    target = HorizonTarget(
        symbol="S0",
        trading_date=date(2024, 1, 2),
        entry_date=date(2024, 1, 3),
        return_end_date=date(2024, 1, 3),
        horizon=1,
        label=1,
        raw_return=0.0,
        benchmark_return=0.0,
        target_return=0.0,
    )
    path = write_horizon_sidecar(
        [target],
        tmp_path,
        source_dataset_fingerprint="a" * 64,
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=2,
    )
    with pytest.raises(ValueError, match="指纹"):
        load_horizon_sidecar(
            path,
            horizon=1,
            source_dataset_fingerprint="b" * 64,
        )


def test_prepare_sidecar_reuses_manifest_h1_exactly(tmp_path) -> None:
    manifest_path = _feature_manifest(tmp_path)
    base = NextDayShardDataset(
        manifest_path,
        date_split=WalkForwardSplit.from_strings(
            train_start="2024-01-02",
            train_end="2024-01-05",
            val_start="2024-01-06",
            val_end="2024-01-08",
            test_start="2024-01-09",
            test_end="2024-01-10",
        ),
        split="train",
    )
    basic_root = tmp_path / "basic"
    basic_root.mkdir()
    dates = list(range(20240102, 20240111))
    prices = pa.table({"value": dates, "S0": [100.0] * len(dates)})
    pq.write_table(prices, basic_root / "open_data.parquet")
    pq.write_table(prices, basic_root / "close_data.parquet")
    benchmark = tmp_path / "benchmark.parquet"
    pq.write_table(
        pa.table(
            {
                "trade_date": dates,
                "open": [100.0] * len(dates),
                "close": [100.0] * len(dates),
            }
        ),
        benchmark,
    )

    sidecar_path = prepare_horizon_sidecar(
        manifest_path,
        basic_root,
        benchmark,
        tmp_path / "sidecar",
        horizons=(1,),
        min_cross_section=2,
    )
    loaded = load_horizon_sidecar(
        sidecar_path,
        horizon=1,
        source_dataset_fingerprint=base.dataset_fingerprint,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (row["symbol"], date.fromisoformat(row["trading_date"])): row for row in manifest["samples"]
    }
    for key, target in loaded.records.items():
        record = expected[key]
        assert target.entry_date == date.fromisoformat(record["label_date"])
        assert target.return_end_date == date.fromisoformat(record["label_date"])
        assert target.label == record["label"]
        assert target.target_return == record["target_return"]
