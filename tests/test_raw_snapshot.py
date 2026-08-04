"""真实沪深 snapshot 适配器的合成数据测试。"""

import json
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from deeplob.nextday.raw_snapshot import (
    RAW_FEATURE_COLUMNS,
    DailyPanel,
    SnapshotPreparationConfig,
    build_dynamic_universe,
    load_snapshot_config,
    normalize_lob_events,
    prepare_snapshot_dataset,
)


def _lob_row(mid: float, size: float) -> list[float]:
    row = []
    for level in range(1, 11):
        row.extend((mid + level, size + level, mid - level, size + level + 1))
    return row


def test_fixed_lob_normalization_preserves_tick_sequence():
    raw = np.asarray([_lob_row(100, 9), _lob_row(101, 19)], dtype=np.float64)
    normalized = normalize_lob_events(raw, price_scale_bps=100, volume_log_scale=10)
    assert normalized.shape == (2, 40)
    assert normalized.dtype == np.float32
    assert normalized[0, 0] == pytest.approx(1.0)
    assert normalized[0, 2] == pytest.approx(-1.0)
    assert normalized[1, 0] > normalized[0, 0]
    assert normalized[1, 1] > normalized[0, 1]


def test_dynamic_universe_uses_only_prior_days():
    dates = tuple(date(2024, 1, 2) + timedelta(days=offset) for offset in range(4))
    symbols = ("000001", "000002")
    prices = np.full((4, 2), 10.0)
    volumes = np.asarray([[100.0, 1.0], [100.0, 1.0], [1.0, 10_000.0], [1.0, 1.0]])
    open_panel = DailyPanel(dates, symbols, prices.copy())
    close_panel = DailyPanel(dates, symbols, prices.copy())
    volume_panel = DailyPanel(dates, symbols, volumes)

    universe = build_dynamic_universe(
        open_panel,
        close_panel,
        volume_panel,
        start_date=dates[2],
        end_date=dates[3],
        top_n=1,
        min_history_days=1,
        liquidity_lookback_days=1,
        min_liquidity_observations=1,
    )
    assert universe[dates[2]] == ("000001",)
    assert universe[dates[3]] == ("000002",)


def test_snapshot_yaml_loads_and_cli_values_override_it(tmp_path):
    config_path = tmp_path / "snapshot.yaml"
    config_path.write_text(
        "\n".join(
            [
                "snapshot_root: /raw/snapshot",
                "basic_root: /raw/basic",
                "benchmark_path: /reference/benchmark.parquet",
                "output_dir: ./data/output",
                "top_n: 10",
                "price_scale_bps: 50.0",
            ]
        ),
        encoding="utf-8",
    )
    config = load_snapshot_config(
        ["--config", str(config_path), "--top-n", "5", "--storage-dtype", "float32"]
    )
    assert config.top_n == 5
    assert config.price_scale_bps == 50.0
    assert config.storage_dtype == "float32"


def _write_wide(path, dates, values_by_symbol):
    content = {"value": [int(value.strftime("%Y%m%d")) for value in dates]}
    content.update(values_by_symbol)
    pq.write_table(pa.table(content), path)


def test_prepare_snapshot_dataset_writes_float16_end_to_end_shards(tmp_path):
    dates = tuple(date(2024, 1, day) for day in (2, 3, 4, 5))
    symbols = ("000001", "000002")
    basic = tmp_path / "basic"
    basic.mkdir()
    _write_wide(basic / "open_data.parquet", dates, {symbol: [100.0] * 4 for symbol in symbols})
    _write_wide(
        basic / "close_data.parquet",
        dates,
        {"000001": [100.0, 100.0, 98.0, 102.0], "000002": [100.0, 100.0, 102.0, 98.0]},
    )
    _write_wide(
        basic / "volume_data.parquet",
        dates,
        {symbol: [1000.0] * 4 for symbol in symbols},
    )

    benchmark = tmp_path / "benchmark.parquet"
    pq.write_table(
        pa.table(
            {
                "trade_date": [value.strftime("%Y%m%d") for value in dates],
                "open": [100.0] * 4,
                "close": [100.0] * 4,
            }
        ),
        benchmark,
    )

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    rows = {"ticker": [], "TradingDay": [], "time_ms": []}
    rows.update({column: [] for column in RAW_FEATURE_COLUMNS})
    for symbol in symbols:
        for trading_date in dates[1:3]:
            for event_index in range(4):
                rows["ticker"].append(symbol)
                rows["TradingDay"].append(int(trading_date.strftime("%Y%m%d")))
                rows["time_ms"].append(18_000_000 + event_index * 1000)
                values = _lob_row(100 + event_index, 10)
                if event_index == 3:
                    values[0] = 0.0
                for column, value in zip(
                    RAW_FEATURE_COLUMNS,
                    values,
                    strict=True,
                ):
                    rows[column].append(value)
    pq.write_table(
        pa.table(rows),
        snapshot_root / "snapshot_202401.parquet",
        row_group_size=4,
    )

    output = tmp_path / "output"
    config = SnapshotPreparationConfig(
        snapshot_root=str(snapshot_root),
        basic_root=str(basic),
        benchmark_path=str(benchmark),
        output_dir=str(output),
        start_date="2024-01-03",
        end_date="2024-01-04",
        chunks_per_sample=1,
        chunk_size=2,
        min_valid_events=2,
        top_n=2,
        min_history_days=1,
        liquidity_lookback_days=1,
        min_liquidity_observations=1,
        min_cross_section=2,
        samples_per_shard=2,
    )
    manifest_path, audit = prepare_snapshot_dataset(config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dtype"] == "float16"
    assert manifest["metadata"]["signal_time_ms"] == 19_500_000
    assert len(manifest["samples"]) == 4
    assert audit["extraction"]["written_samples"] == 4
    assert audit["extraction"]["invalid_lob_rows"] == 4
    assert {sample["valid_events"] for sample in manifest["samples"]} == {2}
    assert np.load(output / manifest["shards"][0]["path"]).dtype == np.float16
