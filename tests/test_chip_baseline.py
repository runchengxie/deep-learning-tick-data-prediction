from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.nextday.chip_baseline import load_daily_chip_bars, run_chip_comparison
from ticknet.nextday.chip_features import ChipFeatures
from ticknet.nextday.formal_targets import FormalNextOpenTarget
from ticknet.nextday.splits import WalkForwardSplit
from ticknet.research.prediction_contract import validate_formal_prediction_artifact


def test_loader_rejects_ambiguous_units_and_excludes_locked_prices(tmp_path):
    path = tmp_path / "bars.parquet"
    rows = [
        {
            "symbol": "000001.SZ",
            "trading_date": date(2025, 12, 31),
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "vwap": 10.0,
            "volume": 100.0,
            "turnover": 0.1,
            "adj_factor": 1.0,
        },
        {
            "symbol": "000001.SZ",
            "trading_date": date(2026, 1, 1),
            "open": -999.0,
            "high": -999.0,
            "low": -999.0,
            "close": -999.0,
            "vwap": -999.0,
            "volume": 100.0,
            "turnover": -999.0,
            "adj_factor": -999.0,
        },
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=1)
    with pytest.raises(ValueError, match="contract"):
        load_daily_chip_bars(path, end_date=date(2025, 12, 31))
    table = table.replace_schema_metadata(
        {
            b"ticknet.chip_daily_contract": b"raw_ohlcv_shares_fractional_turnover_v1",
            b"ticknet.adjustment_contract": b"point_in_time_ratio",
        }
    )
    pq.write_table(table, path, row_group_size=1)
    bars = load_daily_chip_bars(path, end_date=date(2025, 12, 31))
    assert len(bars) == 1
    assert bars[0].vwap == 10
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="crosses the locked"):
        load_daily_chip_bars(path, end_date=date(2025, 12, 31))
    with pytest.raises(ValueError, match="locked"):
        load_daily_chip_bars(path, end_date=date(2026, 1, 1))


def make_targets():
    result = []
    features = {}
    for day in range(18):
        signal = date(2024, 1, 1) + timedelta(days=day)
        for index in range(3):
            symbol = f"{index:06}.SZ"
            target = FormalNextOpenTarget(
                symbol=symbol,
                trading_date=signal,
                label_date=signal + timedelta(days=1),
                label=index,
                target_return=(index - 1) * 0.01,
                raw_return=(index - 1) * 0.01,
                benchmark_return=0.0,
                return_end_date=signal + timedelta(days=2),
                portfolio_return=(index - 1) * 0.01,
                can_buy=True,
                can_sell=True,
                in_universe=True,
            )
            result.append(target)
            features[(signal, symbol)] = ChipFeatures(
                np.full(18, index, dtype=float),
                np.full(5, index, dtype=float),
                signal - timedelta(days=1),
                True,
                0.01,
            )
    return result, features


def test_real_hgb_comparison_purges_boundaries_and_exports_cost_contract(tmp_path):
    targets, features = make_targets()
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-01-06",
        val_start="2024-01-07",
        val_end="2024-01-12",
        test_start="2024-01-13",
        test_end="2024-01-18",
    )
    missing_key = (date(2024, 1, 7), "000000.SZ")
    features[missing_key] = ChipFeatures(np.full(18, np.nan), np.full(5, np.nan), None, False, 1.0)
    result = run_chip_comparison(
        targets,
        features,
        split,
        tmp_path,
        fingerprint="a" * 64,
        top_n=3,
        top_k=1,
        min_symbols_per_day=3,
        max_iter=2,
        min_samples_leaf=2,
        evaluate_test=False,
    )
    assert result["split_rows"] == {"train": 12, "val": 12, "test": 12}
    assert result["coverage"]["val"]["available_rows"] == 11
    for name in ("price", "chip", "combined"):
        path = tmp_path / name / "val-predictions.parquet"
        report = validate_formal_prediction_artifact(path, expected_universe_size=3)
        assert report.row_count == 12
        assert report.date_range[1] < "2024-01-13"
        assert result["models"][name]["val"]["portfolio"]["evaluated_dates"] == 4
        assert not (tmp_path / name / "test-predictions.parquet").exists()
        assert Path(result["models"][name]["checkpoint"]).is_file()


def test_split_cannot_reach_locked_period_even_before_training(tmp_path):
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-06-30",
        test_start="2025-07-01",
        test_end="2026-01-01",
    )
    with pytest.raises(ValueError, match="locked"):
        run_chip_comparison(
            [], {}, split, tmp_path, fingerprint="a" * 64, top_n=3, top_k=1, min_symbols_per_day=3
        )


def test_minute_pairing_preserves_blocked_off_universe_holding_and_seen_oos(tmp_path):
    from dataclasses import replace

    targets, features = make_targets()
    held_key = (date(2024, 1, 8), "000002.SZ")
    target = next(item for item in targets if (item.trading_date, item.symbol) == held_key)
    targets[targets.index(target)] = replace(target, in_universe=False, can_sell=False)
    replacement = replace(target, symbol="000003.SZ")
    targets.append(replacement)
    features[(replacement.trading_date, replacement.symbol)] = features[held_key]
    missing_key = (date(2024, 1, 7), "000000.SZ")
    features[missing_key] = ChipFeatures(np.full(18, np.nan), np.full(5, np.nan), None, False, 1.0)
    minute = {key: np.asarray([index % 3], dtype=float) for index, key in enumerate(features)}
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-01-06",
        val_start="2024-01-07",
        val_end="2024-01-12",
        test_start="2024-01-13",
        test_end="2024-01-18",
    )
    result = run_chip_comparison(
        targets,
        features,
        split,
        tmp_path,
        fingerprint="b" * 64,
        top_n=3,
        top_k=1,
        min_symbols_per_day=3,
        max_iter=2,
        min_samples_leaf=2,
        evaluate_test=True,
        minute_features=minute,
    )
    assert set(result["models"]) == {"price", "chip", "combined", "minute", "minute_chip"}
    for name in result["models"]:
        report = validate_formal_prediction_artifact(
            tmp_path / name / "val-predictions.parquet", expected_universe_size=3
        )
        assert report.status_only_row_count == 1
        assert (tmp_path / name / "test-predictions.parquet").is_file()
    holdings = pq.read_table(tmp_path / "chip" / "val-predictions" / "holdings.parquet").to_pylist()
    assert any(
        row["symbol"] == "000002.SZ" and row["label_date"] == "2024-01-09" for row in holdings
    )
    rows = pq.read_table(tmp_path / "minute" / "val-predictions.parquet").to_pylist()
    missing = next(row for row in rows if (row["trading_date"], row["symbol"]) == missing_key)
    assert missing["feature_available"] is True
    assert missing["chip_feature_available"] is False


def test_output_reuse_cannot_overwrite_a_previous_experiment(tmp_path):
    sentinel = tmp_path / "model.pkl"
    sentinel.write_bytes(b"previous experiment")
    targets, features = make_targets()
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-01-06",
        val_start="2024-01-07",
        val_end="2024-01-12",
        test_start="2024-01-13",
        test_end="2024-01-18",
    )
    with pytest.raises(ValueError, match="empty"):
        run_chip_comparison(
            targets,
            features,
            split,
            tmp_path,
            fingerprint="a" * 64,
            top_n=3,
            top_k=1,
            min_symbols_per_day=3,
        )
    assert sentinel.read_bytes() == b"previous experiment"


def _write_cli_assets(tmp_path):
    basic = tmp_path / "basic"
    basic.mkdir()
    dates = [date(2024, 1, 1) + timedelta(days=day) for day in range(24)]
    symbols = ("000001", "000002", "000003")
    prices = {
        symbol: [10 * (1 + 0.005 * (index - 1)) ** day for day in range(24)]
        for index, symbol in enumerate(symbols)
    }
    for field in ("open", "high", "low", "close", "volume", "st"):
        multiplier = {"high": 1.02, "low": 0.98}.get(field, 1.0)
        values = {
            symbol: (
                [100.0] * 24
                if field == "volume"
                else [0.0] * 24
                if field == "st"
                else [price * multiplier for price in prices[symbol]]
            )
            for symbol in symbols
        }
        pq.write_table(
            pa.table({"value": [int(day.strftime("%Y%m%d")) for day in dates], **values}),
            basic / f"{field}_data.parquet",
        )
    benchmark = tmp_path / "benchmark.parquet"
    pq.write_table(
        pa.table({"trade_date": [day.strftime("%Y%m%d") for day in dates], "open": [1000.0] * 24}),
        benchmark,
    )
    daily = tmp_path / "daily.parquet"
    rows = [
        {
            "symbol": symbol,
            "trading_date": day,
            "open": prices[symbol][index],
            "high": prices[symbol][index] * 1.02,
            "low": prices[symbol][index] * 0.98,
            "close": prices[symbol][index],
            "vwap": prices[symbol][index],
            "volume": 100.0,
            "turnover": 0.1,
            "adj_factor": 1.0,
        }
        for symbol in symbols
        for index, day in enumerate(dates)
    ]
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {
            b"ticknet.chip_daily_contract": b"raw_ohlcv_shares_fractional_turnover_v1",
            b"ticknet.adjustment_contract": b"point_in_time_ratio",
        }
    )
    pq.write_table(table, daily)
    return basic, benchmark, daily


def test_cli_runs_daily_asset_through_formal_targets_features_and_artifacts(tmp_path, capsys):
    import json

    import yaml

    from scripts.run_chip_baseline import main

    basic, benchmark, daily = _write_cli_assets(tmp_path)
    config = tmp_path / "formal.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "basic_root": str(basic),
                "benchmark_path": str(benchmark),
                "start_date": "2024-01-02",
                "end_date": "2024-01-24",
                "top_n": 3,
                "min_history_days": 1,
                "liquidity_lookback_days": 1,
                "min_liquidity_observations": 1,
                "min_cross_section": 3,
                "min_symbols_per_day": 3,
                "train_start": "2024-01-02",
                "train_end": "2024-01-10",
                "val_start": "2024-01-11",
                "val_end": "2024-01-17",
                "test_start": "2024-01-18",
                "test_end": "2024-01-24",
                "target_return_contract": "next_open_to_following_open",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    main(
        [
            "--config",
            str(config),
            "--daily-bars",
            str(daily),
            "--output-dir",
            str(output),
            "--warmup-days",
            "2",
            "--max-iter",
            "2",
            "--min-samples-leaf",
            "2",
            "--top-k",
            "1",
            "--evaluate-test",
        ]
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["coverage"]["val"]["available_rows"] == result["split_rows"]["val"]
    assert len(result["provenance"]["daily_content_fingerprint"]) == 64
    assert len(result["source_sha256"]) == 3
    assert result["settings"]["early_stopping"] is False
    for name in ("price", "chip", "combined"):
        validate_formal_prediction_artifact(
            output / name / "test-predictions.parquet", expected_universe_size=3
        )
    assert (
        json.loads(capsys.readouterr().out)["dataset_fingerprint"] == result["dataset_fingerprint"]
    )
