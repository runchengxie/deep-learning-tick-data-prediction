"""冻结 embedding 下游排序指标的确定性测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ticknet.nextday.embedding_comparison as comparison
from ticknet.nextday.embedding_comparison import (
    ComparisonConfig,
    _ranking_metrics,
    _relevance_by_day,
    _Rows,
    load_comparison_config,
    run_comparison,
)
from ticknet.nextday.formal_targets import FORMAL_TARGET_RETURN_CONTRACT, FormalNextOpenTarget
from ticknet.nextday.minute_baseline import MinuteBaselineConfig, MinuteSample, TargetBuildBundle
from ticknet.nextday.minute_materialization import MaterializedFeatureLoad


def _rows() -> _Rows:
    samples = []
    keys = []
    returns = []
    for day_offset in range(3):
        trading_date = date(2025, 11, 3) + timedelta(days=day_offset)
        for symbol_index in range(120):
            symbol = f"{symbol_index:06d}"
            target_return = float(symbol_index) / 10_000
            keys.append((trading_date, symbol))
            returns.append(target_return)
            samples.append(
                MinuteSample(
                    trading_date=trading_date,
                    symbol=symbol,
                    label_date=trading_date + timedelta(days=1),
                    label=0 if symbol_index < 24 else 2 if symbol_index >= 96 else 1,
                    target_return=target_return,
                    features=np.asarray([symbol_index], dtype=np.float32),
                    return_end_date=trading_date + timedelta(days=2),
                )
            )
    return _Rows(
        keys=tuple(keys),
        samples=tuple(samples),
        minute=np.asarray(returns, dtype=np.float32)[:, None],
        embedding=np.asarray(returns, dtype=np.float32)[:, None],
        labels=np.asarray([sample.label for sample in samples]),
        target_returns=np.asarray(returns),
    )


def test_relevance_and_ranking_metrics_reward_correct_order() -> None:
    rows = _rows()
    config = ComparisonConfig(
        train_start="2025-08-01",
        train_end="2025-10-31",
        validation_start="2025-11-01",
        validation_end="2025-11-30",
        oos_start="2025-12-01",
        oos_end="2025-12-31",
        min_symbols_per_day=100,
    )
    relevance = _relevance_by_day(rows, 5)
    metrics = _ranking_metrics(rows, rows.target_returns, config)

    assert set(np.unique(relevance)) == {0, 1, 2, 3, 4}
    assert metrics["daily_rank_ic_mean"] == 1.0
    assert np.isclose(metrics["ndcg_at_50"], 1.0)
    assert metrics["precision_at_100"] == 1.0


def test_comparison_config_yaml_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "comparison.yaml"
    path.write_text(
        """train_start: '2025-08-01'
train_end: '2025-10-31'
validation_start: '2025-11-01'
validation_end: '2025-11-30'
oos_start: '2025-12-01'
oos_end: '2025-12-31'
top_ks: [10, 20]
min_symbols_per_day: 50
""",
        encoding="utf-8",
    )
    base = load_comparison_config(path)
    assert base.top_ks == (10, 20)

    path.write_text(path.read_text(encoding="utf-8") + "unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知字段"):
        load_comparison_config(path)

    invalid_values = (
        (replace(base, oos_end="2026-01-02"), "locked"),
        (replace(base, top_ks=(50,), min_symbols_per_day=20), "min_symbols"),
        (replace(base, relevance_levels=1), "relevance_levels"),
        (replace(base, top_ks=()), "top_ks"),
        (replace(base, cost_bps=-1), "成本"),
    )
    for invalid, message in invalid_values:
        with pytest.raises(ValueError, match=message):
            invalid.validate()


def _comparison_fixture() -> tuple[list[MinuteSample], list[FormalNextOpenTarget]]:
    signal_dates = (
        date(2025, 8, 1),
        date(2025, 8, 4),
        date(2025, 11, 3),
        date(2025, 11, 4),
        date(2025, 12, 1),
        date(2025, 12, 2),
    )
    samples: list[MinuteSample] = []
    targets: list[FormalNextOpenTarget] = []
    for day_index, trading_date in enumerate(signal_dates):
        for symbol_index in range(60):
            symbol = f"{symbol_index:06d}"
            target_return = (symbol_index - 30) / 10_000 + day_index / 100_000
            label = 0 if symbol_index < 12 else 2 if symbol_index >= 48 else 1
            label_date = trading_date + timedelta(days=1)
            return_end_date = trading_date + timedelta(days=2)
            samples.append(
                MinuteSample(
                    trading_date=trading_date,
                    symbol=symbol,
                    label_date=label_date,
                    label=label,
                    target_return=target_return,
                    features=np.asarray(
                        [target_return, symbol_index % 7, day_index], dtype=np.float32
                    ),
                    return_end_date=return_end_date,
                )
            )
            targets.append(
                FormalNextOpenTarget(
                    symbol=symbol,
                    trading_date=trading_date,
                    label_date=label_date,
                    return_end_date=return_end_date,
                    label=label,
                    raw_return=target_return + 0.001,
                    target_return=target_return,
                    portfolio_return=target_return + 0.001,
                    benchmark_return=0.001,
                    can_buy=True,
                    can_sell=True,
                    in_universe=True,
                )
            )
    return samples, targets


def _minute_config() -> MinuteBaselineConfig:
    return MinuteBaselineConfig(
        basic_root="unused",
        benchmark_path="unused",
        start_date="2021-07-01",
        end_date="2025-12-31",
        top_n=400,
        min_history_days=120,
        liquidity_lookback_days=20,
        min_liquidity_observations=15,
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=400,
        train_start="2021-07-01",
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-06-30",
        test_start="2025-07-01",
        test_end="2025-12-31",
        target_return_contract=FORMAL_TARGET_RETURN_CONTRACT,
    )


def _write_exposures(path: Path, samples: list[MinuteSample]) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "trading_date": sample.trading_date,
                    "symbol": sample.symbol,
                    "industry": f"industry-{int(sample.symbol) % 3}",
                    "size": float(int(sample.symbol) + 1),
                    "liquidity": float(int(sample.symbol) % 11 + 1),
                    "volatility": float(int(sample.symbol) % 13 + 1),
                }
                for sample in samples
            ]
        ),
        path,
    )


def test_run_comparison_small_end_to_end(monkeypatch, tmp_path: Path) -> None:
    samples, targets = _comparison_fixture()
    materialized = MaterializedFeatureLoad(
        samples=samples,
        manifest_path=tmp_path / "minute-manifest.json",
        materialization_identity="minute-identity",
        manifest_fingerprint="minute-fingerprint",
        shard_count=1,
    )
    monkeypatch.setattr(
        comparison,
        "build_target_bundle",
        lambda _config: TargetBuildBundle(targets=targets, universe={}),
    )
    monkeypatch.setattr(
        comparison,
        "load_materialized_minute_features",
        lambda *_args, **_kwargs: materialized,
    )
    keys = [(sample.trading_date, sample.symbol) for sample in samples]
    seed0 = {
        key: np.asarray([sample.target_return, int(sample.symbol) % 5], dtype=np.float32)
        for key, sample in zip(keys, samples, strict=True)
    }
    seed1 = {
        key: np.asarray([sample.target_return * 0.9, int(sample.symbol) % 3], dtype=np.float32)
        for key, sample in zip(keys, samples, strict=True)
    }
    manifests = [
        {
            "dataset_fingerprint": f"embedding-{seed}",
            "contract": {
                "close_cache_fingerprint": "shared-close-cache",
                "encoder": {"seed": seed},
            },
        }
        for seed in range(2)
    ]
    monkeypatch.setattr(
        comparison,
        "_prepare_embedding_sets",
        lambda _roots: (
            [("seed0", seed0, manifests[0]), ("seed1", seed1, manifests[1])],
            set(keys),
            "shared-close-cache",
        ),
    )
    exposures = tmp_path / "exposures.parquet"
    _write_exposures(exposures, samples)
    config = ComparisonConfig(
        train_start="2025-08-01",
        train_end="2025-10-31",
        validation_start="2025-11-01",
        validation_end="2025-11-30",
        oos_start="2025-12-01",
        oos_end="2025-12-31",
        min_symbols_per_day=50,
        top_ks=(10, 20),
        hgb_max_iter=3,
        lambdamart_estimators=3,
        lambdamart_early_stopping_rounds=1,
    )

    result = run_comparison(
        minute_config=_minute_config(),
        minute_features_root=tmp_path,
        embedding_roots=[tmp_path / "seed0", tmp_path / "seed1"],
        comparison_config=config,
        output_root=tmp_path / "output",
        exposure_path=exposures,
    )

    assert result["status"] == "complete"
    assert result["identity"]["embedding_combination"].startswith("independent")
    assert "lambdamart/combined/prediction_ensemble" in result["results"]
    exposures_result = result["results"]["hgb/combined/seed0"]["test"]["exposures"]
    assert exposures_result["status"] == "complete"
    assert (tmp_path / "output" / "comparison.json").is_file()
    assert len(list((tmp_path / "output" / "predictions").rglob("*.parquet"))) == 30
