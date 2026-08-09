"""正式分钟聚合特征物化、恢复与加载测试。"""

from dataclasses import replace
from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.nextday.formal_targets import FORMAL_TARGET_RETURN_CONTRACT, FormalNextOpenTarget
from ticknet.nextday.minute_baseline import MinuteBaselineConfig, MinuteExtractionReport
from ticknet.nextday.minute_materialization import (
    load_materialized_minute_features,
    materialize_minute_features,
)


def _target(symbol: str, trading_date: date) -> FormalNextOpenTarget:
    return FormalNextOpenTarget(
        symbol=symbol,
        trading_date=trading_date,
        label_date=date.fromordinal(trading_date.toordinal() + 1),
        return_end_date=date.fromordinal(trading_date.toordinal() + 2),
        raw_return=0.01,
        portfolio_return=0.01,
        benchmark_return=0.002,
        target_return=0.008,
        label=2,
        can_buy=True,
        can_sell=True,
        in_universe=True,
        execution_status="normal",
    )


def _config(l2_root) -> MinuteBaselineConfig:
    return MinuteBaselineConfig(
        basic_root="basic",
        benchmark_path="benchmark",
        start_date="2025-01-01",
        end_date="2025-12-31",
        top_n=2,
        min_history_days=1,
        liquidity_lookback_days=1,
        min_liquidity_observations=1,
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=2,
        train_start="2025-01-01",
        train_end="2025-06-30",
        val_start="2025-07-01",
        val_end="2025-09-30",
        test_start="2025-10-01",
        test_end="2025-12-31",
        l2_root=str(l2_root),
        feature_source="l2_cache",
        window_minutes=2,
        min_window_minutes=2,
        target_return_contract=FORMAL_TARGET_RETURN_CONTRACT,
    )


def _write_l2(root) -> None:
    rows = [
        (20250102, "000001", 1, 1.0),
        (20250102, "000001", 2, 2.0),
        (20250203, "000001", 1, 3.0),
        (20250203, "000001", 2, 5.0),
    ]
    for modality in ("snapshot", "order", "trade"):
        path = root / "yearly" / modality / "2025.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "date": [row[0] for row in rows],
                    "ticker": [row[1] for row in rows],
                    "minute": [row[2] for row in rows],
                    f"{modality}__value": [row[3] for row in rows],
                }
            ),
            path,
            row_group_size=2,
        )


def test_monthly_materialization_resumes_and_loads_exact_targets(tmp_path) -> None:
    l2_root = tmp_path / "l2"
    _write_l2(l2_root)
    config = _config(l2_root)
    targets = [
        _target("000001", date(2025, 1, 2)),
        _target("000002", date(2025, 1, 2)),
        _target("000001", date(2025, 2, 3)),
    ]
    output = tmp_path / "materialized"
    output.mkdir()
    (output / "manifest.json.tmp").write_text("interrupted", encoding="utf-8")
    updates = []

    partial = materialize_minute_features(
        config,
        targets,
        output,
        periods=["2025-01"],
        on_period=updates.append,
    )
    assert partial["status"] == "in_progress"
    assert partial["summary"]["completed_periods"] == 1
    assert partial["summary"]["imputed_feature_rows"] == 1
    assert updates[-1]["resumed"] is False
    january_sha = partial["shards"][0]["sha256"]

    resumed = materialize_minute_features(
        config,
        targets,
        output,
        periods=["2025-01"],
        on_period=updates.append,
    )
    assert resumed["shards"][0]["sha256"] == january_sha
    assert updates[-1]["resumed"] is True
    with pytest.raises(ValueError, match="尚未完成"):
        load_materialized_minute_features(config, targets, output, MinuteExtractionReport())

    complete = materialize_minute_features(config, targets, output)
    assert complete["status"] == "complete"
    assert complete["summary"]["completed_periods"] == 2
    report = MinuteExtractionReport()
    loaded = load_materialized_minute_features(config, targets, output, report)
    assert len(loaded.samples) == 3
    assert loaded.shard_count == 2
    assert report.materialized_shards == 2
    assert report.materialized_rows == 3
    assert report.written_samples == 2
    assert report.imputed_missing_samples == 1
    missing = next(sample for sample in loaded.samples if sample.symbol == "000002")
    assert missing.feature_available is False
    assert missing.features.shape == (12,)
    assert np.isnan(missing.features).all()

    relabelled = [replace(target, label=0, target_return=-0.01) for target in targets]
    reloaded = load_materialized_minute_features(
        config,
        relabelled,
        output,
        MinuteExtractionReport(),
    )
    assert {sample.label for sample in reloaded.samples} == {0}
    assert {sample.target_return for sample in reloaded.samples} == {-0.01}


def test_materialized_shard_checksum_and_no_resume_are_enforced(tmp_path) -> None:
    l2_root = tmp_path / "l2"
    _write_l2(l2_root)
    config = _config(l2_root)
    targets = [_target("000001", date(2025, 1, 2))]
    output = tmp_path / "materialized"
    manifest = materialize_minute_features(config, targets, output)

    with pytest.raises(ValueError, match="--no-resume"):
        materialize_minute_features(config, targets, output, resume=False)
    shard = output / manifest["shards"][0]["path"]
    with shard.open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_materialized_minute_features(config, targets, output, MinuteExtractionReport())
