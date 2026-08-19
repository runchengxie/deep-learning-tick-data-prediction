"""事件流信号半衰期和交易转换诊断测试。"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.nextday.horizon_labels import HorizonTarget
from ticknet.research import signal_diagnostics as diagnostics_module
from ticknet.research import signal_diagnostics_cli as diagnostics_cli
from ticknet.research.portfolio import CostModel, PortfolioPolicy, evaluate_topk_portfolio
from ticknet.research.signal_diagnostics import (
    PolicyCandidate,
    SignalRow,
    calibrated_scores,
    evaluate_policy,
    half_life_diagnostics,
    load_horizon_maps,
    load_joint_signal_rows,
    load_materialized_signal_rows,
    make_portfolio_predictions,
    percentile_ranks,
    staggered_h5_evaluation,
)
from ticknet.research.signal_diagnostics_market import (
    MarketAttributes,
    build_market_attributes,
    reprice_dynamic_cost,
    reprice_staggered_dynamic_cost,
    risk_attribution,
)


def _signals(
    *,
    days: int = 8,
    symbols: int = 120,
    start: date = date(2025, 10, 8),
) -> list[SignalRow]:
    return [
        SignalRow(
            partition="validation" if day < 4 else "oos",
            trading_date=start + timedelta(days=day),
            symbol=f"{symbol:06d}",
            score=float(symbol + day % 2),
        )
        for day in range(days)
        for symbol in range(symbols)
    ]


def _targets(signals: list[SignalRow], horizon: int) -> dict[tuple[str, date], HorizonTarget]:
    result = {}
    for row in signals:
        value = (int(row.symbol) - 59.5) / 10_000.0
        result[(row.symbol, row.trading_date)] = HorizonTarget(
            symbol=row.symbol,
            trading_date=row.trading_date,
            entry_date=row.trading_date + timedelta(days=1),
            return_end_date=row.trading_date + timedelta(days=horizon),
            horizon=horizon,
            label=2 if int(row.symbol) >= 96 else (0 if int(row.symbol) < 24 else 1),
            raw_return=value + 0.001,
            benchmark_return=0.001,
            target_return=value,
        )
    return result


def test_half_life_reports_ic_spread_and_non_overlapping_anchors() -> None:
    signals = _signals(days=4)
    report = half_life_diagnostics(
        signals,
        {1: _targets(signals, 1), 2: _targets(signals, 2)},
    )

    assert report["1"]["mean_rank_ic"] == pytest.approx(1.0)
    assert report["1"]["mean_extreme_spread"] > 0
    assert report["1"]["mean_precision_at_k"] > 0
    assert len(report["2"]["non_overlapping_anchors"]) == 2


def test_rank_ema_resets_after_symbol_gap() -> None:
    rows = [
        SignalRow("validation", date(2025, 1, 1), "A", 2.0),
        SignalRow("validation", date(2025, 1, 1), "B", 1.0),
        SignalRow("validation", date(2025, 1, 2), "B", 2.0),
        SignalRow("validation", date(2025, 1, 2), "C", 1.0),
        SignalRow("validation", date(2025, 1, 3), "A", 1.0),
        SignalRow("validation", date(2025, 1, 3), "C", 2.0),
    ]
    ranks = percentile_ranks(rows, ema_alpha=0.5)

    assert ranks[(date(2025, 1, 2), "B")] == pytest.approx(0.5)
    assert ranks[(date(2025, 1, 3), "A")] == pytest.approx(0.0)


def test_validation_calibration_and_cash_threshold_are_applied_to_oos() -> None:
    signals = _signals()
    validation = [row for row in signals if row.partition == "validation"]
    oos = [row for row in signals if row.partition == "oos"]
    h5 = _targets(signals, 5)
    h1 = _targets(signals, 1)
    _validation_scores, oos_scores, report = calibrated_scores(
        validation,
        oos,
        h5,
        ema_alpha=1.0,
    )
    evaluation = evaluate_policy(
        oos,
        oos_scores,
        h1,
        PolicyCandidate(1.0, 0.003, 0.0),
    )

    assert report["validation_samples"] == 480
    assert min(oos_scores.values()) < max(oos_scores.values())
    assert evaluation.summary["risk_exposure"]["mean_cash_weight"] > 0
    assert max(row["positions"] for row in evaluation.daily) < 100


def test_staggered_h5_uses_one_fifth_capital_and_fixed_round_trip_cost() -> None:
    signals = _signals(days=5)
    targets = _targets(signals, 5)
    expected = {(row.trading_date, row.symbol): row.score for row in signals}
    evaluation = staggered_h5_evaluation(signals, expected, targets)

    assert evaluation["evaluated_cohorts"] == 5
    assert evaluation["mean_gross_exposure"] == pytest.approx(1.0)
    assert evaluation["mean_one_way_turnover"] == pytest.approx(0.2)
    assert evaluation["rows"][0]["transaction_cost"] == pytest.approx(0.0005)
    assert len(evaluation["sleeves"]) == 5


def test_dynamic_cost_and_risk_attribution_use_adv_attributes() -> None:
    signals = _signals(days=5)
    targets = _targets(signals, 1)
    expected = {(row.trading_date, row.symbol): row.score for row in signals}
    evaluation = evaluate_topk_portfolio(
        make_portfolio_predictions(signals, expected, targets),
        policy=PortfolioPolicy(top_k=100, min_symbols_per_day=100),
        cost_model=CostModel(),
    )
    attributes = {
        (row.trading_date, row.symbol): MarketAttributes(
            log_size=10.0 + int(row.symbol) / 100.0,
            adv20=20_000_000.0 + int(row.symbol) * 100_000.0,
            volatility20=0.01 + int(row.symbol) / 100_000.0,
        )
        for row in signals
    }
    dynamic = reprice_dynamic_cost(evaluation, attributes)
    risk = risk_attribution(evaluation, signals, attributes)
    staggered = staggered_h5_evaluation(signals, expected, _targets(signals, 5))
    staggered_dynamic = reprice_staggered_dynamic_cost(staggered, attributes)

    assert dynamic["trade_liquidity_coverage"] == pytest.approx(1.0)
    assert (
        dynamic["mean_transaction_cost"] > evaluation.summary["turnover"]["mean_transaction_cost"]
    )
    assert risk["status"] == "available"
    assert risk["industry"]["status"] == "unavailable"
    assert staggered_dynamic["trade_liquidity_coverage"] == pytest.approx(1.0)


def _write_manifest(root: Path, value: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def test_materialized_signal_loader_joins_verified_row_identity(tmp_path: Path) -> None:
    keys_root = tmp_path / "keys"
    scores_root = tmp_path / "scores"
    keys_root.mkdir()
    scores_root.mkdir()
    keys = pa.table(
        {
            "partition": ["validation", "oos"],
            "row_index": [0, 0],
            "trading_day": [20251008, 20251103],
            "symbol": ["000001", "000002"],
        }
    )
    pq.write_table(keys, keys_root / "sample-keys.parquet")
    common = {
        "materialized_dataset_fingerprint": "materialized",
        "source_dataset_fingerprint": "source",
        "locked_start": 20260101,
    }
    _write_manifest(
        keys_root,
        {
            "status": "complete",
            "mode": "eventstream_materialized_sample_keys",
            "contract": common,
            "dataset_fingerprint": "keys",
            "artifact": {
                "path": "sample-keys.parquet",
                "rows": 2,
                "bytes": (keys_root / "sample-keys.parquet").stat().st_size,
                "sha256": file_sha256(keys_root / "sample-keys.parquet"),
            },
        },
    )
    artifacts = []
    for partition, day, score in (("validation", 20251008, 0.1), ("oos", 20251103, 0.2)):
        path = scores_root / f"{partition}.parquet"
        pq.write_table(
            pa.table({"row_index": [0], "trading_day": [day], "score": [score]}),
            path,
        )
        artifacts.append(
            {
                "partition": partition,
                "path": path.name,
                "rows": 1,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    _write_manifest(
        scores_root,
        {
            "status": "complete",
            "mode": "eventstream_materialized_day_predictions",
            "contract": {**common, "checkpoint_sha256": "checkpoint"},
            "dataset_fingerprint": "scores",
            "artifacts": artifacts,
            "totals": {"rows": 2},
        },
    )

    rows, identity = load_materialized_signal_rows(keys_root, scores_root)

    assert [(row.partition, row.symbol, row.score) for row in rows] == [
        ("validation", "000001", 0.1),
        ("oos", "000002", 0.2),
    ]
    assert identity["checkpoint_sha256"] == "checkpoint"


def test_joint_loader_normalizes_partition_names(tmp_path: Path) -> None:
    paths = {}
    for partition, day in (("val", 20251008), ("test", 20251103)):
        path = tmp_path / f"{partition}.parquet"
        pq.write_table(
            pa.table({"trading_day": [day], "symbol": ["000001"], "score": [0.1]}),
            path,
        )
        paths[partition] = path

    rows = load_joint_signal_rows(paths)

    assert [row.partition for row in rows] == ["oos", "validation"]


def test_horizon_loader_filters_to_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = _signals(days=1, symbols=2)
    records = _targets(signals, 1)

    def fake_load(*_args, horizon, **_kwargs):
        return SimpleNamespace(
            sidecar_fingerprint="sidecar",
            records={
                key: HorizonTarget(
                    symbol=value.symbol,
                    trading_date=value.trading_date,
                    entry_date=value.entry_date,
                    return_end_date=value.return_end_date,
                    horizon=horizon,
                    label=value.label,
                    raw_return=value.raw_return,
                    benchmark_return=value.benchmark_return,
                    target_return=value.target_return,
                )
                for key, value in records.items()
            },
        )

    monkeypatch.setattr(diagnostics_module, "load_horizon_sidecar", fake_load)
    required = {(signals[0].symbol, signals[0].trading_date)}
    loaded, fingerprint = load_horizon_maps(
        "unused.json",
        source_dataset_fingerprint="source",
        horizons=(1, 2),
        required_keys=required,
    )

    assert fingerprint == "sidecar"
    assert set(loaded) == {1, 2}
    assert set(loaded[1]) == required


def _write_wide(path: Path, dates: list[int], symbols: list[str], values: np.ndarray) -> None:
    columns = {"value": dates}
    columns.update({symbol: values[:, index] for index, symbol in enumerate(symbols)})
    pq.write_table(pa.table(columns), path)


def test_market_attributes_apply_hundred_share_lot_size(tmp_path: Path) -> None:
    basic = tmp_path / "basic"
    basic.mkdir()
    symbols = ["000001"]
    dates = [
        int((date(2025, 1, 1) + timedelta(days=index)).strftime("%Y%m%d")) for index in range(25)
    ]
    close = np.arange(10.0, 35.0)[:, None]
    volume = np.full((25, 1), 100.0)
    size = np.full((25, 1), 1_000_000.0)
    _write_wide(basic / "close_data.parquet", dates, symbols, close)
    _write_wide(basic / "volume_data.parquet", dates, symbols, volume)
    _write_wide(basic / "total_mv_data.parquet", dates, symbols, size)
    signal_date = date(2025, 1, 25)

    attributes = build_market_attributes(
        basic,
        [SignalRow("oos", signal_date, "000001", 1.0)],
    )

    expected_adv = float(np.mean(np.arange(15.0, 35.0) * 100.0 * 100.0))
    assert attributes[(signal_date, "000001")].adv20 == pytest.approx(expected_adv)


def test_full_diagnostic_orchestration_writes_auditable_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjacent = _signals(start=date(2025, 10, 1))
    recent = _signals(start=date(2025, 11, 1))
    combined = [*adjacent, *recent]
    horizons = {horizon: _targets(combined, horizon) for horizon in range(1, 11)}
    attributes = {
        (row.trading_date, row.symbol): MarketAttributes(10.0, 100_000_000.0, 0.02)
        for row in combined
    }
    monkeypatch.setattr(
        diagnostics_cli,
        "load_materialized_signal_rows",
        lambda *_args: (
            adjacent,
            {
                "source_dataset_fingerprint": "source",
                "key_dataset_fingerprint": "keys",
                "score_dataset_fingerprint": "scores",
                "checkpoint_sha256": "checkpoint",
            },
        ),
    )
    monkeypatch.setattr(diagnostics_cli, "load_joint_signal_rows", lambda *_args, **_kwargs: recent)
    monkeypatch.setattr(
        diagnostics_cli,
        "load_horizon_maps",
        lambda *_args, **_kwargs: (horizons, "sidecar"),
    )
    monkeypatch.setattr(
        diagnostics_cli,
        "build_market_attributes",
        lambda _root, signals: {
            key: value
            for key, value in attributes.items()
            if key[0] in {row.trading_date for row in signals}
        },
    )
    recent_validation = tmp_path / "recent-validation.parquet"
    recent_oos = tmp_path / "recent-oos.parquet"
    recent_validation.write_bytes(b"validation")
    recent_oos.write_bytes(b"oos")
    output = tmp_path / "output"
    arguments = argparse.Namespace(
        adjacent_keys=tmp_path / "keys",
        adjacent_scores=tmp_path / "scores",
        recent_validation=recent_validation,
        recent_oos=recent_oos,
        horizon_sidecar=tmp_path / "sidecar.json",
        basic_root=tmp_path / "basic",
        output=output,
    )

    report = diagnostics_cli.run_diagnostics(arguments)

    assert report["status"] == "complete"
    assert len(json.loads((output / "policy-matrix.json").read_text())) == 27
    assert (output / "half-life.svg").is_file()
    assert (output / "policy-matrix.parquet").is_file()
    repeated_svg = output / "half-life-repeated.svg"
    diagnostics_cli._plot_half_life(
        json.loads((output / "half-life.json").read_text()),
        repeated_svg,
    )
    assert file_sha256(output / "half-life.svg") == file_sha256(repeated_svg)
    artifact = next(iter(report["artifacts"].values()))["adjacent_oos"]["daily"]
    assert artifact["path"].startswith("portfolio-details/")
    assert (output / artifact["path"]).is_file()
