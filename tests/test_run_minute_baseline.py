"""分钟 HGB 正式 prediction export 边界测试。"""

import argparse
from dataclasses import replace
from datetime import date

import numpy as np

from scripts.run_minute_baseline import (
    _complete_formal_samples,
    _formal_dataset_fingerprint,
    _load_config,
    _save_formal_test_predictions,
    _split_samples,
    _validate_formal_run,
)
from ticknet.nextday.formal_targets import FormalNextOpenTarget
from ticknet.nextday.minute_baseline import (
    FORMAL_TARGET_RETURN_CONTRACT,
    MinuteBaselineConfig,
    MinuteExtractionReport,
    MinuteSample,
)
from ticknet.nextday.splits import WalkForwardSplit
from ticknet.research.prediction_contract import validate_formal_prediction_artifact


def _target(symbol: str, *, in_universe: bool = True) -> FormalNextOpenTarget:
    return FormalNextOpenTarget(
        symbol=symbol,
        trading_date=date(2025, 7, 1),
        label_date=date(2025, 7, 2),
        return_end_date=date(2025, 7, 3),
        raw_return=0.01,
        portfolio_return=0.01,
        benchmark_return=0.002,
        target_return=0.008,
        label=2,
        can_buy=True,
        can_sell=not symbol.endswith("3"),
        in_universe=in_universe,
        execution_status="normal",
    )


def _config() -> MinuteBaselineConfig:
    return MinuteBaselineConfig(
        basic_root="basic",
        benchmark_path="benchmark",
        start_date="2025-01-01",
        end_date="2025-12-31",
        top_n=2,
        min_history_days=120,
        liquidity_lookback_days=20,
        min_liquidity_observations=15,
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=2,
        train_start="2023-01-01",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        target_return_contract=FORMAL_TARGET_RETURN_CONTRACT,
    )


class _Model:
    def predict_proba(self, features):
        assert features.shape == (2, 4)
        return np.asarray([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])


def test_shared_loader_preserves_formal_target_contract() -> None:
    config = _load_config("configs/nextday-minute-formal-2025.yaml")
    assert config.formal is True
    assert config.top_n == 400
    assert config.feature_source == "l2_cache"


def test_complete_formal_samples_imputes_missing_candidate() -> None:
    targets = [_target("000001"), _target("000002")]
    available = MinuteSample(
        trading_date=targets[0].trading_date,
        symbol=targets[0].symbol,
        label_date=targets[0].label_date,
        label=targets[0].label,
        target_return=targets[0].target_return,
        features=np.arange(4, dtype=np.float32),
        return_end_date=targets[0].return_end_date,
    )
    report = MinuteExtractionReport()
    completed = _complete_formal_samples(targets, [available], report)
    assert len(completed) == 2
    assert report.imputed_missing_samples == 1
    assert completed[1].feature_available is False
    assert np.isnan(completed[1].features).all()


def test_formal_split_purges_return_end_date() -> None:
    sample = MinuteSample(
        trading_date=date(2024, 12, 30),
        symbol="000001",
        label_date=date(2024, 12, 31),
        return_end_date=date(2025, 1, 2),
        label=1,
        target_return=0.0,
        features=np.ones(4),
    )
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-06-30",
        test_start="2025-07-01",
        test_end="2025-12-31",
    )
    assert _split_samples([sample], split) == {"train": [], "val": [], "test": []}


def test_formal_run_requires_exact_top400_and_explicit_export() -> None:
    args = argparse.Namespace(evaluate_test=True, save_predictions="predictions.parquet")
    with np.testing.assert_raises_regex(ValueError, "top_n=400"):
        _validate_formal_run(_config(), args)

    formal = replace(_config(), top_n=400, min_symbols_per_day=400)
    _validate_formal_run(formal, args)
    with np.testing.assert_raises_regex(ValueError, "--evaluate-test"):
        _validate_formal_run(
            formal,
            argparse.Namespace(evaluate_test=None, save_predictions="predictions.parquet"),
        )


def test_formal_export_writes_valid_contract_and_stable_fingerprint(tmp_path) -> None:
    targets = [_target("000001"), _target("000002"), _target("000003", in_universe=False)]
    items = [
        MinuteSample(
            trading_date=target.trading_date,
            symbol=target.symbol,
            label_date=target.label_date,
            return_end_date=target.return_end_date,
            label=target.label,
            target_return=target.target_return,
            features=np.full(4, index, dtype=np.float32),
            feature_available=index == 0,
        )
        for index, target in enumerate(targets[:2])
    ]
    fingerprint = _formal_dataset_fingerprint(_config(), items, targets)
    assert fingerprint == _formal_dataset_fingerprint(_config(), list(reversed(items)), targets)

    path = tmp_path / "formal.parquet"
    report = _save_formal_test_predictions(
        items,
        targets,
        _Model(),
        path,
        dataset_fingerprint=fingerprint,
        expected_universe_size=2,
    )
    assert report["candidate_row_count"] == 2
    assert report["status_only_row_count"] == 1
    validated = validate_formal_prediction_artifact(
        path,
        expected_universe_size=2,
        expected_dataset_fingerprint=fingerprint,
    )
    assert validated.cannot_sell_count == 1
