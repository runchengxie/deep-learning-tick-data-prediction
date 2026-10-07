"""Paired CPU HGB experiments on lagged price and age-layer chip summaries."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from ticknet.nextday.chip_features import (
    CHIP_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    ChipFeatures,
    DailyChipBar,
)
from ticknet.nextday.formal_targets import FormalNextOpenTarget
from ticknet.nextday.metrics import evaluate_predictions
from ticknet.nextday.splits import WalkForwardSplit
from ticknet.research.portfolio import (
    CostModel,
    PortfolioPolicy,
    evaluate_topk_portfolio,
    load_portfolio_predictions,
    write_portfolio_artifacts,
)
from ticknet.research.prediction_contract import (
    attach_formal_prediction_metadata,
    validate_formal_prediction_artifact,
)
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import file_sha256

DAILY_CONTRACT = b"raw_ohlcv_shares_fractional_turnover_v1"


def load_daily_chip_bars(path: Path, *, end_date: date) -> list[DailyChipBar]:
    """Read a published long-table daily asset with explicit units and PIT factors.

    Row-group date statistics exclude locked values before OHLCV reads.
    The date field must be Arrow date32; source factor revisions belong upstream.
    """
    if end_date >= ResearchProtocol().locked_begin:
        raise ValueError("daily chip reader cannot enter the locked period")
    dataset = ds.dataset(path, format="parquet")
    metadata = dataset.schema.metadata or {}
    if (
        metadata.get(b"ticknet.chip_daily_contract") != DAILY_CONTRACT
        or metadata.get(b"ticknet.adjustment_contract") != b"point_in_time_ratio"
    ):
        raise ValueError(
            "daily chip contract must declare shares, fractional turnover and PIT factors"
        )
    required = [field.name for field in fields(DailyChipBar)]
    if set(required) - set(dataset.schema.names):
        raise ValueError(f"daily chip asset requires columns {required}")
    if dataset.schema.field("trading_date").type != pa.date32():
        raise ValueError("trading_date must be Arrow date32")
    bars: list[DailyChipBar] = []
    locked_begin = ResearchProtocol().locked_begin
    for file in dataset.files:
        parquet = pq.ParquetFile(file)
        date_index = parquet.schema_arrow.get_field_index("trading_date")
        for group in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(group).column(date_index).statistics
            if stats is None or not stats.has_min_max or stats.null_count != 0:
                raise ValueError(
                    "daily date row-group statistics are required for locked isolation"
                )
            if stats.min >= locked_begin or stats.min > end_date:
                continue
            if stats.max >= locked_begin:
                raise ValueError("daily row group crosses the locked boundary; republish by date")
            records = parquet.read_row_group(group, columns=required).to_pylist()
            bars.extend(
                DailyChipBar(**record) for record in records if record["trading_date"] <= end_date
            )
    if not bars:
        raise ValueError("daily chip asset has no research rows")
    seen: set[tuple[date, str]] = set()
    for bar in bars:
        bar.validate()
        key = (bar.trading_date, bar.symbol)
        if key in seen:
            raise ValueError(f"duplicate daily bar {key}")
        seen.add(key)
    return bars


def _write_predictions(
    path: Path,
    candidates: Sequence[FormalNextOpenTarget],
    support: Sequence[FormalNextOpenTarget],
    probabilities: np.ndarray,
    features: Mapping[tuple[date, str], ChipFeatures],
    available: np.ndarray,
    fingerprint: str,
    top_n: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate([*candidates, *support]):
        is_candidate = index < len(candidates)
        probability = probabilities[index] if is_candidate else np.asarray([0.0, 1.0, 0.0])
        feature = features.get((target.trading_date, target.symbol))
        rows.append(
            {
                "symbol": target.symbol,
                "trading_date": target.trading_date,
                "label_date": target.label_date,
                "return_end_date": target.return_end_date,
                "target_return": target.portfolio_return,
                "model_target_return": target.target_return,
                "benchmark_return": target.benchmark_return,
                "score": float(probability[2] - probability[0]),
                "prob_down": float(probability[0]),
                "prob_neutral": float(probability[1]),
                "prob_up": float(probability[2]),
                "can_buy": target.can_buy,
                "can_sell": target.can_sell,
                "in_universe": is_candidate,
                "execution_status": target.execution_status,
                "feature_available": bool(is_candidate and available[index]),
                "chip_feature_available": bool(is_candidate and feature and feature.available),
                "chip_feature_date": feature.feature_date if is_candidate and feature else None,
            }
        )
    rows.sort(key=lambda row: (row["label_date"], not row["in_universe"], row["symbol"]))
    table = attach_formal_prediction_metadata(
        pa.Table.from_pylist(rows), dataset_fingerprint=fingerprint
    )
    pq.write_table(table, path)
    validate_formal_prediction_artifact(
        path, expected_universe_size=top_n, expected_dataset_fingerprint=fingerprint
    )


def _evaluate(
    model: HistGradientBoostingClassifier,
    matrix: np.ndarray,
    targets: Sequence[FormalNextOpenTarget],
    support: Sequence[FormalNextOpenTarget],
    features: Mapping[tuple[date, str], ChipFeatures],
    path: Path,
    *,
    fingerprint: str,
    top_n: int,
    top_k: int,
    min_symbols_per_day: int,
) -> dict[str, Any]:
    probabilities = model.predict_proba(matrix)
    metrics = evaluate_predictions(
        np.asarray([target.label for target in targets]),
        probabilities,
        np.asarray([target.target_return for target in targets]),
        [target.label_date for target in targets],
        min_symbols_per_day=min_symbols_per_day,
    )
    _write_predictions(
        path,
        targets,
        support,
        probabilities,
        features,
        np.isfinite(matrix).any(axis=1),
        fingerprint,
        top_n,
    )
    evaluation = evaluate_topk_portfolio(
        load_portfolio_predictions(path),
        policy=PortfolioPolicy(
            top_k=top_k,
            min_symbols_per_day=min_symbols_per_day,
            require_tradability=True,
            require_universe_membership=True,
            missing_holding_policy="error",
        ),
        cost_model=CostModel(),
    )
    artifacts = write_portfolio_artifacts(evaluation, path.parent / path.stem)
    return {
        "ranking": metrics,
        "portfolio": evaluation.summary,
        "predictions": str(path),
        "prediction_sha256": file_sha256(path),
        "portfolio_artifacts": artifacts,
    }


def _partition_targets(
    targets: Sequence[FormalNextOpenTarget],
    split: WalkForwardSplit,
) -> tuple[dict[str, list[FormalNextOpenTarget]], dict[str, list[FormalNextOpenTarget]]]:
    parts: dict[str, list[FormalNextOpenTarget]] = {name: [] for name in ("train", "val", "test")}
    support: dict[str, list[FormalNextOpenTarget]] = {name: [] for name in parts}
    for target in targets:
        if (
            max(target.trading_date, target.label_date, target.return_end_date)
            >= ResearchProtocol().locked_begin
        ):
            raise ValueError("target reaches the locked period")
        name = split.assign(target.trading_date, target.label_date, target.return_end_date)
        if name:
            (parts if target.in_universe else support)[name].append(target)
    return parts, support


def _prepare_matrices(
    parts: Mapping[str, Sequence[FormalNextOpenTarget]],
    features: Mapping[tuple[date, str], ChipFeatures],
    minute_features: Mapping[tuple[date, str], np.ndarray] | None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    coverage: dict[str, Any] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {name: {} for name in parts}
    for name, items in parts.items():
        if not items:
            continue
        values = [features[(item.trading_date, item.symbol)] for item in items]
        chip = np.stack([value.chip for value in values])
        price = np.stack([value.price for value in values])
        matrices[name] = {"price": price, "chip": chip, "combined": np.hstack((price, chip))}
        if minute_features is not None:
            minute = np.stack([minute_features[(item.trading_date, item.symbol)] for item in items])
            matrices[name].update(minute=minute, minute_chip=np.hstack((minute, chip)))
        coverage[name] = {
            "available_rows": sum(value.available for value in values),
            "rows": len(values),
            "max_initial_mass_remaining": max(
                (value.initial_mass_remaining for value in values if value.available), default=None
            ),
        }
    return matrices, coverage


def run_chip_comparison(
    targets: Sequence[FormalNextOpenTarget],
    features: Mapping[tuple[date, str], ChipFeatures],
    split: WalkForwardSplit,
    output_dir: Path,
    *,
    fingerprint: str,
    top_n: int,
    top_k: int = 100,
    min_symbols_per_day: int = 400,
    seed: int = 0,
    max_iter: int = 200,
    min_samples_leaf: int = 40,
    evaluate_test: bool = False,
    minute_features: Mapping[tuple[date, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit equal-budget comparisons with no randomized internal validation split."""
    if split.test.end >= ResearchProtocol().locked_begin:
        raise ValueError("chip experiments cannot enter the locked period")
    if max_iter < 1 or min_samples_leaf < 1:
        raise ValueError("HGB iterations and leaf size must be positive")
    PortfolioPolicy(top_k=top_k, min_symbols_per_day=min_symbols_per_day)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("experiment output directory must be empty")
    parts, support = _partition_targets(targets, split)
    if not parts["train"] or not parts["val"] or (evaluate_test and not parts["test"]):
        raise ValueError("requested train/evaluation split has no eligible candidates")
    result: dict[str, Any] = {
        "evidence": "engineering_experiment_not_broker_reproduction",
        "dataset_fingerprint": fingerprint,
        "seed": seed,
        "split_rows": {name: len(items) for name, items in parts.items()},
        "coverage": {},
        "models": {},
    }
    matrices, result["coverage"] = _prepare_matrices(parts, features, minute_features)
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_set, train_x in matrices["train"].items():
        if not np.isfinite(train_x).any():
            raise ValueError(f"{feature_set}: no finite training features; check coverage/warmup")
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            early_stopping=False,
        )
        model.fit(train_x, np.asarray([target.label for target in parts["train"]]))
        if not np.array_equal(model.classes_, np.arange(3)):
            raise ValueError("training candidates must contain all three classes")
        root = output_dir / feature_set
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "model.pkl"
        with checkpoint.open("wb") as stream:
            pickle.dump(model, stream)
        model_result: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "feature_count": train_x.shape[1],
        }
        for name in ["val", "test"] if evaluate_test else ["val"]:
            model_result[name] = _evaluate(
                model,
                matrices[name][feature_set],
                parts[name],
                support[name],
                features,
                root / f"{name}-predictions.parquet",
                fingerprint=fingerprint,
                top_n=top_n,
                top_k=top_k,
                min_symbols_per_day=min_symbols_per_day,
            )
        result["models"][feature_set] = model_result
    return result


def experiment_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


FEATURE_SCHEMAS = {"chip": CHIP_FEATURE_NAMES, "price": PRICE_FEATURE_NAMES}
