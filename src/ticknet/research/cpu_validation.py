"""Small CPU-only validation path for prediction and backtest integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pyarrow as pa

from .alpha_signal_adapter import build_alpha_signal_table
from .portfolio import (
    CostModel,
    PortfolioEvaluation,
    PortfolioPolicy,
    PortfolioPrediction,
    evaluate_topk_portfolio,
)


@dataclass(frozen=True)
class CpuSmokeResult:
    """Artifacts produced by the bounded CPU integration check."""

    predictions: pa.Table
    alpha_signals: pa.Table
    evaluation: PortfolioEvaluation
    metadata: dict[str, Any]


def _as_date(value: object, *, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{field} 不是 ISO 日期: {value}") from error


def _required_columns(table: pa.Table) -> None:
    required = {
        "symbol",
        "trading_date",
        "label_date",
        "return_end_date",
        "feature",
        "target_return",
        "is_train",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"CPU smoke 输入缺少字段: {missing}")
    if table.num_rows == 0:
        raise ValueError("CPU smoke 输入为空")


def _linear_scores(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> list[float]:
    x = np.asarray([[1.0, float(row["feature"])] for row in train], dtype=np.float64)
    y = np.asarray([float(row["target_return"]) for row in train], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CPU smoke 训练数据必须为有限数值")
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    features = np.asarray([[1.0, float(row["feature"])] for row in test], dtype=np.float64)
    scores = features @ coefficients
    if not np.isfinite(scores).all():
        raise ValueError("CPU smoke 预测结果必须为有限数值")
    return scores.tolist()


def run_cpu_smoke_pipeline(
    table: pa.Table,
    *,
    top_k: int = 2,
    per_side_bps: float = 0.0,
    sell_stamp_tax_bps: float = 0.0,
) -> CpuSmokeResult:
    """Run a deterministic train -> prediction -> alpha -> portfolio smoke test.

    The linear model is intentionally tiny and NumPy-based. It is an integration
    probe, not a replacement for the project's PyTorch training jobs.
    """
    _required_columns(table)
    rows = table.to_pylist()
    train = [row for row in rows if bool(row["is_train"])]
    test = [row for row in rows if not bool(row["is_train"])]
    if not train or not test:
        raise ValueError("CPU smoke 必须同时包含 train rows 和 test rows")
    train_dates = {_as_date(row["trading_date"], field="trading_date") for row in train}
    test_dates = {_as_date(row["trading_date"], field="trading_date") for row in test}
    if train_dates & test_dates:
        raise ValueError("train rows 与 test rows 不得共享 trading_date；请检查 test rows")

    scores = _linear_scores(train, test)
    formal_rows: list[dict[str, Any]] = []
    predictions: list[PortfolioPrediction] = []
    for row, score in zip(test, scores, strict=True):
        trading_date = _as_date(row["trading_date"], field="trading_date")
        label_date = _as_date(row["label_date"], field="label_date")
        return_end_date = _as_date(row["return_end_date"], field="return_end_date")
        symbol = str(row["symbol"]).strip()
        target_return = float(row["target_return"])
        formal_rows.append(
            {
                "symbol": symbol,
                "trading_date": trading_date,
                "label_date": label_date,
                "return_end_date": return_end_date,
                "target_return": target_return,
                "score": score,
                "can_buy": True,
                "can_sell": True,
                "in_universe": True,
            }
        )
        predictions.append(
            PortfolioPrediction(
                symbol=symbol,
                trading_date=trading_date,
                label_date=label_date,
                score=score,
                target_return=target_return,
            )
        )

    formal_table = pa.Table.from_pylist(formal_rows)
    alpha_signals = build_alpha_signal_table(
        formal_table,
        model_version="cpu-smoke-linear-v1",
        feature_set_id="cpu-smoke-feature-v1",
    )
    evaluation = evaluate_topk_portfolio(
        predictions,
        policy=PortfolioPolicy(
            top_k=top_k,
            min_symbols_per_day=len({row["symbol"] for row in test}),
        ),
        cost_model=CostModel(
            per_side_bps=per_side_bps,
            sell_stamp_tax_bps=sell_stamp_tax_bps,
        ),
    )
    return CpuSmokeResult(
        predictions=formal_table,
        alpha_signals=alpha_signals,
        evaluation=evaluation,
        metadata={
            "device": "cpu",
            "model": "numpy_linear_least_squares",
            "train_rows": len(train),
            "test_rows": len(test),
            "top_k": top_k,
        },
    )


def _digest(evaluation: PortfolioEvaluation) -> dict[str, Any]:
    gross = evaluation.summary.get("gross", {})
    net = evaluation.summary.get("net", {})
    return {
        "summary": {
            "gross": gross.get("cumulative"),
            "net": net.get("cumulative"),
            "net_max_drawdown": net.get("max_drawdown"),
        },
        "daily": [
            {
                key: row.get(key)
                for key in (
                    "trading_date",
                    "label_date",
                    "positions",
                    "gross_return",
                    "transaction_cost",
                    "net_return",
                    "one_way_turnover",
                )
            }
            for row in evaluation.daily
        ],
    }


def digest_portfolio_backtester_result(result: Any) -> dict[str, Any]:
    """Convert the portfolio-backtester five-tuple to the local digest shape."""
    _, net_series, gross_series, turnover_series, periods = result[:5]
    net_returns = [float(value) for value in net_series.tolist()]
    gross_returns = [float(value) for value in gross_series.tolist()]
    turnover = [float(value) for value in turnover_series.tolist()]

    def compounded(values: list[float]) -> float:
        return float(np.prod(1.0 + np.asarray(values, dtype=np.float64)) - 1.0)

    def max_drawdown(values: list[float]) -> float:
        curve = np.cumprod(1.0 + np.asarray(values, dtype=np.float64))
        peaks = np.maximum.accumulate(curve)
        return float(np.min(curve / peaks - 1.0))

    daily: list[dict[str, Any]] = []
    for index, period in enumerate(periods):
        gross_return = gross_returns[index]
        net_return = net_returns[index]
        daily.append(
            {
                "trading_date": period.get("period_start", period.get("period_end")),
                "label_date": period.get("period_end", period.get("period_start")),
                "positions": period.get("positions", period.get("target_names")),
                "gross_return": gross_return,
                "transaction_cost": gross_return - net_return,
                "net_return": net_return,
                "one_way_turnover": turnover[index],
            }
        )
    return {
        "summary": {
            "gross": compounded(gross_returns),
            "net": compounded(net_returns),
            "net_max_drawdown": max_drawdown(net_returns),
        },
        "daily": daily,
    }


def compare_portfolio_digests(
    left_digest: Mapping[str, Any],
    right_digest: Mapping[str, Any],
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compare engine-neutral summaries without importing either engine."""
    mismatches: list[dict[str, Any]] = []
    for section in ("summary", "daily"):
        left_values = left_digest[section]
        right_values = right_digest[section]
        if isinstance(left_values, list):
            left_rows = cast(list[dict[str, Any]], left_values)
            right_rows = cast(list[dict[str, Any]], right_values)
            if len(left_rows) != len(right_rows):
                mismatches.append(
                    {
                        "field": f"{section}.length",
                        "left": len(left_values),
                        "right": len(right_values),
                    }
                )
                continue
            pairs = []
            for index, (lrow, rrow) in enumerate(zip(left_rows, right_rows, strict=True)):
                pairs.extend(
                    (f"{section}[{index}].{key}", lrow.get(key), rrow.get(key)) for key in lrow
                )
        else:
            left_map = cast(Mapping[str, Any], left_values)
            right_map = cast(Mapping[str, Any], right_values)
            pairs = [
                (f"{section}.{key}", value, right_map.get(key)) for key, value in left_map.items()
            ]
        for field, left_value, right_value in pairs:
            equal = (
                abs(float(left_value) - float(right_value)) <= atol
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float))
                else left_value == right_value
            )
            if not equal:
                mismatches.append({"field": field, "left": left_value, "right": right_value})
    return {"status": "match" if not mismatches else "mismatch", "mismatches": mismatches}


def compare_portfolio_evaluations(
    left: PortfolioEvaluation,
    right: PortfolioEvaluation,
    *,
    atol: float = 1e-12,
    right_summary_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the local evaluator with another canonical evaluation."""
    left_digest = _digest(left)
    right_digest = _digest(right)
    if right_summary_override is not None:
        right_digest["summary"] = dict(right_summary_override)
    return compare_portfolio_digests(left_digest, right_digest, atol=atol)


__all__ = [
    "CpuSmokeResult",
    "compare_portfolio_digests",
    "compare_portfolio_evaluations",
    "digest_portfolio_backtester_result",
    "run_cpu_smoke_pipeline",
]
