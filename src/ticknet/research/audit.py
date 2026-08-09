"""预测审计（Prediction Audit）：解释指标背离，为 Brainstorm Agent 提供观测接口。

对应 AgentX 论文中 Evaluation Agent 的"解释诊断"职责，以及落地路线 PR-1：
``PredictionTable → audit_predictions() → audit.json``。审计全部由确定性 Python
计算，LLM 不参与"看数字"，只负责解释审计输出并提议下一步。

当前最想回答的问题是第 10 节观察到的矛盾：Rank IC ≈ 0.01 但无成本多空 spread
偏高。审计自动检查极端日贡献、winsorize 敏感性、decile 单调性、月度稳定性，
把背离原因结构化。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class PredictionTable:
    """一组预测明细：symbol、标签日、目标收益、排序分数。"""

    symbols: np.ndarray
    label_dates: np.ndarray
    target_returns: np.ndarray
    scores: np.ndarray
    prob_up: np.ndarray | None = None
    in_universe: np.ndarray | None = None

    @classmethod
    def from_parquet(cls, path: str | Path) -> PredictionTable:
        table = pq.read_table(path)
        return cls(
            symbols=np.asarray(table["symbol"].to_pylist()),
            label_dates=np.asarray(
                [date.fromisoformat(str(value)) for value in table["label_date"]]
            ),
            target_returns=np.asarray(table["target_return"], dtype=np.float64),
            scores=np.asarray(table["score"], dtype=np.float64),
            prob_up=(
                np.asarray(table["prob_up"], dtype=np.float64)
                if "prob_up" in table.column_names
                else None
            ),
            in_universe=(
                np.asarray(table["in_universe"], dtype=np.bool_)
                if "in_universe" in table.column_names
                else None
            ),
        )

    def group_by_date(self) -> dict[date, list[int]]:
        indices: dict[date, list[int]] = defaultdict(list)
        for index, label_date in enumerate(self.label_dates):
            if self.in_universe is not None and not bool(self.in_universe[index]):
                continue
            indices[label_date].append(index)
        return indices


def _rank_correlation(scores: np.ndarray, returns: np.ndarray) -> float:
    if scores.size < 2:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(scores.size)
    order_r = np.argsort(returns, kind="mergesort")
    ranks_r = np.empty(returns.size, dtype=np.float64)
    ranks_r[order_r] = np.arange(returns.size)
    if np.std(ranks) == 0 or np.std(ranks_r) == 0:
        return math.nan
    return float(np.corrcoef(ranks, ranks_r)[0, 1])


@dataclass
class AuditReport:
    """结构化的预测审计输出。"""

    daily_ic_mean: float
    daily_ic_std: float
    daily_ic_ir: float
    positive_days_ratio: float

    monthly_ic: dict[str, float]
    positive_month_ratio: float

    spread_mean: float
    spread_median: float
    decile_returns: list[float]
    decile_monotonicity: float

    top_1_day_contribution: float
    top_5_day_contribution: float
    top_10_day_contribution: float

    winsorized_spread_mean: float

    anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "daily_ic_mean": self.daily_ic_mean,
            "daily_ic_std": self.daily_ic_std,
            "daily_ic_ir": self.daily_ic_ir,
            "positive_days_ratio": self.positive_days_ratio,
            "monthly_ic": self.monthly_ic,
            "positive_month_ratio": self.positive_month_ratio,
            "spread_mean": self.spread_mean,
            "spread_median": self.spread_median,
            "decile_returns": self.decile_returns,
            "decile_monotonicity": self.decile_monotonicity,
            "top_1_day_contribution": self.top_1_day_contribution,
            "top_5_day_contribution": self.top_5_day_contribution,
            "top_10_day_contribution": self.top_10_day_contribution,
            "winsorized_spread_mean": self.winsorized_spread_mean,
            "anomalies": self.anomalies,
        }


def audit_predictions(
    table: PredictionTable,
    *,
    min_symbols_per_day: int = 50,
    portfolio_quantile: float = 0.1,
    winsorize_percentile: float = 99.0,
) -> AuditReport:
    """对预测明细做诊断审计，自动标注异常。"""
    grouped = table.group_by_date()
    eligible_dates = sorted(
        label_date for label_date, indices in grouped.items() if len(indices) >= min_symbols_per_day
    )
    if not eligible_dates:
        raise ValueError("没有可审计的交易日")

    daily_ics: list[float] = []
    daily_spreads: list[float] = []
    winsorized_spreads: list[float] = []

    for label_date in eligible_dates:
        indices = grouped[label_date]
        scores = table.scores[indices]
        returns = table.target_returns[indices]
        order = np.argsort(scores, kind="mergesort")
        tail_count = max(1, math.floor(len(indices) * portfolio_quantile))
        long_returns = returns[order[-tail_count:]]
        short_returns = returns[order[:tail_count]]
        spread = float(np.mean(long_returns) - np.mean(short_returns))
        daily_spreads.append(spread)

        ic = _rank_correlation(scores, returns)
        if math.isfinite(ic):
            daily_ics.append(ic)

        percentile = np.percentile(returns, winsorize_percentile)
        winsorized_returns = np.minimum(returns, percentile)
        winsorized_spread = float(
            np.mean(winsorized_returns[order[-tail_count:]])
            - np.mean(winsorized_returns[order[:tail_count]])
        )
        winsorized_spreads.append(winsorized_spread)

    daily_ic_mean = float(np.mean(daily_ics)) if daily_ics else math.nan
    daily_ic_std = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else math.nan
    daily_ic_ir = (
        daily_ic_mean / daily_ic_std
        if math.isfinite(daily_ic_std) and daily_ic_std > 0
        else math.nan
    )
    positive_days = sum(1 for value in daily_spreads if value > 0)

    monthly_ic: dict[str, float] = {}
    month_groups: dict[str, list[float]] = defaultdict(list)
    for label_date, ic in zip(eligible_dates, daily_ics, strict=True):
        month_groups[f"{label_date.year}-{label_date.month:02d}"].append(ic)
    for month, values in sorted(month_groups.items()):
        monthly_ic[month] = float(np.mean(values))
    positive_months = sum(1 for value in monthly_ic.values() if value > 0)

    spreads = np.asarray(daily_spreads, dtype=np.float64)
    spread_mean = float(np.mean(spreads))
    spread_median = float(np.median(spreads))
    total_spread_sum = float(np.sum(spreads))
    sorted_spreads = np.sort(spreads)
    contribution_1 = float(sorted_spreads[-1] / total_spread_sum)
    contribution_5 = float(np.sum(sorted_spreads[-5:]) / total_spread_sum)
    contribution_10 = float(np.sum(sorted_spreads[-10:]) / total_spread_sum)

    decile_returns: list[float] = []
    for decile in range(10):
        day_means: list[float] = []
        for label_date in eligible_dates:
            indices = grouped[label_date]
            scores = table.scores[indices]
            returns = table.target_returns[indices]
            order = np.argsort(scores, kind="mergesort")
            decile_indices = order[len(indices) * decile // 10 : len(indices) * (decile + 1) // 10]
            day_means.append(float(np.mean(returns[decile_indices])))
        decile_returns.append(float(np.mean(day_means)))
    decile_monotonicity = float(np.corrcoef(np.arange(10), decile_returns)[0, 1])

    winsorized_spread_mean = float(np.mean(winsorized_spreads))

    report = AuditReport(
        daily_ic_mean=daily_ic_mean,
        daily_ic_std=daily_ic_std,
        daily_ic_ir=daily_ic_ir,
        positive_days_ratio=positive_days / len(daily_spreads),
        monthly_ic=monthly_ic,
        positive_month_ratio=positive_months / len(monthly_ic) if monthly_ic else math.nan,
        spread_mean=spread_mean,
        spread_median=spread_median,
        decile_returns=decile_returns,
        decile_monotonicity=decile_monotonicity,
        top_1_day_contribution=contribution_1,
        top_5_day_contribution=contribution_5,
        top_10_day_contribution=contribution_10,
        winsorized_spread_mean=winsorized_spread_mean,
    )

    if math.isfinite(daily_ic_mean) and abs(daily_ic_mean) < 0.02 and spread_mean > 0.002:
        report.anomalies.append(
            {
                "type": "ic_spread_divergence",
                "severity": "high",
                "detail": "Rank IC 接近 0 但无成本多空 spread 偏高，可能存在极端值驱动",
            }
        )
    if contribution_1 > 0.05:
        report.anomalies.append(
            {
                "type": "tail_return_concentration",
                "severity": "high",
                "detail": f"单日贡献 {contribution_1:.1%}，收益高度集中于少数交易日",
            }
        )
    if math.isfinite(decile_monotonicity) and abs(decile_monotonicity) < 0.5:
        report.anomalies.append(
            {
                "type": "weak_decile_monotonicity",
                "severity": "medium",
                "detail": f"decile 单调性 {decile_monotonicity:.2f}，排序信号非线性",
            }
        )
    if math.isfinite(winsorized_spread_mean) and winsorized_spread_mean < spread_mean * 0.5:
        report.anomalies.append(
            {
                "type": "winsorize_sensitivity",
                "severity": "high",
                "detail": "winsorize 后 spread 大幅缩水，收益由极端值贡献",
            }
        )
    return report
