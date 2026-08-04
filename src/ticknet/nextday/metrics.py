"""同时衡量分类质量和每日横截面排序质量。"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from ticknet.dataset import NUM_CLASSES
from ticknet.train import f1_metrics


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _rank_correlation(scores: np.ndarray, returns: np.ndarray) -> float:
    score_ranks = _average_ranks(scores)
    return_ranks = _average_ranks(returns)
    if np.std(score_ranks) == 0 or np.std(return_ranks) == 0:
        return math.nan
    return float(np.corrcoef(score_ranks, return_ranks)[0, 1])


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_returns: np.ndarray,
    label_dates: Sequence[date],
    *,
    scores: np.ndarray | None = None,
    min_symbols_per_day: int = 20,
    portfolio_quantile: float = 0.1,
) -> dict[str, Any]:
    """计算分类指标、Rank IC 和无成本多空收益差。"""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    target_returns = np.asarray(target_returns, dtype=np.float64)
    if labels.ndim != 1 or target_returns.shape != labels.shape:
        raise ValueError("labels 和 target_returns 应为相同长度的一维数组")
    if probabilities.shape != (labels.size, NUM_CLASSES):
        raise ValueError(f"probabilities 应为 {(labels.size, NUM_CLASSES)}")
    if len(label_dates) != labels.size:
        raise ValueError("label_dates 长度与 labels 不一致")
    if labels.size == 0:
        raise ValueError("评估样本不能为空")
    if min_symbols_per_day < 2:
        raise ValueError("min_symbols_per_day 至少为 2")
    if not 0 < portfolio_quantile <= 0.5:
        raise ValueError("portfolio_quantile 应在 (0, 0.5] 内")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(target_returns)):
        raise ValueError("预测概率和目标收益必须是有限值")

    predicted = probabilities.argmax(axis=1)
    metrics: dict[str, Any] = dict(f1_metrics(labels, predicted))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predicted))
    metrics["mcc"] = float(matthews_corrcoef(labels, predicted))
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[labels]
    metrics["brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    if scores is None:
        ranking_scores = probabilities[:, 2] - probabilities[:, 0]
    else:
        ranking_scores = np.asarray(scores, dtype=np.float64)
        if ranking_scores.shape != labels.shape:
            raise ValueError("scores 应为与 labels 相同长度的一维数组")
        if not np.all(np.isfinite(ranking_scores)):
            raise ValueError("连续预测分数必须是有限值")
    indices_by_date: dict[date, list[int]] = defaultdict(list)
    for index, label_date in enumerate(label_dates):
        indices_by_date[label_date].append(index)

    daily_ics: list[float] = []
    daily_spreads: list[float] = []
    for indices in indices_by_date.values():
        if len(indices) < min_symbols_per_day:
            continue
        selected = np.asarray(indices, dtype=np.int64)
        daily_scores = ranking_scores[selected]
        daily_returns = target_returns[selected]
        rank_ic = _rank_correlation(daily_scores, daily_returns)
        if math.isfinite(rank_ic):
            daily_ics.append(rank_ic)

        tail_count = max(1, math.floor(len(indices) * portfolio_quantile))
        order = np.argsort(daily_scores, kind="mergesort")
        long_return = float(np.mean(daily_returns[order[-tail_count:]]))
        short_return = float(np.mean(daily_returns[order[:tail_count]]))
        daily_spreads.append(long_return - short_return)

    ic_mean = float(np.mean(daily_ics)) if daily_ics else math.nan
    ic_std = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else math.nan
    metrics.update(
        {
            "daily_rank_ic_mean": ic_mean,
            "daily_rank_ic_std": ic_std,
            "daily_rank_ic_ir": (
                ic_mean / ic_std if math.isfinite(ic_std) and ic_std > 0 else math.nan
            ),
            "daily_long_short_return_mean": (
                float(np.mean(daily_spreads)) if daily_spreads else math.nan
            ),
            "evaluated_dates": len(daily_spreads),
        }
    )
    return metrics
