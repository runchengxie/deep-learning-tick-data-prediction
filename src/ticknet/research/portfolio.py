"""确定性的 Top-K long-only 组合评估与交易成本核算。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MissingHoldingPolicy = Literal["liquidate", "error"]


def _as_date(value: object, *, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{field} 不是 ISO 日期: {value}") from error


def _as_bool(value: object, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} 应为布尔值: {value}")


@dataclass(frozen=True)
class PortfolioPrediction:
    """一个信号日、一个股票的分数及随后持有期收益。"""

    symbol: str
    trading_date: date
    label_date: date
    score: float
    target_return: float
    can_buy: bool = True
    can_sell: bool = True
    tradability_known: bool = True
    in_universe: bool = True
    universe_membership_known: bool = True

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if not math.isfinite(self.score):
            raise ValueError(f"score 必须为有限值: {self.symbol} {self.label_date}")


@dataclass(frozen=True)
class PortfolioPolicy:
    """Top-K 选股、缓冲区和换仓门槛。"""

    top_k: int
    exit_buffer: int = 0
    min_score_gap: float = 0.0
    min_position_score: float | None = None
    allow_cash: bool = False
    min_symbols_per_day: int = 50
    annualization_days: int = 244
    missing_holding_policy: MissingHoldingPolicy = "liquidate"
    require_tradability: bool = False
    require_universe_membership: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if self.exit_buffer < 0:
            raise ValueError("exit_buffer 不能为负数")
        if self.min_score_gap < 0:
            raise ValueError("min_score_gap 不能为负数")
        if self.min_position_score is not None and not math.isfinite(self.min_position_score):
            raise ValueError("min_position_score 必须为有限值")
        if self.min_position_score is not None and not self.allow_cash:
            raise ValueError("使用 min_position_score 时必须允许现金仓位")
        if self.min_symbols_per_day < self.top_k:
            raise ValueError("min_symbols_per_day 不能小于 top_k")
        if self.annualization_days <= 0:
            raise ValueError("annualization_days 必须为正整数")
        if self.missing_holding_policy not in {"liquidate", "error"}:
            raise ValueError("missing_holding_policy 应为 liquidate 或 error")


@dataclass(frozen=True)
class CostModel:
    """按成交方向收费的 A 股成本模型，输入单位均为 bp。"""

    per_side_bps: float = 10.0
    sell_stamp_tax_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.per_side_bps < 0 or self.sell_stamp_tax_bps < 0:
            raise ValueError("成本参数不能为负数")

    @property
    def buy_rate(self) -> float:
        return self.per_side_bps / 10_000.0

    @property
    def sell_rate(self) -> float:
        return (self.per_side_bps + self.sell_stamp_tax_bps) / 10_000.0


@dataclass(frozen=True)
class PortfolioEvaluation:
    """可写入 artifact、也可由 AgentX 直接消费的评估结果。"""

    summary: dict[str, Any]
    daily: tuple[dict[str, Any], ...]
    holdings: tuple[dict[str, Any], ...]
    trades: tuple[dict[str, Any], ...]


def load_portfolio_predictions(path: str | Path) -> list[PortfolioPrediction]:
    """读取稳定预测契约，并拒绝缺列、空表和股票日重复。"""
    source = Path(path).expanduser().resolve()
    table = pq.read_table(source)
    required = {"symbol", "trading_date", "label_date", "score", "target_return"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"预测明细缺少字段: {sorted(missing)}")
    tradability_columns = {"can_buy", "can_sell"} & set(table.column_names)
    if tradability_columns and tradability_columns != {"can_buy", "can_sell"}:
        raise ValueError("can_buy 与 can_sell 必须同时提供")
    tradability_known = len(tradability_columns) == 2
    universe_membership_known = "in_universe" in table.column_names
    records = table.to_pylist()
    if not records:
        raise ValueError("预测明细为空")

    predictions: list[PortfolioPrediction] = []
    seen: set[tuple[date, str]] = set()
    for record in records:
        label_date = _as_date(record["label_date"], field="label_date")
        symbol = str(record["symbol"])
        key = (label_date, symbol)
        if key in seen:
            raise ValueError(f"预测明细存在重复股票日: {label_date} {symbol}")
        seen.add(key)
        predictions.append(
            PortfolioPrediction(
                symbol=symbol,
                trading_date=_as_date(record["trading_date"], field="trading_date"),
                label_date=label_date,
                score=float(record["score"]),
                target_return=float(record["target_return"]),
                can_buy=_as_bool(record.get("can_buy"), field="can_buy", default=True),
                can_sell=_as_bool(record.get("can_sell"), field="can_sell", default=True),
                tradability_known=tradability_known,
                in_universe=_as_bool(record.get("in_universe"), field="in_universe", default=True),
                universe_membership_known=universe_membership_known,
            )
        )
    return predictions


def _rank_correlation(scores: np.ndarray, returns: np.ndarray) -> float:
    if scores.size < 2 or np.std(scores) == 0 or np.std(returns) == 0:
        return math.nan
    score_order = np.argsort(scores, kind="mergesort")
    return_order = np.argsort(returns, kind="mergesort")
    score_ranks = np.empty(scores.size, dtype=np.float64)
    return_ranks = np.empty(returns.size, dtype=np.float64)
    score_ranks[score_order] = np.arange(scores.size)
    return_ranks[return_order] = np.arange(returns.size)
    return float(np.corrcoef(score_ranks, return_ranks)[0, 1])


def _select_symbols(
    day: dict[str, PortfolioPrediction],
    previous_symbols: set[str],
    policy: PortfolioPolicy,
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    all_ranked = sorted(
        (row for row in day.values() if row.in_universe),
        key=lambda row: (-row.score, row.symbol),
    )
    rank_by_symbol = {row.symbol: rank for rank, row in enumerate(all_ranked, start=1)}
    ranked = [
        row
        for row in all_ranked
        if policy.min_position_score is None or row.score >= policy.min_position_score
    ]
    eligible_symbols = {row.symbol for row in ranked}
    present_previous = previous_symbols & set(day)
    missing_previous = previous_symbols - set(day)
    if missing_previous and policy.missing_holding_policy == "error":
        raise ValueError(
            "已有持仓在当日预测/交易状态中缺失: " + ", ".join(sorted(missing_previous))
        )

    forced = {symbol for symbol in present_previous if not day[symbol].can_sell}
    buffered = {
        symbol
        for symbol in present_previous
        if (
            day[symbol].in_universe
            and symbol in eligible_symbols
            and rank_by_symbol[symbol] <= policy.top_k + policy.exit_buffer
        )
    }
    selected = forced | buffered
    reasons = {
        symbol: ("retained_untradeable" if symbol in forced else "retained_buffer")
        for symbol in selected
    }

    replaceable = sorted(
        {
            symbol
            for symbol in present_previous - selected
            if day[symbol].in_universe and symbol in eligible_symbols
        },
        key=lambda symbol: (-day[symbol].score, symbol),
    )
    entries = [row.symbol for row in ranked if row.symbol not in previous_symbols and row.can_buy]
    slots = max(0, policy.top_k - len(selected))
    while slots:
        incumbent = replaceable[0] if replaceable else None
        challenger = entries[0] if entries else None
        if incumbent is None and challenger is None:
            break
        if incumbent is None:
            chosen = entries.pop(0)
            reasons[chosen] = "entry_rank"
        elif challenger is None:
            chosen = replaceable.pop(0)
            reasons[chosen] = "retained_no_entry"
        elif day[challenger].score - day[incumbent].score >= policy.min_score_gap:
            chosen = entries.pop(0)
            reasons[chosen] = "entry_score_gap"
        else:
            chosen = replaceable.pop(0)
            reasons[chosen] = "retained_score_gap"
        selected.add(chosen)
        slots -= 1

    status_rank = len(all_ranked) + 1
    rank_by_symbol.update(
        {symbol: status_rank for symbol in selected if symbol not in rank_by_symbol}
    )
    ordered = sorted(selected, key=lambda symbol: (rank_by_symbol[symbol], symbol))
    return ordered, reasons, rank_by_symbol


def _trade_reason(
    symbol: str,
    old_weight: float,
    new_weight: float,
    day: dict[str, PortfolioPrediction],
    selection_reasons: dict[str, str],
) -> str:
    if old_weight == 0:
        return selection_reasons.get(symbol, "entry")
    if new_weight == 0:
        return "universe_exit" if symbol not in day or not day[symbol].in_universe else "rank_exit"
    return "equal_weight_rebalance"


def _bounded_equal_weights(
    selected: list[str],
    old_weights: dict[str, float],
    day: dict[str, PortfolioPrediction],
    *,
    target_exposure: float = 1.0,
) -> dict[str, float]:
    """尽量等权，同时不对不可买/不可卖旧持仓做隐式反向成交。"""
    lower = {
        symbol: (
            old_weights.get(symbol, 0.0)
            if symbol in old_weights and not day[symbol].can_sell
            else 0.0
        )
        for symbol in selected
    }
    upper = {
        symbol: (
            old_weights[symbol]
            if symbol in old_weights and (not day[symbol].can_buy or not day[symbol].in_universe)
            else 1.0
        )
        for symbol in selected
    }
    minimum = sum(lower.values())
    maximum = sum(upper.values())
    target_exposure = max(target_exposure, minimum)
    if target_exposure > 1.0 + 1e-12 or maximum < target_exposure - 1e-12:
        raise ValueError("不可交易持仓约束下无法构造目标仓位")

    weights: dict[str, float] = {}
    free = set(selected)
    remaining = target_exposure
    while free:
        target = remaining / len(free)
        below = {symbol for symbol in free if lower[symbol] > target + 1e-15}
        above = {symbol for symbol in free if upper[symbol] < target - 1e-15}
        constrained = below | above
        if not constrained:
            weights.update(dict.fromkeys(free, target))
            break
        for symbol in sorted(constrained):
            weight = lower[symbol] if symbol in below else upper[symbol]
            weights[symbol] = weight
            remaining -= weight
            free.remove(symbol)
    return {symbol: weight for symbol, weight in weights.items() if weight > 1e-15}


def _build_trades(
    *,
    label_date: date,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    day: dict[str, PortfolioPrediction],
    selection_reasons: dict[str, str],
    cost_model: CostModel,
) -> tuple[list[dict[str, Any]], float, float, float]:
    trades: list[dict[str, Any]] = []
    buy_turnover = 0.0
    sell_turnover = 0.0
    transaction_cost = 0.0
    for symbol in sorted(set(old_weights) | set(new_weights)):
        old_weight = old_weights.get(symbol, 0.0)
        new_weight = new_weights.get(symbol, 0.0)
        delta = new_weight - old_weight
        if math.isclose(delta, 0.0, abs_tol=1e-15):
            continue
        action = "buy" if delta > 0 else "sell"
        notional = abs(delta)
        cost_rate = cost_model.buy_rate if action == "buy" else cost_model.sell_rate
        cost = notional * cost_rate
        buy_turnover += notional if action == "buy" else 0.0
        sell_turnover += notional if action == "sell" else 0.0
        transaction_cost += cost
        trades.append(
            {
                "label_date": label_date.isoformat(),
                "symbol": symbol,
                "action": action,
                "reason": _trade_reason(
                    symbol,
                    old_weight,
                    new_weight,
                    day,
                    selection_reasons,
                ),
                "old_weight": old_weight,
                "new_weight": new_weight,
                "weight_change": delta,
                "notional": notional,
                "cost_rate": cost_rate,
                "transaction_cost": cost,
            }
        )
    return trades, buy_turnover, sell_turnover, transaction_cost


def _return_summary(values: list[float], annualization_days: int) -> dict[str, float]:
    returns = np.asarray(values, dtype=np.float64)
    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns, ddof=1)) if returns.size > 1 else math.nan
    annualized_return = mean_daily * annualization_days
    annualized_volatility = std_daily * math.sqrt(annualization_days)
    sharpe = (
        mean_daily / std_daily * math.sqrt(annualization_days)
        if math.isfinite(std_daily) and std_daily > 0
        else math.nan
    )
    cumulative_curve = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], cumulative_curve)))
    drawdowns = cumulative_curve / running_peak[1:] - 1.0
    return {
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "annualized": annualized_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "cumulative": float(cumulative_curve[-1] - 1.0),
        "cumulative_return": float(cumulative_curve[-1] - 1.0),
        "max_drawdown": float(np.min(drawdowns)),
        "positive_days_ratio": float(np.mean(returns > 0)),
    }


def _monthly_stability(daily: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily:
        grouped[str(row["label_date"])[:7]].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for month, rows in sorted(grouped.items()):
        gross = np.asarray([row["gross_return"] for row in rows], dtype=np.float64)
        net = np.asarray([row["net_return"] for row in rows], dtype=np.float64)
        active = np.asarray([row["active_return"] for row in rows], dtype=np.float64)
        net_active = np.asarray([row["net_active_return"] for row in rows], dtype=np.float64)
        result[month] = {
            "days": len(rows),
            "gross_mean": float(np.mean(gross)),
            "net_mean": float(np.mean(net)),
            "active_mean": float(np.mean(active)),
            "net_active_mean": float(np.mean(net_active)),
            "gross_cumulative": float(np.prod(1.0 + gross) - 1.0),
            "net_cumulative": float(np.prod(1.0 + net) - 1.0),
            "positive_net_days_ratio": float(np.mean(net > 0)),
            "positive_net_active_days_ratio": float(np.mean(net_active > 0)),
        }
    return result


def _extreme_day_summary(daily: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(daily, key=lambda row: abs(float(row["gross_return"])), reverse=True)
    active_ordered = sorted(
        daily,
        key=lambda row: abs(float(row["active_return"])),
        reverse=True,
    )
    total_absolute = sum(abs(float(row["gross_return"])) for row in daily)
    total_active_absolute = sum(abs(float(row["active_return"])) for row in daily)

    def contribution(
        count: int,
        *,
        rows: list[dict[str, Any]],
        field: str,
        total: float,
    ) -> float:
        if total == 0:
            return 0.0
        return float(sum(abs(float(row[field])) for row in rows[:count]) / total)

    return {
        "top_1_absolute_contribution": contribution(
            1, rows=ordered, field="gross_return", total=total_absolute
        ),
        "top_5_absolute_contribution": contribution(
            5, rows=ordered, field="gross_return", total=total_absolute
        ),
        "top_10_absolute_contribution": contribution(
            10, rows=ordered, field="gross_return", total=total_absolute
        ),
        "top_1_absolute_active_contribution": contribution(
            1,
            rows=active_ordered,
            field="active_return",
            total=total_active_absolute,
        ),
        "top_5_absolute_active_contribution": contribution(
            5,
            rows=active_ordered,
            field="active_return",
            total=total_active_absolute,
        ),
        "top_10_absolute_active_contribution": contribution(
            10,
            rows=active_ordered,
            field="active_return",
            total=total_active_absolute,
        ),
        "largest_days": [
            {
                "label_date": row["label_date"],
                "gross_return": row["gross_return"],
                "net_return": row["net_return"],
            }
            for row in ordered[:10]
        ],
        "largest_active_days": [
            {
                "label_date": row["label_date"],
                "active_return": row["active_return"],
                "net_active_return": row["net_active_return"],
            }
            for row in active_ordered[:10]
        ],
    }


def evaluate_topk_portfolio(
    predictions: list[PortfolioPrediction],
    *,
    policy: PortfolioPolicy,
    cost_model: CostModel | None = None,
) -> PortfolioEvaluation:
    """按标签日构建 fixed-K long-only 组合并返回完整可审计明细。"""
    if not predictions:
        raise ValueError("预测明细为空")
    if policy.require_tradability and any(not row.tradability_known for row in predictions):
        raise ValueError("正式评估要求预测明细同时提供 can_buy 与 can_sell")
    if policy.require_universe_membership and any(
        not row.universe_membership_known for row in predictions
    ):
        raise ValueError("正式评估要求预测明细提供 in_universe")
    costs = cost_model or CostModel()
    grouped: dict[date, list[PortfolioPrediction]] = defaultdict(list)
    seen: set[tuple[date, str]] = set()
    for row in predictions:
        key = (row.label_date, row.symbol)
        if key in seen:
            raise ValueError(f"预测明细存在重复股票日: {row.label_date} {row.symbol}")
        seen.add(key)
        grouped[row.label_date].append(row)

    eligible_dates = sorted(
        label_date
        for label_date, rows in grouped.items()
        if sum(row.in_universe for row in rows) >= policy.min_symbols_per_day
    )
    if not eligible_dates:
        raise ValueError("没有可评估的交易日")

    old_weights: dict[str, float] = {}
    daily_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for label_date in eligible_dates:
        rows = grouped[label_date]
        signal_dates = {row.trading_date for row in rows}
        if len(signal_dates) != 1:
            raise ValueError(f"同一 label_date 对应多个 trading_date: {label_date}")
        day = {row.symbol: row for row in rows}
        selected, reasons, ranks = _select_symbols(day, set(old_weights), policy)
        target_exposure = len(selected) / policy.top_k if policy.allow_cash else 1.0
        new_weights = _bounded_equal_weights(
            selected,
            old_weights,
            day,
            target_exposure=target_exposure,
        )
        selected = [symbol for symbol in selected if symbol in new_weights]
        day_trades, buy_turnover, sell_turnover, transaction_cost = _build_trades(
            label_date=label_date,
            old_weights=old_weights,
            new_weights=new_weights,
            day=day,
            selection_reasons=reasons,
            cost_model=costs,
        )

        selected_returns = np.asarray(
            [day[symbol].target_return for symbol in selected], dtype=np.float64
        )
        if not np.all(np.isfinite(selected_returns)):
            missing = [
                symbol for symbol in selected if not math.isfinite(day[symbol].target_return)
            ]
            raise ValueError(f"{label_date} 选中持仓缺少收益: {missing}")
        selected_scores = np.asarray([day[symbol].score for symbol in selected], dtype=np.float64)
        weights = np.asarray([new_weights[symbol] for symbol in selected], dtype=np.float64)
        gross_return = float(np.dot(weights, selected_returns))
        if gross_return <= -1.0:
            raise ValueError(f"{label_date} 组合毛收益必须大于 -100%")
        net_return = gross_return - transaction_cost

        finite_universe = [
            row for row in rows if row.in_universe and math.isfinite(row.target_return)
        ]
        universe_return = float(np.mean([row.target_return for row in finite_universe]))
        realized_top = {
            row.symbol
            for row in sorted(
                finite_universe,
                key=lambda row: (-row.target_return, row.symbol),
            )[: min(policy.top_k, len(finite_universe))]
        }
        overlap = len(set(selected) & realized_top) / len(selected) if selected else 0.0
        selected_ic = _rank_correlation(selected_scores, selected_returns)
        one_way_turnover = (buy_turnover + sell_turnover) / 2.0

        daily_rows.append(
            {
                "trading_date": next(iter(signal_dates)).isoformat(),
                "label_date": label_date.isoformat(),
                "universe_size": sum(row.in_universe for row in rows),
                "positions": len(selected),
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "one_way_turnover": one_way_turnover,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "universe_return": universe_return,
                "active_return": gross_return - universe_return,
                "net_active_return": net_return - universe_return,
                "top_k_realized_overlap": overlap,
                "selected_rank_ic": selected_ic,
                "gross_exposure": float(np.sum(np.abs(weights))),
                "net_exposure": float(np.sum(weights)),
                "cash_weight": float(1.0 - np.sum(weights)),
                "max_weight": float(np.max(weights)) if weights.size else 0.0,
                "concentration_hhi": float(np.sum(np.square(weights))),
            }
        )
        for symbol, weight, target_return in zip(selected, weights, selected_returns, strict=True):
            row = day[symbol]
            holding_rows.append(
                {
                    "trading_date": row.trading_date.isoformat(),
                    "label_date": label_date.isoformat(),
                    "symbol": symbol,
                    "rank": ranks[symbol],
                    "score": row.score,
                    "target_return": target_return,
                    "weight": weight,
                    "gross_contribution": weight * target_return,
                    "selection_reason": reasons[symbol],
                    "can_buy": row.can_buy,
                    "can_sell": row.can_sell,
                    "in_universe": row.in_universe,
                }
            )
        trade_rows.extend(day_trades)
        old_weights = {
            symbol: new_weights[symbol] * (1.0 + day[symbol].target_return) / (1.0 + gross_return)
            for symbol in selected
        }

    gross_returns = [float(row["gross_return"]) for row in daily_rows]
    net_returns = [float(row["net_return"]) for row in daily_rows]
    finite_selected_ics = [
        float(row["selected_rank_ic"])
        for row in daily_rows
        if math.isfinite(float(row["selected_rank_ic"]))
    ]
    gross_summary = _return_summary(gross_returns, policy.annualization_days)
    net_summary = (
        gross_summary
        if all(float(row["transaction_cost"]) == 0.0 for row in daily_rows)
        else _return_summary(net_returns, policy.annualization_days)
    )
    summary = {
        "mode": "topk_long_only",
        "policy": {
            "top_k": policy.top_k,
            "exit_buffer": policy.exit_buffer,
            "min_score_gap": policy.min_score_gap,
            "min_position_score": policy.min_position_score,
            "allow_cash": policy.allow_cash,
            "min_symbols_per_day": policy.min_symbols_per_day,
            "annualization_days": policy.annualization_days,
            "missing_holding_policy": policy.missing_holding_policy,
            "require_tradability": policy.require_tradability,
            "require_universe_membership": policy.require_universe_membership,
        },
        "cost_model": {
            "per_side_bps": costs.per_side_bps,
            "sell_stamp_tax_bps": costs.sell_stamp_tax_bps,
        },
        "evaluated_dates": len(daily_rows),
        "skipped_dates": len(grouped) - len(daily_rows),
        "date_range": [daily_rows[0]["label_date"], daily_rows[-1]["label_date"]],
        "gross": gross_summary,
        "net": net_summary,
        "turnover": {
            "mean_buy": float(np.mean([row["buy_turnover"] for row in daily_rows])),
            "mean_sell": float(np.mean([row["sell_turnover"] for row in daily_rows])),
            "mean_one_way": float(np.mean([row["one_way_turnover"] for row in daily_rows])),
            "mean_transaction_cost": float(
                np.mean([row["transaction_cost"] for row in daily_rows])
            ),
        },
        "ranking": {
            "mean_top_k_realized_overlap": float(
                np.mean([row["top_k_realized_overlap"] for row in daily_rows])
            ),
            "mean_active_return": float(np.mean([row["active_return"] for row in daily_rows])),
            "positive_active_days_ratio": float(
                np.mean([row["active_return"] > 0 for row in daily_rows])
            ),
            "mean_selected_rank_ic": (
                float(np.mean(finite_selected_ics)) if finite_selected_ics else math.nan
            ),
        },
        "risk_exposure": {
            "mean_positions": float(np.mean([row["positions"] for row in daily_rows])),
            "mean_gross_exposure": float(np.mean([row["gross_exposure"] for row in daily_rows])),
            "mean_net_exposure": float(np.mean([row["net_exposure"] for row in daily_rows])),
            "mean_cash_weight": float(np.mean([row["cash_weight"] for row in daily_rows])),
            "mean_max_weight": float(np.mean([row["max_weight"] for row in daily_rows])),
            "mean_concentration_hhi": float(
                np.mean([row["concentration_hhi"] for row in daily_rows])
            ),
        },
        "monthly_stability": _monthly_stability(daily_rows),
        "extreme_days": _extreme_day_summary(daily_rows),
        "detail_rows": {
            "daily": len(daily_rows),
            "holdings": len(holding_rows),
            "trades": len(trade_rows),
        },
    }
    return PortfolioEvaluation(
        summary=summary,
        daily=tuple(daily_rows),
        holdings=tuple(holding_rows),
        trades=tuple(trade_rows),
    )


def write_portfolio_artifacts(
    evaluation: PortfolioEvaluation,
    output_dir: str | Path,
) -> dict[str, str]:
    """写出 summary JSON 以及日度、持仓和交易 Parquet。"""
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": target / "summary.json",
        "daily": target / "daily.parquet",
        "holdings": target / "holdings.parquet",
        "trades": target / "trades.parquet",
    }
    paths["summary"].write_text(
        json.dumps(evaluation.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(list(evaluation.daily)), paths["daily"])
    pq.write_table(pa.Table.from_pylist(list(evaluation.holdings)), paths["holdings"])
    pq.write_table(pa.Table.from_pylist(list(evaluation.trades)), paths["trades"])
    return {name: str(path) for name, path in paths.items()}


def evaluate_quantile_long_short(
    predictions_path: str | Path,
    *,
    quantile: float = 0.1,
    cost_bps: float = 10.0,
    stamp_tax: float = 0.0005,
    min_symbols_per_day: int = 50,
    rebalance_days: int = 1,
) -> dict[str, Any]:
    """兼容历史结论的分位数多空诊断，不作为新 Top-K 正式策略。"""
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile 应在 (0, 0.5] 内")
    if cost_bps < 0 or stamp_tax < 0:
        raise ValueError("成本参数不能为负数")
    if min_symbols_per_day < 2:
        raise ValueError("min_symbols_per_day 至少为 2")
    if rebalance_days <= 0:
        raise ValueError("rebalance_days 必须为正整数")

    table = pq.read_table(predictions_path)
    records = table.to_pylist()
    if not records:
        raise ValueError("预测明细为空")
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_as_date(record["label_date"], field="label_date")].append(record)
    eligible_dates = sorted(
        label_date for label_date, rows in grouped.items() if len(rows) >= min_symbols_per_day
    )
    if not eligible_dates:
        raise ValueError("没有可评估的交易日")

    cost_per_side = cost_bps / 10_000.0
    held: tuple[set[str], set[str]] | None = None
    evaluated_dates: list[date] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    transaction_costs: list[float] = []
    for position, label_date in enumerate(eligible_dates):
        rows = grouped[label_date]
        if position % rebalance_days == 0:
            rows.sort(key=lambda record: (-float(record["score"]), str(record["symbol"])))
            tail_count = max(1, math.floor(len(rows) * quantile))
            long_symbols = {str(record["symbol"]) for record in rows[:tail_count]}
            short_symbols = {str(record["symbol"]) for record in rows[-tail_count:]}
            if held is None:
                long_turnover = short_turnover = 1.0
            else:
                long_turnover = len(long_symbols - held[0]) / len(long_symbols)
                short_turnover = len(short_symbols - held[1]) / len(short_symbols)
            turnover = (long_turnover + short_turnover) / 2.0
            transaction_cost = turnover * (2 * cost_per_side + stamp_tax)
            held = (long_symbols, short_symbols)
        else:
            turnover = transaction_cost = 0.0
        if held is None:
            raise RuntimeError("首个交易日必须调仓")
        returns = {str(record["symbol"]): float(record["target_return"]) for record in rows}
        long_values = [returns[symbol] for symbol in held[0] if symbol in returns]
        short_values = [returns[symbol] for symbol in held[1] if symbol in returns]
        if not long_values or not short_values:
            continue
        gross_return = float(np.mean(long_values) - np.mean(short_values))
        evaluated_dates.append(label_date)
        gross_returns.append(gross_return)
        net_returns.append(gross_return - transaction_cost)
        turnovers.append(turnover)
        transaction_costs.append(transaction_cost)

    if not evaluated_dates:
        raise ValueError("没有可评估的交易日")
    return {
        "mode": "legacy_quantile_long_short_diagnostic",
        "predictions": str(predictions_path),
        "quantile": quantile,
        "cost_bps": cost_bps,
        "stamp_tax": stamp_tax,
        "evaluated_dates": len(evaluated_dates),
        "date_range": [evaluated_dates[0].isoformat(), evaluated_dates[-1].isoformat()],
        "mean_turnover": float(np.mean(turnovers)),
        "mean_transaction_cost": float(np.mean(transaction_costs)),
        "gross": _return_summary(gross_returns, 244),
        "net": _return_summary(net_returns, 244),
    }
