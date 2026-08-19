"""信号交易诊断使用的动态流动性成本和风格暴露。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ticknet.nextday.snapshot_io import read_wide_daily_panel
from ticknet.research.portfolio import PortfolioEvaluation
from ticknet.research.signal_diagnostics import SignalRow

CN_STOCK_VOLUME_LOT_SIZE = 100.0


@dataclass(frozen=True)
class MarketAttributes:
    """信号日可知的规模、近 20 日成交额和波动率。"""

    log_size: float
    adv20: float
    volatility20: float


def build_market_attributes(
    basic_root: str | Path,
    signals: list[SignalRow],
    *,
    lookback_days: int = 20,
    volume_lot_size: float = CN_STOCK_VOLUME_LOT_SIZE,
) -> dict[tuple[date, str], MarketAttributes]:
    """从日线宽表构造信号日属性，所有窗口都截止到信号日。"""
    if lookback_days < 2 or volume_lot_size <= 0:
        raise ValueError("lookback_days 至少为 2，volume_lot_size 必须为正数")
    if not signals:
        raise ValueError("信号为空")
    symbols = tuple(sorted({row.symbol for row in signals}))
    start = min(row.trading_date for row in signals) - timedelta(days=lookback_days * 3)
    end = max(row.trading_date for row in signals)
    root = Path(basic_root).expanduser().resolve()
    close = read_wide_daily_panel(
        root / "close_data.parquet",
        symbols=symbols,
        start_date=start,
        end_date=end,
    )
    volume = read_wide_daily_panel(
        root / "volume_data.parquet",
        symbols=symbols,
        start_date=start,
        end_date=end,
    )
    size = read_wide_daily_panel(
        root / "total_mv_data.parquet",
        symbols=symbols,
        start_date=start,
        end_date=end,
    )
    if (close.dates, close.symbols) != (volume.dates, volume.symbols) or (
        close.dates,
        close.symbols,
    ) != (size.dates, size.symbols):
        raise ValueError("close、volume 与 total_mv 日线轴不一致")
    date_index = {value: index for index, value in enumerate(close.dates)}
    symbol_index = {value: index for index, value in enumerate(close.symbols)}
    attributes: dict[tuple[date, str], MarketAttributes] = {}
    requested = {(row.trading_date, row.symbol) for row in signals}
    for trading_date, symbol in sorted(requested):
        row = date_index.get(trading_date)
        column = symbol_index.get(symbol)
        if row is None or column is None or row < lookback_days:
            continue
        close_window = close.values[row - lookback_days : row + 1, column]
        volume_window = volume.values[row - lookback_days + 1 : row + 1, column]
        turnover_close = close.values[row - lookback_days + 1 : row + 1, column]
        current_size = float(size.values[row, column])
        valid_turnover = (
            np.isfinite(turnover_close)
            & (turnover_close > 0)
            & np.isfinite(volume_window)
            & (volume_window > 0)
        )
        valid_close = np.isfinite(close_window) & (close_window > 0)
        if valid_turnover.sum() < lookback_days - 5 or not np.all(valid_close):
            continue
        returns = np.diff(np.log(close_window))
        adv20 = float(
            np.mean(turnover_close[valid_turnover] * volume_window[valid_turnover])
            * volume_lot_size
        )
        volatility = float(np.std(returns, ddof=1))
        if (
            current_size <= 0
            or adv20 <= 0
            or not all(math.isfinite(value) for value in (current_size, adv20, volatility))
        ):
            continue
        attributes[(trading_date, symbol)] = MarketAttributes(
            log_size=math.log(current_size),
            adv20=adv20,
            volatility20=volatility,
        )
    return attributes


def risk_attribution(
    evaluation: PortfolioEvaluation,
    signals: list[SignalRow],
    attributes: dict[tuple[date, str], MarketAttributes],
) -> dict[str, Any]:
    """计算持仓相对当日信号股票池的规模、流动性和波动率 z 暴露。"""
    universe: dict[date, list[str]] = defaultdict(list)
    for row in signals:
        universe[row.trading_date].append(row.symbol)
    holdings: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation.holdings:
        holdings[date.fromisoformat(str(row["trading_date"]))].append(row)

    fields = {
        "size": "log_size",
        "liquidity": "adv20",
        "volatility": "volatility20",
    }
    daily: list[dict[str, Any]] = []
    for trading_date, rows in sorted(holdings.items()):
        pool = [
            (symbol, attributes[(trading_date, symbol)])
            for symbol in universe.get(trading_date, [])
            if (trading_date, symbol) in attributes
        ]
        selected = [(row, attributes.get((trading_date, str(row["symbol"])))) for row in rows]
        selected = [(row, attr) for row, attr in selected if attr is not None]
        if len(pool) < 100 or not selected:
            continue
        result: dict[str, Any] = {
            "trading_date": trading_date.isoformat(),
            "universe_coverage": len(pool) / max(1, len(universe[trading_date])),
            "holding_coverage": len(selected) / len(rows),
        }
        total_weight = sum(float(row["weight"]) for row, _attr in selected)
        for output_name, attribute_name in fields.items():
            values = np.asarray(
                [float(getattr(attr, attribute_name)) for _symbol, attr in pool],
                dtype=np.float64,
            )
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            result[f"{output_name}_z"] = (
                sum(
                    float(row["weight"]) * (float(getattr(attr, attribute_name)) - mean) / std
                    for row, attr in selected
                )
                / total_weight
                if std > 0 and total_weight > 0
                else math.nan
            )
        daily.append(result)
    return {
        "status": "available" if daily else "unavailable",
        "industry": {
            "status": "unavailable",
            "reason": "本地数据源没有带日期的行业分类文件",
        },
        "evaluated_dates": len(daily),
        "mean_universe_coverage": (
            float(np.mean([row["universe_coverage"] for row in daily])) if daily else 0.0
        ),
        "mean_holding_coverage": (
            float(np.mean([row["holding_coverage"] for row in daily])) if daily else 0.0
        ),
        "mean_exposure_z": {
            name: (float(np.mean([row[f"{name}_z"] for row in daily])) if daily else math.nan)
            for name in fields
        },
        "daily": daily,
    }


def _impact_rate(
    notional_weight: float,
    adv20: float | None,
    *,
    portfolio_value_cny: float,
    coefficient_bps: float,
    max_impact_bps: float,
) -> tuple[float, bool]:
    if adv20 is None or not math.isfinite(adv20) or adv20 <= 0:
        return max_impact_bps / 10_000.0, False
    participation = notional_weight * portfolio_value_cny / adv20
    impact_bps = min(max_impact_bps, coefficient_bps * math.sqrt(max(0.0, participation)))
    return impact_bps / 10_000.0, True


def reprice_dynamic_cost(
    evaluation: PortfolioEvaluation,
    attributes: dict[tuple[date, str], MarketAttributes],
    *,
    portfolio_value_cny: float = 100_000_000.0,
    coefficient_bps: float = 10.0,
    max_impact_bps: float = 50.0,
) -> dict[str, Any]:
    """在固定成本之上加入按成交额参与率变化的平方根冲击成本。"""
    if min(portfolio_value_cny, coefficient_bps, max_impact_bps) <= 0:
        raise ValueError("动态成本参数必须为正数")
    daily_by_label = {str(row["label_date"]): row for row in evaluation.daily}
    costs: dict[str, float] = defaultdict(float)
    known = 0
    total = 0
    for trade in evaluation.trades:
        label_date = str(trade["label_date"])
        daily = daily_by_label[label_date]
        trading_date = date.fromisoformat(str(daily["trading_date"]))
        symbol = str(trade["symbol"])
        attribute = attributes.get((trading_date, symbol))
        impact, available = _impact_rate(
            float(trade["notional"]),
            attribute.adv20 if attribute is not None else None,
            portfolio_value_cny=portfolio_value_cny,
            coefficient_bps=coefficient_bps,
            max_impact_bps=max_impact_bps,
        )
        costs[label_date] += float(trade["transaction_cost"]) + float(trade["notional"]) * impact
        known += int(available)
        total += 1
    daily_rows: list[dict[str, Any]] = []
    for row in evaluation.daily:
        dynamic_cost = costs[str(row["label_date"])]
        net = float(row["gross_return"]) - dynamic_cost
        net_active = net - float(row["universe_return"])
        daily_rows.append(
            {
                "label_date": row["label_date"],
                "dynamic_transaction_cost": dynamic_cost,
                "net_return": net,
                "net_active_return": net_active,
            }
        )
    net_returns = [row["net_return"] for row in daily_rows]
    net_active = [row["net_active_return"] for row in daily_rows]
    return {
        "model": "fixed_cost_plus_square_root_adv20_impact",
        "adv20_contract": "close_cny_per_share_times_volume_lots_times_100",
        "portfolio_value_cny": portfolio_value_cny,
        "coefficient_bps": coefficient_bps,
        "max_impact_bps": max_impact_bps,
        "missing_liquidity_fallback": "max_impact_bps",
        "trade_liquidity_coverage": known / total if total else 1.0,
        "mean_transaction_cost": float(
            np.mean([row["dynamic_transaction_cost"] for row in daily_rows])
        ),
        "mean_net_return": float(np.mean(net_returns)),
        "mean_net_active_return": float(np.mean(net_active)),
        "cumulative_net_return": float(np.prod(1.0 + np.asarray(net_returns)) - 1.0),
        "top_5_absolute_active_contribution": _top_contribution(net_active, 5),
        "daily": daily_rows,
    }


def reprice_staggered_dynamic_cost(
    evaluation: dict[str, Any],
    attributes: dict[tuple[date, str], MarketAttributes],
    *,
    top_k: int = 100,
    per_side_bps: float = 10.0,
    sell_stamp_tax_bps: float = 5.0,
    portfolio_value_cny: float = 100_000_000.0,
    coefficient_bps: float = 10.0,
    max_impact_bps: float = 50.0,
) -> dict[str, Any]:
    """为五组 H5 cohort 加入按股票 ADV20 变化的冲击成本。"""
    rows: list[dict[str, Any]] = []
    known = 0
    total = 0
    stock_weight = 1.0 / top_k / 5.0
    for row in evaluation["rows"]:
        trading_date = date.fromisoformat(str(row["trading_date"]))
        cost = 0.0
        for symbol in row["symbols"]:
            attribute = attributes.get((trading_date, str(symbol)))
            impact, available = _impact_rate(
                stock_weight,
                attribute.adv20 if attribute is not None else None,
                portfolio_value_cny=portfolio_value_cny,
                coefficient_bps=coefficient_bps,
                max_impact_bps=max_impact_bps,
            )
            cost += stock_weight * (
                (2.0 * per_side_bps + sell_stamp_tax_bps) / 10_000.0 + 2.0 * impact
            )
            known += int(available)
            total += 1
        net_active = float(row["active_return_contribution"]) - cost
        rows.append(
            {
                "trading_date": row["trading_date"],
                "return_end_date": row["return_end_date"],
                "dynamic_transaction_cost": cost,
                "net_active_return_contribution": net_active,
            }
        )
    values = [row["net_active_return_contribution"] for row in rows]
    return {
        "model": "fixed_cost_plus_square_root_adv20_impact",
        "adv20_contract": "close_cny_per_share_times_volume_lots_times_100",
        "portfolio_value_cny": portfolio_value_cny,
        "coefficient_bps": coefficient_bps,
        "max_impact_bps": max_impact_bps,
        "trade_liquidity_coverage": known / total if total else 1.0,
        "mean_net_active_return": float(np.mean(values)),
        "cumulative_net_active_return": float(np.prod(1.0 + np.asarray(values)) - 1.0),
        "top_5_absolute_active_contribution": _top_contribution(values, 5),
        "rows": rows,
    }


def _top_contribution(values: list[float], count: int) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    total = float(np.sum(absolute))
    if total == 0:
        return 0.0
    return float(np.sum(np.sort(absolute)[-count:]) / total)
