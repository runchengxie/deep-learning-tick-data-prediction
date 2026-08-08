"""成本后多空组合回测：读预测明细 parquet，评估换手与交易成本的影响。

输入是 ``run_minute_baseline.py --save-predictions`` 产出的 parquet，每行一个
股票日样本（symbol、label_date、target_return、score、三分类概率）。

回测口径与 ``evaluate_predictions`` 一致：每日按 score 取 top/bottom
``portfolio_quantile`` 作为多头/空头组合，等权持有到下一交易日开盘前调仓。
成本模型：

- 单边佣金与冲击成本合计 ``cost_bps``（默认 10bp），买卖双边收取；
- 卖出方向额外收取印花税（A 股现行卖出单边 0.05%，可用 ``stamp_tax`` 覆盖）；
- 换手率按相邻两日组合成分差异计算，成本 = 换手率 × 单边成本 × 2。

输出无成本与扣成本后的日度收益、累计收益、年化与夏普，便于判断 Rank IC
量级的信号在真实交易成本下是否仍为正。
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = argparse.ArgumentParser(description="成本后多空组合回测")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stamp-tax", type=float, default=0.0005)
    parser.add_argument("--min-symbols-per-day", type=int, default=50)
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=1,
        help="每 N 个交易日调仓一次（1=每日，5=周频），中间持有不换手",
    )
    args = parser.parse_args(argv)

    if not 0 < args.quantile <= 0.5:
        raise SystemExit("--quantile 应在 (0, 0.5] 内")
    if args.cost_bps < 0 or args.stamp_tax < 0:
        raise SystemExit("成本参数不能为负数")
    if args.min_symbols_per_day < 2:
        raise SystemExit("--min-symbols-per-day 至少为 2")

    table = pq.read_table(args.predictions)
    records = table.to_pylist()
    if not records:
        raise SystemExit("预测明细为空")

    by_date: dict[date, list[dict]] = defaultdict(list)
    for record in records:
        by_date[_parse_date(str(record["label_date"]))].append(record)

    cost_per_side = args.cost_bps / 10000.0
    sorted_dates = sorted(by_date)
    long_dates: list[date] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []

    eligible_dates = [
        label_date
        for label_date in sorted_dates
        if len(by_date[label_date]) >= args.min_symbols_per_day
    ]
    if not eligible_dates:
        raise SystemExit("没有可评估的交易日")

    last_rebalance_symbols: tuple[set[str], set[str]] | None = None
    for position, label_date in enumerate(eligible_dates):
        day_records = by_date[label_date]
        rebalance = position % args.rebalance_days == 0

        if rebalance:
            day_records.sort(key=lambda record: float(record["score"]), reverse=True)
            tail_count = max(1, math.floor(len(day_records) * args.quantile))
            long_records = day_records[:tail_count]
            short_records = day_records[-tail_count:]
            long_symbols = {str(record["symbol"]) for record in long_records}
            short_symbols = {str(record["symbol"]) for record in short_records}
            if last_rebalance_symbols is None:
                long_turnover = 1.0
                short_turnover = 1.0
            else:
                held_long, held_short = last_rebalance_symbols
                long_turnover = len(long_symbols - held_long) / len(long_symbols)
                short_turnover = len(short_symbols - held_short) / len(short_symbols)
            average_turnover = (long_turnover + short_turnover) / 2.0
            stamp_on_short = args.stamp_tax * average_turnover
            transaction_cost = average_turnover * cost_per_side * 2 + stamp_on_short
            last_rebalance_symbols = (long_symbols, short_symbols)
            held_long_symbols = long_symbols
            held_short_symbols = short_symbols
        else:
            average_turnover = 0.0
            transaction_cost = 0.0
            if last_rebalance_symbols is None:
                raise RuntimeError("首个交易日必须调仓")
            held_long_symbols, held_short_symbols = last_rebalance_symbols

        day_by_symbol = {
            str(record["symbol"]): float(record["target_return"]) for record in day_records
        }
        held_long_returns = [
            day_by_symbol[symbol] for symbol in held_long_symbols if symbol in day_by_symbol
        ]
        held_short_returns = [
            day_by_symbol[symbol] for symbol in held_short_symbols if symbol in day_by_symbol
        ]
        if not held_long_returns or not held_short_returns:
            continue
        long_return = float(np.mean(held_long_returns))
        short_return = float(np.mean(held_short_returns))
        spread = long_return - short_return

        long_dates.append(label_date)
        gross_returns.append(spread)
        net_returns.append(spread - transaction_cost)
        turnovers.append(average_turnover)

    if not long_dates:
        raise SystemExit("没有可评估的交易日")

    def summarize(returns: list[float]) -> dict[str, float]:
        values = np.asarray(returns, dtype=np.float64)
        mean_daily = float(np.mean(values))
        std_daily = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
        annualized = mean_daily * 244
        sharpe = (
            (mean_daily / std_daily) * math.sqrt(244)
            if math.isfinite(std_daily) and std_daily > 0
            else math.nan
        )
        cumulative = float(np.prod(1.0 + values) - 1.0)
        return {
            "mean_daily": mean_daily,
            "std_daily": std_daily,
            "annualized": annualized,
            "sharpe": sharpe,
            "cumulative": cumulative,
        }

    result = {
        "predictions": str(args.predictions),
        "quantile": args.quantile,
        "cost_bps": args.cost_bps,
        "stamp_tax": args.stamp_tax,
        "evaluated_dates": len(long_dates),
        "date_range": [long_dates[0].isoformat(), long_dates[-1].isoformat()],
        "mean_turnover": float(np.mean(turnovers)),
        "mean_transaction_cost": float(
            np.mean([net_returns[i] - gross_returns[i] for i in range(len(net_returns))])
        ),
        "gross": summarize(gross_returns),
        "net": summarize(net_returns),
    }
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
