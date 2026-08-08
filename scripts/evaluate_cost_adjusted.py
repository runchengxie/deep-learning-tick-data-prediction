"""统一的 Top-K long-only 与历史分位数多空成本评估入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ticknet.research.portfolio import (
    CostModel,
    PortfolioPolicy,
    evaluate_quantile_long_short,
    evaluate_topk_portfolio,
    load_portfolio_predictions,
    write_portfolio_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-K long-only 成本评估")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        help="启用新 Top-K long-only 内核；省略时保留历史分位数多空诊断",
    )
    parser.add_argument("--exit-buffer", type=int, default=0)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument(
        "--missing-holding-policy",
        choices=("liquidate", "error"),
        default="liquidate",
    )
    parser.add_argument(
        "--require-tradability",
        action="store_true",
        help="正式结果必须提供 can_buy/can_sell 两列",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-symbols-per-day", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)

    legacy = parser.add_argument_group("历史分位数多空诊断")
    legacy.add_argument("--quantile", type=float, default=0.1)
    legacy.add_argument(
        "--stamp-tax",
        type=float,
        default=0.0005,
        help="legacy 模式的卖出印花税小数值；Top-K 模式使用 --stamp-tax-bps",
    )
    legacy.add_argument(
        "--rebalance-days",
        type=int,
        default=1,
        help="legacy 模式每 N 个交易日调仓一次",
    )
    return parser


def _run_topk(args: argparse.Namespace) -> dict[str, Any]:
    try:
        policy = PortfolioPolicy(
            top_k=args.top_k,
            exit_buffer=args.exit_buffer,
            min_score_gap=args.min_score_gap,
            min_symbols_per_day=args.min_symbols_per_day,
            missing_holding_policy=args.missing_holding_policy,
            require_tradability=args.require_tradability,
        )
        cost_model = CostModel(
            per_side_bps=args.cost_bps,
            sell_stamp_tax_bps=args.stamp_tax_bps,
        )
        evaluation = evaluate_topk_portfolio(
            load_portfolio_predictions(args.predictions),
            policy=policy,
            cost_model=cost_model,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = dict(evaluation.summary)
    result["predictions"] = str(args.predictions)
    if args.output_dir is not None:
        result["artifacts"] = write_portfolio_artifacts(evaluation, args.output_dir)
    return result


def _run_legacy(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return evaluate_quantile_long_short(
            args.predictions,
            quantile=args.quantile,
            cost_bps=args.cost_bps,
            stamp_tax=args.stamp_tax,
            min_symbols_per_day=args.min_symbols_per_day,
            rebalance_days=args.rebalance_days,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    result = _run_topk(args) if args.top_k is not None else _run_legacy(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
