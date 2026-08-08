"""Top-K long-only 组合评估内核的合成数据测试。"""

from datetime import date, timedelta

import pyarrow.parquet as pq
import pytest

from ticknet.research.portfolio import (
    CostModel,
    MissingHoldingPolicy,
    PortfolioPolicy,
    PortfolioPrediction,
    evaluate_topk_portfolio,
    write_portfolio_artifacts,
)


def _day(
    day_offset: int,
    scores: dict[str, float],
    *,
    returns: dict[str, float] | None = None,
    cannot_buy: set[str] | None = None,
    cannot_sell: set[str] | None = None,
) -> list[PortfolioPrediction]:
    label_date = date(2024, 1, 3) + timedelta(days=day_offset)
    trading_date = label_date - timedelta(days=1)
    returns = returns or {symbol: score / 100.0 for symbol, score in scores.items()}
    cannot_buy = cannot_buy or set()
    cannot_sell = cannot_sell or set()
    return [
        PortfolioPrediction(
            symbol=symbol,
            trading_date=trading_date,
            label_date=label_date,
            score=score,
            target_return=returns[symbol],
            can_buy=symbol not in cannot_buy,
            can_sell=symbol not in cannot_sell,
        )
        for symbol, score in scores.items()
    ]


def _policy(
    *,
    exit_buffer: int = 0,
    min_score_gap: float = 0.0,
    missing_holding_policy: MissingHoldingPolicy = "liquidate",
    require_tradability: bool = False,
) -> PortfolioPolicy:
    return PortfolioPolicy(
        top_k=2,
        exit_buffer=exit_buffer,
        min_score_gap=min_score_gap,
        min_symbols_per_day=4,
        missing_holding_policy=missing_holding_policy,
        require_tradability=require_tradability,
    )


def test_initial_entry_cost_and_zero_cost_identity() -> None:
    predictions = _day(0, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    charged = evaluate_topk_portfolio(
        predictions,
        policy=_policy(),
        cost_model=CostModel(per_side_bps=10.0, sell_stamp_tax_bps=5.0),
    )
    daily = charged.daily[0]
    assert daily["buy_turnover"] == pytest.approx(1.0)
    assert daily["sell_turnover"] == pytest.approx(0.0)
    assert daily["transaction_cost"] == pytest.approx(0.001)
    assert daily["net_return"] == pytest.approx(daily["gross_return"] - 0.001)
    assert daily["net_active_return"] == pytest.approx(
        daily["net_return"] - daily["universe_return"]
    )
    month = charged.summary["monthly_stability"]["2024-01"]
    assert month["net_active_mean"] == pytest.approx(daily["net_active_return"])
    assert "top_5_absolute_active_contribution" in charged.summary["extreme_days"]

    free = evaluate_topk_portfolio(
        predictions,
        policy=_policy(),
        cost_model=CostModel(per_side_bps=0.0, sell_stamp_tax_bps=0.0),
    )
    assert free.summary["net"] == free.summary["gross"]


def test_exit_buffer_reduces_explainable_turnover() -> None:
    predictions = _day(
        0,
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        returns={"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.01},
    )
    predictions += _day(1, {"C": 4.0, "A": 3.0, "B": 2.0, "D": 1.0})

    no_buffer = evaluate_topk_portfolio(predictions, policy=_policy(exit_buffer=0))
    buffered = evaluate_topk_portfolio(predictions, policy=_policy(exit_buffer=1))

    assert no_buffer.daily[1]["buy_turnover"] == pytest.approx(0.5)
    assert no_buffer.daily[1]["sell_turnover"] == pytest.approx(0.5)
    assert {row["symbol"] for row in no_buffer.holdings if row["label_date"] == "2024-01-04"} == {
        "A",
        "C",
    }
    assert buffered.daily[1]["one_way_turnover"] == pytest.approx(0.0)
    assert {row["symbol"] for row in buffered.holdings if row["label_date"] == "2024-01-04"} == {
        "A",
        "B",
    }


def test_score_gap_blocks_marginal_replacement() -> None:
    predictions = _day(
        0,
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        returns={"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.01},
    )
    predictions += _day(1, {"C": 3.05, "A": 3.0, "B": 2.9, "D": 1.0})
    evaluation = evaluate_topk_portfolio(
        predictions,
        policy=_policy(min_score_gap=0.2),
    )
    assert evaluation.daily[1]["one_way_turnover"] == pytest.approx(0.0)
    reasons = {
        row["symbol"]: row["selection_reason"]
        for row in evaluation.holdings
        if row["label_date"] == "2024-01-04"
    }
    assert reasons["B"] == "retained_score_gap"


def test_dynamic_universe_exit_is_explicit_trade() -> None:
    predictions = _day(0, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    predictions += _day(1, {"C": 4.0, "A": 3.0, "D": 2.0, "E": 1.0})
    evaluation = evaluate_topk_portfolio(predictions, policy=_policy())
    second_day_trades = [row for row in evaluation.trades if row["label_date"] == "2024-01-04"]
    assert any(
        row["symbol"] == "B" and row["action"] == "sell" and row["reason"] == "universe_exit"
        for row in second_day_trades
    )
    assert any(row["symbol"] == "C" and row["action"] == "buy" for row in second_day_trades)


def test_untradeable_existing_position_is_retained() -> None:
    predictions = _day(
        0,
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        returns={"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.01},
    )
    predictions += _day(
        1,
        {"C": 4.0, "A": 3.0, "D": 2.0, "B": 1.0},
        cannot_sell={"B"},
    )
    evaluation = evaluate_topk_portfolio(predictions, policy=_policy())
    holdings = [row for row in evaluation.holdings if row["label_date"] == "2024-01-04"]
    assert {row["symbol"] for row in holdings} == {"A", "B"}
    assert next(row for row in holdings if row["symbol"] == "B")["selection_reason"] == (
        "retained_untradeable"
    )


def test_untradeable_weight_is_not_implicitly_sold_to_fund_new_entry() -> None:
    predictions = _day(
        0,
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        cannot_buy={"B", "C", "D"},
    )
    predictions += _day(
        1,
        {"B": 4.0, "A": 3.0, "C": 2.0, "D": 1.0},
        cannot_sell={"A"},
    )
    evaluation = evaluate_topk_portfolio(predictions, policy=_policy())
    second_holdings = [row for row in evaluation.holdings if row["label_date"] == "2024-01-04"]
    second_trades = [row for row in evaluation.trades if row["label_date"] == "2024-01-04"]
    assert [(row["symbol"], row["weight"]) for row in second_holdings] == [("A", 1.0)]
    assert second_trades == []


def test_return_drift_is_rebalanced_and_charged() -> None:
    predictions = _day(
        0,
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        returns={"A": 0.1, "B": 0.0, "C": 0.0, "D": 0.0},
    )
    predictions += _day(1, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    evaluation = evaluate_topk_portfolio(predictions, policy=_policy())
    expected_rebalance = 0.5 * 1.1 / 1.05 - 0.5
    assert evaluation.daily[1]["buy_turnover"] == pytest.approx(expected_rebalance)
    assert evaluation.daily[1]["sell_turnover"] == pytest.approx(expected_rebalance)


def test_selected_missing_return_fails_instead_of_lookahead_filtering() -> None:
    predictions = _day(
        0,
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        returns={"A": float("nan"), "B": 0.03, "C": 0.02, "D": 0.01},
    )
    with pytest.raises(ValueError, match="选中持仓缺少收益"):
        evaluate_topk_portfolio(predictions, policy=_policy())


def test_missing_holding_can_be_strictly_rejected() -> None:
    predictions = _day(0, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    predictions += _day(1, {"A": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})
    with pytest.raises(ValueError, match=r"已有持仓.*缺失"):
        evaluate_topk_portfolio(
            predictions,
            policy=_policy(missing_holding_policy="error"),
        )


def test_formal_mode_rejects_missing_tradability_status() -> None:
    predictions = [
        PortfolioPrediction(
            symbol=row.symbol,
            trading_date=row.trading_date,
            label_date=row.label_date,
            score=row.score,
            target_return=row.target_return,
            tradability_known=False,
        )
        for row in _day(0, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    ]
    with pytest.raises(ValueError, match="正式评估要求"):
        evaluate_topk_portfolio(predictions, policy=_policy(require_tradability=True))


def test_artifacts_include_daily_holdings_trades_and_summary(tmp_path) -> None:
    predictions = _day(0, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    evaluation = evaluate_topk_portfolio(predictions, policy=_policy())
    paths = write_portfolio_artifacts(evaluation, tmp_path / "artifacts")
    assert set(paths) == {"summary", "daily", "holdings", "trades"}
    assert pq.read_table(paths["daily"]).num_rows == 1
    assert pq.read_table(paths["holdings"]).num_rows == 2
    assert pq.read_table(paths["trades"]).num_rows == 2
