"""正式 open-to-following-open 标签与交易状态测试。"""

from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.nextday.formal_targets import (
    FormalMarketPanels,
    build_formal_next_open_targets,
    open_execution_state,
)
from ticknet.nextday.snapshot_config import DailyPanel, SnapshotPreparationConfig


def _panel(dates, symbols, values) -> DailyPanel:
    return DailyPanel(tuple(dates), tuple(symbols), np.asarray(values, dtype=np.float64))


def _market_panels():
    dates = [date(2025, 1, 2) + timedelta(days=index) for index in range(6)]
    symbols = ("000001", "000002", "000003")
    opens = np.full((6, 3), 100.0)
    highs = opens.copy()
    lows = opens.copy()
    closes = opens.copy()
    volumes = np.asarray(
        [
            [100.0, 90.0, 10.0],
            [100.0, 1.0, 200.0],
            [100.0, 1.0, 200.0],
            [100.0, 1.0, 200.0],
            [100.0, 1.0, 200.0],
            [100.0, 1.0, 200.0],
        ]
    )
    # 000002 在第 4 个交易日一字跌停，调出股票池后无法卖出；次日恢复交易。
    opens[3, 1] = highs[3, 1] = lows[3, 1] = closes[3, 1] = 90.0
    opens[4, 1] = highs[4, 1] = lows[4, 1] = closes[4, 1] = 90.0
    return (
        dates,
        FormalMarketPanels(
            open=_panel(dates, symbols, opens),
            high=_panel(dates, symbols, highs),
            low=_panel(dates, symbols, lows),
            close=_panel(dates, symbols, closes),
            volume=_panel(dates, symbols, volumes),
            st=_panel(dates, symbols, np.zeros((6, 3))),
        ),
    )


def _benchmark(path, dates) -> None:
    pq.write_table(
        pa.table(
            {
                "trade_date": [value.strftime("%Y%m%d") for value in dates],
                "open": np.full(len(dates), 1000.0),
            }
        ),
        path,
    )


def test_open_execution_state_handles_suspension_and_one_price_limits() -> None:
    suspended = open_execution_state(
        symbol="600000",
        trading_date=date(2025, 1, 3),
        open_price=float("nan"),
        high_price=float("nan"),
        low_price=float("nan"),
        volume=0.0,
        previous_close=10.0,
        is_st=False,
    )
    assert suspended is not None
    assert suspended.mark_price == 10.0
    assert (suspended.can_buy, suspended.can_sell, suspended.status) == (
        False,
        False,
        "suspended",
    )

    limit_up = open_execution_state(
        symbol="600000",
        trading_date=date(2025, 1, 3),
        open_price=11.0,
        high_price=11.0,
        low_price=11.0,
        volume=100.0,
        previous_close=10.0,
        is_st=False,
    )
    assert limit_up is not None
    assert (limit_up.can_buy, limit_up.can_sell, limit_up.status) == (
        False,
        True,
        "one_price_limit_up",
    )

    limit_down = open_execution_state(
        symbol="300001",
        trading_date=date(2025, 1, 3),
        open_price=8.0,
        high_price=8.0,
        low_price=8.0,
        volume=100.0,
        previous_close=10.0,
        is_st=False,
    )
    assert limit_down is not None
    assert (limit_down.can_buy, limit_down.can_sell, limit_down.status) == (
        True,
        False,
        "one_price_limit_down",
    )

    beijing_limit_up = open_execution_state(
        symbol="830001",
        trading_date=date(2025, 1, 3),
        open_price=13.0,
        high_price=13.0,
        low_price=13.0,
        volume=100.0,
        previous_close=10.0,
        is_st=False,
    )
    assert beijing_limit_up is not None
    assert beijing_limit_up.status == "one_price_limit_up"


def test_formal_targets_keep_unsellable_removed_symbol_until_exit(tmp_path) -> None:
    dates, panels = _market_panels()
    benchmark = tmp_path / "benchmark.parquet"
    _benchmark(benchmark, dates)
    config = SnapshotPreparationConfig(
        snapshot_root="unused",
        basic_root="unused",
        benchmark_path=str(benchmark),
        output_dir="unused",
        start_date=dates[1].isoformat(),
        end_date=dates[5].isoformat(),
        top_n=2,
        min_history_days=1,
        liquidity_lookback_days=1,
        min_liquidity_observations=1,
        min_cross_section=2,
    )
    targets, universe, report = build_formal_next_open_targets(config, panels)

    candidates = [target for target in targets if target.in_universe]
    statuses = [target for target in targets if not target.in_universe]
    assert report.complete_signal_dates == 3
    assert report.candidate_targets == 6
    assert report.status_only_targets == 2
    assert all(
        sum(target.label_date == label_date for target in candidates) == 2
        for label_date in {target.label_date for target in candidates}
    )
    assert set(universe[dates[1]]) == {"000001", "000002"}
    assert set(universe[dates[2]]) == {"000001", "000003"}

    blocked, resolved = [target for target in statuses if target.symbol == "000002"]
    assert blocked.trading_date == dates[2]
    assert blocked.execution_status == "one_price_limit_down"
    assert blocked.can_sell is False
    assert blocked.portfolio_return == pytest.approx(0.0)
    assert resolved.trading_date == dates[3]
    assert resolved.can_sell is True
    assert resolved.return_end_date == dates[5]


def test_formal_market_panels_require_identical_axes() -> None:
    dates, panels = _market_panels()
    mismatched = _panel(dates[:-1], panels.open.symbols, np.ones((5, 3)))
    with pytest.raises(ValueError, match="high 日线轴"):
        FormalMarketPanels(
            open=panels.open,
            high=mismatched,
            low=panels.low,
            close=panels.close,
            volume=panels.volume,
            st=panels.st,
        )
