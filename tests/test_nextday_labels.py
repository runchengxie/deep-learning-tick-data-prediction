"""次日收益标签和日期切分测试。"""

from datetime import date

import pytest

from ticknet.nextday.labels import DailyBar, build_next_day_targets
from ticknet.nextday.splits import DateRange, WalkForwardSplit


def _bars(symbols, dates):
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, trading_date in enumerate(dates):
            open_price = 100.0
            next_move = (symbol_index - 2) * 0.01 + date_index * 0.001
            rows.append(
                DailyBar(
                    symbol=symbol,
                    trading_date=trading_date,
                    open=open_price,
                    close=open_price * (1 + next_move),
                )
            )
    return rows


def test_cross_sectional_labels_use_next_trading_day():
    dates = [date(2024, 1, day) for day in (2, 3, 4)]
    symbols = [f"S{index}" for index in range(5)]
    targets = build_next_day_targets(
        _bars(symbols, dates),
        dates,
        min_cross_section=5,
        lower_quantile=0.2,
        upper_quantile=0.8,
    )

    assert len(targets) == 10
    first_day = [target for target in targets if target.trading_date == dates[0]]
    assert {target.label_date for target in first_day} == {dates[1]}
    assert [sum(target.label == label for target in first_day) for label in range(3)] == [1, 3, 1]
    assert min(first_day, key=lambda item: item.target_return).label == 0
    assert max(first_day, key=lambda item: item.target_return).label == 2


def test_missing_immediate_next_day_is_not_bridged():
    dates = [date(2024, 1, day) for day in (2, 3, 4)]
    bars = _bars(["A", "B"], dates)
    bars = [bar for bar in bars if not (bar.symbol == "A" and bar.trading_date == dates[1])]
    targets = build_next_day_targets(
        bars,
        dates,
        label_method="fixed",
        neutral_threshold=0.0,
    )
    keys = {(target.symbol, target.trading_date, target.label_date) for target in targets}
    assert ("A", dates[0], dates[2]) not in keys
    assert all(
        not (target.symbol == "A" and target.trading_date in dates[:2]) for target in targets
    )


def test_cross_sectional_ties_are_not_broken_by_symbol_name():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    bars = []
    for symbol in ("A", "B", "C", "D"):
        bars.extend(
            [
                DailyBar(symbol, dates[0], 100, 100),
                DailyBar(symbol, dates[1], 100, 100),
            ]
        )
    targets = build_next_day_targets(bars, dates, min_cross_section=4)
    assert {target.label for target in targets} == {1}


def test_benchmark_return_is_subtracted_before_fixed_label():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    bars = [
        DailyBar("A", dates[0], 100, 100),
        DailyBar("A", dates[1], 100, 101),
    ]
    target = build_next_day_targets(
        bars,
        dates,
        label_method="fixed",
        neutral_threshold=0.002,
        benchmark_returns={dates[1]: 0.009},
    )[0]
    assert target.raw_return == pytest.approx(0.01)
    assert target.target_return == pytest.approx(0.001)
    assert target.label == 1


def test_historical_universe_filters_each_input_date():
    dates = [date(2024, 1, day) for day in (2, 3, 4)]
    bars = _bars(["A", "B", "C"], dates)
    targets = build_next_day_targets(
        bars,
        dates,
        label_method="fixed",
        neutral_threshold=0.0,
        universe={dates[0]: ["A", "B"], dates[1]: ["C"]},
    )
    assert {(target.symbol, target.trading_date) for target in targets} == {
        ("A", dates[0]),
        ("B", dates[0]),
        ("C", dates[1]),
    }


def test_walk_forward_split_purges_boundary_label():
    split = WalkForwardSplit(
        train=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
        val=DateRange(date(2024, 1, 4), date(2024, 1, 5)),
        test=DateRange(date(2024, 1, 8), date(2024, 1, 9)),
    )
    assert split.assign(date(2024, 1, 2), date(2024, 1, 3)) == "train"
    assert split.assign(date(2024, 1, 3), date(2024, 1, 4)) is None
    assert split.assign(date(2024, 1, 4), date(2024, 1, 5)) == "val"
    with pytest.raises(ValueError, match="晚于"):
        split.assign(date(2024, 1, 3), date(2024, 1, 3))


def test_walk_forward_ranges_cannot_overlap():
    with pytest.raises(ValueError, match="训练区间"):
        WalkForwardSplit(
            train=DateRange(date(2024, 1, 1), date(2024, 1, 3)),
            val=DateRange(date(2024, 1, 3), date(2024, 1, 4)),
            test=DateRange(date(2024, 1, 5), date(2024, 1, 6)),
        )
