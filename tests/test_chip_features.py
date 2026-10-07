from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from ticknet.nextday.chip_features import (
    CHIP_FEATURE_NAMES,
    ChipState,
    DailyChipBar,
    build_lagged_chip_features,
)


def bar(day: int, *, price: float = 10, turnover: float = 0, factor: float = 1):
    return DailyChipBar(
        symbol="000001.SZ",
        trading_date=date(2024, 1, 1) + timedelta(days=day),
        open=price,
        high=price,
        low=price,
        close=price,
        vwap=price,
        volume=100,
        turnover=turnover,
        adj_factor=factor,
    )


def summary(state):
    return dict(zip(CHIP_FEATURE_NAMES, state.summary(), strict=True))


def test_turnover_replaces_mass_and_ages_survivors():
    state = ChipState(bar(0))
    state.update(bar(1, price=20, turnover=0.5))
    state.update(bar(2, price=20))
    values = summary(state)
    assert values["age_1_2_mass"] == pytest.approx(0.5)
    assert values["age_3_10_mass"] == pytest.approx(0.5)
    assert values["profit_ratio"] == pytest.approx(0.5)
    assert values["avg_cost_close_ratio"] == pytest.approx(0.75, abs=0.005)
    assert values["avg_age"] == pytest.approx(2.5)
    assert state.initial_mass_remaining == pytest.approx(0.5)


def test_day_100_is_middle_age_and_day_101_is_long_age():
    state = ChipState(bar(0))
    for day in range(1, 100):
        state.update(bar(day))
    assert summary(state)["age_11_100_mass"] == pytest.approx(1)
    state.update(bar(100))
    assert summary(state)["age_101_plus_mass"] == pytest.approx(1)
    assert summary(state)["avg_age"] == pytest.approx(101)


def test_corporate_action_preserves_relative_costs():
    state = ChipState(bar(0))
    state.update(bar(1, price=5, factor=2))
    assert summary(state)["avg_cost_close_ratio"] == pytest.approx(1)


@pytest.mark.parametrize("turnover", [-0.01, 1.01, float("nan")])
def test_turnover_units_are_not_silently_clipped(turnover):
    with pytest.raises(ValueError, match="turnover"):
        ChipState(bar(0, turnover=turnover))


def test_invalid_vwap_and_repeated_dates_are_rejected():
    with pytest.raises(ValueError, match="VWAP"):
        ChipState(replace(bar(0), low=9, high=11, vwap=12))
    state = ChipState(bar(0))
    with pytest.raises(ValueError, match="increasing"):
        state.update(bar(0))


def test_signal_day_and_future_bars_cannot_change_features():
    bars = [bar(day, turnover=0.1) for day in range(5)]
    key = (date(2024, 1, 4), "000001.SZ")
    original = build_lagged_chip_features(bars, [key], warmup_days=2)
    changed = build_lagged_chip_features(
        [*bars[:3], bar(3, price=50), bar(4, price=100)], [key], warmup_days=2
    )
    np.testing.assert_array_equal(original[key].chip, changed[key].chip)
    np.testing.assert_array_equal(original[key].price, changed[key].price)
    assert original[key].feature_date == date(2024, 1, 3)


def test_missing_previous_session_and_warmup_keep_nan_rows():
    bars = [bar(0), bar(1), bar(3), bar(4)]
    other = replace(bar(2), symbol="000002.SZ")
    key = (date(2024, 1, 4), "000001.SZ")
    result = build_lagged_chip_features([*bars, other], [key], warmup_days=2)
    assert not result[key].available
    assert np.isnan(result[key].chip).all()
    early = build_lagged_chip_features(bars, [(date(2024, 1, 2), "000001.SZ")], warmup_days=2)
    assert not next(iter(early.values())).available


@pytest.mark.parametrize("vwap", [9.0, 10.0, 11.0])
def test_triangular_injection_conserves_mass_even_with_endpoint_modes(vwap):
    state = ChipState(replace(bar(0), low=9.0, high=11.0, vwap=vwap))
    for day in range(1, 4):
        state.update(replace(bar(day, turnover=0.25), low=9.0, high=11.0, vwap=vwap))
    values = summary(state)
    assert sum(
        values[f"{name}_mass"] for name in ("age_1_2", "age_3_10", "age_11_100", "age_101_plus")
    ) == pytest.approx(1)
    # The mean of a triangle is (low + mode + high) / 3.
    assert values["avg_cost_close_ratio"] == pytest.approx((20 + vwap) / 30, abs=0.005)
