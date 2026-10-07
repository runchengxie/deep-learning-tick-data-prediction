"""Causal, lagged summaries of estimated holdings by cost and trading-session age."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

AGE_NAMES = ("age_1_2", "age_3_10", "age_11_100", "age_101_plus")
CHIP_FEATURE_NAMES = (
    "profit_ratio",
    "avg_cost_close_ratio",
    "peak_cost_close_ratio",
    "concentration",
    "entropy",
    "avg_age",
    *(f"{name}_mass" for name in AGE_NAMES),
    *(f"{name}_profit_mass" for name in AGE_NAMES),
    *(f"{name}_cost_mass_ratio" for name in AGE_NAMES),
)
PRICE_FEATURE_NAMES = (
    "intraday_return",
    "range_close_ratio",
    "vwap_close_ratio",
    "turnover",
    "log_volume",
)


@dataclass(frozen=True)
class DailyChipBar:
    """Published daily input: raw prices, shares, fractional turnover, adjustment factor."""

    symbol: str
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    turnover: float
    adj_factor: float

    def validate(self) -> None:
        prices = (self.open, self.high, self.low, self.close, self.vwap, self.adj_factor)
        if not self.symbol or not all(np.isfinite(value) and value > 0 for value in prices):
            raise ValueError("symbol and positive finite prices/adj_factor are required")
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("OHLC prices must lie within low/high")
        if not self.low <= self.vwap <= self.high:
            raise ValueError("VWAP must lie within low/high")
        if not np.isfinite(self.turnover) or not 0 <= self.turnover <= 1:
            raise ValueError("turnover must be a fraction in [0, 1], not percentage points")
        if not np.isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite nonnegative shares")
        if self.volume == 0 and self.turnover != 0:
            raise ValueError("zero-volume sessions must have zero turnover")


def _triangle_cdf(edges: np.ndarray, low: float, mode: float, high: float) -> np.ndarray:
    """Integrate a triangular distribution, including modes at either endpoint."""
    clipped = np.clip(edges, low, high)
    result = np.zeros_like(edges)
    if mode > low:
        left = clipped <= mode
        result[left] = (clipped[left] - low) ** 2 / ((high - low) * (mode - low))
    if mode < high:
        right = clipped > mode
        result[right] = 1 - (high - clipped[right]) ** 2 / ((high - low) * (high - mode))
    result[edges >= high] = 1
    return result


class ChipState:
    """100 recent cohorts plus an old-cohort mass and first age moment.

    The fixed log-price grid is anchored to the first observed adjusted VWAP,
    never to future price extrema. Its 1% log spacing is an approximation;
    prices outside its wide support are rejected rather than clipped.
    """

    def __init__(self, bar: DailyChipBar) -> None:
        bar.validate()
        self.costs = bar.vwap * bar.adj_factor * np.exp(np.arange(-1024, 1025) * 0.01)
        self.edges = np.concatenate((self.costs * np.exp(-0.005), [self.costs[-1] * np.exp(0.005)]))
        self.recent = np.zeros((100, len(self.costs)), dtype=np.float64)
        self.old = np.zeros_like(self.costs)
        self.old_age_mass = np.zeros_like(self.costs)
        self.recent[0] = self._injection(bar)
        self.bar = bar
        self.observations = 1
        self.initial_mass_remaining = 1.0

    def _injection(self, bar: DailyChipBar) -> np.ndarray:
        low, mode, high = np.asarray((bar.low, bar.vwap, bar.high)) * bar.adj_factor
        if low < self.edges[0] or high >= self.edges[-1]:
            raise ValueError("adjusted prices exceed the fixed chip cost grid")
        if high == low:
            mass = np.zeros_like(self.costs)
            mass[np.searchsorted(self.edges, mode, side="right") - 1] = 1
            return mass
        return np.diff(_triangle_cdf(self.edges, low, mode, high))

    def update(self, bar: DailyChipBar) -> None:
        bar.validate()
        if bar.symbol != self.bar.symbol or bar.trading_date <= self.bar.trading_date:
            raise ValueError("one symbol with strictly increasing dates is required")
        injection = self._injection(bar)
        survival = 1 - bar.turnover
        self.old_age_mass = survival * (self.old_age_mass + self.old + 101 * self.recent[-1])
        self.old = survival * (self.old + self.recent[-1])
        self.recent[1:] = survival * self.recent[:-1]
        self.recent[0] = bar.turnover * injection
        self.initial_mass_remaining *= survival
        self.observations += 1
        self.bar = bar

    def summary(self) -> np.ndarray:
        layers = np.stack(
            (
                self.recent[:2].sum(axis=0),
                self.recent[2:10].sum(axis=0),
                self.recent[10:].sum(axis=0),
                self.old,
            )
        )
        mass = layers.sum(axis=0)
        relative = self.costs / (self.bar.close * self.bar.adj_factor)
        # Treat the bin containing the close as neutral, avoiding false profits
        # caused solely by rounding a same-price acquisition to a bin center.
        profitable = self.edges[1:] <= self.bar.close * self.bar.adj_factor
        positive = mass[mass > 0]
        age = np.dot(np.arange(1, 101), self.recent.sum(axis=1)) + self.old_age_mass.sum()
        return np.asarray(
            [
                mass[profitable].sum(),
                np.dot(mass, relative),
                relative[np.argmax(mass)],
                np.dot(mass, mass),
                -np.dot(positive, np.log(positive)),
                age,
                *layers.sum(axis=1),
                *layers[:, profitable].sum(axis=1),
                *(layers @ relative),
            ],
            dtype=np.float64,
        )

    def price_summary(self) -> np.ndarray:
        bar = self.bar
        return np.asarray(
            [
                bar.close / bar.open - 1,
                (bar.high - bar.low) / bar.close,
                bar.vwap / bar.close,
                bar.turnover,
                np.log1p(bar.volume),
            ]
        )


@dataclass(frozen=True)
class ChipFeatures:
    chip: np.ndarray
    price: np.ndarray
    feature_date: date | None
    available: bool
    initial_mass_remaining: float


def build_lagged_chip_features(
    bars: Sequence[DailyChipBar],
    keys: Sequence[tuple[date, str]],
    *,
    warmup_days: int = 300,
) -> dict[tuple[date, str], ChipFeatures]:
    """Use only the immediately preceding market session for each signal date.

    A missing stock session resets state; suspended sessions must be explicit
    zero-volume rows. Missing/warmup rows remain NaN, preserving the candidate pool.
    The union of input dates is the session calendar, including non-candidate stocks.
    """
    if warmup_days < 1:
        raise ValueError("warmup_days must be positive")
    calendar = sorted({bar.trading_date for bar in bars} | {key[0] for key in keys})
    previous = dict(zip(calendar[1:], calendar[:-1], strict=True))
    by_symbol: dict[str, list[DailyChipBar]] = defaultdict(list)
    for bar in bars:
        by_symbol[bar.symbol].append(bar)
    requested: dict[str, set[date]] = defaultdict(set)
    keys_by_symbol: dict[str, list[date]] = defaultdict(list)
    result = {
        key: ChipFeatures(
            np.full(len(CHIP_FEATURE_NAMES), np.nan),
            np.full(len(PRICE_FEATURE_NAMES), np.nan),
            None,
            False,
            1.0,
        )
        for key in keys
    }
    for signal_date, symbol in keys:
        keys_by_symbol[symbol].append(signal_date)
        if signal_date in previous:
            requested[symbol].add(previous[signal_date])
    for symbol, feature_dates in requested.items():
        last_feature_date = max(feature_dates)
        states: dict[date, ChipFeatures] = {}
        state: ChipState | None = None
        for bar in sorted(by_symbol[symbol], key=lambda item: item.trading_date):
            if bar.trading_date > last_feature_date:
                break
            if state is not None and bar.trading_date == state.bar.trading_date:
                raise ValueError(f"duplicate daily bar: {symbol} {bar.trading_date}")
            if state is None or previous.get(bar.trading_date) != state.bar.trading_date:
                state = ChipState(bar)
            else:
                state.update(bar)
            if bar.trading_date in feature_dates and state.observations >= warmup_days:
                states[bar.trading_date] = ChipFeatures(
                    state.summary(),
                    state.price_summary(),
                    bar.trading_date,
                    True,
                    state.initial_mass_remaining,
                )
        for signal_date in keys_by_symbol[symbol]:
            if previous.get(signal_date) in states:
                result[(signal_date, symbol)] = states[previous[signal_date]]
    return result
