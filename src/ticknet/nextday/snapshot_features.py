"""snapshot 特征工程：十档盘口归一化与动态流动性股票池。

从 ``raw_snapshot.py`` 拆出。归一化是无拟合参数的固定尺度变换，供推理链路复用；
股票池构建只用输入日之前的数据，避免标签泄漏。
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from ticknet.nextday.snapshot_config import (
    PRICE_INDICES,
    RAW_FEATURE_COLUMNS,
    VOLUME_INDICES,
    DailyPanel,
)


def normalize_lob_events(
    raw_events: np.ndarray,
    *,
    price_scale_bps: float = 100.0,
    volume_log_scale: float = 16.0,
    clip: float = 32.0,
) -> np.ndarray:
    """使用固定、无拟合参数的尺度变换保留逐 tick 盘口状态。"""
    events = np.asarray(raw_events, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != len(RAW_FEATURE_COLUMNS):
        raise ValueError(f"raw_events 应为 N × {len(RAW_FEATURE_COLUMNS)}")
    if events.shape[0] < 1:
        raise ValueError("raw_events 不能为空")
    prices = events[:, PRICE_INDICES]
    volumes = events[:, VOLUME_INDICES]
    if not np.all(np.isfinite(events)) or np.any(prices <= 0) or np.any(volumes < 0):
        raise ValueError("raw_events 包含无效价格或数量")
    reference_mid = (events[0, 0] + events[0, 2]) / 2.0
    if not math.isfinite(reference_mid) or reference_mid <= 0:
        raise ValueError("首个盘口缺少有效中间价")

    normalized = np.empty_like(events, dtype=np.float64)
    normalized[:, PRICE_INDICES] = ((prices / reference_mid - 1.0) * 10_000.0) / (price_scale_bps)
    normalized[:, VOLUME_INDICES] = np.log1p(volumes) / volume_log_scale
    np.clip(normalized, -clip, clip, out=normalized)
    return normalized.astype(np.float32)


def valid_lob_event_rows(raw_events: np.ndarray) -> np.ndarray:
    """返回价格、数量和有限值均有效的逐事件布尔掩码。"""
    events = np.asarray(raw_events)
    if events.ndim != 2 or events.shape[1] != len(RAW_FEATURE_COLUMNS):
        raise ValueError(f"raw_events 应为 N × {len(RAW_FEATURE_COLUMNS)}")
    prices = events[:, PRICE_INDICES]
    volumes = events[:, VOLUME_INDICES]
    return (
        np.all(np.isfinite(events), axis=1)
        & np.all(prices > 0, axis=1)
        & np.all(volumes >= 0, axis=1)
    )


def build_dynamic_universe(
    open_panel: DailyPanel,
    close_panel: DailyPanel,
    volume_panel: DailyPanel,
    *,
    start_date: date,
    end_date: date,
    top_n: int,
    min_history_days: int,
    liquidity_lookback_days: int,
    min_liquidity_observations: int,
) -> dict[date, tuple[str, ...]]:
    """只用输入日前数据生成历史动态流动性股票池。"""
    if (open_panel.dates, open_panel.symbols) != (close_panel.dates, close_panel.symbols):
        raise ValueError("open 和 close 日线轴不一致")
    if (open_panel.dates, open_panel.symbols) != (volume_panel.dates, volume_panel.symbols):
        raise ValueError("open 和 volume 日线轴不一致")
    if top_n < 1 or min_history_days < 1 or liquidity_lookback_days < 1:
        raise ValueError("股票池参数必须为正整数")
    if not 1 <= min_liquidity_observations <= liquidity_lookback_days:
        raise ValueError("min_liquidity_observations 超出流动性窗口")

    historical_valid = (
        np.isfinite(open_panel.values)
        & (open_panel.values > 0)
        & np.isfinite(close_panel.values)
        & (close_panel.values > 0)
        & np.isfinite(volume_panel.values)
        & (volume_panel.values > 0)
    )
    has_history = historical_valid.any(axis=0)
    first_seen = np.where(has_history, historical_valid.argmax(axis=0), len(open_panel.dates))
    universe: dict[date, tuple[str, ...]] = {}

    for day_index, trading_date in enumerate(open_panel.dates):
        if trading_date < start_date or trading_date > end_date:
            continue
        window_start = max(0, day_index - liquidity_lookback_days)
        if day_index - window_start < liquidity_lookback_days:
            continue
        prior_close = close_panel.values[window_start:day_index]
        prior_volume = volume_panel.values[window_start:day_index]
        valid = historical_valid[window_start:day_index]
        observations = valid.sum(axis=0)
        turnover = np.where(valid, prior_close * prior_volume, 0.0)
        liquidity = np.divide(
            turnover.sum(axis=0),
            observations,
            out=np.full(observations.shape, np.nan, dtype=np.float64),
            where=observations > 0,
        )
        eligible = (
            (day_index - first_seen >= min_history_days)
            & (observations >= min_liquidity_observations)
            & np.isfinite(liquidity)
            & (liquidity > 0)
        )
        candidates = np.flatnonzero(eligible)
        if candidates.size == 0:
            universe[trading_date] = ()
            continue
        order = np.argsort(liquidity[candidates], kind="mergesort")[::-1]
        selected = candidates[order[:top_n]]
        universe[trading_date] = tuple(sorted(open_panel.symbols[index] for index in selected))
    return universe
