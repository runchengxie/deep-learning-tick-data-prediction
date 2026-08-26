"""平方根冲击定律观测。

在多个 participation rate（成交量占比）下，把 TWAP 子单序列注入回放会话，
记录价格冲击，估计冲击与执行量的对数斜率（理论值约 0.5）。

纯 CPU 后处理：背景流可用 stub generator 跑通流程，真实权重就绪后替换。
"""

from __future__ import annotations

from dataclasses import dataclass

from .generator import GenerationContext, OrderGenerator
from .impact import ExecutionSchedule, ImpactEstimator
from .matching import MatchingEngine
from .replay import ReplaySession


@dataclass
class SqrtLawResult:
    participation_rates: list[float]
    impacts: list[float]
    slope: float  # log(impact) ~ slope * log(executed_volume)


def estimate_sqrt_law(
    initial_bid: tuple[int, int],
    initial_ask: tuple[int, int],
    generator: OrderGenerator,
    participation_rates: list[float],
    base_volume: int = 10000,
    window_ms: int = 300_000,
) -> SqrtLawResult:
    """在多个 participation rate 下注入 TWAP 子单，观测冲击斜率。"""
    impacts: list[float] = []
    executed: list[float] = []
    for pr in participation_rates:
        engine = MatchingEngine()
        session = ReplaySession(engine=engine, initial_bid=initial_bid, initial_ask=initial_ask)
        # 背景流填充（stub 或真实模型）
        ctx = GenerationContext(
            initial_bid=initial_bid,
            initial_ask=initial_ask,
            n_orders=16,
            history=[],
        )
        for o in generator.generate(ctx):
            session.add_background(time_ms=o.time_ms, side=o.side, price=o.price, volume=o.volume)

        total = int(base_volume * pr)
        sched = ExecutionSchedule(
            side=1,
            total_volume=total,
            n_child=8,
            start_time_ms=1000,
            end_time_ms=1000 + window_ms,
        )
        est = ImpactEstimator(session=session)
        res = est.estimate(sched)
        impacts.append(res.slippage_bps)
        executed.append(float(total))

    # 双对数拟合斜率：log(impact) = slope * log(volume) + c
    import math

    xs = [math.log(v) for v in executed]
    ys = [math.log(max(abs(i), 1e-6)) for i in impacts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx > 0 else 0.0
    return SqrtLawResult(participation_rates=participation_rates, impacts=impacts, slope=slope)
