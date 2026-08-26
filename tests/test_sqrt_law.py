"""RED: 平方根定律观测脚本。

验证 estimate_sqrt_law 能跑通流程并输出斜率。stub 生成器下斜率不保证 ~0.5，
仅验证管线连通与单调性（量越大冲击越大）。
"""

from __future__ import annotations

from ticknet.simulator.generator import OrderGenerator
from ticknet.simulator.sqrt_law import estimate_sqrt_law


def test_sqrt_law_runs_and_monotonic():
    gen = OrderGenerator(model=None, stub=True)
    result = estimate_sqrt_law(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        generator=gen,
        participation_rates=[0.001, 0.005, 0.01, 0.05],
    )
    assert len(result.impacts) == 4
    # 单调性：participation 越大，冲击不减
    for a, b in zip(result.impacts, result.impacts[1:], strict=False):
        assert b >= a
    assert isinstance(result.slope, float)


def test_sqrt_law_slope_finite():
    gen = OrderGenerator(model=None, stub=True)
    result = estimate_sqrt_law(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        generator=gen,
        participation_rates=[0.01, 0.02, 0.04, 0.08],
    )
    assert result.slope == result.slope  # 非 NaN
