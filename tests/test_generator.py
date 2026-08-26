"""RED: generator 接入 eventstream Transformer 生成背景订单流。

generator 给定一个 replay 上下文（初始盘口 prefix + 已发生事件），
自回归生成下一批背景订单，注入回放会话。
"""

from __future__ import annotations

from ticknet.simulator.generator import GenerationContext, OrderGenerator
from ticknet.simulator.matching import MatchingEngine
from ticknet.simulator.replay import ReplaySession


def test_generator_produces_orders_into_replay():
    # 用一个确定性的 stub 生成器：每次生成固定模式买单
    gen = OrderGenerator(model=None, stub=True)

    ctx = GenerationContext(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        history=[],
        n_orders=3,
    )
    orders = gen.generate(ctx)
    assert len(orders) == 3
    # stub 默认生成买一附近的小单
    for o in orders:
        assert o.side in (1, -1)
        assert o.volume > 0


def test_generator_orders_feed_replay_and_change_lob():
    engine = MatchingEngine()
    session = ReplaySession(engine=engine, initial_bid=(1000, 500), initial_ask=(1001, 300))
    gen = OrderGenerator(model=None, stub=True)
    ctx = GenerationContext(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        history=[],
        n_orders=2,
    )
    orders = gen.generate(ctx)
    for o in orders:
        session.add_background(time_ms=o.time_ms, side=o.side, price=o.price, volume=o.volume)
    session.run()
    # 至少产生了轨迹
    assert len(session.ticks) >= 1
