"""RED: 生成式回放闭环。

prefix（初始盘口）+ 背景订单流（Transformer 生成）+ 外部干预订单
→ 经撮合引擎回放 → 合成市场轨迹。
"""

from __future__ import annotations

from ticknet.simulator.matching import MatchingEngine
from ticknet.simulator.replay import InterventionOrder, ReplaySession


def test_replay_session_initializes_from_snapshot():
    engine = MatchingEngine()
    ReplaySession(engine=engine, initial_bid=(1000, 500), initial_ask=(1001, 300))
    # 初始盘口应已注入撮合引擎
    assert engine.lob.best_bid() == (1000, 500)
    assert engine.lob.best_ask() == (1001, 300)


def test_intervention_order_injected_into_stream():
    engine = MatchingEngine()
    session = ReplaySession(engine=engine, initial_bid=(1000, 500), initial_ask=(1001, 300))
    # 外部买入大单，应吃掉卖一 300 股，剩余 200 挂在 1001
    session.add_intervention(
        InterventionOrder(
            time_ms=1000,
            side=1,
            price=1001,
            volume=500,
        )
    )
    session.run()
    # 卖一被吃完，买一 1000 剩 500，买一 1001 挂 200
    assert engine.lob.best_ask() is None or engine.lob.best_ask()[0] > 1001
    assert engine.lob.best_bid() == (1001, 200)


def test_background_order_stream_consumed():
    engine = MatchingEngine()
    session = ReplaySession(engine=engine, initial_bid=(1000, 500), initial_ask=(1001, 300))
    # 注入一笔背景卖单（模型生成），价格 1000，应部分吃掉买一
    session.add_background(time_ms=500, side=-1, price=1000, volume=200)
    session.run()
    assert engine.lob.best_bid() == (1000, 300)
