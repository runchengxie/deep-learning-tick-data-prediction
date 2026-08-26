"""RED: 确定性撮合引擎正确性。

消费保留 ID 的 simulator 事件流，维护十档订单簿，
撮合结果与手算一致。
"""

from __future__ import annotations

from ticknet.simulator.matching import LimitOrderBook, MatchingEngine


def test_limit_order_resting():
    lob = LimitOrderBook()
    lob.apply_order(order_id="O1", side=1, price=1000, volume=500)
    assert lob.best_bid() == (1000, 500)
    assert lob.best_ask() is None


def test_marketable_order_executes_against_resting():
    engine = MatchingEngine()
    engine.apply_order(order_id="O1", side=1, price=1000, volume=500)
    # 卖单价格 <= 1000 才能成交；这里卖 999 应吃掉买一
    trade = engine.apply_order(order_id="O2", side=-1, price=999, volume=200)
    assert trade is not None
    assert trade.volume == 200
    # 剩余买一：1000 元剩 300 股
    assert engine.lob.best_bid() == (1000, 300)


def test_cancel_removes_specific_order():
    engine = MatchingEngine()
    engine.apply_order(order_id="O1", side=1, price=1000, volume=500)
    engine.apply_order(order_id="O2", side=1, price=1000, volume=300)
    # 只撤 O1，O2 仍在
    engine.cancel_order(order_id="O1")
    assert engine.lob.best_bid() == (1000, 300)


def test_time_priority_within_same_price():
    engine = MatchingEngine()
    engine.apply_order(order_id="O1", side=1, price=1000, volume=500)
    engine.apply_order(order_id="O2", side=1, price=1000, volume=500)
    # 卖单 1000 先吃先到者 O1
    trade = engine.apply_order(order_id="O3", side=-1, price=1000, volume=500)
    assert trade.buy_id == "O1"
    assert engine.lob.best_bid() == (1000, 500)  # O2 剩
