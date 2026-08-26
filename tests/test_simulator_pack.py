"""RED: simulator pack 必须保留原始 OrderID/BuyID/SellID/DealID，用于确定性撮合。"""

from __future__ import annotations

from ticknet.simulator.pack import SimulatorEvent, build_simulator_pack


def test_simulator_pack_preserves_order_ids():
    # 合成原始 L2 风格输入：orders / trades / snapshots，含原始 ID
    raw_orders = [
        {
            "time_ms": 100,
            "order_id": "O1",
            "price": 1000,
            "volume": 500,
            "order_type": 0,
            "side": 1,
            "last_price": 1000,
        },
        {
            "time_ms": 200,
            "order_id": "O2",
            "price": 1001,
            "volume": 300,
            "order_type": 0,
            "side": -1,
            "last_price": 1001,
        },
    ]
    raw_trades = [
        {
            "time_ms": 150,
            "deal_id": "D1",
            "buy_id": "O1",
            "sell_id": "O2",
            "price": 1000,
            "volume": 200,
            "side": 1,
        },
    ]
    raw_snapshots = [
        {
            "time_ms": 50,
            "bid": [(999, 400)],
            "ask": [(1001, 300)],
            "order_count": {"bid": [1], "ask": [1]},
        },
    ]

    pack = build_simulator_pack(raw_orders, raw_trades, raw_snapshots)

    # 关键断言：原始 ID 不得被丢弃
    events = pack.events
    by_id = {e.order_id: e for e in events if e.order_id}
    assert "O1" in by_id
    assert "O2" in by_id
    # 成交事件须关联买卖双方原始 ID
    deal = next(e for e in events if e.kind == "trade")
    assert deal.buy_id == "O1"
    assert deal.sell_id == "O2"


def test_simulator_event_has_matching_fields():
    ev = SimulatorEvent(
        time_ms=1,
        kind="order",
        order_id="O1",
        side=1,
        price=1000,
        volume=500,
        order_type=0,
    )
    assert ev.order_id == "O1"
    assert ev.price == 1000
