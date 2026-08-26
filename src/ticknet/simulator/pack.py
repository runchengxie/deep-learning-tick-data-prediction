"""Simulator pack：保留原始 OrderID 的 L2 事件流。

与 eventstream pack 解耦：预测用的 pack 故意丢弃原始 ID（提炼为年龄特征），
但撮合引擎需要精确识别撤单对应的挂单，因此 simulator pack 必须保留
OrderID / BuyID / SellID / DealID。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimulatorEvent:
    time_ms: int
    kind: Literal["order", "trade", "snapshot", "cancel"]
    order_id: str = ""
    side: int = 0  # 1 买, -1 卖, 0 未知
    price: int = 0  # 分
    volume: int = 0  # 股
    order_type: int = 0
    buy_id: str = ""
    sell_id: str = ""
    deal_id: str = ""
    # snapshot 自有字段：盘口期望值，用于 correctness 验证
    expected_bid: tuple[int, int] | None = None
    expected_ask: tuple[int, int] | None = None
    # 真实数据可携带完整十档（价格分, 量股），供初始盘口注入或深度对比
    bid_levels: tuple[tuple[int, int], ...] | None = None
    ask_levels: tuple[tuple[int, int], ...] | None = None


@dataclass
class SimulatorPack:
    events: list[SimulatorEvent] = field(default_factory=list)
    # snapshot 索引，便于按时间定位初始盘口
    snapshots: list[SimulatorEvent] = field(default_factory=list)


def build_simulator_pack(
    raw_orders: Iterable[dict],
    raw_trades: Iterable[dict],
    raw_snapshots: Iterable[dict],
) -> SimulatorPack:
    """从原始 L2 风格字典构建保留 ID 的 simulator pack。"""
    events: list[SimulatorEvent] = []
    snapshots: list[SimulatorEvent] = []

    for o in raw_orders:
        ev = SimulatorEvent(
            time_ms=int(o["time_ms"]),
            kind="order",
            order_id=str(o.get("order_id", "")),
            side=int(o.get("side", 0)),
            price=int(o["price"]),
            volume=int(o["volume"]),
            order_type=int(o.get("order_type", 0)),
        )
        events.append(ev)

    for t in raw_trades:
        ev = SimulatorEvent(
            time_ms=int(t["time_ms"]),
            kind="trade",
            deal_id=str(t.get("deal_id", "")),
            buy_id=str(t.get("buy_id", "")),
            sell_id=str(t.get("sell_id", "")),
            side=int(t.get("side", 0)),
            price=int(t["price"]),
            volume=int(t["volume"]),
        )
        events.append(ev)

    for s in raw_snapshots:
        bid = s.get("bid")
        ask = s.get("ask")
        ev = SimulatorEvent(
            time_ms=int(s["time_ms"]),
            kind="snapshot",
            price=int(s.get("last_price", 0)),
            expected_bid=tuple(bid[0]) if bid else None,
            expected_ask=tuple(ask[0]) if ask else None,
        )
        events.append(ev)
        snapshots.append(ev)

    events.sort(key=lambda e: e.time_ms)
    return SimulatorPack(events=events, snapshots=snapshots)
