"""确定性撮合引擎：价格优先 + 时间优先（FIFO）。

仅负责物理结算，不学习任何行为。消费保留 OrderID 的 simulator 事件，
维护十档订单簿。
"""

from __future__ import annotations

from dataclasses import dataclass

from .pack import SimulatorEvent


@dataclass
class RestingOrder:
    order_id: str
    side: int
    price: int
    volume: int
    seq: int  # 到达序号，用于时间优先


@dataclass
class Trade:
    buy_id: str
    sell_id: str
    price: int
    volume: int


class LimitOrderBook:
    """价格优先 + 时间优先的限价订单簿。"""

    def __init__(self) -> None:
        # price -> list[RestingOrder]，list 保持时间顺序
        self._bids: dict[int, list[RestingOrder]] = {}
        self._asks: dict[int, list[RestingOrder]] = {}
        self._by_id: dict[str, RestingOrder] = {}
        self._seq = 0

    def apply_order(self, order_id: str, side: int, price: int, volume: int) -> Trade | None:
        """挂单并尝试与对手方撮合。返回成交（若有）。"""
        if volume <= 0:
            return None
        if side == 1:
            trade = self._match_against(self._asks, order_id, side, price, volume)
        else:
            trade = self._match_against(self._bids, order_id, side, price, volume)
        # 未成交部分挂入深度
        remaining = volume - (trade.volume if trade else 0)
        if remaining > 0:
            book = self._bids if side == 1 else self._asks
            resting = RestingOrder(order_id, side, price, remaining, self._seq)
            self._seq += 1
            book.setdefault(price, []).append(resting)
            self._by_id[order_id] = resting
        return trade

    def _match_against(
        self,
        book: dict[int, list[RestingOrder]],
        order_id: str,
        side: int,
        price: int,
        volume: int,
    ) -> Trade | None:
        # 买单从最低卖价向上吃；卖单从最高买价向下吃
        counterparty_prices = sorted(
            (p for p in book if (p <= price if side == 1 else p >= price)),
            reverse=side == -1,
        )
        if not counterparty_prices:
            return None
        remaining = volume
        counterparty_id = ""
        trade_price = 0
        for p in counterparty_prices:
            queue = book[p]
            while queue and remaining > 0:
                top = queue[0]
                fill = min(top.volume, remaining)
                top.volume -= fill
                remaining -= fill
                counterparty_id = top.order_id
                trade_price = p
                if top.volume == 0:
                    queue.pop(0)
                    self._by_id.pop(top.order_id, None)
            if not queue:
                book.pop(p, None)
            if remaining == 0:
                break
        if remaining < volume:
            return Trade(
                buy_id=order_id if side == 1 else counterparty_id,
                sell_id=counterparty_id if side == 1 else order_id,
                price=trade_price,
                volume=volume - remaining,
            )
        return None

    def cancel_order(self, order_id: str) -> bool:
        resting = self._by_id.pop(order_id, None)
        if resting is None:
            return False
        book = self._bids if resting.side == 1 else self._asks
        queue = book.get(resting.price, [])
        queue[:] = [o for o in queue if o.order_id != order_id]
        if not queue:
            book.pop(resting.price, None)
        return True

    def seed_level(self, side: int, price: int, volume: int, order_id: str) -> None:
        """注入初始盘口档位（不撮合）。用于从真实快照重建起始账本。"""
        if volume <= 0 or price <= 0:
            return
        book = self._bids if side == 1 else self._asks
        resting = RestingOrder(order_id, side, price, volume, self._seq)
        self._seq += 1
        book.setdefault(price, []).append(resting)
        self._by_id[order_id] = resting

    def reduce_level(self, side: int, price: int, volume: int) -> bool:
        """匿名扣减指定档位数量（FIFO）。

        用于真实数据里撤单指向账本外订单的场景（如集合竞价残留单）：
        撤单消息自带价格与数量，无需订单 ID 即可对齐账本。
        """
        if volume <= 0:
            return False
        book = self._bids if side == 1 else self._asks
        queue = book.get(price)
        if not queue:
            return False
        remaining = volume
        for top in queue:
            fill = min(top.volume, remaining)
            top.volume -= fill
            remaining -= fill
            if top.volume == 0:
                self._by_id.pop(top.order_id, None)
        queue[:] = [o for o in queue if o.volume > 0]
        if not queue:
            book.pop(price, None)
        return remaining < volume

    def best_bid(self) -> tuple[int, int] | None:
        if not self._bids:
            return None
        p = max(self._bids)
        vol = sum(o.volume for o in self._bids[p])
        return (p, vol)

    def best_ask(self) -> tuple[int, int] | None:
        if not self._asks:
            return None
        p = min(self._asks)
        vol = sum(o.volume for o in self._asks[p])
        return (p, vol)


class MatchingEngine:
    """消费 simulator 事件流并维护 LOB。"""

    def __init__(self) -> None:
        self.lob = LimitOrderBook()

    def apply_order(self, order_id: str, side: int, price: int, volume: int) -> Trade | None:
        return self.lob.apply_order(order_id, side, price, volume)

    def cancel_order(self, order_id: str) -> bool:
        return self.lob.cancel_order(order_id)

    def consume(self, event: SimulatorEvent) -> Trade | None:
        if event.kind == "order":
            return self.apply_order(event.order_id, event.side, event.price, event.volume)
        if event.kind == "cancel":
            self.cancel_order(event.order_id)
        return None
