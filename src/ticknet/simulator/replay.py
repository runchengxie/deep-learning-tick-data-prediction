"""生成式回放闭环。

以初始盘口为 prefix，将背景订单流（Transformer 生成）与外部干预订单
（如 TWAP 子单）按时间顺序送入确定性撮合引擎，产出合成市场轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass

from .matching import MatchingEngine


@dataclass
class InterventionOrder:
    """外部干预订单（如算法拆单子单），由用户显式插入模拟上下文。"""

    time_ms: int
    side: int
    price: int
    volume: int
    order_id: str = "INTERVENE"


@dataclass
class BackgroundOrder:
    """背景订单流中的一笔（模型自回归生成）。"""

    time_ms: int
    side: int
    price: int
    volume: int
    order_id: str = ""


@dataclass
class Tick:
    time_ms: int
    best_bid: tuple[int, int] | None
    best_ask: tuple[int, int] | None
    source: str  # "background" | "intervention" | "init"


class ReplaySession:
    """初始盘口 prefix + 背景流 + 外部干预 → 撮合回放 → 轨迹。"""

    def __init__(
        self,
        engine: MatchingEngine,
        initial_bid: tuple[int, int],
        initial_ask: tuple[int, int],
        init_time_ms: int = 0,
    ) -> None:
        self.engine = engine
        self._background: list[BackgroundOrder] = []
        self._interventions: list[InterventionOrder] = []
        self.ticks: list[Tick] = []
        # 注入初始盘口 prefix
        bid_px, bid_vol = initial_bid
        ask_px, ask_vol = initial_ask
        self.engine.apply_order("INIT_BID", 1, bid_px, bid_vol)
        self.engine.apply_order("INIT_ASK", -1, ask_px, ask_vol)
        self.ticks.append(
            Tick(init_time_ms, self.engine.lob.best_bid(), self.engine.lob.best_ask(), "init")
        )

    def add_background(
        self, time_ms: int, side: int, price: int, volume: int, order_id: str = ""
    ) -> None:
        self._background.append(BackgroundOrder(time_ms, side, price, volume, order_id))

    def add_intervention(self, order: InterventionOrder) -> None:
        self._interventions.append(order)

    def run(self) -> list[Tick]:
        """按时间顺序消费背景流与干预订单，返回合成轨迹。"""
        bg = sorted(self._background, key=lambda o: o.time_ms)
        iv = sorted(self._interventions, key=lambda o: o.time_ms)
        bi = ivi = 0
        while bi < len(bg) or ivi < len(iv):
            next_bg = bg[bi].time_ms if bi < len(bg) else None
            next_iv = iv[ivi].time_ms if ivi < len(iv) else None
            if next_iv is not None and (next_bg is None or next_iv <= next_bg):
                o = iv[ivi]
                self.engine.apply_order(o.order_id, o.side, o.price, o.volume)
                self.ticks.append(
                    Tick(
                        o.time_ms,
                        self.engine.lob.best_bid(),
                        self.engine.lob.best_ask(),
                        "intervention",
                    )
                )
                ivi += 1
            else:
                o = bg[bi]
                oid = o.order_id or f"BG{bi}"
                self.engine.apply_order(oid, o.side, o.price, o.volume)
                self.ticks.append(
                    Tick(
                        o.time_ms,
                        self.engine.lob.best_bid(),
                        self.engine.lob.best_ask(),
                        "background",
                    )
                )
                bi += 1
        return self.ticks
