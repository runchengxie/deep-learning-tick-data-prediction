"""冲击成本估计。

把候选执行单（ExecutionSchedule）注入回放会话作为干预订单，
对比执行前后中间价，估计 slippage（基点）与成交量。
"""

from __future__ import annotations

from dataclasses import dataclass

from .replay import InterventionOrder, ReplaySession


@dataclass
class ExecutionSchedule:
    side: int  # 1 买, -1 卖
    total_volume: int
    n_child: int = 1
    start_time_ms: int = 1000
    end_time_ms: int = 5000


@dataclass
class ImpactResult:
    mid_before: float
    mid_after: float
    slippage_bps: float
    filled_volume: int


def _mid(session: ReplaySession) -> float:
    bid = session.engine.lob.best_bid()
    ask = session.engine.lob.best_ask()
    if bid is not None and ask is not None:
        return (bid[0] + ask[0]) / 2.0
    if ask is not None:
        return float(ask[0])
    if bid is not None:
        return float(bid[0])
    return 0.0
    return (bid[0] + ask[0]) / 2.0


class ImpactEstimator:
    """估计执行单对市场的冲击。"""

    def __init__(self, session: ReplaySession) -> None:
        self.session = session

    def estimate(self, schedule: ExecutionSchedule) -> ImpactResult:
        mid_before = _mid(self.session)
        child_vol = schedule.total_volume // max(schedule.n_child, 1)
        span = max(schedule.end_time_ms - schedule.start_time_ms, 1)
        for i in range(schedule.n_child):
            t = schedule.start_time_ms + span * i // max(schedule.n_child, 1)
            vol = (
                child_vol
                if i < schedule.n_child - 1
                else (schedule.total_volume - child_vol * (schedule.n_child - 1))
            )
            if vol <= 0:
                continue
            # 激进执行：市价单用对手方最优价
            if schedule.side == 1:
                ask = self.session.engine.lob.best_ask()
                price = ask[0] if ask is not None else int(mid_before + 1)
            else:
                bid = self.session.engine.lob.best_bid()
                price = bid[0] if bid is not None else int(mid_before - 1)
            self.session.add_intervention(
                InterventionOrder(time_ms=t, side=schedule.side, price=int(price), volume=vol)
            )
        self.session.run()
        mid_after = _mid(self.session)
        if mid_before > 0:
            slippage = (mid_after - mid_before) / mid_before * 10000.0
            # 买入冲击应为正（价格推高），卖出为负；统一取方向修正
            slippage *= schedule.side
        else:
            slippage = 0.0
        return ImpactResult(
            mid_before=mid_before,
            mid_after=mid_after,
            slippage_bps=slippage,
            filled_volume=schedule.total_volume,
        )
