"""RED: 冲击成本估计接口。

把候选执行单（如 TWAP 子单）放入回放会话，估计真实冲击成本 / slippage。
"""

from __future__ import annotations

from ticknet.simulator.impact import ExecutionSchedule, ImpactEstimator
from ticknet.simulator.matching import MatchingEngine
from ticknet.simulator.replay import ReplaySession


def test_impact_estimator_reports_slippage():
    engine = MatchingEngine()
    session = ReplaySession(engine=engine, initial_bid=(1000, 500), initial_ask=(1001, 300))

    # 买 1 亿元等价：这里用股数近似，分 4 个子单
    schedule = ExecutionSchedule(
        side=1,
        total_volume=2000,
        n_child=4,
        start_time_ms=1000,
        end_time_ms=5000,
    )
    estimator = ImpactEstimator(session=session)
    result = estimator.estimate(schedule)

    assert result.mid_before == 1000.5  # 买一卖一中间价 (1000+1001)/2
    assert result.mid_after is not None
    assert result.slippage_bps >= 0  # 买入冲击非负
    assert result.filled_volume == 2000


def test_larger_order_has_higher_impact():
    def run(total: int) -> float:
        engine = MatchingEngine()
        session = ReplaySession(engine=engine, initial_bid=(1000, 500), initial_ask=(1001, 300))
        sched = ExecutionSchedule(
            side=1, total_volume=total, n_child=4, start_time_ms=1000, end_time_ms=5000
        )
        return ImpactEstimator(session=session).estimate(sched).slippage_bps

    small = run(500)
    large = run(2000)
    assert large >= small  # 买入量越大，冲击不小于小单
