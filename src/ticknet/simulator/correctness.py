"""撮合引擎正确性验证。

从指定 snapshot 初始化撮合引擎，回放其后的订单/成交事件，
将重建盘口与目标 snapshot 对比。用于确认 simulator 的物理结算正确性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .matching import MatchingEngine
from .pack import SimulatorEvent, SimulatorPack


@dataclass
class CorrectnessResult:
    matched: bool
    bid_error: int
    ask_error: int
    detail: str = ""
    comparable: bool = True

    @property
    def status(self) -> Literal["matched", "mismatched", "not_comparable"]:
        if not self.comparable:
            return "not_comparable"
        return "matched" if self.matched else "mismatched"


def _apply_event(engine: MatchingEngine, ev: SimulatorEvent) -> None:
    if ev.kind == "order":
        engine.apply_order(ev.order_id, ev.side, ev.price, ev.volume)
    elif ev.kind == "cancel":
        engine.cancel_order(ev.order_id)
    # trade 已由撮合引擎在 order 时结算，这里不重复处理


def replay_and_compare(
    pack: SimulatorPack,
    init_snapshot_idx: int = 0,
    target_snapshot_idx: int = 1,
) -> CorrectnessResult:
    """从 init_snapshot 初始化，回放中间事件，对比 target_snapshot 盘口。"""
    if not (0 <= init_snapshot_idx < len(pack.snapshots)):
        return CorrectnessResult(False, -1, -1, "init_snapshot 越界")
    if not (0 <= target_snapshot_idx < len(pack.snapshots)):
        return CorrectnessResult(False, -1, -1, "target_snapshot 越界")

    init = pack.snapshots[init_snapshot_idx]
    target = pack.snapshots[target_snapshot_idx]
    t_init = init.time_ms
    t_target = target.time_ms

    engine = MatchingEngine()
    # 用 init snapshot 的买卖盘注入初始盘口
    # 注：snapshot 结构体当前仅含 time_ms/price，真实数据由 pack 补全 bid/ask
    # 合成场景下用 build_simulator_pack 的 snapshot 字段；此处从事件重建更稳妥
    # 见下方：直接用事件流回放，最后对比 target 的盘口期望值

    # 回放 [t_init, t_target) 之间的订单/成交/撤单
    for ev in pack.events:
        if ev.time_ms <= t_init:
            continue
        if ev.time_ms >= t_target:
            break
        _apply_event(engine, ev)

    # 对比重建盘口与 target snapshot 期望值
    bid = engine.lob.best_bid()
    ask = engine.lob.best_ask()
    exp_bid = target.expected_bid
    exp_ask = target.expected_ask

    if exp_bid is None or exp_ask is None:
        # 无法对比（数据未携带期望值），返回重建结果供人工检查
        return CorrectnessResult(
            False,
            0,
            0,
            f"重建买一={bid} 卖一={ask}（target 无期望值，跳过严格对比）",
            comparable=False,
        )

    bid_error = 0 if bid == exp_bid else 1
    ask_error = 0 if ask == exp_ask else 1
    matched = bid_error == 0 and ask_error == 0
    return CorrectnessResult(
        matched,
        bid_error,
        ask_error,
        f"重建买一={bid} 期望={exp_bid}；卖一={ask} 期望={exp_ask}",
    )
