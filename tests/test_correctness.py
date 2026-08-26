"""RED: 撮合引擎正确性验证。

从 snapshot 初始化撮合引擎，回放其后的订单流，重建盘口须与
下一个真实 snapshot 在容差内一致。无真实数据时用合成序列验证逻辑。
"""

from __future__ import annotations

from ticknet.simulator.correctness import replay_and_compare
from ticknet.simulator.pack import build_simulator_pack


def _synthetic_pack():
    """构造合成序列：snapshot@0 -> 挂单 -> 撤单 -> snapshot@1000。"""
    orders = [
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
        {
            "time_ms": 300,
            "order_id": "O3",
            "price": 1000,
            "volume": 200,
            "order_type": 0,
            "side": 1,
            "last_price": 1000,
        },
    ]
    trades = []
    snapshots = [
        {
            "time_ms": 0,
            "bid": [(999, 400)],
            "ask": [(1002, 250)],
            "order_count": {"bid": [1], "ask": [1]},
        },
        # 回放后预期盘口：买一 1000(700)、卖一 1001(300)
        {
            "time_ms": 1000,
            "bid": [(1000, 700)],
            "ask": [(1001, 300)],
            "order_count": {"bid": [2], "ask": [1]},
        },
    ]
    return build_simulator_pack(orders, trades, snapshots)


def test_replay_reconstructs_matching_snapshot():
    pack = _synthetic_pack()
    # 用第一个 snapshot 初始化，回放订单，对比第二个 snapshot
    result = replay_and_compare(pack, init_snapshot_idx=0, target_snapshot_idx=1)
    assert result.matched, result.detail
    assert result.bid_error == 0
    assert result.ask_error == 0


def test_replay_detects_mismatch():
    pack = _synthetic_pack()
    # 故意改坏目标 snapshot 的期望值，应报不匹配
    pack.snapshots[1].expected_bid = (999, 999)
    result = replay_and_compare(pack, init_snapshot_idx=0, target_snapshot_idx=1)
    assert not result.matched
