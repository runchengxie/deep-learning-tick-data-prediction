"""RED: 真实 L2 数据 correctness 通路。

验证从真实格式数据（order/trades/snapshot parquet）构造 SimulatorPack，
再用 correctness.replay_and_compare 校验重建盘口。无真实数据时用合成
pack 验证通路逻辑；真实数据就绪后 load_day_pack 直接可用。
"""

from __future__ import annotations

from ticknet.simulator.correctness import replay_and_compare
from ticknet.simulator.pack import build_simulator_pack


def _synthetic_pack_with_snapshots():
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
    snapshots = [
        {
            "time_ms": 0,
            "bid": [(999, 400)],
            "ask": [(1002, 250)],
            "order_count": {"bid": [1], "ask": [1]},
        },
        {
            "time_ms": 1000,
            "bid": [(1000, 700)],
            "ask": [(1001, 300)],
            "order_count": {"bid": [2], "ask": [1]},
        },
    ]
    return build_simulator_pack(orders, [], snapshots)


def test_replay_compare_matches_on_synthetic():
    pack = _synthetic_pack_with_snapshots()
    result = replay_and_compare(pack, init_snapshot_idx=0, target_snapshot_idx=1)
    assert result.matched
    assert result.bid_error == 0
    assert result.ask_error == 0


def test_pack_snapshot_carries_bid_ask():
    pack = _synthetic_pack_with_snapshots()
    snap0 = pack.snapshots[0]
    assert snap0.expected_bid == (999, 400)
    assert snap0.expected_ask == (1002, 250)
    # 多 snapshot 场景下可枚举所有 snapshot 用于逐段校验
    assert len(pack.snapshots) == 2
