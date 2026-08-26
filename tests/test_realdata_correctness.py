"""真实 schema 的合成 parquet 验证 realdata 映射与 correctness 校验。

字段名、单位（价格元）、snapshot 宽表布局与真实数据一致：
- order 表无 side 列，方向由 OrderType 推导，撤单行 OrderID 即被撤订单
- snapshot 为月度宽表 BidPrice1..10 等 20 个标量列，按 TradingDay 过滤
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ticknet.eventstream.config import MARKET_END_MS
from ticknet.simulator.realdata import (
    default_snapshot_event_lag_ms,
    load_day_pack,
    verify_day_correctness,
)

DAY = 20210104
TICKER = "000001"
OTHER = "999999"

ORDER_COLS = ["ticker", "TradingDay", "time_ms", "OrderID", "Price", "Volume", "OrderType"]
SNAP_COLS = (
    ["ticker", "TradingDay", "time_ms"]
    + [f"BidPrice{i}" for i in range(1, 11)]
    + [f"BidVolume{i}" for i in range(1, 11)]
    + [f"AskPrice{i}" for i in range(1, 11)]
    + [f"AskVolume{i}" for i in range(1, 11)]
)


def _order_row(t: int, oid: str, px: float, vol: int, ot: int, tk: str = TICKER) -> dict:
    return {
        "ticker": tk,
        "TradingDay": DAY,
        "time_ms": t,
        "OrderID": oid,
        "Price": px,
        "Volume": vol,
        "OrderType": ot,
    }


def _snap_row(
    t: int,
    bids: list[tuple[float, int]],
    asks: list[tuple[float, int]],
    tk: str = TICKER,
) -> dict:
    row: dict = {"ticker": tk, "TradingDay": DAY, "time_ms": t}
    for k in range(1, 11):
        row[f"BidPrice{k}"] = bids[k - 1][0] if k <= len(bids) else None
        row[f"BidVolume{k}"] = bids[k - 1][1] if k <= len(bids) else 0
        row[f"AskPrice{k}"] = asks[k - 1][0] if k <= len(asks) else None
        row[f"AskVolume{k}"] = asks[k - 1][1] if k <= len(asks) else 0
    return row


def _write_parquets(root: Path, orders: list[dict], snaps: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    odir = root / "order" / "202101"
    sdir = root / "snapshot"
    odir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    def to_table(rows: list[dict], cols: list[str]) -> pa.Table:
        cols_data = {c: [r[c] for r in rows] for c in cols}
        return pa.table(cols_data)

    order_name = f"{DAY // 10000}-{str(DAY)[4:6]}-{str(DAY)[6:]}"
    pq.write_table(to_table(orders, ORDER_COLS), odir / f"order_{order_name}.parquet")
    month = str(DAY)[:6]
    pq.write_table(to_table(snaps, SNAP_COLS), sdir / f"snapshot_{month}.parquet")


def _day_events() -> tuple[list[dict], list[dict]]:
    """买 10.00 两笔 + 9.95 一笔；卖 10.10；中间一笔卖单吃掉买一 600 股；撤掉 B。"""
    orders = [
        _order_row(100, "A", 10.00, 500, ot=1),
        _order_row(105, "E", 9.95, 700, ot=1),
        _order_row(110, "B", 10.00, 400, ot=1),
        _order_row(120, "C", 10.10, 300, ot=11),
        # 其他股票的行必须被过滤
        _order_row(130, "X", 5.00, 100, ot=1, tk=OTHER),
        # D 市价吃 600 股：A 全成交 500，B 剩 300
        _order_row(140, "D", 10.00, 600, ot=12),
        # 撤掉 B：真实数据中撤单行自带原订单的价格与剩余量
        _order_row(160, "B", 10.00, 400, ot=-1),
    ]
    snaps = [
        _snap_row(125, [(10.00, 900), (9.95, 700)], [(10.10, 300)]),
        _snap_row(150, [(10.00, 300), (9.95, 700)], [(10.10, 300)]),
        _snap_row(170, [(9.95, 700)], [(10.10, 300)]),
        # 其他股票的快照必须被过滤
        _snap_row(180, [(5.00, 100)], [(5.01, 100)], tk=OTHER),
    ]
    return orders, snaps


def test_default_snapshot_event_lag_is_market_aware():
    assert default_snapshot_event_lag_ms("000001") == 140
    assert default_snapshot_event_lag_ms("300001") == 140
    assert default_snapshot_event_lag_ms("600000") == 0
    assert default_snapshot_event_lag_ms("688001") == 0


@pytest.fixture
def realdata_root(tmp_path: Path) -> Path:
    orders, snaps = _day_events()
    _write_parquets(tmp_path, orders, snaps)
    return tmp_path


def test_load_day_pack_maps_real_schema(realdata_root: Path):
    pack = load_day_pack(DAY, realdata_root, TICKER)
    kinds = [(e.kind, e.time_ms) for e in pack.events]
    # 其他股票被过滤；撤单转为 cancel 事件
    assert ("cancel", 160) in kinds
    assert all(e.order_id != "X" for e in pack.events)
    # 价格元转分
    first_order = next(e for e in pack.events if e.kind == "order")
    assert first_order.price == 1000
    assert first_order.volume == 500
    # 方向推导：ot=1 买 / ot>=10 卖
    sides = {e.order_id: e.side for e in pack.events if e.kind == "order"}
    assert sides["A"] == 1
    assert sides["C"] == -1
    assert sides["D"] == -1
    order_types = {e.order_id: e.order_type for e in pack.events if e.kind == "order"}
    assert order_types["D"] == 12
    # snapshot 宽表组装为十档 levels + L1 期望值
    assert len(pack.snapshots) == 3
    s0 = pack.snapshots[0]
    assert s0.expected_bid == (1000, 900)
    assert s0.expected_ask == (1010, 300)
    assert s0.bid_levels is not None
    assert s0.bid_levels[1] == (995, 700)
    assert pack.snapshots[-1].expected_bid == (995, 700)


def test_verify_day_correctness_matches(realdata_root: Path):
    results = verify_day_correctness(DAY, realdata_root, TICKER, event_lag_ms=0)
    # 首个开盘快照用于注入初始账本，不参与对比
    assert len(results) == 2
    assert all(r.matched for r in results), [r.detail for r in results]


def test_verify_day_detects_mismatch(realdata_root: Path):
    orders, snaps = _day_events()
    # 篡改第二个快照的买一量，回放结果应对不上
    snaps[1]["BidVolume1"] = 299
    _write_parquets(realdata_root, orders, snaps)
    results = verify_day_correctness(DAY, realdata_root, TICKER, event_lag_ms=0)
    assert results[0].matched is False
    assert results[0].bid_error == 1


def test_verify_day_marks_missing_snapshot_not_comparable(tmp_path: Path):
    orders, snaps = _day_events()
    snaps[2]["AskPrice1"] = None
    snaps[2]["AskVolume1"] = 0
    _write_parquets(tmp_path, orders, snaps)

    results = verify_day_correctness(DAY, tmp_path, TICKER, event_lag_ms=0)

    assert results[-1].status == "not_comparable"
    assert results[-1].matched is False


def test_verify_day_interval_mode_resets_at_each_snapshot(tmp_path: Path):
    orders = [
        _order_row(130, "A", 10.00, 50, ot=1),
        _order_row(150, "B", 10.00, 50, ot=1),
    ]
    snaps = [
        _snap_row(125, [(10.00, 100)], [(10.10, 100)]),
        # 第一段故意漏掉 A，形成 mismatch；interval 模式应以这张快照重新校准。
        _snap_row(140, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(160, [(10.00, 150)], [(10.10, 100)]),
    ]
    _write_parquets(tmp_path, orders, snaps)

    results = verify_day_correctness(DAY, tmp_path, TICKER, mode="interval", event_lag_ms=0)

    assert [r.status for r in results] == ["mismatched", "matched"]


def test_verify_day_uses_own_side_best_price_for_type3_buy(tmp_path: Path):
    orders = [_order_row(130, "B", 20.00, 50, ot=3)]
    snaps = [
        _snap_row(125, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(140, [(10.00, 150)], [(10.10, 100)]),
    ]
    _write_parquets(tmp_path, orders, snaps)

    results = verify_day_correctness(
        DAY, tmp_path, TICKER, mode="interval", event_lag_ms=0
    )

    assert results[0].status == "matched"


def test_verify_day_uses_own_side_best_price_for_type13_sell(tmp_path: Path):
    orders = [_order_row(130, "S", 1.00, 50, ot=13)]
    snaps = [
        _snap_row(125, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(140, [(10.00, 100)], [(10.10, 150)]),
    ]
    _write_parquets(tmp_path, orders, snaps)

    results = verify_day_correctness(
        DAY, tmp_path, TICKER, mode="interval", event_lag_ms=0
    )

    assert results[0].status == "matched"


def test_verify_day_interval_mode_uses_snapshot_event_lag(tmp_path: Path):
    orders = [_order_row(150, "B", 10.00, 50, ot=2)]
    snaps = [
        _snap_row(125, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(140, [(10.00, 150)], [(10.10, 100)]),
    ]
    _write_parquets(tmp_path, orders, snaps)

    without_lag = verify_day_correctness(
        DAY, tmp_path, TICKER, mode="interval", event_lag_ms=0
    )
    with_lag = verify_day_correctness(
        DAY, tmp_path, TICKER, mode="interval", event_lag_ms=10
    )

    assert without_lag[0].status == "mismatched"
    assert with_lag[0].status == "matched"


def test_verify_day_interval_mode_does_not_seed_from_incomplete_snapshot(tmp_path: Path):
    orders = [_order_row(170, "B", 10.00, 50, ot=2)]
    snaps = [
        _snap_row(125, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(140, [(10.00, 100)], []),
        _snap_row(160, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(180, [(10.00, 150)], [(10.10, 100)]),
    ]
    _write_parquets(tmp_path, orders, snaps)

    results = verify_day_correctness(DAY, tmp_path, TICKER, mode="interval", event_lag_ms=0)

    assert [r.status for r in results] == [
        "not_comparable",
        "not_comparable",
        "matched",
    ]


def test_verify_day_excludes_closing_auction_snapshots(tmp_path: Path):
    orders: list[dict] = []
    snaps = [
        _snap_row(MARKET_END_MS - 20, [(10.00, 100)], [(10.10, 100)]),
        _snap_row(MARKET_END_MS - 10, [(10.00, 100)], [(10.10, 100)]),
        # 14:57 后进入收盘集合竞价，买卖指示价可相同，不属于连续撮合协议。
        _snap_row(MARKET_END_MS + 10, [(10.05, 500)], [(10.05, 500)]),
    ]
    _write_parquets(tmp_path, orders, snaps)

    results = verify_day_correctness(
        DAY, tmp_path, TICKER, mode="interval", event_lag_ms=0
    )

    assert len(results) == 1
    assert results[0].status == "matched"


def test_verify_day_skips_auction_and_handles_ghost_cancel(tmp_path: Path):
    """竞价段快照跳过；指向账本外订单的撤单按价格/数量匿名扣减。"""
    orders = [
        _order_row(100, "A", 10.00, 500, ot=1),
        _order_row(120, "C", 10.10, 300, ot=11),
        # 撤的是注入账本外的订单（ID 不在事件流中），扣 (10.00) 档 200 股
        _order_row(130, "NOPE", 10.00, 200, ot=-1),
    ]
    snaps = [
        _snap_row(-1000, [(9.90, 999)], [(9.91, 999)]),  # 竞价段：指示价，跳过
        _snap_row(125, [(10.00, 900), (9.95, 700)], [(10.10, 300)]),
        _snap_row(135, [(10.00, 700), (9.95, 700)], [(10.10, 300)]),
    ]
    _write_parquets(tmp_path, orders, snaps)
    results = verify_day_correctness(DAY, tmp_path, TICKER, event_lag_ms=0)
    assert len(results) == 1
    assert results[0].matched, results[0].detail


def test_missing_ticker_raises():
    with pytest.raises(ValueError, match="ticker"):
        load_day_pack(DAY)
