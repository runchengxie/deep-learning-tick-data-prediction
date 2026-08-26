"""真实 L2 数据的 simulator 接入与 correctness 校验通路。

读取仓库约定的 order/trades/snapshot parquet（见 eventstream.config.day_input_files），
构造保留 OrderID 的 SimulatorPack，并逐 snapshot 段校验撮合引擎重建精度。

无真实数据时可用 build_simulator_pack 的合成数据走相同校验逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .correctness import CorrectnessResult, replay_and_compare
from .pack import SimulatorEvent, SimulatorPack


def load_day_pack(day: int, raw_root: Path) -> SimulatorPack:
    """读取某交易日的真实 L2 parquet，构造 SimulatorPack。

    order 表字段（分）：time_ms, price, volume, OrderType, side, LastPrice, OrderID
    trades 表字段（分）：time_ms, price, volume, side, BuyID, SellID, DealID
    snapshot 表字段：time_ms, bid_px[], ask_px[], bid_vol[], ask_vol[]（每档一列）
    """
    from ticknet.eventstream.config import day_input_files

    paths = day_input_files(day, Path(raw_root))
    orders = _read_orders(paths["order"])
    trades = _read_trades(paths["trades"])
    snapshots = _read_snapshots(paths["snap"], day)
    return SimulatorPack(events=orders + trades + snapshots, snapshots=snapshots)


def _read_orders(path: Path) -> list[SimulatorEvent]:
    if not Path(path).exists():
        return []
    t = pq.read_table(path)
    d = t.to_pydict()
    out: list[SimulatorEvent] = []
    for i in range(len(d["time_ms"])):
        out.append(
            SimulatorEvent(
                time_ms=int(d["time_ms"][i]),
                kind="order",
                order_id=str(d.get("OrderID", [None] * len(d["time_ms"]))[i] or ""),
                side=int(d.get("side", [0] * len(d["time_ms"]))[i] or 0),
                price=int(d["price"][i]),
                volume=int(d["volume"][i]),
                order_type=int(d.get("OrderType", [0] * len(d["time_ms"]))[i] or 0),
            )
        )
    return out


def _read_trades(path: Path) -> list[SimulatorEvent]:
    if not Path(path).exists():
        return []
    t = pq.read_table(path)
    d = t.to_pydict()
    out: list[SimulatorEvent] = []
    for i in range(len(d["time_ms"])):
        out.append(
            SimulatorEvent(
                time_ms=int(d["time_ms"][i]),
                kind="trade",
                deal_id=str(d.get("DealID", [None] * len(d["time_ms"]))[i] or ""),
                buy_id=str(d.get("BuyID", [None] * len(d["time_ms"]))[i] or ""),
                sell_id=str(d.get("SellID", [None] * len(d["time_ms"]))[i] or ""),
                side=int(d.get("side", [0] * len(d["time_ms"]))[i] or 0),
                price=int(d["price"][i]),
                volume=int(d["volume"][i]),
            )
        )
    return out


def _read_snapshots(path: Path, day: int) -> list[SimulatorEvent]:
    if not Path(path).exists():
        return []
    t = pq.read_table(path)
    d = t.to_pydict()
    # snapshot 按月整文件；真实数据的当日过滤字段待数据到位后补充，这里全量返回
    out: list[SimulatorEvent] = []
    times = d["time_ms"]
    for i in range(len(times)):
        # 仅在匹配当日的 snapshot 行构造（iso 信息可由调用方过滤，这里全量返回）
        bid_px = d.get("bid_px")
        ask_px = d.get("ask_px")
        bid_vol = d.get("bid_vol")
        ask_vol = d.get("ask_vol")
        exp_bid = (int(bid_px[i][0]), int(bid_vol[i][0])) if bid_px else None
        exp_ask = (int(ask_px[i][0]), int(ask_vol[i][0])) if ask_px else None
        out.append(
            SimulatorEvent(
                time_ms=int(times[i]),
                kind="snapshot",
                expected_bid=exp_bid,
                expected_ask=exp_ask,
            )
        )
    return out


def verify_day_correctness(day: int, raw_root: Path, step: int = 1) -> list[CorrectnessResult]:
    """逐相邻 snapshot 对校验撮合引擎重建精度，返回每段结果。"""
    pack = load_day_pack(day, raw_root)
    results: list[CorrectnessResult] = []
    n = len(pack.snapshots)
    for i in range(0, n - 1, max(step, 1)):
        results.append(replay_and_compare(pack, init_snapshot_idx=i, target_snapshot_idx=i + 1))
    return results
