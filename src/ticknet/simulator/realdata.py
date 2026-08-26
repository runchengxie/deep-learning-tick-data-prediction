"""真实 L2 数据的 simulator 接入与 correctness 校验通路。

读取仓库约定的 order/trades/snapshot parquet（见 eventstream.config.day_input_files），
构造保留 OrderID 的 SimulatorPack，并逐 snapshot 校验撮合引擎重建精度。

字段与单位约定与 eventstream.pack 对齐：
- 价格原始单位为元，统一转分（round(x * 100)）
- 订单方向由 OrderType 推导（dataset._featurize 同规则）：
  撤单 = OrderType 属于 (-1, -11)，其 OrderID 即被撤订单的 ID；
  非撤单 ot >= 10 为卖，否则为买
- snapshot 为月度宽表，按 TradingDay + ticker 过滤后组装十档数组
- trades 不进入回放：成交已由撮合引擎在订单流内自行结算
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from ticknet.eventstream.config import RAW_L2_ROOT, day_input_files
from ticknet.simulator.matching import MatchingEngine
from ticknet.simulator.pack import SimulatorEvent, SimulatorPack

from .correctness import CorrectnessResult

CANCEL_TYPES = frozenset((-1, -11))
_SNAP_COLS = (
    ["ticker", "TradingDay", "time_ms"]
    + [f"BidPrice{i}" for i in range(1, 11)]
    + [f"BidVolume{i}" for i in range(1, 11)]
    + [f"AskPrice{i}" for i in range(1, 11)]
    + [f"AskVolume{i}" for i in range(1, 11)]
)


def _to_cents(values: list) -> list[int]:
    """原始价格单位为元，转分。"""
    return [round(float(v) * 100) for v in values]


def load_day_pack(
    day: int,
    raw_root: Path = RAW_L2_ROOT,
    ticker: str = "",
) -> SimulatorPack:
    """读取某交易日的真实 L2 parquet，构造单只股票的 SimulatorPack。

    ticker 为空时取当日全部股票（数据量大，慎用）。order 表无 side 列，
    方向按 OrderType 规则推导；撤单行转为 kind="cancel" 事件。
    """
    if not ticker:
        raise ValueError("真实全市场数据必须指定 ticker 以过滤")
    paths = day_input_files(day, Path(raw_root))
    orders = _read_order_events(paths["order"], ticker)
    snapshots = _read_snapshot_events(paths["snap"], day, ticker)
    # 同一毫秒内订单先于快照结算：order=0, snapshot=1
    events = [*orders, *snapshots]
    events.sort(key=lambda e: (e.time_ms, 0 if e.kind != "snapshot" else 1))
    return SimulatorPack(events=events, snapshots=snapshots)


def verify_day_correctness(
    day: int,
    raw_root: Path = RAW_L2_ROOT,
    ticker: str = "",
) -> list[CorrectnessResult]:
    """全天连续回放，在每个 snapshot 处对比重建盘口与真实盘口。

    协议（与真实数据语义对齐，见模块 docstring）：
    - 委托流不含集合竞价期订单，竞价段快照为指示性价格，跳过不比
    - 以开盘后第一个快照的十档注入初始账本（每档一个聚合单）
    - 撤单优先按 OrderID 回链；账本外订单（竞价残留）按消息自带的
      价格与数量匿名扣减对应档位
    """
    pack = load_day_pack(day, raw_root, ticker)
    results: list[CorrectnessResult] = []
    engine: MatchingEngine | None = None
    for ev in pack.events:
        if engine is None:
            # 开盘后首个快照之前的委托属于集合竞价（其成交不在委托流），
            # 全部忽略，从该快照的十档注入干净的初始账本
            if ev.kind == "snapshot" and ev.time_ms >= 0:
                engine = MatchingEngine()
                lob = engine.lob
                for i, (p, v) in enumerate(ev.bid_levels or ()):
                    lob.seed_level(1, p, v, f"INIT-B{i}")
                for i, (p, v) in enumerate(ev.ask_levels or ()):
                    lob.seed_level(-1, p, v, f"INIT-A{i}")
            continue
        if ev.kind == "snapshot":
            bid = engine.lob.best_bid()
            ask = engine.lob.best_ask()
            exp_bid, exp_ask = ev.expected_bid, ev.expected_ask
            if exp_bid is None or exp_ask is None:
                results.append(CorrectnessResult(True, 0, 0, f"t={ev.time_ms} 快照缺档，跳过"))
                continue
            bid_error = 0 if bid == exp_bid else 1
            ask_error = 0 if ask == exp_ask else 1
            results.append(
                CorrectnessResult(
                    bid_error == 0 and ask_error == 0,
                    bid_error,
                    ask_error,
                    f"t={ev.time_ms} 重建买一={bid} 期望={exp_bid}；卖一={ask} 期望={exp_ask}",
                )
            )
            continue
        if ev.kind == "cancel":
            if not engine.cancel_order(ev.order_id):
                # 账本外订单：按撤单自带的价格/数量匿名扣减
                side = 1 if ev.order_type == -1 else -1
                engine.lob.reduce_level(side, ev.price, ev.volume)
        elif ev.kind == "order":
            engine.apply_order(ev.order_id, ev.side, ev.price, ev.volume)
    return results


def _read_order_events(path: Path, ticker: str) -> list[SimulatorEvent]:
    if not Path(path).exists():
        raise FileNotFoundError(f"order parquet 不存在: {path}")
    t = pq.read_table(
        path,
        columns=["ticker", "time_ms", "OrderID", "Price", "Volume", "OrderType"],
        filters=[("ticker", "=", ticker)],
    )
    d = t.to_pydict()
    out: list[SimulatorEvent] = []
    n = len(d["time_ms"])
    prices = _to_cents(d["Price"])
    for i in range(n):
        ot = int(d["OrderType"][i])
        oid = str(d["OrderID"][i])
        common = {
            "time_ms": int(d["time_ms"][i]),
            "price": prices[i],
            "volume": int(d["Volume"][i]),
        }
        if ot in CANCEL_TYPES:
            out.append(SimulatorEvent(kind="cancel", order_id=oid, order_type=ot, **common))
        else:
            side = -1 if ot >= 10 else 1
            out.append(SimulatorEvent(kind="order", order_id=oid, side=side, **common))
    return out


def _read_snapshot_events(path: Path, day: int, ticker: str) -> list[SimulatorEvent]:
    if not Path(path).exists():
        raise FileNotFoundError(f"snapshot parquet 不存在: {path}")
    t = pq.read_table(
        path,
        columns=_SNAP_COLS,
        filters=[("ticker", "=", ticker), ("TradingDay", "=", int(day))],
    )
    if t.num_rows == 0:
        return []
    d = t.to_pydict()
    n = t.num_rows
    out: list[SimulatorEvent] = []

    def level(px_key: str, vol_key: str, i: int) -> tuple[int, int] | None:
        px, vol = d[px_key][i], d[vol_key][i]
        if px is None or float(px) <= 0:
            return None
        return (round(float(px) * 100), int(vol))

    for i in range(n):
        bid_levels = tuple(
            lv
            for k in range(1, 11)
            if (lv := level(f"BidPrice{k}", f"BidVolume{k}", i)) is not None
        )
        ask_levels = tuple(
            lv
            for k in range(1, 11)
            if (lv := level(f"AskPrice{k}", f"AskVolume{k}", i)) is not None
        )
        out.append(
            SimulatorEvent(
                time_ms=int(d["time_ms"][i]),
                kind="snapshot",
                price=bid_levels[0][0] if bid_levels else 0,
                expected_bid=bid_levels[0] if bid_levels else None,
                expected_ask=ask_levels[0] if ask_levels else None,
                bid_levels=bid_levels or None,
                ask_levels=ask_levels or None,
            )
        )
    out.sort(key=lambda e: e.time_ms)
    return out
