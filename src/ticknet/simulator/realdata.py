"""真实 L2 数据的 simulator 接入与 correctness 校验通路。

读取仓库约定的 order/trades/snapshot parquet（见 eventstream.config.day_input_files），
构造保留 OrderID 的 SimulatorPack，并逐 snapshot 校验撮合引擎重建精度。

字段与单位约定与 eventstream.pack 对齐：
- 价格在 raw L2 Parquet 中已经是整数分，直接转为 int
- 订单方向由 OrderType 推导（dataset._featurize 同规则）：
  撤单 = OrderType 属于 (-1, -11)，其 OrderID 即被撤订单的 ID；
  非撤单 ot >= 10 为卖，否则为买
- snapshot 为月度宽表，按 TradingDay + ticker 过滤后组装十档数组
- trades 不进入回放：成交已由撮合引擎在订单流内自行结算
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from ticknet.eventstream.config import MARKET_END_MS, RAW_L2_ROOT, day_input_files, day_preopen_file
from ticknet.simulator.matching import MatchingEngine
from ticknet.simulator.pack import SimulatorEvent, SimulatorPack

from .correctness import CorrectnessResult

CANCEL_TYPES = frozenset((-1, -11))
OWN_SIDE_BEST_TYPES = frozenset((3, 13))
# 深市原始 snapshot 的 time_ms 比 order/trades 事件时钟早 140ms。
# 已用 snapshot.Volume/DealNum 对 raw trades 在多股票、多交易日逐区间核对确认；
# 沪市未发现同样干净的固定偏移，因此默认保持 0ms，等待单独研究。
SNAPSHOT_EVENT_LAG_MS = 140


def is_shenzhen_ticker(ticker: str) -> bool:
    """按当前数据湖代码格式判断是否为深市证券。"""
    return ticker[:1] in {"0", "1", "2", "3"}


def default_snapshot_event_lag_ms(ticker: str) -> int:
    """返回当前已验证的市场默认 snapshot→event 时钟偏移。"""
    return SNAPSHOT_EVENT_LAG_MS if is_shenzhen_ticker(ticker) else 0


_SNAP_COLS = (
    ["ticker", "TradingDay", "time_ms"]
    + [f"BidPrice{i}" for i in range(1, 11)]
    + [f"BidVolume{i}" for i in range(1, 11)]
    + [f"AskPrice{i}" for i in range(1, 11)]
    + [f"AskVolume{i}" for i in range(1, 11)]
)


def _to_price_units(values: list) -> list[int]:
    """读取 raw L2 已缩放的整数价格，保持为分。"""
    return [round(float(v)) for v in values]


def _apply_realdata_order(engine: MatchingEngine, event: SimulatorEvent):
    """按真实深市订单语义把 order 事件应用到撮合引擎。"""
    price = event.price
    if event.order_type in OWN_SIDE_BEST_TYPES:
        own_best = engine.lob.best_bid() if event.side == 1 else engine.lob.best_ask()
        if own_best is None:
            return None
        price = own_best[0]
    return engine.apply_order(event.order_id, event.side, price, event.volume)


def _seed_engine(snapshot: SimulatorEvent) -> MatchingEngine:
    engine = MatchingEngine()
    for index, (price, volume) in enumerate(snapshot.bid_levels or ()):
        engine.lob.seed_level(1, price, volume, f"INIT-B{index}")
    for index, (price, volume) in enumerate(snapshot.ask_levels or ()):
        engine.lob.seed_level(-1, price, volume, f"INIT-A{index}")
    return engine


def _apply_realdata_event(engine: MatchingEngine, event: SimulatorEvent) -> None:
    if event.kind == "cancel":
        if engine.has_order(event.order_id):
            engine.cancel_order(event.order_id, event.volume)
        else:
            side = 1 if event.order_type == -1 else -1
            engine.lob.reduce_level(side, event.price, event.volume)
    elif event.kind == "order":
        _apply_realdata_order(engine, event)


def _compare_snapshot(
    engine: MatchingEngine,
    target: SimulatorEvent,
    *,
    baseline_comparable: bool,
) -> CorrectnessResult:
    bid = engine.lob.best_bid()
    ask = engine.lob.best_ask()
    exp_bid, exp_ask = target.expected_bid, target.expected_ask
    if not baseline_comparable:
        return CorrectnessResult(
            False,
            0,
            0,
            f"t={target.time_ms} 起点快照缺档，无法建立回放起点",
            comparable=False,
        )
    if exp_bid is None or exp_ask is None:
        return CorrectnessResult(
            False,
            0,
            0,
            f"t={target.time_ms} 快照缺档，跳过",
            comparable=False,
        )
    bid_error = 0 if bid == exp_bid else 1
    ask_error = 0 if ask == exp_ask else 1
    return CorrectnessResult(
        bid_error == 0 and ask_error == 0,
        bid_error,
        ask_error,
        f"t={target.time_ms} 重建买一={bid} 期望={exp_bid}；卖一={ask} 期望={exp_ask}",
    )


def load_day_pack(
    day: int,
    raw_root: Path = RAW_L2_ROOT,
    ticker: str = "",
) -> SimulatorPack:
    """读取某交易日的真实 L2 parquet，构造单只股票的 SimulatorPack。

    ticker 为空时取当日全部股票（数据量大，慎用）。order 表无 side 列，
    order 和 order_preopen 均按 OrderType 推导方向；撤单行转为 kind="cancel" 事件。
    """
    if not ticker:
        raise ValueError("真实全市场数据必须指定 ticker 以过滤")
    paths = day_input_files(day, Path(raw_root))
    orders = [*_read_order_events(paths["order"], ticker)]
    preopen_path = day_preopen_file(day, Path(raw_root))
    if preopen_path.exists():
        orders = [*_read_order_events(preopen_path, ticker), *orders]
    snapshots = _read_snapshot_events(paths["snap"], day, ticker)
    # 同一毫秒内订单先于快照结算：order=0, snapshot=1
    events = [*orders, *snapshots]
    events.sort(key=lambda e: (e.time_ms, 0 if e.kind != "snapshot" else 1))
    return SimulatorPack(events=events, snapshots=snapshots)


def verify_day_correctness(
    day: int,
    raw_root: Path = RAW_L2_ROOT,
    ticker: str = "",
    mode: Literal["continuous", "interval"] = "continuous",
    event_lag_ms: int | None = None,
) -> list[CorrectnessResult]:
    """回放订单流，并在每个 snapshot 处对比重建盘口与真实盘口。

    协议（与真实数据语义对齐，见模块 docstring）：
    - 委托流不含集合竞价期订单，竞价段快照为指示性价格，跳过不比
    - 真实 snapshot 的 ``time_ms`` 映射到事件流 ``time_ms + event_lag_ms``
    - 以开盘后第一个快照的十档注入初始账本（每档一个聚合单）
    - 撤单优先按 OrderID 回链；账本外订单（竞价残留）按消息自带的
      价格与数量匿名扣减对应档位
    - continuous 模式只在首个快照初始化；interval 模式每个区间从
      起点真实快照重新初始化，用于隔离局部重建误差
    """
    if event_lag_ms is None:
        event_lag_ms = default_snapshot_event_lag_ms(ticker)
    pack = load_day_pack(day, raw_root, ticker)
    snapshots = [snapshot for snapshot in pack.snapshots if 0 <= snapshot.time_ms < MARKET_END_MS]
    if len(snapshots) < 2:
        return []

    stream_events = [event for event in pack.events if event.kind in ("order", "cancel")]
    event_index = 0
    first_event_time = snapshots[0].time_ms + event_lag_ms
    while (
        event_index < len(stream_events) and stream_events[event_index].time_ms <= first_event_time
    ):
        event_index += 1

    results: list[CorrectnessResult] = []
    continuous_engine = _seed_engine(snapshots[0])
    continuous_baseline_comparable = (
        snapshots[0].expected_bid is not None and snapshots[0].expected_ask is not None
    )

    for start, target in pairwise(snapshots):
        interval_events: list[SimulatorEvent] = []
        target_event_time = target.time_ms + event_lag_ms
        while (
            event_index < len(stream_events)
            and stream_events[event_index].time_ms <= target_event_time
        ):
            interval_events.append(stream_events[event_index])
            event_index += 1

        if mode == "interval":
            baseline_comparable = start.expected_bid is not None and start.expected_ask is not None
            engine = _seed_engine(start)
            for event in interval_events:
                _apply_realdata_event(engine, event)
            results.append(
                _compare_snapshot(engine, target, baseline_comparable=baseline_comparable)
            )
            continue

        for event in interval_events:
            _apply_realdata_event(continuous_engine, event)
        results.append(
            _compare_snapshot(
                continuous_engine,
                target,
                baseline_comparable=continuous_baseline_comparable,
            )
        )

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
    prices = _to_price_units(d["Price"])
    for i in range(n):
        ot = int(d["OrderType"][i])
        oid = str(d["OrderID"][i])
        time_ms = int(d["time_ms"][i])
        price = prices[i]
        volume = int(d["Volume"][i])
        if ot in CANCEL_TYPES:
            out.append(
                SimulatorEvent(
                    time_ms=time_ms,
                    kind="cancel",
                    order_id=oid,
                    price=price,
                    volume=volume,
                    order_type=ot,
                )
            )
        else:
            side = -1 if ot >= 10 else 1
            out.append(
                SimulatorEvent(
                    time_ms=time_ms,
                    kind="order",
                    order_id=oid,
                    side=side,
                    price=price,
                    volume=volume,
                    order_type=ot,
                )
            )
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
        return (round(float(px)), int(vol))

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
