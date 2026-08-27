"""扫描 raw L2 盘前订单覆盖和关联文件完整性。"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import pyarrow.parquet as pq

from ticknet.eventstream.config import RAW_L2_ROOT

_DAY_FILE_RE = re.compile(r"order_(\d{4})-(\d{2})-(\d{2})\.parquet$")
_CANCEL_TYPES = frozenset((-1, -11))


class RelatedPaths(TypedDict):
    order: Path
    trades: Path
    snapshot: Path | None


@dataclass(frozen=True)
class CoverageRow:
    """一个交易日和股票的 raw L2 覆盖状态。"""

    day: int
    ticker: str
    year: int
    month: str
    market: str
    batch: str
    preopen_file_present: bool
    preopen_ticker_present: bool
    order_file_present: bool
    order_ticker_present: bool
    trades_file_present: bool
    trades_ticker_present: bool
    snapshot_file_present: bool
    snapshot_ticker_present: bool
    preopen_order_count: int
    preopen_order_volume: int
    opening_trade_count: int
    opening_trade_volume: int


def market_for_ticker(ticker: str) -> str:
    """按当前数据湖代码格式返回市场名。"""
    return "shenzhen" if ticker[:1] in {"0", "1", "2", "3"} else "shanghai"


def scan_preopen_coverage(
    raw_root: Path = RAW_L2_ROOT,
    *,
    limit_days: int | None = None,
) -> tuple[CoverageRow, ...]:
    """扫描所有盘前文件，并关联订单、成交和快照文件。"""
    preopen_root = Path(raw_root) / "order_preopen"
    paths = sorted(preopen_root.glob("*/order_*.parquet"))
    paths = paths[:limit_days] if limit_days is not None else paths
    rows: list[CoverageRow] = []
    for preopen_path in paths:
        parsed = _parse_day(preopen_path)
        if parsed is None:
            continue
        day, year, month = parsed
        aggregates = _read_preopen_aggregates(preopen_path)
        preopen_tickers = set(aggregates)
        related = _related_paths(Path(raw_root), day)
        order_tickers = (
            _read_order_coverage(related["order"], preopen_tickers)
            if related["order"].exists()
            else set()
        )
        trade_coverage = (
            _read_trades_coverage(related["trades"], preopen_tickers)
            if related["trades"].exists()
            else {}
        )
        snapshot_tickers = (
            _read_snapshot_coverage(related["snapshot"], day, preopen_tickers)
            if related["snapshot"] is not None
            else set()
        )
        for ticker, (order_count, order_volume) in sorted(aggregates.items()):
            order_present = related["order"].exists()
            trades_present = related["trades"].exists()
            snapshot_present = related["snapshot"] is not None
            order_ticker = ticker in order_tickers
            trades_ticker = ticker in trade_coverage
            opening_count, opening_volume = trade_coverage.get(ticker, (0, 0))
            snapshot_ticker = ticker in snapshot_tickers
            rows.append(
                CoverageRow(
                    day=day,
                    ticker=ticker,
                    year=year,
                    month=month,
                    market=market_for_ticker(ticker),
                    batch=preopen_path.parent.name,
                    preopen_file_present=True,
                    preopen_ticker_present=True,
                    order_file_present=order_present,
                    order_ticker_present=order_ticker,
                    trades_file_present=trades_present,
                    trades_ticker_present=trades_ticker,
                    snapshot_file_present=snapshot_present,
                    snapshot_ticker_present=snapshot_ticker,
                    preopen_order_count=order_count,
                    preopen_order_volume=order_volume,
                    opening_trade_count=opening_count,
                    opening_trade_volume=opening_volume,
                )
            )
    return tuple(rows)


def summarize_coverage(rows: Sequence[CoverageRow]) -> dict[str, dict[str, dict[str, int]]]:
    """按年份、月份、市场和文件批次汇总覆盖行。"""
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for dimension in ("year", "month", "market", "batch"):
        groups: dict[str, dict[str, int]] = defaultdict(_empty_group)
        for row in rows:
            group = groups[str(getattr(row, dimension))]
            group["samples"] += 1
            group["preopen_tickers"] += int(row.preopen_ticker_present)
            group["opening_trade_samples"] += int(row.opening_trade_count > 0)
            group["complete_file_samples"] += int(
                row.order_file_present and row.trades_file_present and row.snapshot_file_present
            )
            group["complete_ticker_samples"] += int(
                row.order_ticker_present
                and row.trades_ticker_present
                and row.snapshot_ticker_present
            )
            group["preopen_order_count"] += row.preopen_order_count
            group["preopen_order_volume"] += row.preopen_order_volume
            group["opening_trade_count"] += row.opening_trade_count
            group["opening_trade_volume"] += row.opening_trade_volume
        summary[dimension] = dict(sorted(groups.items()))
    return summary


def coverage_row_dict(row: CoverageRow) -> dict:
    """返回适合 JSON 或 CSV 的稳定字段字典。"""
    return asdict(row)


def _empty_group() -> dict[str, int]:
    return {
        "samples": 0,
        "preopen_tickers": 0,
        "opening_trade_samples": 0,
        "complete_file_samples": 0,
        "complete_ticker_samples": 0,
        "preopen_order_count": 0,
        "preopen_order_volume": 0,
        "opening_trade_count": 0,
        "opening_trade_volume": 0,
    }


def _parse_day(path: Path) -> tuple[int, int, str] | None:
    match = _DAY_FILE_RE.search(path.name)
    if match is None:
        return None
    year, month, day = (int(value) for value in match.groups())
    return year * 10000 + month * 100 + day, year, f"{year:04d}{month:02d}"


def _read_preopen_aggregates(path: Path) -> dict[str, tuple[int, int]]:
    aggregates: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    parquet = pq.ParquetFile(path)
    columns = ["ticker", "Volume"]
    if "OrderType" in parquet.schema.names:
        columns.append("OrderType")
    for batch in parquet.iter_batches(columns=columns, batch_size=131072):
        data = batch.to_pydict()
        for index, ticker_value in enumerate(data["ticker"]):
            order_type = int(data["OrderType"][index]) if "OrderType" in data else 0
            if order_type in _CANCEL_TYPES:
                continue
            ticker = str(ticker_value)
            aggregates[ticker][0] += 1
            aggregates[ticker][1] += int(data["Volume"][index] or 0)
    return {ticker: (values[0], values[1]) for ticker, values in aggregates.items()}


def _related_paths(raw_root: Path, day: int) -> RelatedPaths:
    text = str(day)
    iso = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    month = text[:6]
    snapshot_candidates = (
        raw_root / "snapshot" / f"snapshot_{iso}.parquet",
        raw_root / "snapshot" / f"snapshot_{text}.parquet",
        raw_root / "snapshot" / f"snapshot_{month}.parquet",
    )
    return {
        "order": raw_root / "order" / month / f"order_{iso}.parquet",
        "trades": raw_root / "trades" / month / f"trades_{iso}.parquet",
        "snapshot": next((path for path in snapshot_candidates if path.exists()), None),
    }


def _read_order_coverage(path: Path, tickers: set[str]) -> set[str]:
    found: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["ticker"], batch_size=131072):
        found.update(str(value) for value in batch.column("ticker").unique().to_pylist())
        if tickers <= found:
            break
    return found & tickers


def _read_trades_coverage(path: Path, tickers: set[str]) -> dict[str, tuple[int, int]]:
    table = pq.read_table(
        path,
        columns=["ticker", "time_ms", "Volume"],
        filters=[("ticker", "in", sorted(tickers)), ("time_ms", "<=", 0)],
    )
    data = table.to_pydict()
    aggregates: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ticker, volume in zip(data["ticker"], data["Volume"], strict=True):
        values = aggregates[str(ticker)]
        values[0] += 1
        values[1] += int(volume or 0)
    return {ticker: (values[0], values[1]) for ticker, values in aggregates.items()}


def _read_snapshot_coverage(path: Path, day: int, tickers: set[str]) -> set[str]:
    if path is None:
        return set()
    table = pq.read_table(
        path,
        columns=["ticker", "TradingDay"],
        filters=[("ticker", "in", sorted(tickers)), ("TradingDay", "=", day)],
    )
    return {str(value) for value in table.column("ticker").to_pylist()}
