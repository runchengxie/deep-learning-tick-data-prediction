"""eventstream 测试共享数据：合成数据湖 + 打包 + 标签。"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.eventstream.pack import pack_day

DAY = 20210104


def _write_table(rows: list[dict], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _snap_row(
    ticker: str,
    t: int,
    *,
    last: float,
    volume: int,
    turnover: int,
    dealnum: int,
    bid1: float,
    ask1: float,
) -> dict:
    row: dict = {
        "ticker": ticker,
        "TradingDay": DAY,
        "time_ms": t,
        "TickTimeDiff": 0,
        "Price": last,
        "DealNum": dealnum,
        "Volume": volume,
        "Turnover": turnover,
        "TotalDealNum": dealnum,
        "TotalVolume": volume,
        "TotalTurnover": turnover,
        "TotalBidVolume": 1000,
        "TotalAskVolume": 1200,
        "WeightBidPrice": bid1,
        "WeightAskPrice": ask1,
    }
    for i in range(1, 11):
        row[f"BidPrice{i}"] = bid1 - (i - 1) * 0.01
        row[f"AskPrice{i}"] = ask1 + (i - 1) * 0.01
        row[f"BidVolume{i}"] = 100 * i
        row[f"AskVolume{i}"] = 100 * i
        row[f"BidOrder{i}"] = i
        row[f"AskOrder{i}"] = i
    return row


def build_synthetic_lake(root: Path) -> tuple[Path, Path, Path]:
    """构造合成数据湖并打包，返回 (raw_root, pack_root, label_path)。"""
    raw = root / "raw"
    order_dir = raw / "order" / "202101"
    trades_dir = raw / "trades" / "202101"
    order_dir.mkdir(parents=True)
    trades_dir.mkdir(parents=True)
    (raw / "snapshot").mkdir(parents=True)
    (raw / "basic").mkdir(parents=True)

    _write_table(
        [
            {
                "ticker": "600000",
                "TradingDay": DAY,
                "time_ms": 1000,
                "OrderID": "B1",
                "Price": 10.00,
                "Volume": 100,
                "OrderType": 0,
                "LastPrice": 10.00,
            },
            {
                "ticker": "600000",
                "TradingDay": DAY,
                "time_ms": 1200,
                "OrderID": "S1",
                "Price": 10.01,
                "Volume": 200,
                "OrderType": 0,
                "LastPrice": 10.01,
            },
            {
                "ticker": "600000",
                "TradingDay": DAY,
                "time_ms": 2000,
                "OrderID": "B1",
                "Price": 10.00,
                "Volume": 100,
                "OrderType": -1,
                "LastPrice": 10.00,
            },
        ],
        order_dir / "order_2021-01-04.parquet",
    )
    _write_table(
        [
            {
                "ticker": "600000",
                "TradingDay": DAY,
                "time_ms": 1500,
                "DealID": "D1",
                "Price": 10.00,
                "Volume": 60,
                "Side": 0,
                "bsflag": 1,
                "BuyID": "B1",
                "SellID": "S1",
            },
        ],
        trades_dir / "trades_2021-01-04.parquet",
    )
    _write_table(
        [
            _snap_row(
                "600000",
                1000,
                last=10.00,
                volume=100,
                turnover=1000,
                dealnum=2,
                bid1=9.99,
                ask1=10.01,
            ),
            _snap_row(
                "600000",
                1500,
                last=10.00,
                volume=150,
                turnover=1500,
                dealnum=3,
                bid1=9.99,
                ask1=10.01,
            ),
        ],
        raw / "snapshot" / "snapshot_202101.parquet",
    )
    _write_table(
        [{"value": 20201231, "600000": 10.00}, {"value": DAY, "600000": 10.10}],
        raw / "basic" / "close_data.parquet",
    )

    pack_root = root / "pack"
    pack_day(DAY, raw_root=raw, pack_root=pack_root)

    label_path = root / "labels.parquet"
    _write_table([{"value": DAY, "600000": 0.5}], label_path)
    return raw, pack_root, label_path


@pytest.fixture(scope="session")
def packed_day(tmp_path_factory) -> dict:
    """会话级：合成数据湖 + 已打包日 + 标签文件。"""
    tmp = tmp_path_factory.mktemp("eventstream_lake")
    raw, pack, labels = build_synthetic_lake(tmp)
    return {"raw_root": raw, "pack_root": pack, "label_path": labels, "day": DAY}
