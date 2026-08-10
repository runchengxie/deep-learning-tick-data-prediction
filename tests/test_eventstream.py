"""L2 事件流无损打包 + 数据集窗口采样（合成数据，无真实行情依赖）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.eventstream.config import ORDER_DTYPE, SNAP_DTYPE, TRADE_DTYPE, day_pack_paths
from ticknet.eventstream.dataset import L2WindowDataset
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


def _make_lake(tmp_path: Path) -> tuple[Path, Path]:
    """构造合成数据湖（order/trades 按日、snapshot 按月、basic close 宽表）。"""
    raw = tmp_path / "raw"
    order_dir = raw / "order" / "202101"
    trades_dir = raw / "trades" / "202101"
    order_dir.mkdir(parents=True)
    trades_dir.mkdir(parents=True)
    (raw / "snapshot").mkdir(parents=True)
    (raw / "basic").mkdir(parents=True)

    # ---- orders：B1 买加 + S1 卖加 + C1 撤（撤 B1）----
    _write_table(
        [
            {"ticker": "600000", "TradingDay": DAY, "time_ms": 1000, "OrderID": "B1",
             "Price": 10.00, "Volume": 100, "OrderType": 0, "LastPrice": 10.00},
            {"ticker": "600000", "TradingDay": DAY, "time_ms": 1200, "OrderID": "S1",
             "Price": 10.01, "Volume": 200, "OrderType": 0, "LastPrice": 10.01},
            {"ticker": "600000", "TradingDay": DAY, "time_ms": 2000, "OrderID": "B1",
             "Price": 10.00, "Volume": 100, "OrderType": -1, "LastPrice": 10.00},
        ],
        order_dir / "order_2021-01-04.parquet",
    )

    # ---- trades：成交 D1 撮合 B1 x S1 ----
    _write_table(
        [
            {"ticker": "600000", "TradingDay": DAY, "time_ms": 1500, "DealID": "D1",
             "Price": 10.00, "Volume": 60, "Side": 0, "bsflag": 1,
             "BuyID": "B1", "SellID": "S1"},
        ],
        trades_dir / "trades_2021-01-04.parquet",
    )

    # ---- snapshot（月度文件，只含当日两行）----
    _write_table(
        [
            _snap_row("600000", 1000, last=10.00, volume=100, turnover=1000, dealnum=2,
                      bid1=9.99, ask1=10.01),
            _snap_row("600000", 1500, last=10.00, volume=150, turnover=1500, dealnum=3,
                      bid1=9.99, ask1=10.01),
        ],
        raw / "snapshot" / "snapshot_202101.parquet",
    )

    # ---- basic close 宽表：20201231 收盘 10.00 ----
    _write_table(
        [{"value": 20201231, "600000": 10.00}, {"value": 20210104, "600000": 10.10}],
        raw / "basic" / "close_data.parquet",
    )

    return raw, tmp_path / "pack"


class TestPack:
    def test_lossless_and_linkage(self, tmp_path):
        raw, pack_root = _make_lake(tmp_path)
        pack_day(DAY, raw_root=raw, pack_root=pack_root)

        paths = day_pack_paths(DAY, pack_root)
        assert all(p.exists() for p in paths.values())

        orders = np.fromfile(paths["order"], dtype=ORDER_DTYPE)
        assert len(orders) == 3
        cancel = orders[orders["order_type"] == -1][0]
        assert cancel["cancel_age_ms"] == 2000 - 1000
        assert cancel["cancel_orig_vol"] == 100
        assert orders[0]["price"] == 1000  # 10.00 元 -> 分
        assert orders[1]["price"] == 1001

        trades = np.fromfile(paths["trade"], dtype=TRADE_DTYPE)
        assert len(trades) == 1
        assert trades[0]["buy_age_ms"] == 1500 - 1000
        assert trades[0]["sell_age_ms"] == 1500 - 1200
        assert trades[0]["volume"] == 60

        snaps = np.fromfile(paths["snap"], dtype=SNAP_DTYPE)
        assert len(snaps) == 2
        assert snaps[1]["d_volume"] == 50
        assert snaps[1]["d_turnover"] == 500
        assert snaps[1]["bid_px"][0] == 999

        index = np.load(paths["index"], allow_pickle=False)
        assert list(index["tickers"].astype(str)) == ["600000"]
        assert index["o_len"][0] == 3
        assert index["t_len"][0] == 1
        assert index["s_len"][0] == 2
        assert index["prev_close"][0] == pytest.approx(10.0)

    def test_skip_missing_day(self, tmp_path):
        raw, pack_root = _make_lake(tmp_path)
        (raw / "order" / "202101" / "order_2021-01-04.parquet").unlink()
        pack_day(DAY, raw_root=raw, pack_root=pack_root)
        assert not (pack_root / f"orders_{DAY}.bin").exists()


class TestDataset:
    def test_window_shapes_and_targets(self, tmp_path):
        raw, pack_root = _make_lake(tmp_path)
        pack_day(DAY, raw_root=raw, pack_root=pack_root)

        ds = L2WindowDataset(
            [DAY], seq_len=8, min_events=2, samples_per_day=4,
            root=pack_root, label_path=None, seed=0,
        )
        assert len(ds) == 4
        sample = ds[0]
        x, sid, _oid, tgt_sid, _tgt_oid, tgt_reg, _tgt_day, day_valid, valid, day = sample
        assert x.shape == (8, 80)
        span = int(valid.sum())
        assert set(np.unique(sid[: span + 1].numpy())).issubset({1, 2, 3})
        assert span >= 1
        assert int(day) == DAY
        assert day_valid.item() == 0.0  # 无 label 文件
        # 目标 = 下一事件的流类型/回归特征，必须自洽
        span = int(valid.sum())
        for i in range(span - 1):
            assert int(tgt_sid[i]) == int(sid[i + 1])
            assert tgt_reg[i, 0] == pytest.approx(x[i + 1, 1])
            assert tgt_reg[i, 1] == pytest.approx(x[i + 1, 0])
        # 窗口内最后位置的目标指向窗口外的下一个事件（sid 持有 span+1 个真实事件）
        assert int(tgt_sid[span - 1]) == int(sid[span])

    def test_eval_mode_takes_last_window(self, tmp_path):
        raw, pack_root = _make_lake(tmp_path)
        pack_day(DAY, raw_root=raw, pack_root=pack_root)
        _write_table(
            [{"value": DAY, "600000": 0.5}],
            tmp_path / "labels.parquet",
        )
        ds = L2WindowDataset(
            [DAY], seq_len=8, min_events=2, root=pack_root,
            label_path=tmp_path / "labels.parquet", eval_mode=True,
        )
        assert len(ds) == 1
        _, _, _, _, _, _, tgt_day, day_valid, _, _ = ds[0]
        assert day_valid.item() == 1.0
        assert tgt_day.item() == pytest.approx(0.5)

    def test_no_packed_days_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no packed days"):
            L2WindowDataset([20250101], root=tmp_path / "none")
