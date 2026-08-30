from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.simulator.realdata import load_day_pack

DAY = 20210104
TICKER = "000001"


def _snapshot_row() -> dict[str, object]:
    row: dict[str, object] = {"ticker": TICKER, "TradingDay": DAY, "time_ms": 200}
    for level in range(1, 11):
        row[f"BidPrice{level}"] = 1000 - level + 1
        row[f"BidVolume{level}"] = 100
        row[f"AskPrice{level}"] = 1010 + level - 1
        row[f"AskVolume{level}"] = 100
    return row


def test_realdata_preserves_exchange_sequence_and_uses_it_within_one_channel(
    tmp_path: Path,
) -> None:
    order_dir = tmp_path / "order" / "202101"
    snapshot_dir = tmp_path / "snapshot"
    order_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "ticker": TICKER,
                    "TradingDay": DAY,
                    "time_ms": 100,
                    "OrderID": "later",
                    "Price": 1000,
                    "Volume": 100,
                    "OrderType": 1,
                    "ChannelNo": 7,
                    "ApplSeqNum": 12,
                },
                {
                    "ticker": TICKER,
                    "TradingDay": DAY,
                    "time_ms": 100,
                    "OrderID": "earlier",
                    "Price": 1000,
                    "Volume": 100,
                    "OrderType": 1,
                    "ChannelNo": 7,
                    "ApplSeqNum": 11,
                },
            ]
        ),
        order_dir / "order_2021-01-04.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([_snapshot_row()]),
        snapshot_dir / "snapshot_202101.parquet",
    )

    pack = load_day_pack(DAY, tmp_path, TICKER)

    order_ids = [event.order_id for event in pack.events if event.kind == "order"]
    assert order_ids == ["earlier", "later"]
    assert pack.ordering_provenance["exchange_sequence_available"] is True
    assert pack.ordering_provenance["channel_column"] == "ChannelNo"
    assert pack.ordering_provenance["sequence_column"] == "ApplSeqNum"
    assert pack.ordering_provenance["ordering_mode"] == "timestamp_then_channel_sequence"
    assert pack.ordering_provenance["cross_channel_total_order"] is False
