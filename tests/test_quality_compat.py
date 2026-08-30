from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.simulator.opening_ledger import OpeningOrder
from ticknet.simulator.quality_compat import (
    audit_opening_ledger_compat,
    profile_parquet_compat,
)


def test_compat_profiler_runs_without_platform_package(tmp_path: Path) -> None:
    path = tmp_path / "order_2024-01-02.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"ticker": "000001", "TradingDay": 20240102, "time_ms": -1, "OrderID": 1}]
        ),
        path,
    )

    report = profile_parquet_compat(path)

    assert report["rows"] == 1
    assert report["trading_days"] == [20240102]


def test_compat_opening_audit_preserves_local_result_type() -> None:
    result = audit_opening_ledger_compat(
        [OpeningOrder("B1", 1, 1000, 100)],
        [],
        [],
        expected_bid_levels=((1000, 100),),
        expected_ask_levels=(),
    )

    assert result.status == "matched"
    assert type(result).__name__ == "OpeningLedgerAudit"
