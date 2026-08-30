"""Adapt market-data-platform canonical L2 tables to ticknet raw names.

The adapter is deliberately explicit about units: canonical prices are
integer cents, while the ticknet parquet contract stores prices in yuan and
the packer converts them back to cents.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_TIME_COLUMNS = {"order": "OrderTime", "deal": "DealTime", "snapshot": "TickTime"}
_REQUIRED = {
    "order": ("ticker", "TradingDay", "time_ms", "OrderID", "Price", "Volume", "OrderType"),
    "deal": ("ticker", "TradingDay", "time_ms", "DealID", "Price", "Volume", "Side"),
    "snapshot": ("ticker", "TradingDay", "time_ms", "Price", "Volume", "DealNum"),
}
_PRICE_RE = re.compile(r"^(?:Bid|Ask)Price\d+$")


def _ticker(value: object) -> str:
    return str(int(value)).zfill(6) if isinstance(value, (int, float)) else str(value).zfill(6)


def _is_price(name: str) -> bool:
    return name in {"Price", "LastPrice", "WeightBidPrice", "WeightAskPrice"} or bool(
        _PRICE_RE.match(name)
    )


def adapt_canonical_table(table: pa.Table, kind: str) -> pa.Table:
    """Return a renamed/unit-adapted table without changing the input table."""
    if kind not in _TIME_COLUMNS:
        raise ValueError(f"unsupported canonical L2 kind: {kind!r}")
    time_column = _TIME_COLUMNS[kind]
    if "SecuCode" not in table.column_names or time_column not in table.column_names:
        raise ValueError(f"canonical {kind} table lacks SecuCode/{time_column}")

    arrays: list[pa.ChunkedArray | pa.Array] = []
    names: list[str] = []
    for name in table.column_names:
        output_name = {"SecuCode": "ticker", time_column: "time_ms"}.get(name, name)
        column = table[name]
        if name == "SecuCode":
            column = pa.array([_ticker(value) for value in column.to_pylist()])
        elif _is_price(name):
            column = pc.divide(column.cast(pa.float64()), 100.0)
        arrays.append(column)
        names.append(output_name)
    return pa.table(dict(zip(names, arrays, strict=True)))


def validate_adapter_schema(table: pa.Table, kind: str) -> None:
    """Fail closed when an adapted table cannot be consumed by ticknet."""
    missing = set(_REQUIRED[kind]) - set(table.column_names)
    if missing:
        raise ValueError(f"adapted {kind} table missing columns: {sorted(missing)}")


def adapt_canonical_file(
    source: str,
    target: str,
    kind: str,
    *,
    batch_size: int = 262_144,
) -> dict[str, object]:
    """Convert one canonical parquet file with bounded memory usage."""
    source_path, target_path = Path(source), Path(target)
    reader = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in reader.iter_batches(batch_size=batch_size):
            adapted = adapt_canonical_table(pa.Table.from_batches([batch]), kind)
            validate_adapter_schema(adapted, kind)
            if writer is None:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(target_path, adapted.schema)
            writer.write_table(adapted)
            rows += adapted.num_rows
    finally:
        if writer is not None:
            writer.close()
    return {
        "kind": kind,
        "source": str(source_path),
        "target": str(target_path),
        "rows": rows,
        "price_unit_conversion": "integer_cent_to_yuan (divide by 100)",
        "columns": pq.ParquetFile(target_path).schema_arrow.names,
    }
