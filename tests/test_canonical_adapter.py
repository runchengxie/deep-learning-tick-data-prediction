import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.eventstream.canonical_adapter import adapt_canonical_file, adapt_canonical_table


def test_adapt_order_maps_names_and_integer_cents_to_yuan() -> None:
    source = pa.table(
        {
            "SecuCode": [2202],
            "TradingDay": [20260424],
            "OrderTime": [123],
            "OrderID": [7],
            "Price": [1234],
            "LastPrice": [1200],
            "Volume": [100],
            "OrderType": [1],
        }
    )
    output = adapt_canonical_table(source, "order")
    assert output.column_names == [
        "ticker", "TradingDay", "time_ms", "OrderID", "Price", "LastPrice", "Volume", "OrderType"
    ]
    assert output["ticker"].to_pylist() == ["002202"]
    assert output["time_ms"].to_pylist() == [123]
    assert output["Price"].to_pylist() == [12.34]
    assert output["LastPrice"].to_pylist() == [12.0]


def test_adapt_snapshot_scales_book_prices_and_keeps_zero_sentinels() -> None:
    source = pa.table(
        {
            "SecuCode": [2202],
            "TradingDay": [20260424],
            "TickTime": [123],
            "Price": [1234],
            "Volume": [100],
            "DealNum": [2],
            "BidPrice1": [1230],
            "BidVolume1": [50],
            "AskPrice1": [1240],
            "AskVolume1": [60],
        }
    )
    output = adapt_canonical_table(source, "snapshot")
    assert output["ticker"].to_pylist() == ["002202"]
    assert output["time_ms"].to_pylist() == [123]
    assert output["BidPrice1"].to_pylist() == [12.3]
    assert output["AskPrice1"].to_pylist() == [12.4]


def test_adapt_canonical_file_streams_and_reports_manifest(tmp_path) -> None:
    source = tmp_path / "order.parquet"
    target = tmp_path / "out" / "order.parquet"
    pq.write_table(
        pa.table(
            {
                "SecuCode": [2202, 2202],
                "TradingDay": [20260424, 20260424],
                "OrderTime": [1, 2],
                "OrderID": [1, 2],
                "Price": [1000, 1100],
                "Volume": [10, 20],
                "OrderType": [1, 1],
            }
        ),
        source,
    )
    report = adapt_canonical_file(str(source), str(target), "order", batch_size=1)
    assert report["rows"] == 2
    assert pq.read_table(target)["Price"].to_pylist() == [10.0, 11.0]
