"""审计跨股票、跨日期的开盘数据契约和沪市事件时差。

示例：
    python scripts/audit_shanghai_contract.py \
        --sample 20220615:600000 \
        --sample 20250613:600000 \
        --trace 20220615:600000:bid:777 \
        --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
        --json-output /tmp/shanghai-opening.json \
        --csv-output /tmp/shanghai-opening.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from ticknet.simulator.opening_ledger import (
    OpeningDayLagScan,
    OpeningLevelTrace,
    level_differences,
    scan_opening_day_lags,
    trace_opening_day_level,
)


def _parse_sample(value: str) -> tuple[int, str]:
    try:
        day_text, ticker = value.split(":", maxsplit=1)
        day = int(day_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("样本格式必须是 YYYYMMDD:TICKER") from error
    if len(day_text) != 8 or not ticker:
        raise argparse.ArgumentTypeError("样本格式必须是 YYYYMMDD:TICKER")
    return day, ticker


def _parse_trace(value: str) -> tuple[int, str, int, int]:
    try:
        day_text, ticker, side_text, price_text = value.split(":", maxsplit=3)
        day = int(day_text)
        price = int(price_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("追踪格式必须是 YYYYMMDD:TICKER:bid|ask:PRICE") from error
    if len(day_text) != 8 or side_text not in {"bid", "ask"} or not ticker:
        raise argparse.ArgumentTypeError("追踪格式必须是 YYYYMMDD:TICKER:bid|ask:PRICE")
    return day, ticker, 1 if side_text == "bid" else -1, price


def _lag_values(lag_min: int, lag_max: int, lag_step: int) -> tuple[int, ...]:
    if lag_step <= 0 or lag_min > lag_max:
        raise ValueError("lag 范围或步长无效")
    return tuple(range(lag_min, lag_max + 1, lag_step))


def _level_diff_rows(scan: OpeningDayLagScan) -> list[dict]:
    audit = scan.best.audit
    return [
        asdict(difference)
        for difference in (
            *level_differences(audit.bid_levels, audit.expected_bid_levels, side=1),
            *level_differences(audit.ask_levels, audit.expected_ask_levels, side=-1),
        )
    ]


def _sample_payload(scan: OpeningDayLagScan) -> dict:
    audit = asdict(scan.best.audit)
    return {
        "day": scan.day,
        "ticker": scan.ticker,
        "snapshot_time_ms": scan.snapshot_time_ms,
        "coverage_status": scan.coverage_status,
        "preopen_file_present": scan.preopen_file_present,
        "preopen_ticker_present": scan.preopen_ticker_present,
        "best_lag_ms": scan.best.lag_ms,
        "best_audit": audit,
        "level_differences": _level_diff_rows(scan),
        "candidates": [
            {
                "lag_ms": candidate.lag_ms,
                "status": candidate.audit.status,
                "unknown_trade_volume": candidate.audit.unknown_trade_volume,
                "unknown_cancel_volume": candidate.audit.unknown_cancel_volume,
                "overdrawn_volume": candidate.audit.overdrawn_volume,
            }
            for candidate in scan.candidates
        ],
    }


def _summary(scans: list[OpeningDayLagScan]) -> dict:
    status_counts = {"matched": 0, "mismatched": 0, "not_comparable": 0}
    coverage_counts: dict[str, int] = {}
    lag_counts: dict[str, int] = {}
    comparable = 0
    for scan in scans:
        status_counts[scan.best.audit.status] += 1
        coverage_counts[scan.coverage_status] = coverage_counts.get(scan.coverage_status, 0) + 1
        lag_key = str(scan.best.lag_ms)
        lag_counts[lag_key] = lag_counts.get(lag_key, 0) + 1
        comparable += scan.best.audit.status in {"matched", "mismatched"}
    return {
        "total_samples": len(scans),
        **status_counts,
        "coverage_status_counts": coverage_counts,
        "best_lag_counts": lag_counts,
        "comparable_match_rate": (status_counts["matched"] / comparable if comparable else None),
    }


def _write_csv(path: Path, scans: list[OpeningDayLagScan]) -> None:
    fields = [
        "day",
        "ticker",
        "coverage_status",
        "snapshot_time_ms",
        "best_lag_ms",
        "status",
        "preopen_file_present",
        "preopen_ticker_present",
        "unknown_trade_volume",
        "unknown_cancel_volume",
        "overdrawn_volume",
        "level_differences_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scan in scans:
            audit = scan.best.audit
            writer.writerow(
                {
                    "day": scan.day,
                    "ticker": scan.ticker,
                    "coverage_status": scan.coverage_status,
                    "snapshot_time_ms": scan.snapshot_time_ms,
                    "best_lag_ms": scan.best.lag_ms,
                    "status": audit.status,
                    "preopen_file_present": scan.preopen_file_present,
                    "preopen_ticker_present": scan.preopen_ticker_present,
                    "unknown_trade_volume": audit.unknown_trade_volume,
                    "unknown_cancel_volume": audit.unknown_cancel_volume,
                    "overdrawn_volume": audit.overdrawn_volume,
                    "level_differences_json": json.dumps(
                        _level_diff_rows(scan), ensure_ascii=False
                    ),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", type=_parse_sample, default=[])
    parser.add_argument("--sample-file", type=Path, default=None, help="每行一个 YYYYMMDD:TICKER")
    parser.add_argument("--trace", action="append", type=_parse_trace, default=[])
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--lag-min", type=int, default=-200)
    parser.add_argument("--lag-max", type=int, default=200)
    parser.add_argument("--lag-step", type=int, default=10)
    parser.add_argument("--trace-lag", type=int, default=None, help="追踪时覆盖最佳 lag")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser


def _read_sample_file(path: Path | None) -> list[tuple[int, str]]:
    if path is None:
        return []
    values: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(_parse_sample(value))
    return values


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    samples = [*arguments.sample, *_read_sample_file(arguments.sample_file)]
    if not samples:
        raise SystemExit("至少需要一个 --sample 或 --sample-file")
    lags = _lag_values(arguments.lag_min, arguments.lag_max, arguments.lag_step)
    scans = [
        scan_opening_day_lags(day, ticker, arguments.raw_root, lags) for day, ticker in samples
    ]
    traces: list[dict] = []
    for day, ticker, side, price in arguments.trace:
        rows: tuple[OpeningLevelTrace, ...] = trace_opening_day_level(
            day,
            ticker,
            side=side,
            price=price,
            raw_root=arguments.raw_root,
            event_lag_ms=(
                arguments.trace_lag
                if arguments.trace_lag is not None
                else next(
                    scan.best.lag_ms for scan in scans if scan.day == day and scan.ticker == ticker
                )
            ),
        )
        traces.append(
            {
                "day": day,
                "ticker": ticker,
                "side": side,
                "price": price,
                "orders": [asdict(row) for row in rows],
            }
        )
    payload = {
        "summary": _summary(scans),
        "samples": [_sample_payload(scan) for scan in scans],
        "traces": traces,
    }
    arguments.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(arguments.csv_output, scans)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
