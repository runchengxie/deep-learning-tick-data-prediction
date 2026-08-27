"""审计盘前订单身份账本对开盘十档快照的重建能力。

示例：
    python scripts/audit_opening_ledger.py \
        --sample 20210104:000001 \
        --sample 20230301:000001 \
        --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ticknet.simulator.opening_ledger import (
    OpeningDayAudit,
    audit_opening_day,
    summarize_opening_audits,
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


def _serialise(result: OpeningDayAudit) -> dict:
    return asdict(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        type=_parse_sample,
        metavar="YYYYMMDD:TICKER",
        help="要审计的股票日，可重复传入",
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="raw L2 根目录")
    parser.add_argument(
        "--event-lag-ms", type=int, default=None, help="覆盖 snapshot 到事件时钟偏移"
    )
    parser.add_argument("--output", type=Path, default=None, help="可选 JSON 输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    results: list[OpeningDayAudit] = []
    for day, ticker in arguments.sample:
        if arguments.raw_root is None:
            result = audit_opening_day(day, ticker, event_lag_ms=arguments.event_lag_ms)
        else:
            result = audit_opening_day(
                day, ticker, arguments.raw_root, event_lag_ms=arguments.event_lag_ms
            )
        results.append(result)
    payload = {
        "samples": [_serialise(result) for result in results],
        "summary": asdict(summarize_opening_audits(results)),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
