#!/usr/bin/env python3
"""用真实 L2 数据校验撮合引擎重建精度。

读取原始 order/snapshot parquet，回放单只股票全天订单流，
在每个快照处对比重建盘口与真实盘口。

用法：
    python scripts/verify_realdata.py --day 20210104 --ticker 000001 \
        [--root ROOT] [--mode continuous|interval] [--event-lag-ms N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticknet.eventstream.config import RAW_L2_ROOT
from ticknet.simulator.realdata import default_snapshot_event_lag_ms, verify_day_correctness


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", type=int, required=True, help="交易日，如 20210104")
    ap.add_argument("--ticker", required=True, help="股票代码，如 000001")
    ap.add_argument("--root", type=Path, default=RAW_L2_ROOT, help="raw L2 根目录")
    ap.add_argument(
        "--mode",
        choices=("continuous", "interval"),
        default="continuous",
        help="continuous=全天连续回放；interval=每个快照后重新校准",
    )
    ap.add_argument(
        "--event-lag-ms",
        type=int,
        default=None,
        help="snapshot→event 毫秒偏移；不填则按市场使用已验证默认值",
    )
    args = ap.parse_args()
    event_lag_ms = (
        args.event_lag_ms
        if args.event_lag_ms is not None
        else default_snapshot_event_lag_ms(args.ticker)
    )

    results = verify_day_correctness(
        args.day,
        args.root,
        args.ticker,
        mode=args.mode,
        event_lag_ms=event_lag_ms,
    )
    comparable = [r for r in results if r.comparable]
    skipped = len(results) - len(comparable)
    n = len(comparable)
    ok = sum(r.matched for r in comparable)
    bad = n - ok
    bid_ok = sum(1 - r.bid_error for r in comparable)
    ask_ok = sum(1 - r.ask_error for r in comparable)
    print(
        f"[{args.ticker} @ {args.day}] mode={args.mode} "
        f"event_lag_ms={event_lag_ms}"
    )
    print(f"  快照结果: {len(results)}（可比较 {n}，跳过 {skipped}）")
    print(f"  完全一致: {ok}/{n}")
    print(f"  不一致: {bad}/{n}")
    print(f"  买一一致: {bid_ok}/{n}  卖一一致: {ask_ok}/{n}")
    if bad:
        print("  前 5 个不一致样本:")
        for r in [x for x in comparable if not x.matched][:5]:
            print(f"    {r.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
