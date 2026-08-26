#!/usr/bin/env python3
"""用真实 L2 数据校验撮合引擎重建精度。

读取原始 order/snapshot parquet，回放单只股票全天订单流，
在每个快照处对比重建盘口与真实盘口。

用法：
    python scripts/verify_realdata.py --day 20210104 --ticker 000001 [--root ROOT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticknet.eventstream.config import RAW_L2_ROOT
from ticknet.simulator.realdata import verify_day_correctness


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", type=int, required=True, help="交易日，如 20210104")
    ap.add_argument("--ticker", required=True, help="股票代码，如 000001")
    ap.add_argument("--root", type=Path, default=RAW_L2_ROOT, help="raw L2 根目录")
    args = ap.parse_args()

    results = verify_day_correctness(args.day, args.root, args.ticker)
    n = len(results)
    ok = sum(r.matched for r in results)
    bid_ok = sum(1 - r.bid_error for r in results)
    ask_ok = sum(1 - r.ask_error for r in results)
    print(f"[{args.ticker} @ {args.day}] 快照对比 {n} 个")
    print(f"  完全一致: {ok}/{n}")
    print(f"  买一一致: {bid_ok}/{n}  卖一一致: {ask_ok}/{n}")
    if ok < n:
        print("  前 5 个不一致样本:")
        for r in [x for x in results if not x.matched][:5]:
            print(f"    {r.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
