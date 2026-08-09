"""把正式 Top-400 L2 聚合特征按月物化为可恢复的 Parquet 分片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ticknet.nextday.minute_baseline import build_target_bundle, load_minute_baseline_config
from ticknet.nextday.minute_materialization import materialize_minute_features
from ticknet.nextday.splits import parse_date
from ticknet.research.protocol import ResearchProtocol


def _load_config(path: str | Path):
    try:
        return load_minute_baseline_config(path)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


def _period_update(shard: dict[str, Any]) -> None:
    action = "已校验并跳过" if shard["resumed"] else "已物化"
    elapsed_label = "original_elapsed" if shard["resumed"] else "elapsed"
    print(
        f"{action} {shard['period']}：rows={shard['row_count']:,} "
        f"missing={shard['imputed_feature_rows']:,} "
        f"{elapsed_label}={float(shard['elapsed_seconds']):.1f}s "
        f"peak_rss={float(shard['peak_rss_mb']):.1f}MB",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="按月物化正式分钟 HGB 聚合特征")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--period",
        action="append",
        default=None,
        help="只处理指定 YYYY-MM，可重复；省略时处理全部缺失月份",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="校验并跳过已有分片；--no-resume 会拒绝已有 manifest",
    )
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if not config.formal:
        raise SystemExit("分钟特征物化要求正式 next_open_to_following_open 配置")
    if config.top_n != 400 or config.min_symbols_per_day != 400:
        raise SystemExit("正式分钟特征物化要求 top_n=min_symbols_per_day=400")
    if parse_date(config.end_date) > ResearchProtocol().validation_end_date:
        raise SystemExit("研究阶段分钟特征物化不能进入 2026 locked 区间")
    if args.period is not None:
        for period in args.period:
            try:
                parse_date(f"{period}-01")
            except ValueError as error:
                raise SystemExit(f"--period 应为 YYYY-MM：{period}") from error

    bundle = build_target_bundle(config)
    targets = [target for target in bundle.targets if target.in_universe]
    manifest = materialize_minute_features(
        config,
        targets,
        args.output,
        resume=args.resume,
        periods=args.period,
        on_period=_period_update,
    )
    result = {
        "manifest": str(args.output.expanduser().resolve() / "manifest.json"),
        "materialization_identity": manifest["materialization_identity"],
        "status": manifest["status"],
        "summary": manifest["summary"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
