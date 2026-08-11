"""审计一个月全天 L2 eventstream pilot 的原始输入与打包产物。"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ticknet.eventstream.config import PACK_ROOT, RAW_L2_ROOT, day_pack_paths

MONTH_PATTERN = re.compile(r"^(\d{4})-?(\d{2})$")


def normalize_month(value: str) -> str:
    match = MONTH_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"月份应使用 YYYY-MM 或 YYYYMM 格式，收到 {value!r}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"无效月份：{value}")
    return f"{year:04d}{month:02d}"


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _daily_files(root: Path, stream: str, month: str) -> dict[int, Path]:
    prefix = "order" if stream == "order" else "trades"
    files: dict[int, Path] = {}
    for path in sorted((root / stream / month).glob(f"{prefix}_????-??-??.parquet")):
        compact = path.stem.split("_", 1)[1].replace("-", "")
        files[int(compact)] = path
    return files


def inventory_raw_month(month: str, raw_root: Path = RAW_L2_ROOT) -> dict[str, Any]:
    """只读取文件元数据，返回一个月三条原始流的容量清单。"""
    compact_month = normalize_month(month)
    order_files = _daily_files(raw_root, "order", compact_month)
    trade_files = _daily_files(raw_root, "trades", compact_month)
    snapshot = raw_root / "snapshot" / f"snapshot_{compact_month}.parquet"
    common_days = sorted(set(order_files) & set(trade_files))
    order_only = sorted(set(order_files) - set(trade_files))
    trade_only = sorted(set(trade_files) - set(order_files))
    order_bytes = sum(path.stat().st_size for path in order_files.values())
    trade_bytes = sum(path.stat().st_size for path in trade_files.values())
    snapshot_bytes = snapshot.stat().st_size if snapshot.is_file() else 0
    return {
        "month": f"{compact_month[:4]}-{compact_month[4:]}",
        "raw_root": str(raw_root),
        "status": (
            "complete"
            if common_days and not order_only and not trade_only and snapshot.is_file()
            else "incomplete"
        ),
        "common_days": common_days,
        "order_only_days": order_only,
        "trade_only_days": trade_only,
        "streams": {
            "order": {"files": len(order_files), "bytes": order_bytes},
            "trades": {"files": len(trade_files), "bytes": trade_bytes},
            "snapshot": {"files": int(snapshot.is_file()), "bytes": snapshot_bytes},
        },
        "total_unique_input_bytes": order_bytes + trade_bytes + snapshot_bytes,
        "daily_files": [
            {
                "day": day,
                "order_bytes": order_files[day].stat().st_size,
                "trade_bytes": trade_files[day].stat().st_size,
            }
            for day in common_days
        ],
    }


def build_manifest_universe(manifest_path: Path, month: str) -> dict[str, Any]:
    """从 snapshot manifest 提取按日历史股票池，供 eventstream pack 复用。"""
    compact_month = normalize_month(month)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("特征 manifest 缺少 samples 列表")
    fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("特征 manifest 缺少有效 dataset_fingerprint")

    universes: dict[int, set[str]] = defaultdict(set)
    for sample in manifest["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("manifest sample 应为对象")
        trading_date = str(sample.get("trading_date", ""))
        symbol = str(sample.get("symbol", ""))
        if trading_date.replace("-", "")[:6] == compact_month and symbol:
            universes[int(trading_date.replace("-", ""))].add(symbol)
    if not universes:
        raise ValueError(f"manifest 在 {compact_month} 没有股票池记录")
    ordered = {str(day): sorted(symbols) for day, symbols in sorted(universes.items())}
    counts = [len(symbols) for symbols in ordered.values()]
    return {
        "schema_version": 1,
        "mode": "daily_manifest_universe",
        "month": f"{compact_month[:4]}-{compact_month[4:]}",
        "source_manifest": str(manifest_path),
        "source_dataset_fingerprint": fingerprint,
        "days": len(ordered),
        "symbols_per_day_min": min(counts),
        "symbols_per_day_max": max(counts),
        "union_symbols": len(set().union(*(set(symbols) for symbols in ordered.values()))),
        "universes": ordered,
    }


def inventory_packed_month(
    expected_days: list[int],
    pack_root: Path = PACK_ROOT,
) -> dict[str, Any]:
    """读取 index 元数据，汇总已完成和部分完成的按日 pack。"""
    packed_days: list[dict[str, Any]] = []
    missing_days: list[int] = []
    partial_days: list[dict[str, Any]] = []
    totals = {"order": 0, "trade": 0, "snap": 0, "index": 0}
    event_totals = {"order": 0, "trade": 0, "snap": 0}
    for day in expected_days:
        paths = day_pack_paths(day, pack_root)
        existing = sorted(name for name, path in paths.items() if path.is_file())
        if not existing:
            missing_days.append(day)
            continue
        if len(existing) != len(paths):
            partial_days.append({"day": day, "existing": existing})
            continue
        sizes = {name: path.stat().st_size for name, path in paths.items()}
        with np.load(paths["index"], allow_pickle=False) as index:
            events = {
                "order": int(np.sum(index["o_len"])),
                "trade": int(np.sum(index["t_len"])),
                "snap": int(np.sum(index["s_len"])),
            }
            tickers = len(index["tickers"])
        for name, size in sizes.items():
            totals[name] += size
        for name, count in events.items():
            event_totals[name] += count
        packed_days.append(
            {"day": day, "tickers": tickers, "bytes": sum(sizes.values()), "events": events}
        )
    return {
        "pack_root": str(pack_root),
        "status": (
            "complete"
            if expected_days and len(packed_days) == len(expected_days)
            else "partial"
            if packed_days or partial_days
            else "not_started"
        ),
        "expected_days": len(expected_days),
        "packed_days": packed_days,
        "missing_days": missing_days,
        "partial_days": partial_days,
        "stream_bytes": totals,
        "event_counts": event_totals,
        "total_bytes": sum(totals.values()),
    }


def run_pilot_audit(
    *,
    month: str,
    raw_root: Path,
    pack_root: Path,
    feature_manifest: Path,
    universe_output: Path,
    output: Path,
    require_complete_pack: bool = False,
) -> dict[str, Any]:
    raw = inventory_raw_month(month, raw_root)
    universe = build_manifest_universe(feature_manifest, month)
    _atomic_json(universe_output, universe)
    expected_days = sorted(set(raw["common_days"]) & {int(day) for day in universe["universes"]})
    packed = inventory_packed_month(expected_days, pack_root)
    packed_month_bytes = packed["total_bytes"] if packed["status"] == "complete" else None
    report = {
        "schema_version": 1,
        "mode": "eventstream_month_pilot",
        "month": raw["month"],
        "status": packed["status"],
        "raw": raw,
        "universe": {key: value for key, value in universe.items() if key != "universes"},
        "pack": packed,
        "projections": {
            "five_month_raw_bytes": raw["total_unique_input_bytes"] * 5,
            "five_month_packed_bytes": (
                None if packed_month_bytes is None else packed_month_bytes * 5
            ),
            "drive_soft_limit_bytes": 150 * 10**9,
            "five_month_pack_fits_drive_soft_limit": (
                None if packed_month_bytes is None else packed_month_bytes * 5 <= 150 * 10**9
            ),
        },
        "artifacts": {"universe": str(universe_output), "report": str(output)},
    }
    _atomic_json(output, report)
    if require_complete_pack and packed["status"] != "complete":
        raise RuntimeError("月度 eventstream pack 尚未完成")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="审计一个月全天 L2 eventstream pilot")
    parser.add_argument("--month", required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW_L2_ROOT)
    parser.add_argument("--pack-root", type=Path, default=PACK_ROOT)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--universe-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete-pack", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_pilot_audit(
        month=arguments.month,
        raw_root=arguments.raw_root.expanduser().resolve(),
        pack_root=arguments.pack_root.expanduser().resolve(),
        feature_manifest=arguments.feature_manifest.expanduser().resolve(),
        universe_output=arguments.universe_output.expanduser().resolve(),
        output=arguments.output.expanduser().resolve(),
        require_complete_pack=arguments.require_complete_pack,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
