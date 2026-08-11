"""把多周期长表转换成 eventstream 使用的 fold 级宽表标签。"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.nextday.dataset import manifest_fingerprint
from ticknet.nextday.horizon_labels import HORIZON_RETURN_CONTRACT, load_horizon_sidecar
from ticknet.nextday.splits import WalkForwardSplit

FORMAT_VERSION = 1
PURGE_CONTRACT = "signal_entry_return_end_must_share_split"


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _feature_fingerprint(path: Path) -> str:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("特征 manifest 根节点应为对象")
    computed = manifest_fingerprint(manifest)
    stored = manifest.get("dataset_fingerprint")
    if stored is not None and stored != computed:
        raise ValueError("特征 manifest dataset_fingerprint 与内容不一致")
    return str(stored or computed)


def _split_for_trading_date(split: WalkForwardSplit, trading_date) -> str | None:
    for name in ("train", "val", "test"):
        if split.range_for(name).contains(trading_date):
            return name
    return None


def _write_wide_labels(path: Path, rows: dict[int, dict[str, float]]) -> None:
    symbols = sorted({symbol for values in rows.values() for symbol in values})
    if not rows or not symbols:
        raise ValueError("purge 后没有可写入的 eventstream 标签")
    days = sorted(rows)
    columns: dict[str, pa.Array] = {
        "value": pa.array(days, type=pa.int32()),
    }
    for symbol in symbols:
        columns[symbol] = pa.array(
            [rows[day].get(symbol) for day in days],
            type=pa.float32(),
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.table(columns), temporary, compression="zstd")
    os.replace(temporary, path)


def prepare_eventstream_fold_labels(
    *,
    sidecar_manifest: Path,
    feature_manifest: Path,
    output_dir: Path,
    split: WalkForwardSplit,
    horizons: tuple[int, ...] = (3, 5),
) -> Path:
    """生成 H3/H5 宽表，并清除跨 train、val、test 边界的收益标签。"""
    selected = tuple(sorted({int(value) for value in horizons}))
    if not selected or any(value < 1 for value in selected):
        raise ValueError("horizons 必须包含正整数")
    source_fingerprint = _feature_fingerprint(feature_manifest)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    source_sidecar_fingerprint = ""

    for horizon in selected:
        loaded = load_horizon_sidecar(
            sidecar_manifest,
            horizon=horizon,
            source_dataset_fingerprint=source_fingerprint,
        )
        source_sidecar_fingerprint = loaded.sidecar_fingerprint
        rows: dict[int, dict[str, float]] = defaultdict(dict)
        accepted = dict.fromkeys(("train", "val", "test"), 0)
        purged = dict.fromkeys(("train", "val", "test"), 0)
        for target in loaded.records.values():
            trading_split = _split_for_trading_date(split, target.trading_date)
            if trading_split is None:
                continue
            assigned = split.assign(
                target.trading_date,
                target.entry_date,
                target.return_end_date,
            )
            if assigned is None:
                purged[trading_split] += 1
                continue
            day = int(target.trading_date.strftime("%Y%m%d"))
            rows[day][target.symbol] = target.target_return
            accepted[assigned] += 1

        labels_path = root / f"h{horizon}.parquet"
        _write_wide_labels(labels_path, rows)
        artifacts[str(horizon)] = {
            "path": labels_path.name,
            "bytes": labels_path.stat().st_size,
            "sha256": file_sha256(labels_path),
            "days": len(rows),
            "symbols": len({symbol for values in rows.values() for symbol in values}),
            "accepted_by_split": accepted,
            "purged_by_split": purged,
        }

    report = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "mode": "eventstream_horizon_fold_labels",
        "return_contract": HORIZON_RETURN_CONTRACT,
        "purge_contract": PURGE_CONTRACT,
        "source_feature_manifest": str(feature_manifest.expanduser().resolve()),
        "source_dataset_fingerprint": source_fingerprint,
        "source_horizon_sidecar": str(sidecar_manifest.expanduser().resolve()),
        "source_sidecar_fingerprint": source_sidecar_fingerprint,
        "split": {
            "train": [split.train.start.isoformat(), split.train.end.isoformat()],
            "val": [split.val.start.isoformat(), split.val.end.isoformat()],
            "test": [split.test.start.isoformat(), split.test.end.isoformat()],
        },
        "artifacts": artifacts,
    }
    report_path = root / "manifest.json"
    _atomic_json(report_path, report)
    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 eventstream fold 级 H3/H5 标签")
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-start", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    args = parser.parse_args(argv)
    split = WalkForwardSplit.from_strings(
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    path = prepare_eventstream_fold_labels(
        sidecar_manifest=args.sidecar,
        feature_manifest=args.feature_manifest,
        output_dir=args.output_dir,
        split=split,
        horizons=tuple(args.horizons),
    )
    print(f"eventstream fold 标签：{path}")


if __name__ == "__main__":
    main()
