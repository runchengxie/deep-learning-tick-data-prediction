"""物化事件流联合微调所需的分钟特征、目标和尾盘缓存行号。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.eventstream.close_cache import (
    PARTITIONS as CLOSE_PARTITIONS,
)
from ticknet.eventstream.close_cache import (
    load_close_cache_manifest,
    verify_close_cache,
)
from ticknet.eventstream.fingerprint import file_sha256, git_sha
from ticknet.nextday.embedding_comparison import ComparisonConfig, load_comparison_config
from ticknet.nextday.minute_baseline import (
    MinuteBaselineConfig,
    MinuteExtractionReport,
    MinuteSample,
    build_target_bundle,
    load_minute_baseline_config,
)
from ticknet.nextday.minute_materialization import load_materialized_minute_features
from ticknet.nextday.splits import WalkForwardSplit

SCHEMA_VERSION = 1
MODE = "eventstream_joint_feature_cache"
MANIFEST_NAME = "manifest.json"
PARTITIONS = ("train", "val", "test")
CLOSE_PARTITION = {"train": "train", "val": "validation", "test": "oos"}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"联合特征路径必须是安全的相对路径：{value}")
    return path.as_posix()


def _atomic_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _assigned_partition(sample: MinuteSample, split: WalkForwardSplit) -> str | None:
    partition = split.assign(sample.trading_date, sample.label_date)
    if (
        partition is not None
        and sample.return_end_date is not None
        and not split.range_for(partition).contains(sample.return_end_date)
    ):
        return None
    return partition


def _close_locations(
    close_root: Path,
    manifest: dict[str, Any],
) -> dict[tuple[date, str], tuple[str, str, int]]:
    result: dict[tuple[date, str], tuple[str, str, int]] = {}
    for record in manifest["shards"]:
        files = {
            Path(item["path"]).stem: close_root / str(item["path"]) for item in record["files"]
        }
        days = np.load(files["day"], mmap_mode="r", allow_pickle=False)
        symbols = np.load(files["symbol"], mmap_mode="r", allow_pickle=False)
        shard = Path(str(files["day"].relative_to(close_root))).parent.as_posix()
        partition = str(record["partition"])
        for row, (raw_day, raw_symbol) in enumerate(zip(days, symbols, strict=True)):
            text = str(int(raw_day))
            key = (
                date(int(text[:4]), int(text[4:6]), int(text[6:])),
                bytes(raw_symbol).decode("ascii"),
            )
            if key in result:
                raise ValueError(f"尾盘缓存股票日重复：{key}")
            result[key] = (partition, shard, row)
    if len(result) != int(manifest["totals"]["samples"]):
        raise ValueError("尾盘缓存股票日索引行数与 manifest 不一致")
    return result


def _comparison_contract(config: ComparisonConfig) -> dict[str, Any]:
    result = asdict(config)
    result["top_ks"] = list(config.top_ks)
    return result


def _joint_table(
    rows: list[tuple[MinuteSample, str, int]],
    *,
    feature_count: int,
) -> pa.Table:
    matrix = np.stack([sample.features for sample, _shard, _row in rows]).astype(
        np.float32,
        copy=False,
    )
    if matrix.shape[1] != feature_count or np.isinf(matrix).any():
        raise ValueError("联合特征维度无效或包含无穷值")
    return pa.table(
        {
            "trading_day": pa.array(
                [int(sample.trading_date.strftime("%Y%m%d")) for sample, _shard, _row in rows],
                type=pa.int32(),
            ),
            "symbol": [sample.symbol for sample, _shard, _row in rows],
            "label_date": pa.array(
                [sample.label_date for sample, _shard, _row in rows], type=pa.date32()
            ),
            "return_end_date": pa.array(
                [sample.return_end_date for sample, _shard, _row in rows], type=pa.date32()
            ),
            "label": pa.array([sample.label for sample, _shard, _row in rows], type=pa.int8()),
            "ranking_target_return": pa.array(
                [sample.target_return for sample, _shard, _row in rows], type=pa.float64()
            ),
            "feature_available": [sample.feature_available for sample, _shard, _row in rows],
            "minute_features": _fixed_list(matrix),
            "close_shard": [shard for _sample, shard, _row in rows],
            "close_row": pa.array([row for _sample, _shard, row in rows], type=pa.int32()),
        }
    )


def _portfolio_target_table(targets: list[Any], selected_days: set[date]) -> pa.Table:
    rows = sorted(
        (target for target in targets if target.trading_date in selected_days),
        key=lambda target: (target.trading_date, target.symbol),
    )
    if not rows:
        raise ValueError("联合特征缓存缺少组合评估目标")
    return pa.table(
        {
            "trading_date": pa.array([target.trading_date for target in rows], type=pa.date32()),
            "symbol": [target.symbol for target in rows],
            "label_date": pa.array([target.label_date for target in rows], type=pa.date32()),
            "portfolio_return": pa.array(
                [target.portfolio_return for target in rows], type=pa.float64()
            ),
            "can_buy": [bool(target.can_buy) for target in rows],
            "can_sell": [bool(target.can_sell) for target in rows],
            "in_universe": [bool(target.in_universe) for target in rows],
        }
    )


def _artifact(path: Path, root: Path, *, partition: str, rows: int) -> dict[str, Any]:
    return {
        "partition": partition,
        "path": _safe_relative_path(path.relative_to(root).as_posix()),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest["contract"],
        "artifacts": manifest["artifacts"],
        "portfolio_targets": manifest["portfolio_targets"],
        "totals": manifest["totals"],
    }


def validate_joint_cache_manifest(manifest: dict[str, Any]) -> None:
    """验证联合特征缓存的结构和逻辑指纹。"""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != MODE:
        raise ValueError("联合特征缓存格式或版本无效")
    if manifest.get("status") != "complete":
        raise ValueError("联合特征缓存尚未完成")
    contract = manifest.get("contract")
    artifacts = manifest.get("artifacts")
    targets = manifest.get("portfolio_targets")
    totals = manifest.get("totals")
    if (
        not isinstance(contract, dict)
        or not isinstance(artifacts, list)
        or not isinstance(targets, dict)
        or not isinstance(totals, dict)
    ):
        raise ValueError("联合特征缓存缺少合同、分片或汇总")
    if manifest.get("contract_sha256") != _canonical_sha256(contract):
        raise ValueError("联合特征缓存合同指纹不匹配")
    if int(contract.get("feature_count", 0)) < 1:
        raise ValueError("联合特征缓存分钟特征维度无效")
    if {str(row.get("partition")) for row in artifacts} != set(PARTITIONS):
        raise ValueError("联合特征缓存必须覆盖 train、val 和 test")
    for row in [*artifacts, targets]:
        _safe_relative_path(str(row.get("path", "")))
        if int(row.get("rows", 0)) < 1 or int(row.get("bytes", -1)) < 0:
            raise ValueError("联合特征缓存文件记录无效")
        if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise ValueError("联合特征缓存文件缺少 SHA-256")
    expected = {
        "rows": sum(int(row["rows"]) for row in artifacts),
        "bytes": sum(int(row["bytes"]) for row in artifacts) + int(targets["bytes"]),
        "partitions": {
            partition: sum(int(row["rows"]) for row in artifacts if row["partition"] == partition)
            for partition in PARTITIONS
        },
    }
    if totals != expected:
        raise ValueError("联合特征缓存汇总不一致")
    if manifest.get("dataset_fingerprint") != _canonical_sha256(_manifest_payload(manifest)):
        raise ValueError("联合特征缓存数据指纹不匹配")


def _verify_parquet(root: Path, record: dict[str, Any], feature_count: int) -> None:
    path = root / str(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"联合特征缓存文件缺失或大小漂移：{record['path']}")
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"联合特征缓存文件内容漂移：{record['path']}")
    table = pq.read_table(path)
    if table.num_rows != int(record["rows"]):
        raise ValueError(f"联合特征缓存文件行数漂移：{record['path']}")
    if "minute_features" in table.column_names:
        feature_type = table.schema.field("minute_features").type
        if not pa.types.is_fixed_size_list(feature_type) or feature_type.list_size != feature_count:
            raise ValueError("联合特征缓存分钟特征 schema 漂移")
        keys = list(zip(table["trading_day"].to_pylist(), table["symbol"].to_pylist(), strict=True))
    else:
        keys = list(
            zip(table["trading_date"].to_pylist(), table["symbol"].to_pylist(), strict=True)
        )
    if len(keys) != len(set(keys)):
        raise ValueError(f"联合特征缓存文件存在重复股票日：{record['path']}")


def load_joint_cache_manifest(root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("联合特征缓存 manifest 顶层应为对象")
    validate_joint_cache_manifest(manifest)
    if verify_files:
        feature_count = int(manifest["contract"]["feature_count"])
        for record in [*manifest["artifacts"], manifest["portfolio_targets"]]:
            _verify_parquet(Path(root), record, feature_count)
    return manifest


def build_joint_cache(
    *,
    minute_config: MinuteBaselineConfig,
    minute_features_root: Path,
    comparison_config: ComparisonConfig,
    close_cache_root: Path,
    output_root: Path,
    source_revision: str,
) -> dict[str, Any]:
    """按 frozen E2 的股票日交集物化联合微调轻量特征缓存。"""
    if not minute_config.formal:
        raise ValueError("联合特征缓存要求正式 open-to-following-open 分钟配置")
    if not source_revision or source_revision == "unknown":
        raise ValueError("联合特征缓存需要有效的源码 revision")
    comparison_config.validate()
    close_cache_root = Path(close_cache_root)
    verify_close_cache(close_cache_root, partitions=CLOSE_PARTITIONS)
    close_manifest = load_close_cache_manifest(close_cache_root)
    close_locations = _close_locations(close_cache_root, close_manifest)

    bundle = build_target_bundle(minute_config)
    candidates = [target for target in bundle.targets if target.in_universe]
    report = MinuteExtractionReport()
    materialized = load_materialized_minute_features(
        minute_config,
        candidates,
        minute_features_root,
        report,
    )
    split = comparison_config.split()
    selected: dict[str, list[tuple[MinuteSample, str, int]]] = defaultdict(list)
    for sample in materialized.samples:
        partition = _assigned_partition(sample, split)
        location = close_locations.get((sample.trading_date, sample.symbol))
        if partition is None or location is None:
            continue
        close_partition, shard, row = location
        if close_partition != CLOSE_PARTITION[partition]:
            raise ValueError(f"联合特征与尾盘缓存分区不一致：{sample.trading_date} {sample.symbol}")
        selected[partition].append((sample, shard, row))
    for partition in PARTITIONS:
        selected[partition].sort(key=lambda item: (item[0].trading_date, item[0].symbol))
        if not selected[partition]:
            raise ValueError(f"联合特征缓存分区为空：{partition}")

    feature_count = int(selected["train"][0][0].features.shape[0])
    contract = {
        "source_revision": source_revision,
        "close_cache_fingerprint": close_manifest["dataset_fingerprint"],
        "close_cache_contract_sha256": close_manifest["contract_sha256"],
        "minute_materialization_identity": materialized.materialization_identity,
        "minute_manifest_fingerprint": materialized.manifest_fingerprint,
        "target_return_contract": minute_config.target_return_contract,
        "comparison_config": _comparison_contract(comparison_config),
        "feature_count": feature_count,
        "close_reference": "relative_shard_and_row_v1",
    }
    output_root = Path(output_root)
    manifest_path = output_root / MANIFEST_NAME
    if manifest_path.exists():
        existing = load_joint_cache_manifest(output_root)
        if existing["contract"] != contract:
            raise ValueError("已有联合特征缓存与本次合同不同")
        return existing
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("联合特征缓存输出目录非空且缺少 manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        table = _joint_table(selected[partition], feature_count=feature_count)
        path = output_root / "shards" / f"{partition}.parquet"
        _atomic_parquet(path, table)
        artifacts.append(_artifact(path, output_root, partition=partition, rows=table.num_rows))

    selected_days = {
        sample.trading_date for rows in selected.values() for sample, _shard, _row in rows
    }
    portfolio_table = _portfolio_target_table(list(bundle.targets), selected_days)
    portfolio_path = output_root / "portfolio-targets.parquet"
    _atomic_parquet(portfolio_path, portfolio_table)
    portfolio_record = _artifact(
        portfolio_path,
        output_root,
        partition="portfolio_targets",
        rows=portfolio_table.num_rows,
    )
    totals = {
        "rows": sum(int(row["rows"]) for row in artifacts),
        "bytes": sum(int(row["bytes"]) for row in artifacts) + int(portfolio_record["bytes"]),
        "partitions": {
            partition: sum(int(row["rows"]) for row in artifacts if row["partition"] == partition)
            for partition in PARTITIONS
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "complete",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "artifacts": artifacts,
        "portfolio_targets": portfolio_record,
        "totals": totals,
    }
    manifest["dataset_fingerprint"] = _canonical_sha256(_manifest_payload(manifest))
    validate_joint_cache_manifest(manifest)
    _atomic_json(manifest_path, manifest)
    return load_joint_cache_manifest(output_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="物化并核对事件流联合微调特征缓存")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="生成 frozen E2 同口径联合特征缓存")
    build.add_argument("--minute-config", type=Path, required=True)
    build.add_argument("--minute-features", type=Path, required=True)
    build.add_argument("--comparison-config", type=Path, required=True)
    build.add_argument("--close-cache", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-revision", default="")
    verify = commands.add_parser("verify", help="逐文件核对联合特征缓存")
    verify.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        report = build_joint_cache(
            minute_config=load_minute_baseline_config(arguments.minute_config),
            minute_features_root=arguments.minute_features.expanduser().resolve(),
            comparison_config=load_comparison_config(arguments.comparison_config),
            close_cache_root=arguments.close_cache.expanduser().resolve(),
            output_root=arguments.output.expanduser().resolve(),
            source_revision=arguments.source_revision or git_sha(Path.cwd()),
        )
    else:
        report = load_joint_cache_manifest(arguments.root.expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
