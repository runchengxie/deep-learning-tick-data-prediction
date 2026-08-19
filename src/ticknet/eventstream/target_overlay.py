"""为物化事件窗口生成轻量、可校验的日级训练标签覆盖层。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.eventstream.materialized import (
    assert_materialized_compatible,
    build_source_datasets,
    load_materialized_manifest,
    verify_materialized_dataset,
)
from ticknet.eventstream.storage_readiness import validate_storage_manifest
from ticknet.eventstream.train import EventstreamConfig

SCHEMA_VERSION = 1
MODE = "eventstream_day_target_overlay"
TRANSFORM_NAME = "daily_cross_sectional_winsorized_z_v1"
PARTITION = "train"
MAD_MULTIPLIER = 5.0
MIN_SCALE = 1e-12


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return value


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"标签覆盖层路径必须是安全的相对路径：{value}")
    return path.as_posix()


def _record_file(record: dict[str, Any], name: str) -> str:
    matches = [row for row in record["files"] if Path(str(row["path"])).stem == name]
    if len(matches) != 1:
        raise ValueError(f"物化分片缺少唯一 {name}.npy：{record['partition']} {record['month']}")
    return str(matches[0]["path"])


def daily_winsorized_z(values: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """按中位数加减 5 倍 MAD 去极值，再做总体标准差 z 标准化。"""
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2 or not np.isfinite(raw).all():
        raise ValueError("每日截面至少需要两个有限标签")
    median = float(np.median(raw))
    mad = float(np.median(np.abs(raw - median)))
    if mad <= MIN_SCALE:
        raise ValueError("每日截面标签的 MAD 过小，无法执行预注册去极值")
    lower = median - MAD_MULTIPLIER * mad
    upper = median + MAD_MULTIPLIER * mad
    clipped = np.clip(raw, lower, upper)
    mean = float(clipped.mean())
    scale = float(clipped.std(ddof=0))
    if scale <= MIN_SCALE:
        raise ValueError("每日截面去极值后的标准差过小，无法执行 z 标准化")
    transformed = ((clipped - mean) / scale).astype(np.float32)
    return transformed, {
        "symbols": int(raw.size),
        "raw_median": median,
        "raw_mad": mad,
        "lower_bound": lower,
        "upper_bound": upper,
        "clipped_symbols": int(((raw < lower) | (raw > upper)).sum()),
        "winsorized_mean": mean,
        "winsorized_population_std": scale,
        "transformed_mean": float(transformed.astype(np.float64).mean()),
        "transformed_population_std": float(transformed.astype(np.float64).std(ddof=0)),
    }


def _label_cross_sections(
    path: Path,
    *,
    selected_days: set[int],
) -> tuple[dict[int, dict[str, float]], list[dict[str, Any]]]:
    table = pq.read_table(path)
    names = table.column_names
    if "value" in names:
        day_column = "value"
    else:
        day_column = "__index_level_0__" if "__index_level_0__" in names else names[0]
    ticker_columns = [name for name in names if name != day_column]
    mappings: dict[int, dict[str, float]] = {}
    day_stats: list[dict[str, Any]] = []
    for row, raw_day in enumerate(table.column(day_column).to_pylist()):
        day = int(raw_day)
        if day not in selected_days:
            continue
        symbols: list[str] = []
        values: list[float] = []
        for ticker in ticker_columns:
            value = table.column(ticker)[row].as_py()
            if value is not None and math.isfinite(float(value)):
                symbols.append(str(ticker))
                values.append(float(value))
        if not values:
            continue
        transformed, stats = daily_winsorized_z(np.asarray(values, dtype=np.float64))
        mappings[day] = dict(zip(symbols, (float(value) for value in transformed), strict=True))
        day_stats.append({"trading_day": day, **stats})
    return mappings, sorted(day_stats, key=lambda row: int(row["trading_day"]))


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest["contract"],
        "files": manifest["files"],
        "totals": manifest["totals"],
        "day_stats": manifest["day_stats"],
    }


def _validate_overlay_contract(
    contract: dict[str, Any],
    *,
    expected_materialized_fingerprint: str | None,
    expected_partition: str | None,
) -> None:
    expected_transform = {
        "name": TRANSFORM_NAME,
        "winsorization": "median_plus_minus_5_raw_mad",
        "mad_multiplier": MAD_MULTIPLIER,
        "centering": "winsorized_cross_section_mean",
        "scaling": "winsorized_cross_section_population_std",
        "evaluation_target": "raw_h5_return",
    }
    if contract.get("transform") != expected_transform:
        raise ValueError("日级标签覆盖层转换合同无效")
    if expected_materialized_fingerprint is not None and (
        contract.get("materialized_dataset_fingerprint") != expected_materialized_fingerprint
    ):
        raise ValueError("日级标签覆盖层与物化数据指纹不一致")
    if expected_partition is not None and contract.get("partition") != expected_partition:
        raise ValueError("日级标签覆盖层分区不一致")
    if contract.get("partition") != PARTITION:
        raise ValueError("日级标签覆盖层只允许替换 train 分区")


def _verify_overlay_files(root: Path, files: list[Any]) -> tuple[int, int]:
    samples = 0
    valid_samples = 0
    seen_months: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("日级标签覆盖层文件记录无效")
        month = str(record.get("month", ""))
        if len(month) != 6 or not month.isdigit() or month in seen_months:
            raise ValueError("日级标签覆盖层月份无效或重复")
        seen_months.add(month)
        relative = _safe_relative_path(str(record.get("path", "")))
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"日级标签覆盖层文件大小漂移：{relative}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"日级标签覆盖层文件内容漂移：{relative}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.dtype(np.float32) or values.shape != (
            int(record.get("samples", -1)),
        ):
            raise ValueError(f"日级标签覆盖层数组合同无效：{relative}")
        if not np.isfinite(values).all():
            raise ValueError(f"日级标签覆盖层包含非有限值：{relative}")
        samples += int(record["samples"])
        valid_samples += int(record["valid_samples"])
    return samples, valid_samples


def _validate_day_stats(day_stats: list[Any]) -> None:
    days: list[int] = []
    for row in day_stats:
        if not isinstance(row, dict):
            raise ValueError("日级标签覆盖层逐日统计无效")
        day = int(row.get("trading_day", 0))
        if day < 20000101 or int(row.get("symbols", 0)) < 2:
            raise ValueError("日级标签覆盖层逐日日期或股票数无效")
        if (
            float(row.get("raw_mad", 0.0)) <= MIN_SCALE
            or float(row.get("winsorized_population_std", 0.0)) <= MIN_SCALE
        ):
            raise ValueError("日级标签覆盖层逐日尺度无效")
        if abs(float(row.get("transformed_mean", math.inf))) > 1e-5 or not math.isclose(
            float(row.get("transformed_population_std", 0.0)),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError("日级标签覆盖层逐日 z 标准化统计无效")
        days.append(day)
    if days != sorted(set(days)):
        raise ValueError("日级标签覆盖层逐日统计应唯一并按日期排序")


def load_target_overlay_manifest(
    root: Path,
    *,
    expected_materialized_fingerprint: str | None = None,
    expected_partition: str | None = None,
) -> dict[str, Any]:
    """加载并逐文件核对标签覆盖层。"""
    root = Path(root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("mode") != MODE
        or manifest.get("status") != "complete"
    ):
        raise ValueError("日级标签覆盖层格式、版本或状态无效")
    contract = manifest.get("contract")
    files = manifest.get("files")
    totals = manifest.get("totals")
    day_stats = manifest.get("day_stats")
    if (
        not isinstance(contract, dict)
        or not isinstance(files, list)
        or not isinstance(totals, dict)
        or not isinstance(day_stats, list)
    ):
        raise ValueError("日级标签覆盖层缺少合同、文件、汇总或逐日统计")
    _validate_overlay_contract(
        contract,
        expected_materialized_fingerprint=expected_materialized_fingerprint,
        expected_partition=expected_partition,
    )
    samples, valid_samples = _verify_overlay_files(root, files)
    _validate_day_stats(day_stats)
    if files != sorted(
        files,
        key=lambda record: str(record.get("month", "")) if isinstance(record, dict) else "",
    ):
        raise ValueError("日级标签覆盖层文件应按月份排序")
    if totals != {
        "files": len(files),
        "samples": samples,
        "valid_samples": valid_samples,
        "days": len(day_stats),
    }:
        raise ValueError("日级标签覆盖层汇总不一致")
    if manifest.get("dataset_fingerprint") != _canonical_sha256(_manifest_payload(manifest)):
        raise ValueError("日级标签覆盖层数据指纹不匹配")
    return manifest


def _write_overlay_files(
    *,
    source: Any,
    materialized_root: Path,
    output_root: Path,
    records: list[dict[str, Any]],
    transformed_by_day: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    source_indices_by_month: dict[str, list[int]] = defaultdict(list)
    for index, (day, _ticker_index, _start) in enumerate(source.entries):
        source_indices_by_month[str(day)[:6]].append(index)

    files: list[dict[str, Any]] = []
    for record in records:
        month = str(record["month"])
        source_indices = source_indices_by_month[month]
        if len(source_indices) != int(record["samples"]):
            raise ValueError(f"train-{month} 的源样本数与物化记录不同")
        days = np.load(
            materialized_root / _record_file(record, "day"),
            mmap_mode="r",
            allow_pickle=False,
        )
        raw_targets = np.load(
            materialized_root / _record_file(record, "tgt_day"),
            mmap_mode="r",
            allow_pickle=False,
        )
        day_valid = np.load(
            materialized_root / _record_file(record, "day_valid"),
            mmap_mode="r",
            allow_pickle=False,
        )
        overlay = np.zeros(len(source_indices), dtype=np.float32)
        valid_samples = 0
        for local_index, source_index in enumerate(source_indices):
            day, ticker = source.sample_key(source_index)
            if int(days[local_index]) != day:
                raise ValueError(f"train-{month} 第 {local_index} 行交易日不一致")
            _entry_day, ticker_index, _start = source.entries[source_index]
            source_raw = float(source.index[day]["label"][ticker_index])
            if day_valid[local_index] <= 0:
                if float(raw_targets[local_index]) != 0.0 or math.isfinite(source_raw):
                    raise ValueError(f"train-{month} 第 {local_index} 行无效标签合同不一致")
                continue
            if not math.isfinite(source_raw) or not math.isclose(
                float(raw_targets[local_index]), source_raw, rel_tol=1e-6, abs_tol=1e-7
            ):
                raise ValueError(f"train-{month} 第 {local_index} 行原始标签不一致")
            try:
                overlay[local_index] = transformed_by_day[day][ticker]
            except KeyError as error:
                raise ValueError(f"train-{month} 缺少 {day} {ticker} 的转换标签") from error
            valid_samples += 1
        relative = f"targets/train-{month}.npy"
        path = output_root / relative
        _atomic_npy(path, overlay)
        files.append(
            {
                "partition": PARTITION,
                "month": month,
                "path": relative,
                "samples": int(overlay.size),
                "valid_samples": valid_samples,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return files


def build_target_overlay(
    config: EventstreamConfig,
    *,
    storage_manifest_path: Path,
    materialized_root: Path,
    output_root: Path,
    source_revision: str,
) -> dict[str, Any]:
    """从 H5 原始标签和物化样本身份生成 train 分区标签覆盖层。"""
    if not source_revision or source_revision == "unknown":
        raise ValueError("标签覆盖层需要有效的源码 revision")
    config.validate()
    if not config.label_path:
        raise ValueError("标签覆盖层需要 H5 标签文件")
    storage = _load_json(storage_manifest_path)
    validate_storage_manifest(storage)
    materialized_root = Path(materialized_root).expanduser().resolve()
    materialized = load_materialized_manifest(materialized_root)
    assert_materialized_compatible(materialized, config)
    if materialized["contract"]["source_inventory_sha256"] != storage["inventory_sha256"]:
        raise ValueError("存储清单与物化缓存的源库存指纹不同")
    label_sha256 = file_sha256(config.label_path)
    if label_sha256 != materialized["contract"]["label_sha256"]:
        raise ValueError("H5 标签与物化缓存合同不同")
    verify_materialized_dataset(materialized_root, partitions=(PARTITION,))

    source = build_source_datasets(config, storage["contract"]["splits"])[PARTITION]
    train_days = {int(day) for day in storage["contract"]["splits"][PARTITION]}
    transformed_by_day, day_stats = _label_cross_sections(
        Path(config.label_path),
        selected_days=train_days,
    )
    contract = {
        "source_revision": source_revision,
        "materialized_dataset_fingerprint": materialized["dataset_fingerprint"],
        "materialized_source_revision": materialized["contract"]["source_revision"],
        "source_inventory_sha256": storage["inventory_sha256"],
        "label_sha256": label_sha256,
        "partition": PARTITION,
        "locked_start": materialized["contract"]["locked_start"],
        "seed": config.seed,
        "transform": {
            "name": TRANSFORM_NAME,
            "winsorization": "median_plus_minus_5_raw_mad",
            "mad_multiplier": MAD_MULTIPLIER,
            "centering": "winsorized_cross_section_mean",
            "scaling": "winsorized_cross_section_population_std",
            "evaluation_target": "raw_h5_return",
        },
    }
    output_root = Path(output_root).expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and any(output_root.iterdir()):
        existing = load_target_overlay_manifest(
            output_root,
            expected_materialized_fingerprint=str(materialized["dataset_fingerprint"]),
            expected_partition=PARTITION,
        )
        if existing["contract"] != contract:
            raise ValueError("已有标签覆盖层合同与本次运行不同")
        return existing
    output_root.mkdir(parents=True, exist_ok=True)

    records = [record for record in materialized["shards"] if record["partition"] == PARTITION]
    files = _write_overlay_files(
        source=source,
        materialized_root=materialized_root,
        output_root=output_root,
        records=records,
        transformed_by_day=transformed_by_day,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "complete",
        "contract": contract,
        "files": files,
        "totals": {
            "files": len(files),
            "samples": sum(int(record["samples"]) for record in files),
            "valid_samples": sum(int(record["valid_samples"]) for record in files),
            "days": len(day_stats),
        },
        "day_stats": day_stats,
    }
    manifest["dataset_fingerprint"] = _canonical_sha256(_manifest_payload(manifest))
    _atomic_json(manifest_path, manifest)
    return load_target_overlay_manifest(
        output_root,
        expected_materialized_fingerprint=str(materialized["dataset_fingerprint"]),
        expected_partition=PARTITION,
    )


def _load_config(path: Path) -> EventstreamConfig:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("事件流配置应为 YAML 对象")
    return EventstreamConfig.from_mapping(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成并核对每日截面去极值 z 训练标签覆盖层")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--storage-manifest", type=Path, required=True)
    build.add_argument("--materialized-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-revision", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--materialized-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        result = build_target_overlay(
            _load_config(arguments.config),
            storage_manifest_path=arguments.storage_manifest,
            materialized_root=arguments.materialized_root,
            output_root=arguments.output,
            source_revision=arguments.source_revision,
        )
    else:
        materialized = load_materialized_manifest(arguments.materialized_root)
        result = load_target_overlay_manifest(
            arguments.root,
            expected_materialized_fingerprint=str(materialized["dataset_fingerprint"]),
            expected_partition=PARTITION,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
