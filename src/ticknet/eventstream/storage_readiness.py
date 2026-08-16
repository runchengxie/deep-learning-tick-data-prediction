"""事件流正式训练的存储清单与运行盘预检。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ticknet.eventstream.config import day_pack_paths

SCHEMA_VERSION = 1
MODE = "eventstream_storage_manifest"
SPLITS = ("train", "validation", "oos")
HASH_NAMES = ("sha256", "md5")
REQUIRED_ARTIFACT_PATHS = frozenset(
    {
        "fold-labels/manifest.json",
        "fold-labels/h3.parquet",
        "fold-labels/h5.parquet",
    }
)
DEFAULT_LOCKED_START = 20260101
_CHUNK_BYTES = 8 << 20


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"存储清单路径必须是安全的相对路径：{value}")
    return path.as_posix()


def _file_record(path: Path, logical_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    hashers = {name: hashlib.new(name) for name in HASH_NAMES}
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_BYTES), b""):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {
        "path": _safe_relative_path(logical_path),
        "bytes": path.stat().st_size,
        "hashes": {name: hasher.hexdigest() for name, hasher in hashers.items()},
    }


def _load_split_ranges(config_path: Path) -> dict[str, tuple[int, int]]:
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("事件流配置应为 YAML 对象")
    fields = {
        "train": ("train_start", "train_end"),
        "validation": ("val_start", "val_end"),
        "oos": ("test_start", "test_end"),
    }
    ranges: dict[str, tuple[int, int]] = {}
    for split, (start_name, end_name) in fields.items():
        start, end = int(raw.get(start_name, 0)), int(raw.get(end_name, 0))
        if not 0 < start <= end:
            raise ValueError(f"配置缺少有效的 {start_name}/{end_name}")
        ranges[split] = (start, end)
    ordered = [ranges[name] for name in SPLITS]
    if any(left[1] >= right[0] for left, right in pairwise(ordered)):
        raise ValueError("train、validation 和 OOS 日期区间应按时间排列且不能重叠")
    return ranges


def _expected_days_from_universes(
    universe_paths: list[Path],
    ranges: dict[str, tuple[int, int]],
    *,
    locked_start: int,
) -> tuple[dict[str, list[int]], str]:
    if not universe_paths:
        raise ValueError("至少需要一个按日股票池文件")
    days: set[int] = set()
    source_fingerprints: set[str] = set()
    for path in universe_paths:
        with path.open(encoding="utf-8") as file:
            universe = json.load(file)
        if not isinstance(universe, dict) or not isinstance(universe.get("universes"), dict):
            raise ValueError(f"股票池文件缺少 universes 对象：{path}")
        fingerprint = universe.get("source_dataset_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"股票池文件缺少有效的源数据指纹：{path}")
        source_fingerprints.add(fingerprint)
        declared_days = int(universe.get("days", -1))
        actual_days = {int(day) for day in universe["universes"]}
        if declared_days != len(actual_days):
            raise ValueError(f"股票池文件的 days 与 universes 数量不一致：{path}")
        if days & actual_days:
            raise ValueError(f"股票池文件包含重复交易日：{path}")
        days.update(actual_days)
    if len(source_fingerprints) != 1:
        raise ValueError("全部按日股票池必须来自同一个特征数据指纹")

    split_days = {split: [] for split in SPLITS}
    for day in sorted(days):
        if day >= locked_start:
            raise ValueError(f"股票池触及锁定区：{day}")
        matches = [split for split, (start, end) in ranges.items() if start <= day <= end]
        if len(matches) != 1:
            raise ValueError(f"交易日没有落入唯一的 train、validation 或 OOS 区间：{day}")
        split_days[matches[0]].append(day)
    empty = [split for split, values in split_days.items() if not values]
    if empty:
        raise ValueError(f"日期合同缺少分区：{empty}")
    return split_days, source_fingerprints.pop()


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest["contract"],
        "months": manifest["months"],
        "artifacts": manifest["artifacts"],
        "totals": manifest["totals"],
    }


def build_storage_manifest(
    *,
    config_path: Path,
    pack_root: Path,
    universe_paths: list[Path],
    artifacts: dict[str, Path],
    locked_start: int = DEFAULT_LOCKED_START,
) -> dict[str, Any]:
    """扫描固定日期合同，生成按月、带内容哈希的逻辑存储清单。"""
    artifact_paths = {_safe_relative_path(path) for path in artifacts}
    if artifact_paths != REQUIRED_ARTIFACT_PATHS:
        raise ValueError(f"正式 H5/H3 清单需要附属产物：{sorted(REQUIRED_ARTIFACT_PATHS)}")
    ranges = _load_split_ranges(config_path)
    if max(end for _start, end in ranges.values()) >= locked_start:
        raise ValueError("事件流配置触及锁定区")
    split_days, source_fingerprint = _expected_days_from_universes(
        universe_paths,
        ranges,
        locked_start=locked_start,
    )
    split_by_day = {day: split for split, days in split_days.items() for day in days}
    month_rows: dict[str, dict[str, Any]] = {}
    for day in sorted(split_by_day):
        month = str(day)[:6]
        row = month_rows.setdefault(
            month,
            {"month": f"{month[:4]}-{month[4:]}", "days": [], "files": [], "bytes": 0},
        )
        row["days"].append(day)
        paths = day_pack_paths(day, pack_root)
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"交易日 {day} 缺少 pack 文件：{missing}")
        for path in paths.values():
            record = _file_record(path, f"pack/{path.name}")
            row["files"].append(record)
            row["bytes"] += record["bytes"]

    artifact_rows = [
        _file_record(source, logical_path) for logical_path, source in sorted(artifacts.items())
    ]
    pack_files = sum(len(row["files"]) for row in month_rows.values())
    pack_bytes = sum(int(row["bytes"]) for row in month_rows.values())
    artifact_bytes = sum(int(row["bytes"]) for row in artifact_rows)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "complete",
        "contract": {
            "splits": split_days,
            "ranges": {
                split: {"start": start, "end": end} for split, (start, end) in ranges.items()
            },
            "locked_start": locked_start,
            "source_dataset_fingerprint": source_fingerprint,
        },
        "months": list(month_rows.values()),
        "artifacts": artifact_rows,
        "totals": {
            "days": len(split_by_day),
            "pack_files": pack_files,
            "artifact_files": len(artifact_rows),
            "files": pack_files + len(artifact_rows),
            "pack_bytes": pack_bytes,
            "artifact_bytes": artifact_bytes,
            "bytes": pack_bytes + artifact_bytes,
        },
    }
    manifest["inventory_sha256"] = _canonical_sha256(_manifest_payload(manifest))
    validate_storage_manifest(manifest)
    return manifest


def _validate_contract(manifest: dict[str, Any]) -> list[int]:
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("splits"), dict):
        raise ValueError("存储清单缺少日期合同")
    splits = contract["splits"]
    if set(splits) != set(SPLITS) or any(not isinstance(splits[name], list) for name in SPLITS):
        raise ValueError("存储清单必须包含 train、validation 和 OOS 日期")
    all_days = [int(day) for split in SPLITS for day in splits[split]]
    if len(all_days) != len(set(all_days)) or all_days != sorted(all_days):
        raise ValueError("存储清单交易日应唯一并按时间排列")
    locked_start = int(contract.get("locked_start", 0))
    if not locked_start or any(day >= locked_start for day in all_days):
        raise ValueError("存储清单触及锁定区")
    return all_days


def _validate_layout(
    manifest: dict[str, Any],
    all_days: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    months = manifest.get("months")
    artifacts = manifest.get("artifacts")
    if not isinstance(months, list) or not isinstance(artifacts, list):
        raise ValueError("存储清单缺少月份或附属产物")
    if any(not isinstance(record, dict) for record in artifacts):
        raise ValueError("存储清单附属产物记录无效")
    artifact_paths = {str(record.get("path", "")) for record in artifacts}
    if artifact_paths != REQUIRED_ARTIFACT_PATHS or len(artifacts) != len(REQUIRED_ARTIFACT_PATHS):
        raise ValueError("存储清单缺少正式 H5/H3 附属产物")

    expected_pack_paths = {
        f"pack/{path.name}" for day in all_days for path in day_pack_paths(day, Path(".")).values()
    }
    actual_pack_paths: set[str] = set()
    month_days: list[int] = []
    for month in months:
        if not isinstance(month, dict) or not isinstance(month.get("files"), list):
            raise ValueError("存储清单月份记录无效")
        if any(not isinstance(record, dict) for record in month["files"]):
            raise ValueError("存储清单月份文件记录无效")
        label = str(month.get("month", ""))
        days = month.get("days")
        if not isinstance(days, list) or days != sorted({int(day) for day in days}):
            raise ValueError(f"存储清单月份交易日无效：{label}")
        compact_month = label.replace("-", "")
        if len(compact_month) != 6 or any(str(day)[:6] != compact_month for day in days):
            raise ValueError(f"存储清单月份与交易日不一致：{label}")
        month_bytes = sum(int(record.get("bytes", -1)) for record in month["files"])
        if month_bytes != int(month.get("bytes", -1)):
            raise ValueError(f"存储清单月份字节汇总不一致：{label}")
        actual_pack_paths.update(str(record.get("path", "")) for record in month["files"])
        month_days.extend(int(day) for day in days)
    if month_days != all_days or actual_pack_paths != expected_pack_paths:
        raise ValueError("存储清单月份没有完整覆盖日期合同的四类 pack 文件")
    return months, artifacts, expected_pack_paths


def _validate_records_and_totals(
    manifest: dict[str, Any],
    *,
    all_days: list[int],
    months: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    expected_pack_paths: set[str],
) -> None:
    records = list(_required_records_unchecked(manifest))
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("存储清单包含重复路径")
    for record in records:
        _safe_relative_path(str(record.get("path", "")))
        if int(record.get("bytes", -1)) < 0:
            raise ValueError(f"存储清单文件大小无效：{record.get('path')}")
        hashes = record.get("hashes")
        if not isinstance(hashes, dict) or any(
            not isinstance(hashes.get(name), str) or not hashes[name] for name in HASH_NAMES
        ):
            raise ValueError(f"存储清单缺少内容哈希：{record.get('path')}")
    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("存储清单缺少汇总")
    actual_bytes = sum(int(record["bytes"]) for record in records)
    expected_totals = {
        "days": len(all_days),
        "pack_files": len(expected_pack_paths),
        "artifact_files": len(artifacts),
        "files": len(records),
        "pack_bytes": sum(int(record["bytes"]) for month in months for record in month["files"]),
        "artifact_bytes": sum(int(record["bytes"]) for record in artifacts),
        "bytes": actual_bytes,
    }
    if totals != expected_totals:
        raise ValueError("存储清单汇总与文件记录不一致")


def validate_storage_manifest(manifest: dict[str, Any]) -> None:
    """验证清单结构、汇总值和逻辑指纹。"""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != MODE:
        raise ValueError("存储清单格式或版本无效")
    if manifest.get("status") != "complete":
        raise ValueError("只接受状态为 complete 的存储清单")
    all_days = _validate_contract(manifest)
    months, artifacts, expected_pack_paths = _validate_layout(manifest, all_days)
    _validate_records_and_totals(
        manifest,
        all_days=all_days,
        months=months,
        artifacts=artifacts,
        expected_pack_paths=expected_pack_paths,
    )
    if manifest.get("inventory_sha256") != _canonical_sha256(_manifest_payload(manifest)):
        raise ValueError("存储清单逻辑指纹不匹配")


def _required_records_unchecked(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    months = manifest.get("months")
    artifacts = manifest.get("artifacts")
    if not isinstance(months, list) or not isinstance(artifacts, list):
        raise ValueError("存储清单缺少月份或附属产物")
    records: list[dict[str, Any]] = []
    for month in months:
        if not isinstance(month, dict) or not isinstance(month.get("files"), list):
            raise ValueError("存储清单月份记录无效")
        records.extend(month["files"])
    records.extend(artifacts)
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("存储清单文件记录无效")
    return records


def required_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_storage_manifest(manifest)
    return _required_records_unchecked(manifest)


def verify_staged_dataset(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """逐文件校验已落盘的数据，训练前调用会拒绝缺失或内容漂移。"""
    records = required_records(manifest)
    verified_bytes = 0
    for expected in records:
        path = root / str(expected["path"])
        actual = _file_record(path, str(expected["path"]))
        if actual["bytes"] != expected["bytes"] or actual["hashes"] != expected["hashes"]:
            raise ValueError(f"已落盘文件与存储清单不一致：{expected['path']}")
        verified_bytes += int(actual["bytes"])
    return {
        "status": "complete",
        "mode": "eventstream_staged_storage_preflight",
        "inventory_sha256": manifest["inventory_sha256"],
        "files": len(records),
        "bytes": verified_bytes,
    }


def verify_direct_remote_listing(
    manifest: dict[str, Any],
    listing: list[dict[str, Any]],
) -> dict[str, Any]:
    """核对 rclone lsjson 直存布局，要求每个文件至少有一个共享内容哈希。"""
    records = required_records(manifest)
    remote = {str(row.get("Path", "")): row for row in listing if isinstance(row, dict)}
    for expected in records:
        path = str(expected["path"])
        actual = remote.get(path)
        if actual is None:
            raise FileNotFoundError(f"远端缺少存储清单文件：{path}")
        if int(actual.get("Size", -1)) != int(expected["bytes"]):
            raise ValueError(f"远端文件大小与存储清单不一致：{path}")
        raw_hashes = actual.get("Hashes")
        hashes = (
            {str(name).lower(): str(value).lower() for name, value in raw_hashes.items()}
            if isinstance(raw_hashes, dict)
            else {}
        )
        shared = [name for name in HASH_NAMES if name in hashes]
        if not shared:
            raise ValueError(f"远端没有可与清单核对的内容哈希：{path}")
        if any(hashes[name] != str(expected["hashes"][name]).lower() for name in shared):
            raise ValueError(f"远端文件内容哈希与存储清单不一致：{path}")
    return {
        "status": "complete",
        "mode": "eventstream_direct_remote_preflight",
        "inventory_sha256": manifest["inventory_sha256"],
        "files": len(records),
        "bytes": int(manifest["totals"]["bytes"]),
    }


def check_full_copy_capacity(
    manifest: dict[str, Any],
    path: Path,
    *,
    reserve_bytes: int,
    headroom_ratio: float,
    available_bytes: int | None = None,
) -> dict[str, Any]:
    """检查完整复制布局所需运行盘，空间不足时确定性停止。"""
    validate_storage_manifest(manifest)
    if reserve_bytes < 0 or headroom_ratio < 1:
        raise ValueError("运行盘预留不能为负，容量系数不能小于 1")
    dataset_bytes = int(manifest["totals"]["bytes"])
    required_bytes = math.ceil(dataset_bytes * headroom_ratio) + reserve_bytes
    available = shutil.disk_usage(path).free if available_bytes is None else available_bytes
    report = {
        "status": "complete" if available >= required_bytes else "insufficient",
        "mode": "eventstream_full_copy_capacity_preflight",
        "inventory_sha256": manifest["inventory_sha256"],
        "dataset_bytes": dataset_bytes,
        "reserve_bytes": reserve_bytes,
        "headroom_ratio": headroom_ratio,
        "required_bytes": required_bytes,
        "available_bytes": available,
    }
    if available < required_bytes:
        raise RuntimeError(f"运行盘不足：需要 {required_bytes} 字节，当前可用 {available} 字节")
    return report


def _parse_artifacts(values: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        logical_path, separator, source = value.partition("=")
        if not separator or not source:
            raise ValueError("--artifact 应使用 逻辑路径=本地路径 格式")
        safe_path = _safe_relative_path(logical_path)
        if safe_path in artifacts:
            raise ValueError(f"重复附属产物路径：{safe_path}")
        artifacts[safe_path] = Path(source).expanduser().resolve()
    if not artifacts:
        raise ValueError("至少需要一个附属产物")
    return artifacts


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        content = json.load(file)
    if not isinstance(content, dict):
        raise ValueError("存储清单应为 JSON 对象")
    return content


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成并核对事件流正式训练存储清单")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="扫描本地 pack 和标签，生成按月逻辑清单")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--pack-root", type=Path, required=True)
    build.add_argument("--universe", type=Path, action="append", required=True)
    build.add_argument("--artifact", action="append", default=[])
    build.add_argument("--locked-start", type=int, default=DEFAULT_LOCKED_START)
    build.add_argument("--output", type=Path, required=True)

    staged = commands.add_parser("verify-staged", help="训练前逐文件核对已落盘数据")
    staged.add_argument("--manifest", type=Path, required=True)
    staged.add_argument("--root", type=Path, required=True)
    staged.add_argument("--output", type=Path)

    remote = commands.add_parser("verify-direct-remote", help="核对 rclone 直存文件列表")
    remote.add_argument("--manifest", type=Path, required=True)
    remote.add_argument("--listing", type=Path, required=True)
    remote.add_argument("--output", type=Path)

    capacity = commands.add_parser("check-full-copy-capacity", help="检查完整复制所需运行盘")
    capacity.add_argument("--manifest", type=Path, required=True)
    capacity.add_argument("--path", type=Path, required=True)
    capacity.add_argument("--reserve-gib", type=float, default=20.0)
    capacity.add_argument("--headroom-ratio", type=float, default=1.05)
    capacity.add_argument("--output", type=Path)
    return parser


def _emit(report: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _atomic_json(output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        report = build_storage_manifest(
            config_path=arguments.config.expanduser().resolve(),
            pack_root=arguments.pack_root.expanduser().resolve(),
            universe_paths=[path.expanduser().resolve() for path in arguments.universe],
            artifacts=_parse_artifacts(arguments.artifact),
            locked_start=arguments.locked_start,
        )
        _atomic_json(arguments.output.expanduser().resolve(), report)
    elif arguments.command == "verify-staged":
        report = verify_staged_dataset(
            _load_manifest(arguments.manifest.expanduser().resolve()),
            arguments.root.expanduser().resolve(),
        )
        _emit(report, arguments.output)
    elif arguments.command == "verify-direct-remote":
        with arguments.listing.expanduser().resolve().open(encoding="utf-8") as file:
            listing = json.load(file)
        if not isinstance(listing, list):
            raise ValueError("rclone lsjson 文件应为 JSON 列表")
        report = verify_direct_remote_listing(
            _load_manifest(arguments.manifest.expanduser().resolve()),
            listing,
        )
        _emit(report, arguments.output)
    elif arguments.command == "check-full-copy-capacity":
        report = check_full_copy_capacity(
            _load_manifest(arguments.manifest.expanduser().resolve()),
            arguments.path.expanduser().resolve(),
            reserve_bytes=math.ceil(arguments.reserve_gib * 2**30),
            headroom_ratio=arguments.headroom_ratio,
        )
        _emit(report, arguments.output)
    else:
        raise ValueError(f"未知命令：{arguments.command}")


if __name__ == "__main__":
    main()
