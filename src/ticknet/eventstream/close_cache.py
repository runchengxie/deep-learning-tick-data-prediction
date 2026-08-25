"""物化三组事件流 checkpoint 共用的尾盘窗口缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Subset

from ticknet.eventstream.dataset import N_FEATURES, L2WindowDataset
from ticknet.eventstream.fingerprint import file_sha256, git_sha
from ticknet.eventstream.storage_readiness import validate_storage_manifest

SCHEMA_VERSION = 1
MODE = "eventstream_daily_close_windows"
MANIFEST_NAME = "manifest.json"
PARTITIONS = ("train", "validation", "oos")
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "x": np.dtype(np.float32),
    "sid": np.dtype(np.int16),
    "oid": np.dtype(np.int16),
    "day": np.dtype(np.int32),
    "symbol": np.dtype("S6"),
}


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


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"缓存路径必须是安全的相对路径：{value}")
    return path.as_posix()


def _array_tail_shapes(seq_len: int) -> dict[str, tuple[int, ...]]:
    return {
        "x": (seq_len, N_FEATURES),
        "sid": (seq_len,),
        "oid": (seq_len,),
        "day": (),
        "symbol": (),
    }


def _array_contract(seq_len: int) -> dict[str, dict[str, Any]]:
    tails = _array_tail_shapes(seq_len)
    return {
        name: {"dtype": dtype.str, "tail_shape": list(tails[name])}
        for name, dtype in ARRAY_DTYPES.items()
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return value


def _contract(
    storage: dict[str, Any],
    *,
    pack_root: Path,
    seq_len: int,
    min_events: int,
    source_revision: str,
    use_lob_prefix: bool = False,
    use_session_anchors: bool = False,
) -> dict[str, Any]:
    if not source_revision or source_revision == "unknown":
        raise ValueError("尾盘窗口物化需要有效的源码 revision")
    if seq_len < 1 or min_events < 2:
        raise ValueError("seq_len 和 min_events 必须为正数")
    if use_session_anchors and not use_lob_prefix:
        raise ValueError("use_session_anchors 需要同时启用 use_lob_prefix")
    ranges = storage["contract"]["ranges"]
    if int(storage["contract"]["locked_start"]) <= max(
        int(ranges[name]["end"]) for name in PARTITIONS
    ):
        raise ValueError("尾盘窗口日期不能进入 locked 区间")
    return {
        "source_inventory_sha256": storage["inventory_sha256"],
        "source_dataset_fingerprint": storage["contract"]["source_dataset_fingerprint"],
        "source_revision": source_revision,
        "pack_root": str(pack_root.expanduser().resolve()),
        "locked_start": storage["contract"]["locked_start"],
        "splits": {name: storage["contract"]["splits"][name] for name in PARTITIONS},
        "ranges": {name: ranges[name] for name in PARTITIONS},
        "seq_len": seq_len,
        "min_events": min_events,
        "use_lob_prefix": bool(use_lob_prefix),
        "use_session_anchors": bool(use_session_anchors),
        "selection_policy": "all_pack_tickers_min_events_v1",
        "anchor_policy": (
            "lob_prefix_last_events_before_market_close_v2"
            if use_lob_prefix
            else "last_seq_len_events_before_market_close_v1"
        ),
        "arrays": _array_contract(seq_len),
    }


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest["contract"],
        "shards": manifest["shards"],
        "totals": manifest["totals"],
    }


def _expected_totals(shards: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "shards": len(shards),
        "samples": sum(int(row["samples"]) for row in shards),
        "bytes": sum(int(item["bytes"]) for row in shards for item in row["files"]),
        "partitions": {
            partition: sum(int(row["samples"]) for row in shards if row["partition"] == partition)
            for partition in PARTITIONS
        },
    }


def _validate_shard_manifest_record(record: dict[str, Any]) -> tuple[str, str]:
    partition = str(record.get("partition", ""))
    month = str(record.get("month", ""))
    if partition not in PARTITIONS or len(month) != 6 or not month.isdigit():
        raise ValueError("尾盘窗口缓存分片范围无效")
    samples = int(record.get("samples", 0))
    days = record.get("days")
    if not isinstance(days, list):
        raise ValueError("尾盘窗口缓存分片日期无效")
    normalized_days = [int(day) for day in days]
    if samples < 1 or normalized_days != sorted(set(normalized_days)):
        raise ValueError("尾盘窗口缓存分片样本数或日期无效")
    files = record.get("files")
    if not isinstance(files, list) or len(files) != len(ARRAY_DTYPES):
        raise ValueError("尾盘窗口缓存分片文件集合不完整")
    names: set[str] = set()
    for item in files:
        path = _safe_relative_path(str(item.get("path", "")))
        name = Path(path).stem
        names.add(name)
        if name not in ARRAY_DTYPES or int(item.get("bytes", -1)) < 0:
            raise ValueError(f"尾盘窗口缓存文件记录无效：{path}")
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"尾盘窗口缓存文件缺少 SHA-256：{path}")
    if names != set(ARRAY_DTYPES):
        raise ValueError("尾盘窗口缓存张量文件不完整")
    return partition, month


def _validate_representation_contract(contract: dict[str, Any]) -> None:
    use_lob_prefix = bool(contract.get("use_lob_prefix", False))
    use_session_anchors = bool(contract.get("use_session_anchors", False))
    if use_session_anchors and not use_lob_prefix:
        raise ValueError("尾盘窗口缓存 session anchor 缺少 LOB prefix")
    expected_anchor = (
        "lob_prefix_last_events_before_market_close_v2"
        if use_lob_prefix
        else "last_seq_len_events_before_market_close_v1"
    )
    if contract.get("anchor_policy") != expected_anchor:
        raise ValueError("尾盘窗口缓存锚点合同无效")


def validate_close_cache_manifest(
    manifest: dict[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """验证尾盘窗口缓存的结构和逻辑指纹。"""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != MODE:
        raise ValueError("尾盘窗口缓存格式或版本无效")
    status = manifest.get("status")
    if status not in {"in_progress", "complete"}:
        raise ValueError("尾盘窗口缓存状态无效")
    if require_complete and status != "complete":
        raise ValueError("尾盘窗口缓存尚未完成")
    contract = manifest.get("contract")
    shards = manifest.get("shards")
    totals = manifest.get("totals")
    if (
        not isinstance(contract, dict)
        or not isinstance(shards, list)
        or not isinstance(totals, dict)
    ):
        raise ValueError("尾盘窗口缓存缺少合同、分片或汇总")
    seq_len = int(contract.get("seq_len", 0))
    if contract.get("arrays") != _array_contract(seq_len):
        raise ValueError("尾盘窗口缓存张量合同无效")
    if contract.get("selection_policy") != "all_pack_tickers_min_events_v1":
        raise ValueError("尾盘窗口缓存股票选择合同无效")
    _validate_representation_contract(contract)
    if manifest.get("contract_sha256") != _canonical_sha256(contract):
        raise ValueError("尾盘窗口缓存合同指纹不匹配")

    seen: set[tuple[str, str]] = set()
    for record in shards:
        if not isinstance(record, dict):
            raise ValueError("尾盘窗口缓存分片记录无效")
        partition, month = _validate_shard_manifest_record(record)
        if (partition, month) in seen:
            raise ValueError(f"尾盘窗口缓存分片重复：{partition}-{month}")
        seen.add((partition, month))

    expected = _expected_totals(shards)
    if totals != expected:
        raise ValueError("尾盘窗口缓存汇总不一致")
    if status == "complete":
        if any(expected["partitions"][name] < 1 for name in PARTITIONS):
            raise ValueError("完整尾盘窗口缓存必须覆盖三个分区")
        if manifest.get("dataset_fingerprint") != _canonical_sha256(_manifest_payload(manifest)):
            raise ValueError("尾盘窗口缓存数据指纹不匹配")


def load_close_cache_manifest(root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    manifest = _load_json(Path(root) / MANIFEST_NAME)
    validate_close_cache_manifest(manifest, require_complete=require_complete)
    return manifest


def _verify_shard(root: Path, record: dict[str, Any], contract: dict[str, Any]) -> int:
    samples = int(record["samples"])
    verified_bytes = 0
    keys: list[tuple[int, str]] = []
    for item in record["files"]:
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"尾盘窗口缓存文件缺失或大小漂移：{item['path']}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"尾盘窗口缓存文件内容漂移：{item['path']}")
        name = path.stem
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        tail = tuple(contract["arrays"][name]["tail_shape"])
        dtype = np.dtype(contract["arrays"][name]["dtype"])
        if array.shape != (samples, *tail) or array.dtype != dtype:
            raise ValueError(f"尾盘窗口缓存文件形状或类型漂移：{item['path']}")
        if name == "day":
            keys = [(int(value), "") for value in array]
        if name == "symbol":
            if not keys:
                keys = [(0, bytes(value).decode("ascii")) for value in array]
            else:
                keys = [
                    (day, bytes(symbol).decode("ascii"))
                    for (day, _), symbol in zip(keys, array, strict=True)
                ]
        verified_bytes += path.stat().st_size
    if len(keys) != samples or len(set(keys)) != samples:
        raise ValueError("尾盘窗口缓存股票日键缺失或重复")
    return verified_bytes


def verify_close_cache(
    root: Path,
    *,
    partitions: tuple[str, ...] = PARTITIONS,
) -> dict[str, Any]:
    """逐文件核对指定分区，并避免提前读取未授权分区。"""
    root = Path(root)
    manifest = load_close_cache_manifest(root)
    selected = tuple(dict.fromkeys(partitions))
    if not selected or any(name not in PARTITIONS for name in selected):
        raise ValueError(f"尾盘窗口核对分区应来自 {PARTITIONS}")
    records = [row for row in manifest["shards"] if row["partition"] in selected]
    verified_bytes = sum(_verify_shard(root, row, manifest["contract"]) for row in records)
    return {
        "status": "complete",
        "mode": "eventstream_daily_close_preflight",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "partitions": list(selected),
        "shards": len(records),
        "samples": sum(int(row["samples"]) for row in records),
        "bytes": verified_bytes,
    }


def _build_datasets(
    storage: dict[str, Any],
    *,
    pack_root: Path,
    seq_len: int,
    min_events: int,
    use_lob_prefix: bool = False,
    use_session_anchors: bool = False,
) -> dict[str, L2WindowDataset]:
    return {
        partition: L2WindowDataset(
            list(storage["contract"]["splits"][partition]),
            seq_len=seq_len,
            min_events=min_events,
            samples_per_day=1,
            root=pack_root,
            label_path=None,
            seed=0,
            eval_mode=True,
            eval_tickers=0,
            fixed_windows=True,
            require_eval_labels=False,
            use_lob_prefix=use_lob_prefix,
            use_session_anchors=use_session_anchors,
        )
        for partition in PARTITIONS
    }


def _write_shard(
    dataset: L2WindowDataset,
    indices: list[int],
    *,
    root: Path,
    partition: str,
    month: str,
    seq_len: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    relative = Path("shards") / f"{partition}-{month}"
    final_dir = root / relative
    temporary = root / "shards" / f".{partition}-{month}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if final_dir.exists():
        return _record_shard(
            dataset,
            indices,
            root=root,
            relative=relative,
            partition=partition,
            month=month,
            seq_len=seq_len,
        )
    temporary.mkdir(parents=True)
    tails = _array_tail_shapes(seq_len)
    arrays = {
        name: np.lib.format.open_memmap(
            temporary / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(len(indices), *tails[name]),
        )
        for name, dtype in ARRAY_DTYPES.items()
    }
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    output_index = 0
    for batch in loader:
        x, sid, oid = batch[:3]
        rows = int(x.shape[0])
        batch_indices = indices[output_index : output_index + rows]
        keys = [dataset.sample_key(index) for index in batch_indices]
        arrays["x"][output_index : output_index + rows] = x.numpy().astype(np.float32, copy=False)
        arrays["sid"][output_index : output_index + rows] = sid.numpy().astype(np.int16, copy=False)
        arrays["oid"][output_index : output_index + rows] = oid.numpy().astype(np.int16, copy=False)
        arrays["day"][output_index : output_index + rows] = np.asarray(
            [day for day, _symbol in keys], dtype=np.int32
        )
        arrays["symbol"][output_index : output_index + rows] = np.asarray(
            [symbol.encode("ascii") for _day, symbol in keys], dtype="S6"
        )
        output_index += rows
        if output_index % 1000 < rows:
            print(f"[close-cache] {partition}-{month}: {output_index}/{len(indices)}")
    if output_index != len(indices):
        raise RuntimeError("尾盘窗口缓存写入样本数不一致")
    for array in arrays.values():
        array.flush()
    del arrays
    os.replace(temporary, final_dir)
    return _record_shard(
        dataset,
        indices,
        root=root,
        relative=relative,
        partition=partition,
        month=month,
        seq_len=seq_len,
    )


def _record_shard(
    dataset: L2WindowDataset,
    indices: list[int],
    *,
    root: Path,
    relative: Path,
    partition: str,
    month: str,
    seq_len: int,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    tails = _array_tail_shapes(seq_len)
    for name, dtype in ARRAY_DTYPES.items():
        path = root / relative / f"{name}.npy"
        if not path.is_file():
            raise ValueError(f"尾盘窗口缓存分片不完整：{path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(indices), *tails[name]) or array.dtype != dtype:
            raise ValueError(f"尾盘窗口缓存分片形状或类型异常：{path}")
        files.append(
            {
                "path": _safe_relative_path(path.relative_to(root).as_posix()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    days = sorted({dataset.sample_key(index)[0] for index in indices})
    return {
        "partition": partition,
        "month": month,
        "days": days,
        "samples": len(indices),
        "files": files,
    }


def _empty_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "in_progress",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "shards": [],
        "totals": _expected_totals([]),
    }


def build_close_cache(
    *,
    storage_manifest_path: Path,
    pack_root: Path,
    output_root: Path,
    seq_len: int,
    min_events: int,
    batch_size: int,
    num_workers: int,
    source_revision: str,
    use_lob_prefix: bool = False,
    use_session_anchors: bool = False,
) -> dict[str, Any]:
    """物化 seed 无关的尾盘窗口，支持按月原子恢复。"""
    storage = _load_json(storage_manifest_path)
    validate_storage_manifest(storage)
    contract = _contract(
        storage,
        pack_root=pack_root,
        seq_len=seq_len,
        min_events=min_events,
        source_revision=source_revision,
        use_lob_prefix=use_lob_prefix,
        use_session_anchors=use_session_anchors,
    )
    output_root = Path(output_root)
    manifest_path = output_root / MANIFEST_NAME
    if manifest_path.exists():
        manifest = load_close_cache_manifest(output_root, require_complete=False)
        if manifest["contract"] != contract:
            raise ValueError("已有尾盘窗口缓存与本次合同不一致")
        for record in manifest["shards"]:
            _verify_shard(output_root, record, contract)
        if manifest["status"] == "complete":
            return manifest
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("尾盘窗口缓存目录非空且缺少 manifest.json")
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = _empty_manifest(contract)
        _atomic_json(manifest_path, manifest)

    datasets = _build_datasets(
        storage,
        pack_root=pack_root,
        seq_len=seq_len,
        min_events=min_events,
        use_lob_prefix=use_lob_prefix,
        use_session_anchors=use_session_anchors,
    )
    per_sample = sum(
        int(np.prod(shape, dtype=np.int64) if shape else 1) * ARRAY_DTYPES[name].itemsize
        for name, shape in _array_tail_shapes(seq_len).items()
    )
    estimated = sum(len(dataset) for dataset in datasets.values()) * per_sample
    remaining = max(0, estimated - int(manifest["totals"]["bytes"]))
    required_free = math.ceil(remaining * 1.05) + 2**30
    if shutil.disk_usage(output_root).free < required_free:
        raise RuntimeError(f"尾盘窗口缓存磁盘空间不足，需要至少 {required_free} 字节")

    completed = {(row["partition"], row["month"]) for row in manifest["shards"]}
    for partition in PARTITIONS:
        dataset = datasets[partition]
        by_month: dict[str, list[int]] = defaultdict(list)
        for index, (day, _ticker_index, _start) in enumerate(dataset.entries):
            by_month[str(day)[:6]].append(index)
        for month, indices in sorted(by_month.items()):
            if (partition, month) in completed:
                continue
            manifest["shards"].append(
                _write_shard(
                    dataset,
                    indices,
                    root=output_root,
                    partition=partition,
                    month=month,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            )
            manifest["totals"] = _expected_totals(manifest["shards"])
            _atomic_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["totals"] = _expected_totals(manifest["shards"])
    manifest["dataset_fingerprint"] = _canonical_sha256(_manifest_payload(manifest))
    validate_close_cache_manifest(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="物化并核对事件流尾盘窗口缓存")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="生成三组 checkpoint 共用的尾盘窗口")
    build.add_argument("--storage-manifest", type=Path, required=True)
    build.add_argument("--pack-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seq-len", type=int, default=512)
    build.add_argument("--min-events", type=int, default=256)
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--num-workers", type=int, default=4)
    build.add_argument("--source-revision", default="")
    build.add_argument(
        "--use-lob-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    build.add_argument(
        "--use-session-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    verify = commands.add_parser("verify", help="逐文件核对尾盘窗口缓存")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument(
        "--partition",
        action="append",
        choices=PARTITIONS,
        dest="partitions",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        report = build_close_cache(
            storage_manifest_path=arguments.storage_manifest.expanduser().resolve(),
            pack_root=arguments.pack_root.expanduser().resolve(),
            output_root=arguments.output.expanduser().resolve(),
            seq_len=arguments.seq_len,
            min_events=arguments.min_events,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            source_revision=arguments.source_revision or git_sha(Path.cwd()),
            use_lob_prefix=arguments.use_lob_prefix,
            use_session_anchors=arguments.use_session_anchors,
        )
    else:
        report = verify_close_cache(
            arguments.root.expanduser().resolve(),
            partitions=tuple(arguments.partitions or PARTITIONS),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
