"""把固定事件窗口物化为可校验、可恢复的远端训练集。"""

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
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from ticknet.eventstream.dataset import N_FEATURES, L2WindowDataset
from ticknet.eventstream.fingerprint import file_sha256, git_sha
from ticknet.eventstream.storage_readiness import validate_storage_manifest

SCHEMA_VERSION = 1
MODE = "eventstream_materialized_windows"
MANIFEST_NAME = "manifest.json"
PARTITIONS = ("train", "validation", "oos", "monitor_validation", "monitor_oos")
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "x": np.dtype(np.float32),
    "sid": np.dtype(np.int16),
    "oid": np.dtype(np.int16),
    "tgt_sid": np.dtype(np.int16),
    "tgt_oid": np.dtype(np.int16),
    "tgt_reg": np.dtype(np.float32),
    "tgt_day": np.dtype(np.float32),
    "day_valid": np.dtype(np.float32),
    "valid": np.dtype(np.float32),
    "day": np.dtype(np.int64),
}


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
        raise ValueError(f"物化路径必须是安全的相对路径：{value}")
    return path.as_posix()


def _array_tail_shapes(seq_len: int) -> dict[str, tuple[int, ...]]:
    return {
        "x": (seq_len, N_FEATURES),
        "sid": (seq_len,),
        "oid": (seq_len,),
        "tgt_sid": (seq_len,),
        "tgt_oid": (seq_len,),
        "tgt_reg": (seq_len, 3),
        "tgt_day": (),
        "day_valid": (),
        "valid": (seq_len,),
        "day": (),
    }


def _array_contract(seq_len: int) -> dict[str, dict[str, Any]]:
    tails = _array_tail_shapes(seq_len)
    return {
        name: {"dtype": dtype.name, "tail_shape": list(tails[name])}
        for name, dtype in ARRAY_DTYPES.items()
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return value


def _storage_artifact_hash(storage: dict[str, Any], path: str) -> str:
    matches = [row for row in storage["artifacts"] if row.get("path") == path]
    if len(matches) != 1:
        raise ValueError(f"存储清单缺少唯一附属产物：{path}")
    return str(matches[0]["hashes"]["sha256"])


def _materialization_contract(
    config: Any,
    storage: dict[str, Any],
    *,
    source_revision: str,
) -> dict[str, Any]:
    if not source_revision or source_revision == "unknown":
        raise ValueError("物化需要有效的源码 revision")
    splits = storage["contract"]["splits"]
    ranges = storage["contract"]["ranges"]
    expected_ranges = {
        "train": {"start": config.train_start, "end": config.train_end},
        "validation": {"start": config.val_start, "end": config.val_end},
        "oos": {"start": config.test_start, "end": config.test_end},
    }
    if ranges != expected_ranges:
        raise ValueError("事件流配置的日期区间与正式存储清单不一致")
    label_sha256 = file_sha256(config.label_path)
    monitor_label_sha256 = file_sha256(config.monitor_label_path)
    if label_sha256 != _storage_artifact_hash(storage, "fold-labels/h5.parquet"):
        raise ValueError("H5 标签与正式存储清单不一致")
    if monitor_label_sha256 != _storage_artifact_hash(storage, "fold-labels/h3.parquet"):
        raise ValueError("H3 监控标签与正式存储清单不一致")
    use_lob_prefix = bool(getattr(config, "use_lob_prefix", False))
    use_session_anchors = bool(getattr(config, "use_session_anchors", False))
    return {
        "source_inventory_sha256": storage["inventory_sha256"],
        "source_revision": source_revision,
        "source_dataset_fingerprint": storage["contract"]["source_dataset_fingerprint"],
        "locked_start": storage["contract"]["locked_start"],
        "splits": splits,
        "ranges": ranges,
        "seed": config.seed,
        "seq_len": config.seq_len,
        "min_events": config.min_events,
        "samples_per_day": config.samples_per_day,
        "eval_tickers": config.eval_tickers,
        "label_sha256": label_sha256,
        "monitor_label_sha256": monitor_label_sha256,
        "monitor_name": config.monitor_name,
        "use_lob_prefix": use_lob_prefix,
        "use_session_anchors": use_session_anchors,
        "sampling_policy": (
            "seeded_fixed_window_v2" if use_lob_prefix else "seeded_fixed_window_v1"
        ),
        "arrays": _array_contract(config.seq_len),
    }


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest["contract"],
        "shards": manifest["shards"],
        "totals": manifest["totals"],
    }


def _validate_shard_record(record: dict[str, Any], contract: dict[str, Any]) -> None:
    partition = str(record.get("partition", ""))
    if partition not in PARTITIONS:
        raise ValueError(f"物化分片的 partition 无效：{partition}")
    samples = int(record.get("samples", -1))
    if samples < 1:
        raise ValueError("物化分片样本数应为正整数")
    days = record.get("days")
    if not isinstance(days, list) or days != sorted({int(day) for day in days}):
        raise ValueError("物化分片日期应唯一并按顺序排列")
    files = record.get("files")
    if not isinstance(files, list) or len(files) != len(ARRAY_DTYPES):
        raise ValueError("物化分片缺少张量文件")
    names: set[str] = set()
    for file_record in files:
        if not isinstance(file_record, dict):
            raise ValueError("物化分片文件记录无效")
        path = _safe_relative_path(str(file_record.get("path", "")))
        name = Path(path).stem
        names.add(name)
        if name not in contract["arrays"]:
            raise ValueError(f"物化分片包含未知张量：{name}")
        if int(file_record.get("bytes", -1)) < 0:
            raise ValueError(f"物化分片文件大小无效：{path}")
        sha256 = file_record.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"物化分片缺少 SHA-256：{path}")
    if names != set(ARRAY_DTYPES):
        raise ValueError("物化分片张量集合不完整")


def _validate_representation_contract(contract: dict[str, Any]) -> None:
    use_lob_prefix = bool(contract.get("use_lob_prefix", False))
    use_session_anchors = bool(contract.get("use_session_anchors", False))
    if use_session_anchors and not use_lob_prefix:
        raise ValueError("物化清单 session anchor 缺少 LOB prefix")
    expected_policy = "seeded_fixed_window_v2" if use_lob_prefix else "seeded_fixed_window_v1"
    if contract.get("sampling_policy") != expected_policy:
        raise ValueError("物化清单采样策略无效")


def validate_materialized_manifest(
    manifest: dict[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """验证物化清单结构、汇总和逻辑指纹。"""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != MODE:
        raise ValueError("物化清单格式或版本无效")
    status = manifest.get("status")
    if status not in {"in_progress", "complete"}:
        raise ValueError("物化清单状态无效")
    if require_complete and status != "complete":
        raise ValueError("物化清单尚未完成")
    contract = manifest.get("contract")
    shards = manifest.get("shards")
    totals = manifest.get("totals")
    if (
        not isinstance(contract, dict)
        or not isinstance(shards, list)
        or not isinstance(totals, dict)
    ):
        raise ValueError("物化清单缺少合同、分片或汇总")
    if contract.get("arrays") != _array_contract(int(contract.get("seq_len", 0))):
        raise ValueError("物化清单张量合同无效")
    _validate_representation_contract(contract)
    if manifest.get("contract_sha256") != _canonical_sha256(contract):
        raise ValueError("物化合同指纹不匹配")
    seen: set[tuple[str, str]] = set()
    for record in shards:
        if not isinstance(record, dict):
            raise ValueError("物化分片记录无效")
        _validate_shard_record(record, contract)
        key = (str(record["partition"]), str(record["month"]))
        if key in seen:
            raise ValueError(f"物化清单包含重复分片：{key}")
        seen.add(key)
    expected_totals = {
        "shards": len(shards),
        "samples": sum(int(record["samples"]) for record in shards),
        "bytes": sum(
            int(file_record["bytes"]) for record in shards for file_record in record["files"]
        ),
        "partitions": {
            partition: sum(
                int(record["samples"]) for record in shards if record["partition"] == partition
            )
            for partition in PARTITIONS
        },
    }
    if totals != expected_totals:
        raise ValueError("物化清单汇总不一致")
    if status == "complete":
        if any(expected_totals["partitions"][name] < 1 for name in PARTITIONS):
            raise ValueError("完整物化清单必须覆盖五个训练与评估分区")
        if manifest.get("dataset_fingerprint") != _canonical_sha256(_manifest_payload(manifest)):
            raise ValueError("物化数据指纹不匹配")


def load_materialized_manifest(root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    manifest = _load_json(Path(root) / MANIFEST_NAME)
    validate_materialized_manifest(manifest, require_complete=require_complete)
    return manifest


def _verify_shard(root: Path, record: dict[str, Any], contract: dict[str, Any]) -> int:
    samples = int(record["samples"])
    verified_bytes = 0
    for file_record in record["files"]:
        path = root / str(file_record["path"])
        if not path.is_file() or path.stat().st_size != int(file_record["bytes"]):
            raise ValueError(f"物化文件缺失或大小漂移：{file_record['path']}")
        if file_sha256(path) != file_record["sha256"]:
            raise ValueError(f"物化文件内容漂移：{file_record['path']}")
        name = path.stem
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_tail = tuple(contract["arrays"][name]["tail_shape"])
        expected_dtype = np.dtype(contract["arrays"][name]["dtype"])
        if array.shape != (samples, *expected_tail) or array.dtype != expected_dtype:
            raise ValueError(f"物化文件形状或类型漂移：{file_record['path']}")
        verified_bytes += path.stat().st_size
    return verified_bytes


def verify_materialized_dataset(
    root: Path,
    *,
    partitions: tuple[str, ...] = PARTITIONS,
) -> dict[str, Any]:
    """逐文件校验指定物化分区，不读取未授权的 OOS 分片。"""
    root = Path(root)
    manifest = load_materialized_manifest(root)
    selected = tuple(dict.fromkeys(partitions))
    if not selected or any(partition not in PARTITIONS for partition in selected):
        raise ValueError(f"物化核对分区应来自 {PARTITIONS}")
    records = [record for record in manifest["shards"] if record["partition"] in selected]
    verified_bytes = sum(_verify_shard(root, record, manifest["contract"]) for record in records)
    return {
        "status": "complete",
        "mode": "eventstream_materialized_preflight",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "partitions": list(selected),
        "shards": len(records),
        "samples": sum(int(record["samples"]) for record in records),
        "bytes": verified_bytes,
    }


def _as_numpy(sample: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray[Any, Any]]:
    return {
        name: tensor.detach().cpu().numpy().astype(ARRAY_DTYPES[name], copy=False)
        for name, tensor in zip(ARRAY_DTYPES, sample, strict=True)
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
    relative_dir = Path("shards") / f"{partition}-{month}"
    final_dir = root / relative_dir
    temporary = root / "shards" / f".{partition}-{month}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if final_dir.exists():
        return _record_existing_shard(
            dataset,
            indices,
            root=root,
            relative_dir=relative_dir,
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
    next_progress = 1000
    for batch in loader:
        values = _as_numpy(batch)
        rows = int(values["x"].shape[0])
        for name, array in arrays.items():
            array[output_index : output_index + rows] = values[name]
        output_index += rows
        while output_index >= next_progress:
            print(f"[materialize] {partition}-{month}: {next_progress}/{len(indices)}")
            next_progress += 1000
    if output_index != len(indices):
        raise RuntimeError(f"物化分片样本数不一致：{output_index} != {len(indices)}")
    for array in arrays.values():
        array.flush()
    del arrays
    os.replace(temporary, final_dir)
    return _record_existing_shard(
        dataset,
        indices,
        root=root,
        relative_dir=relative_dir,
        partition=partition,
        month=month,
        seq_len=seq_len,
    )


def _record_existing_shard(
    dataset: L2WindowDataset,
    indices: list[int],
    *,
    root: Path,
    relative_dir: Path,
    partition: str,
    month: str,
    seq_len: int,
) -> dict[str, Any]:
    """核对并登记已经原子落盘、尚未来得及写入清单的分片。"""
    final_dir = root / relative_dir
    tails = _array_tail_shapes(seq_len)
    files = []
    for name in ARRAY_DTYPES:
        path = final_dir / f"{name}.npy"
        if not path.is_file():
            raise ValueError(f"未登记的物化分片不完整：{path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(indices), *tails[name]) or array.dtype != ARRAY_DTYPES[name]:
            raise ValueError(f"未登记的物化分片形状或类型不一致：{path}")
        files.append(
            {
                "path": _safe_relative_path(path.relative_to(root).as_posix()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    days = sorted({int(dataset.entries[index][0]) for index in indices})
    return {
        "partition": partition,
        "month": month,
        "days": days,
        "samples": len(indices),
        "files": files,
    }


def build_source_datasets(
    config: Any,
    splits: dict[str, list[int]],
) -> dict[str, L2WindowDataset]:
    """按正式物化合同重建各分区的确定性源数据集。"""
    primary_label = Path(config.label_path)
    monitor_label = Path(config.monitor_label_path)

    def build(
        days: list[int],
        label_path: Path,
        *,
        eval_mode: bool,
        eval_tickers: int,
    ) -> L2WindowDataset:
        return L2WindowDataset(
            days,
            seq_len=int(config.seq_len),
            min_events=int(config.min_events),
            samples_per_day=int(config.samples_per_day),
            root=Path(config.pack_root),
            label_path=label_path,
            seed=int(config.seed),
            eval_mode=eval_mode,
            eval_tickers=eval_tickers,
            fixed_windows=True,
            use_lob_prefix=bool(getattr(config, "use_lob_prefix", False)),
            use_session_anchors=bool(getattr(config, "use_session_anchors", False)),
        )

    return {
        "train": build(
            splits["train"],
            primary_label,
            eval_mode=False,
            eval_tickers=0,
        ),
        "validation": build(
            splits["validation"],
            primary_label,
            eval_mode=True,
            eval_tickers=int(config.eval_tickers),
        ),
        "oos": build(
            splits["oos"],
            primary_label,
            eval_mode=True,
            eval_tickers=0,
        ),
        "monitor_validation": build(
            splits["validation"],
            monitor_label,
            eval_mode=True,
            eval_tickers=int(config.eval_tickers),
        ),
        "monitor_oos": build(
            splits["oos"],
            monitor_label,
            eval_mode=True,
            eval_tickers=0,
        ),
    }


def _empty_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "in_progress",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "shards": [],
        "totals": {
            "shards": 0,
            "samples": 0,
            "bytes": 0,
            "partitions": dict.fromkeys(PARTITIONS, 0),
        },
    }


def _refresh_totals(manifest: dict[str, Any]) -> None:
    shards = manifest["shards"]
    manifest["totals"] = {
        "shards": len(shards),
        "samples": sum(int(record["samples"]) for record in shards),
        "bytes": sum(
            int(file_record["bytes"]) for record in shards for file_record in record["files"]
        ),
        "partitions": {
            partition: sum(
                int(record["samples"]) for record in shards if record["partition"] == partition
            )
            for partition in PARTITIONS
        },
    }


def _required_bytes(datasets: dict[str, L2WindowDataset], seq_len: int) -> int:
    tails = _array_tail_shapes(seq_len)
    per_sample = sum(
        int(np.prod(tails[name], dtype=np.int64) if tails[name] else 1) * dtype.itemsize
        for name, dtype in ARRAY_DTYPES.items()
    )
    return sum(len(dataset) for dataset in datasets.values()) * per_sample


def build_materialized_dataset(
    config: Any,
    *,
    storage_manifest_path: Path,
    output_root: Path,
    source_revision: str,
) -> dict[str, Any]:
    """按正式日期合同物化五个训练与评估分区，支持按月断点续跑。"""
    config.validate()
    if not config.label_path or not config.monitor_label_path:
        raise ValueError("正式物化需要 H5 主标签和 H3 监控标签")
    storage = _load_json(storage_manifest_path)
    validate_storage_manifest(storage)
    contract = _materialization_contract(config, storage, source_revision=source_revision)
    output_root = Path(output_root)
    manifest_path = output_root / MANIFEST_NAME
    if manifest_path.exists():
        manifest = load_materialized_manifest(output_root, require_complete=False)
        if manifest["contract"] != contract:
            raise ValueError("已有物化目录的合同与本次运行不同")
        for record in manifest["shards"]:
            _verify_shard(output_root, record, contract)
        if manifest["status"] == "complete":
            return manifest
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("物化输出目录非空且缺少 manifest.json")
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = _empty_manifest(contract)
        _atomic_json(manifest_path, manifest)

    datasets = build_source_datasets(config, storage["contract"]["splits"])
    estimated_bytes = _required_bytes(datasets, config.seq_len)
    remaining_bytes = max(0, estimated_bytes - int(manifest["totals"]["bytes"]))
    required_free = math.ceil(remaining_bytes * 1.05) + 2**30
    available = shutil.disk_usage(output_root).free
    if available < required_free:
        raise RuntimeError(f"物化盘空间不足：需要 {required_free} 字节，当前可用 {available} 字节")

    completed = {(row["partition"], row["month"]) for row in manifest["shards"]}
    for partition in PARTITIONS:
        dataset = datasets[partition]
        by_month: dict[str, list[int]] = defaultdict(list)
        for index, (day, _ticker_index, _start) in enumerate(dataset.entries):
            by_month[str(day)[:6]].append(index)
        for month, indices in sorted(by_month.items()):
            if (partition, month) in completed:
                continue
            record = _write_shard(
                dataset,
                indices,
                root=output_root,
                partition=partition,
                month=month,
                seq_len=config.seq_len,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            manifest["shards"].append(record)
            _refresh_totals(manifest)
            _atomic_json(manifest_path, manifest)

    manifest["status"] = "complete"
    _refresh_totals(manifest)
    manifest["dataset_fingerprint"] = _canonical_sha256(_manifest_payload(manifest))
    validate_materialized_manifest(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def assert_materialized_compatible(manifest: dict[str, Any], config: Any) -> None:
    """拒绝与训练配置不一致的 seed、日期、窗口或表征合同。"""
    contract = manifest["contract"]
    expected = {
        "seed": config.seed,
        "seq_len": config.seq_len,
        "min_events": config.min_events,
        "samples_per_day": config.samples_per_day,
        "eval_tickers": config.eval_tickers,
        "monitor_name": config.monitor_name,
        "ranges": {
            "train": {"start": config.train_start, "end": config.train_end},
            "validation": {"start": config.val_start, "end": config.val_end},
            "oos": {"start": config.test_start, "end": config.test_end},
        },
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"物化训练集与配置的 {name} 不一致")
    representation = {
        "use_lob_prefix": bool(getattr(config, "use_lob_prefix", False)),
        "use_session_anchors": bool(getattr(config, "use_session_anchors", False)),
    }
    for name, value in representation.items():
        if bool(contract.get(name, False)) != value:
            raise ValueError(f"物化训练集与配置的 {name} 不一致")
    expected_source_revision = config.materialized_source_revision or config.source_revision
    if expected_source_revision and contract.get("source_revision") != expected_source_revision:
        raise ValueError("物化训练集与配置的源码 revision 不一致")


class MaterializedWindowDataset(Dataset):
    """按需 mmap 物化分片，返回与 ``L2WindowDataset`` 相同的张量合同。"""

    def __init__(
        self,
        root: Path,
        partition: str,
        *,
        target_overlay_root: Path | None = None,
    ):
        self.root = Path(root)
        self.manifest = load_materialized_manifest(self.root)
        if partition not in PARTITIONS:
            raise ValueError(f"未知物化分区：{partition}")
        self.partition = partition
        self.records = [
            record for record in self.manifest["shards"] if record["partition"] == partition
        ]
        if not self.records:
            raise ValueError(f"物化训练集缺少分区：{partition}")
        self.offsets = np.cumsum([0, *(int(record["samples"]) for record in self.records)])
        self._arrays: dict[int, dict[str, np.ndarray[Any, Any]]] = {}
        self.target_overlay_root = (
            Path(target_overlay_root).expanduser().resolve()
            if target_overlay_root is not None
            else None
        )
        self.target_overlay_fingerprint: str | None = None
        self.target_overlay_records: list[dict[str, Any]] = []
        self._target_overlay_arrays: dict[int, np.ndarray[Any, Any]] = {}
        if self.target_overlay_root is not None:
            from ticknet.eventstream.target_overlay import load_target_overlay_manifest

            overlay = load_target_overlay_manifest(
                self.target_overlay_root,
                expected_materialized_fingerprint=str(self.manifest["dataset_fingerprint"]),
                expected_partition=partition,
            )
            self.target_overlay_records = list(overlay["files"])
            expected_rows = [
                (str(record["month"]), int(record["samples"])) for record in self.records
            ]
            overlay_rows = [
                (str(record["month"]), int(record["samples"]))
                for record in self.target_overlay_records
            ]
            if overlay_rows != expected_rows:
                raise ValueError("日级标签覆盖层与物化训练分片不一致")
            self.target_overlay_fingerprint = str(overlay["dataset_fingerprint"])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _shard_arrays(self, shard_index: int) -> dict[str, np.ndarray[Any, Any]]:
        arrays = self._arrays.get(shard_index)
        if arrays is None:
            record = self.records[shard_index]
            files = {Path(row["path"]).stem: row["path"] for row in record["files"]}
            arrays = {
                name: np.load(self.root / files[name], mmap_mode="r", allow_pickle=False)
                for name in ARRAY_DTYPES
            }
            self._arrays[shard_index] = arrays
        return arrays

    def _target_overlay_array(self, shard_index: int) -> np.ndarray[Any, Any] | None:
        if self.target_overlay_root is None:
            return None
        values = self._target_overlay_arrays.get(shard_index)
        if values is None:
            record = self.target_overlay_records[shard_index]
            values = np.load(
                self.target_overlay_root / str(record["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            self._target_overlay_arrays[shard_index] = values
        return values

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        row = index - int(self.offsets[shard_index])
        values = self._shard_arrays(shard_index)
        target_overlay = self._target_overlay_array(shard_index)

        def copy(name: str, dtype: np.dtype[Any] | None = None) -> torch.Tensor:
            value = np.array(values[name][row], copy=True)
            if dtype is not None:
                value = value.astype(dtype, copy=False)
            return torch.from_numpy(value)

        return (
            copy("x"),
            copy("sid", np.dtype(np.int64)),
            copy("oid", np.dtype(np.int64)),
            copy("tgt_sid", np.dtype(np.int64)),
            copy("tgt_oid", np.dtype(np.int64)),
            copy("tgt_reg"),
            (
                torch.from_numpy(np.array(target_overlay[row], copy=True))
                if target_overlay is not None
                else copy("tgt_day")
            ),
            copy("day_valid"),
            copy("valid"),
            copy("day"),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="物化并核对事件流固定训练窗口")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="从正式 pack 生成按月物化训练集")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--storage-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-revision", default="")

    verify = commands.add_parser("verify", help="逐文件核对物化训练集")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    verify.add_argument(
        "--partition",
        action="append",
        choices=PARTITIONS,
        dest="partitions",
        help="只核对指定分区，可重复使用。默认核对全部分区",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        from ticknet.eventstream.train import EventstreamConfig

        with arguments.config.expanduser().resolve().open(encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("事件流配置应为 YAML 对象")
        config = EventstreamConfig.from_mapping(raw)
        report = build_materialized_dataset(
            config,
            storage_manifest_path=arguments.storage_manifest.expanduser().resolve(),
            output_root=arguments.output.expanduser().resolve(),
            source_revision=arguments.source_revision or git_sha(Path.cwd()),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    elif arguments.command == "verify":
        report = verify_materialized_dataset(
            arguments.root.expanduser().resolve(),
            partitions=tuple(arguments.partitions or PARTITIONS),
        )
        if arguments.output is not None:
            _atomic_json(arguments.output.expanduser().resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        raise ValueError(f"未知命令：{arguments.command}")


if __name__ == "__main__":
    main()
