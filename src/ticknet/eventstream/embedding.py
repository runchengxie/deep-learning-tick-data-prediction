"""用冻结事件流 checkpoint 导出按股票日对齐的尾盘 embedding。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from ticknet.eventstream.close_cache import (
    PARTITIONS,
    load_close_cache_manifest,
    verify_close_cache,
)
from ticknet.eventstream.fingerprint import file_sha256, git_sha
from ticknet.eventstream.materialized import load_materialized_manifest
from ticknet.eventstream.model import CONFIGS, L2FoundationModel, build_eventstream_model
from ticknet.train import resolve_device

SCHEMA_VERSION = 1
MODE = "eventstream_frozen_daily_embeddings"
MANIFEST_NAME = "manifest.json"


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return value


def _checkpoint_contract(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    *,
    model_name: str,
    training_manifest: dict[str, Any],
    close_manifest: dict[str, Any],
) -> dict[str, Any]:
    experiment = checkpoint.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("事件流 checkpoint 缺少实验签名")
    if experiment.get("model") != model_name:
        raise ValueError("事件流 checkpoint 模型名称与导出配置不一致")
    if experiment.get("dataset_fingerprint") != training_manifest.get("dataset_fingerprint"):
        raise ValueError("事件流 checkpoint 与固定窗口训练集指纹不一致")
    training_contract = training_manifest["contract"]
    close_contract = close_manifest["contract"]
    expected = {
        "seq_len": close_contract["seq_len"],
        "min_events": close_contract["min_events"],
        "ranges": close_contract["ranges"],
    }
    actual = {
        "seq_len": experiment.get("seq_len"),
        "min_events": experiment.get("min_events"),
        "ranges": {
            "train": {
                "start": experiment.get("train_start"),
                "end": experiment.get("train_end"),
            },
            "validation": {
                "start": experiment.get("val_start"),
                "end": experiment.get("val_end"),
            },
            "oos": {
                "start": experiment.get("test_start"),
                "end": experiment.get("test_end"),
            },
        },
    }
    if actual != expected:
        raise ValueError("事件流 checkpoint 的窗口或日期合同与尾盘缓存不一致")
    if training_contract.get("source_dataset_fingerprint") != close_contract.get(
        "source_dataset_fingerprint"
    ):
        raise ValueError("训练缓存与尾盘缓存的源数据指纹不一致")
    seed = int(experiment.get("seed", -1))
    if seed < 0:
        raise ValueError("事件流 checkpoint 缺少有效 seed")
    source_revision = str(experiment.get("source_revision", ""))
    if not source_revision or source_revision == "unknown":
        raise ValueError("事件流 checkpoint 缺少有效源码 revision")
    return {
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_source_revision": source_revision,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_selection_value": float(checkpoint.get("best_selection_value", math.nan)),
        "training_dataset_fingerprint": training_manifest["dataset_fingerprint"],
        "source_dataset_fingerprint": close_contract["source_dataset_fingerprint"],
        "model": model_name,
        "seed": seed,
        "seq_len": int(close_contract["seq_len"]),
        "min_events": int(close_contract["min_events"]),
        "embedding_dimension": int(CONFIGS[model_name].d_model),
    }


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest["contract"],
        "artifacts": manifest["artifacts"],
        "totals": manifest["totals"],
    }


def validate_embedding_manifest(manifest: dict[str, Any]) -> None:
    """验证冻结 embedding 合同、分片和数据指纹。"""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != MODE:
        raise ValueError("冻结 embedding manifest 格式或版本无效")
    if manifest.get("status") != "complete":
        raise ValueError("冻结 embedding 尚未完整导出")
    contract = manifest.get("contract")
    artifacts = manifest.get("artifacts")
    totals = manifest.get("totals")
    if (
        not isinstance(contract, dict)
        or not isinstance(artifacts, list)
        or not isinstance(totals, dict)
    ):
        raise ValueError("冻结 embedding manifest 缺少合同、分片或汇总")
    if contract.get("pooling") != "last_valid_hidden" or contract.get("anchor") != "market_close":
        raise ValueError("冻结 embedding 池化或锚点合同无效")
    if manifest.get("contract_sha256") != _canonical_sha256(contract):
        raise ValueError("冻结 embedding 合同指纹不一致")
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        partition = str(artifact.get("partition", ""))
        month = str(artifact.get("month", ""))
        if partition not in PARTITIONS or (partition, month) in seen:
            raise ValueError("冻结 embedding 分片范围无效或重复")
        seen.add((partition, month))
        if int(artifact.get("rows", 0)) < 1 or int(artifact.get("bytes", -1)) < 0:
            raise ValueError("冻结 embedding 分片行数或大小无效")
        sha256 = artifact.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError("冻结 embedding 分片缺少 SHA-256")
    expected = {
        "artifacts": len(artifacts),
        "rows": sum(int(row["rows"]) for row in artifacts),
        "bytes": sum(int(row["bytes"]) for row in artifacts),
        "partitions": {
            name: sum(int(row["rows"]) for row in artifacts if row["partition"] == name)
            for name in PARTITIONS
        },
    }
    if totals != expected or any(expected["partitions"][name] < 1 for name in PARTITIONS):
        raise ValueError("冻结 embedding 汇总不一致或缺少分区")
    if manifest.get("dataset_fingerprint") != _canonical_sha256(_manifest_payload(manifest)):
        raise ValueError("冻结 embedding 数据指纹不一致")


def load_embedding_manifest(root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    root = Path(root)
    manifest = _load_json(root / MANIFEST_NAME)
    validate_embedding_manifest(manifest)
    if verify_files:
        dimension = int(manifest["contract"]["encoder"]["embedding_dimension"])
        seen_keys: set[tuple[int, str]] = set()
        for artifact in manifest["artifacts"]:
            path = root / str(artifact["path"])
            if not path.is_file() or path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"冻结 embedding 分片缺失或大小漂移：{artifact['path']}")
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"冻结 embedding 分片内容漂移：{artifact['path']}")
            table = pq.read_table(path, columns=["trading_day", "symbol", "embedding"])
            field_type = table.schema.field("embedding").type
            if (
                table.num_rows != int(artifact["rows"])
                or not pa.types.is_fixed_size_list(field_type)
                or field_type.list_size != dimension
            ):
                raise ValueError(f"冻结 embedding 分片 schema 漂移：{artifact['path']}")
            for day, symbol in zip(
                table["trading_day"].to_pylist(),
                table["symbol"].to_pylist(),
                strict=True,
            ):
                key = (int(day), str(symbol))
                if key in seen_keys:
                    raise ValueError(f"冻结 embedding 股票日重复：{key}")
                seen_keys.add(key)
    return manifest


class _CloseShardDataset(Dataset):
    def __init__(self, root: Path, record: dict[str, Any]):
        files = {Path(item["path"]).stem: root / str(item["path"]) for item in record["files"]}
        self.x = np.load(files["x"], mmap_mode="r", allow_pickle=False)
        self.sid = np.load(files["sid"], mmap_mode="r", allow_pickle=False)
        self.oid = np.load(files["oid"], mmap_mode="r", allow_pickle=False)
        self.day = np.load(files["day"], mmap_mode="r", allow_pickle=False)
        self.symbol = np.load(files["symbol"], mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(np.array(self.x[index], copy=True)),
            torch.from_numpy(np.array(self.sid[index], dtype=np.int64, copy=True)),
            torch.from_numpy(np.array(self.oid[index], dtype=np.int64, copy=True)),
            torch.tensor(int(self.day[index]), dtype=torch.int32),
            torch.from_numpy(np.frombuffer(bytes(self.symbol[index]), dtype=np.uint8).copy()),
        )


def _fixed_embedding_array(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


@torch.no_grad()
def _encode_shard(
    model: L2FoundationModel,
    dataset: _CloseShardDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pa.Table:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    days: list[int] = []
    symbols: list[str] = []
    embeddings: list[np.ndarray] = []
    model.eval()
    for x, sid, oid, day, symbol in loader:
        x = x.to(device)
        sid = sid.to(device)
        oid = oid.to(device)
        valid_lengths = (sid != 0).sum(dim=1)
        if torch.any(valid_lengths < 1):
            raise ValueError("尾盘窗口包含没有有效事件的样本")
        hidden = model.backbone(x, sid, oid)
        rows = torch.arange(hidden.shape[0], device=device)
        selected = hidden[rows, valid_lengths - 1]
        embeddings.append(selected.float().cpu().numpy())
        days.extend(int(value) for value in day.tolist())
        symbols.extend(bytes(row.tolist()).decode("ascii") for row in symbol)
    matrix = np.concatenate(embeddings).astype(np.float32, copy=False)
    if matrix.shape[0] != len(dataset) or not np.all(np.isfinite(matrix)):
        raise ValueError("冻结 embedding 行数或有限值检查失败")
    return pa.table(
        {
            "trading_day": pa.array(days, type=pa.int32()),
            "symbol": pa.array(symbols, type=pa.string()),
            "embedding": _fixed_embedding_array(matrix),
        }
    )


def export_frozen_embeddings(
    *,
    close_cache_root: Path,
    checkpoint_path: Path,
    training_manifest_root: Path,
    model_name: str,
    output_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    allow_oos: bool,
    source_revision: str,
) -> dict[str, Any]:
    """从共享尾盘窗口导出一个 checkpoint 的冻结表示。"""
    if not allow_oos:
        raise ValueError("导出完整冻结 embedding 必须显式批准读取 OOS 分区")
    close_cache_root = Path(close_cache_root)
    verify_close_cache(close_cache_root, partitions=PARTITIONS)
    close_manifest = load_close_cache_manifest(close_cache_root)
    training_manifest = load_materialized_manifest(training_manifest_root)
    checkpoint_path = Path(checkpoint_path)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("事件流 checkpoint 顶层应为对象")
    encoder = _checkpoint_contract(
        checkpoint_path,
        raw_checkpoint,
        model_name=model_name,
        training_manifest=training_manifest,
        close_manifest=close_manifest,
    )
    if not source_revision or source_revision == "unknown":
        raise ValueError("冻结 embedding 导出需要有效的源码 revision")
    contract = {
        "source_revision": source_revision,
        "close_cache_fingerprint": close_manifest["dataset_fingerprint"],
        "close_cache_contract_sha256": close_manifest["contract_sha256"],
        "anchor": "market_close",
        "input_window": "last_seq_len_events",
        "pooling": "last_valid_hidden",
        "dtype": "float32",
        "partitions": list(PARTITIONS),
        "encoder": encoder,
    }
    output_root = Path(output_root)
    manifest_path = output_root / MANIFEST_NAME
    if manifest_path.exists():
        existing = load_embedding_manifest(output_root)
        if existing["contract"] != contract:
            raise ValueError("已有冻结 embedding 与本次合同不一致")
        return existing
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("冻结 embedding 输出目录非空且缺少 manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    model = build_eventstream_model(model_name).to(resolved_device)
    model.load_state_dict(raw_checkpoint["model"])
    del raw_checkpoint
    artifacts: list[dict[str, Any]] = []
    for record in close_manifest["shards"]:
        partition = str(record["partition"])
        month = str(record["month"])
        dataset = _CloseShardDataset(close_cache_root, record)
        table = _encode_shard(
            model,
            dataset,
            device=resolved_device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        relative = Path("shards") / f"{partition}-{month}.parquet"
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        artifacts.append(
            {
                "partition": partition,
                "month": month,
                "path": relative.as_posix(),
                "rows": table.num_rows,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        print(f"[embedding] {partition}-{month}: {table.num_rows}")
    totals = {
        "artifacts": len(artifacts),
        "rows": sum(int(row["rows"]) for row in artifacts),
        "bytes": sum(int(row["bytes"]) for row in artifacts),
        "partitions": {
            name: sum(int(row["rows"]) for row in artifacts if row["partition"] == name)
            for name in PARTITIONS
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "complete",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "artifacts": artifacts,
        "totals": totals,
    }
    manifest["dataset_fingerprint"] = _canonical_sha256(_manifest_payload(manifest))
    validate_embedding_manifest(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="导出冻结事件流尾盘 embedding")
    parser.add_argument("--close-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest-root", type=Path, required=True)
    parser.add_argument("--model", default="capacity100m", choices=sorted(CONFIGS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--allow-oos", action="store_true")
    parser.add_argument("--source-revision", default="")
    arguments = parser.parse_args(argv)
    report = export_frozen_embeddings(
        close_cache_root=arguments.close_cache.expanduser().resolve(),
        checkpoint_path=arguments.checkpoint.expanduser().resolve(),
        training_manifest_root=arguments.training_manifest_root.expanduser().resolve(),
        model_name=arguments.model,
        output_root=arguments.output.expanduser().resolve(),
        device=arguments.device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        allow_oos=arguments.allow_oos,
        source_revision=arguments.source_revision or git_sha(Path.cwd()),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
