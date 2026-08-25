"""为正式物化窗口导出可恢复身份的股票级预测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.eventstream.materialized import (
    MaterializedWindowDataset,
    assert_materialized_compatible,
    build_source_datasets,
    load_materialized_manifest,
    verify_materialized_dataset,
)
from ticknet.eventstream.model import CONFIGS, build_eventstream_model
from ticknet.eventstream.storage_readiness import validate_storage_manifest
from ticknet.eventstream.train import EventstreamConfig
from ticknet.train import resolve_device

PREDICTION_MODE = "eventstream_materialized_day_predictions"
KEY_MODE = "eventstream_materialized_sample_keys"
SCHEMA_VERSION = 1
EVALUATION_PARTITIONS = ("validation", "oos")


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


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return value


def _validate_partitions(partitions: tuple[str, ...], *, allow_oos: bool) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(partitions))
    if not selected or any(partition not in EVALUATION_PARTITIONS for partition in selected):
        raise ValueError(f"预测分区应来自 {EVALUATION_PARTITIONS}")
    if "oos" in selected and not allow_oos:
        raise ValueError("导出 OOS 预测需要显式传入 allow_oos=True")
    return selected


def _record_file(record: dict[str, Any], name: str) -> str:
    matches = [row for row in record["files"] if Path(str(row["path"])).stem == name]
    if len(matches) != 1:
        raise ValueError(f"物化分片缺少唯一 {name}.npy：{record['partition']} {record['month']}")
    return str(matches[0]["path"])


def _source_partition_rows(
    *,
    partition: str,
    source_dataset: Any,
    materialized_root: Path,
    materialized_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    records = [
        record for record in materialized_manifest["shards"] if record["partition"] == partition
    ]
    rows: list[dict[str, Any]] = []
    output_index = 0
    for record in records:
        source_indices = [
            index
            for index, (day, _ticker_index, _start) in enumerate(source_dataset.entries)
            if str(day)[:6] == str(record["month"])
        ]
        if len(source_indices) != int(record["samples"]):
            raise ValueError(f"{partition}-{record['month']} 的源样本数与物化记录不同")
        days = np.load(
            materialized_root / _record_file(record, "day"),
            mmap_mode="r",
            allow_pickle=False,
        )
        targets = np.load(
            materialized_root / _record_file(record, "tgt_day"),
            mmap_mode="r",
            allow_pickle=False,
        )
        for local_index, source_index in enumerate(source_indices):
            day, ticker_index, _start = source_dataset.entries[source_index]
            ticker = str(source_dataset.index[day]["tickers"][ticker_index])
            target = float(source_dataset.index[day]["label"][ticker_index])
            if int(days[local_index]) != int(day):
                raise ValueError(f"{partition} 第 {output_index} 行交易日与物化缓存不同")
            if not math.isfinite(target) or not math.isclose(
                float(targets[local_index]), target, rel_tol=1e-6, abs_tol=1e-7
            ):
                raise ValueError(f"{partition} 第 {output_index} 行标签与物化缓存不同")
            rows.append(
                {
                    "partition": partition,
                    "row_index": output_index,
                    "trading_day": int(day),
                    "symbol": ticker,
                    "training_target_return": target,
                }
            )
            output_index += 1
    if output_index != int(materialized_manifest["totals"]["partitions"][partition]):
        raise ValueError(f"{partition} 的身份行数与物化清单不同")
    return rows


def export_materialized_sample_keys(
    *,
    config: EventstreamConfig,
    storage_manifest_path: Path,
    materialized_root: Path,
    output_dir: Path,
    partitions: tuple[str, ...] = EVALUATION_PARTITIONS,
    allow_oos: bool = False,
) -> dict[str, Any]:
    """从正式 pack 索引重建物化评估行的股票代码，并逐行核对日期与标签。"""
    selected = _validate_partitions(partitions, allow_oos=allow_oos)
    config.validate()
    storage = _load_json(storage_manifest_path)
    validate_storage_manifest(storage)
    materialized_root = materialized_root.expanduser().resolve()
    manifest = load_materialized_manifest(materialized_root)
    assert_materialized_compatible(manifest, config)
    if manifest["contract"]["source_inventory_sha256"] != storage["inventory_sha256"]:
        raise ValueError("存储清单与物化缓存的源库存指纹不同")
    if file_sha256(config.label_path) != manifest["contract"]["label_sha256"]:
        raise ValueError("H5 标签与物化缓存合同不同")

    datasets = build_source_datasets(config, storage["contract"]["splits"])
    rows = [
        row
        for partition in selected
        for row in _source_partition_rows(
            partition=partition,
            source_dataset=datasets[partition],
            materialized_root=materialized_root,
            materialized_manifest=manifest,
        )
    ]
    table = pa.Table.from_pylist(rows)
    root = output_dir.expanduser().resolve()
    path = root / "sample-keys.parquet"
    _atomic_parquet(table, path)
    contract = {
        "materialized_dataset_fingerprint": manifest["dataset_fingerprint"],
        "source_inventory_sha256": storage["inventory_sha256"],
        "source_dataset_fingerprint": manifest["contract"]["source_dataset_fingerprint"],
        "label_sha256": manifest["contract"]["label_sha256"],
        "partitions": list(selected),
        "locked_start": manifest["contract"]["locked_start"],
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": KEY_MODE,
        "status": "complete",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "artifact": {
            "path": path.name,
            "rows": table.num_rows,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows_by_partition": {
                partition: sum(row["partition"] == partition for row in rows)
                for partition in selected
            },
        },
    }
    report["dataset_fingerprint"] = _canonical_sha256(
        {"contract": contract, "artifact": report["artifact"]}
    )
    _atomic_json(root / "manifest.json", report)
    return report


def _checkpoint_experiment(checkpoint: dict[str, Any]) -> dict[str, Any]:
    experiment = checkpoint.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("checkpoint 缺少实验签名")
    return experiment


def _representation_contract(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "use_lob_prefix": bool(experiment.get("use_lob_prefix", False)),
        "use_session_anchors": bool(experiment.get("use_session_anchors", False)),
        "use_vq": bool(experiment.get("use_vq", False)),
        "vq_codebook_size": int(experiment.get("vq_codebook_size", 1024)),
        "vq_dim": int(experiment.get("vq_dim", 64)),
    }


def _validate_checkpoint_representation(
    experiment: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    representation = _representation_contract(experiment)
    contract = manifest["contract"]
    for name in ("use_lob_prefix", "use_session_anchors"):
        if representation[name] != bool(contract.get(name, False)):
            raise ValueError(f"checkpoint 与物化缓存的 {name} 不一致")
    return representation


@torch.no_grad()
def _score_partition(
    *,
    model: torch.nn.Module,
    dataset: MaterializedWindowDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pa.Table:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    row_indices: list[int] = []
    trading_days: list[int] = []
    scores: list[float] = []
    target_returns: list[float] = []
    output_index = 0
    use_amp = device.type == "cuda"
    model.eval()
    for batch in loader:
        x, sid, oid, _, _, _, tgt_day, day_valid, valid, day = (
            value.to(device, non_blocking=True) for value in batch
        )
        if not bool(torch.all(day_valid > 0).item()):
            raise ValueError("物化评估分区包含缺少日级标签的样本")
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            output = model(x, sid, oid)
        last = valid.sum(-1).clamp(min=1).long() - 1
        batch_scores = output["day"].float().gather(1, last[:, None]).squeeze(1)
        rows = int(batch_scores.shape[0])
        row_indices.extend(range(output_index, output_index + rows))
        trading_days.extend(int(value) for value in day.cpu().numpy())
        scores.extend(float(value) for value in batch_scores.cpu().numpy())
        target_returns.extend(float(value) for value in tgt_day.cpu().numpy())
        output_index += rows
    if output_index != len(dataset):
        raise RuntimeError(f"预测行数与物化分区不同：{output_index} != {len(dataset)}")
    return pa.table(
        {
            "row_index": pa.array(row_indices, type=pa.int64()),
            "trading_day": pa.array(trading_days, type=pa.int32()),
            "score": pa.array(scores, type=pa.float64()),
            "training_target_return": pa.array(target_returns, type=pa.float64()),
        }
    )


def export_materialized_predictions(
    *,
    checkpoint_path: Path,
    materialized_root: Path,
    model_name: str,
    output_dir: Path,
    device_name: str = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    partitions: tuple[str, ...] = EVALUATION_PARTITIONS,
    allow_oos: bool = False,
    source_revision: str = "",
) -> dict[str, Any]:
    """只运行 day 头推理，导出可与股票身份侧车逐行连接的分数。"""
    started_at = time.perf_counter()
    selected = _validate_partitions(partitions, allow_oos=allow_oos)
    if model_name not in CONFIGS:
        raise ValueError(f"model 应为 {sorted(CONFIGS)} 之一")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size 应为正整数，num_workers 不能为负数")
    if not source_revision or len(source_revision) < 7:
        raise ValueError("source_revision 应为至少 7 位标识")

    materialized_root = materialized_root.expanduser().resolve()
    manifest = load_materialized_manifest(materialized_root)
    verify_materialized_dataset(materialized_root, partitions=selected)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    device = resolve_device(device_name)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint 缺少模型权重")
    experiment = _checkpoint_experiment(checkpoint)
    if experiment.get("dataset_fingerprint") != manifest["dataset_fingerprint"]:
        raise ValueError("checkpoint 与物化缓存的数据指纹不同")
    if experiment.get("model") != model_name:
        raise ValueError("checkpoint 与请求的模型配置不同")
    representation = _validate_checkpoint_representation(experiment, manifest)

    model = build_eventstream_model(
        model_name,
        use_vq=representation["use_vq"],
        vq_codebook_size=representation["vq_codebook_size"],
        vq_dim=representation["vq_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    root = output_dir.expanduser().resolve()
    artifacts: list[dict[str, Any]] = []
    for partition in selected:
        table = _score_partition(
            model=model,
            dataset=MaterializedWindowDataset(materialized_root, partition),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        path = root / f"{partition}.parquet"
        _atomic_parquet(table, path)
        artifacts.append(
            {
                "partition": partition,
                "path": path.name,
                "rows": table.num_rows,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    del model

    contract = {
        "source_revision": source_revision,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_training_revision": experiment.get("source_revision", ""),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "model": model_name,
        "materialized_dataset_fingerprint": manifest["dataset_fingerprint"],
        "source_dataset_fingerprint": manifest["contract"]["source_dataset_fingerprint"],
        "partitions": list(selected),
        "locked_start": manifest["contract"]["locked_start"],
        "pooling": "last_valid_day_head",
        "representation": representation,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": PREDICTION_MODE,
        "status": "complete",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "artifacts": artifacts,
        "totals": {
            "rows": sum(int(row["rows"]) for row in artifacts),
            "bytes": sum(int(row["bytes"]) for row in artifacts),
        },
        "duration_seconds": time.perf_counter() - started_at,
    }
    report["dataset_fingerprint"] = _canonical_sha256(
        {"contract": contract, "artifacts": artifacts}
    )
    _atomic_json(root / "manifest.json", report)
    return report


def _load_config(path: Path) -> EventstreamConfig:
    with path.expanduser().resolve().open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("事件流配置应为 YAML 对象")
    return EventstreamConfig.from_mapping(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出物化事件流窗口的股票身份与 day 头预测")
    commands = parser.add_subparsers(dest="command", required=True)

    keys = commands.add_parser("keys", help="从正式 pack 索引恢复物化行的股票代码")
    keys.add_argument("--config", type=Path, required=True)
    keys.add_argument("--storage-manifest", type=Path, required=True)
    keys.add_argument("--materialized-root", type=Path, required=True)
    keys.add_argument("--output", type=Path, required=True)
    keys.add_argument("--partition", action="append", choices=EVALUATION_PARTITIONS)
    keys.add_argument("--allow-oos", action="store_true")

    score = commands.add_parser("score", help="从 checkpoint 批量导出物化评估分数")
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--materialized-root", type=Path, required=True)
    score.add_argument("--model", default="capacity100m", choices=tuple(CONFIGS))
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--num-workers", type=int, default=0)
    score.add_argument("--partition", action="append", choices=EVALUATION_PARTITIONS)
    score.add_argument("--allow-oos", action="store_true")
    score.add_argument("--source-revision", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    partitions = tuple(arguments.partition or EVALUATION_PARTITIONS)
    if arguments.command == "keys":
        report = export_materialized_sample_keys(
            config=_load_config(arguments.config),
            storage_manifest_path=arguments.storage_manifest,
            materialized_root=arguments.materialized_root,
            output_dir=arguments.output,
            partitions=partitions,
            allow_oos=arguments.allow_oos,
        )
    elif arguments.command == "score":
        report = export_materialized_predictions(
            checkpoint_path=arguments.checkpoint,
            materialized_root=arguments.materialized_root,
            model_name=arguments.model,
            output_dir=arguments.output,
            device_name=arguments.device,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            partitions=partitions,
            allow_oos=arguments.allow_oos,
            source_revision=arguments.source_revision,
        )
    else:
        raise ValueError(f"未知命令：{arguments.command}")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
