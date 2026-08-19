"""事件流多任务对共享主干的确定性梯度审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.eventstream.materialized import (
    MaterializedWindowDataset,
    assert_materialized_compatible,
    load_materialized_manifest,
    verify_materialized_dataset,
)
from ticknet.eventstream.model import (
    LOSS_WEIGHTS,
    L2FoundationModel,
    build_eventstream_model,
    compute_loss_components,
)
from ticknet.eventstream.train import EventstreamConfig
from ticknet.train import resolve_device, set_seed

SCHEMA_VERSION = 1
MODE = "eventstream_gradient_audit"
EXPERIMENT_ID = "EVT-GRAD-AUDIT-001"
TASKS = ("stream", "otype", "reg", "day")
GENERATIVE_TASKS = ("stream", "otype", "reg")
DEFAULT_LOSS_SCALE = 1024.0
WEAK_DAY_RATIO_THRESHOLD = 0.1
CONFLICT_COSINE_THRESHOLD = -0.1
CONFLICT_NEGATIVE_FRACTION_THRESHOLD = 0.75


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("事件流 checkpoint 缺少模型权重")
    return checkpoint


def _fixed_indices(length: int, *, batch_size: int, batches: int) -> list[int]:
    if length < 1 or batch_size < 1 or batches < 1:
        raise ValueError("数据集、batch_size 和 batches 应为正数")
    samples = min(length, batch_size * batches)
    if samples == 1:
        return [0]
    return [index * (length - 1) // (samples - 1) for index in range(samples)]


def _tensor_fingerprint(
    batches: list[tuple[torch.Tensor, ...]],
    *,
    dataset_fingerprint: str,
    partition: str,
    indices: list[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(dataset_fingerprint.encode())
    digest.update(partition.encode())
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    for batch in batches:
        for tensor in batch:
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_fixed_batches(
    dataset: MaterializedWindowDataset,
    *,
    batch_size: int,
    batches: int,
) -> tuple[list[tuple[torch.Tensor, ...]], list[int], str]:
    indices = _fixed_indices(len(dataset), batch_size=batch_size, batches=batches)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    fixed_batches = [tuple(tensor for tensor in batch) for batch in loader]
    fingerprint = _tensor_fingerprint(
        fixed_batches,
        dataset_fingerprint=str(dataset.manifest["dataset_fingerprint"]),
        partition=dataset.partition,
        indices=indices,
    )
    return fixed_batches, indices, fingerprint


def _backbone_parameters(model: L2FoundationModel) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("head_") and parameter.requires_grad
    )
    if not parameters:
        raise ValueError("事件流模型没有可审计的共享主干参数")
    return parameters


def _gradient_dot(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> float:
    total: torch.Tensor | None = None
    for first, second in zip(left, right, strict=True):
        if first is None or second is None:
            continue
        value = (first.float() * second.float()).sum()
        total = value if total is None else total + value
    return 0.0 if total is None else float(total.detach().cpu())


def _audit_batch(
    model: L2FoundationModel,
    batch: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    amp: bool,
    loss_scale: float,
    batch_index: int,
) -> dict[str, Any]:
    values = tuple(tensor.to(device, non_blocking=True) for tensor in batch)
    x, sid, oid, tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid, day = values
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        out = model(x, sid, oid)
        components = compute_loss_components(
            out,
            tgt_sid,
            tgt_oid,
            tgt_reg,
            tgt_day,
            day_valid,
            valid,
        )

    parameters = _backbone_parameters(model)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for task_index, task in enumerate(TASKS):
        weighted = components[task] * LOSS_WEIGHTS[task] * loss_scale
        gradients[task] = torch.autograd.grad(
            weighted,
            parameters,
            retain_graph=task_index < len(TASKS) - 1,
            allow_unused=True,
        )

    norms = {
        task: math.sqrt(max(_gradient_dot(gradients[task], gradients[task]), 0.0)) / loss_scale
        for task in TASKS
    }
    norm_total = sum(norms.values())
    generative_median = float(np.median([norms[task] for task in GENERATIVE_TASKS]))
    task_rows = {
        task: {
            "loss": float(components[task].detach().float().cpu()),
            "weight": LOSS_WEIGHTS[task],
            "weighted_loss": float((components[task] * LOSS_WEIGHTS[task]).detach().float().cpu()),
            "gradient_norm": norms[task],
            "gradient_norm_fraction": norms[task] / norm_total if norm_total > 0 else None,
        }
        for task in TASKS
    }
    cosines: dict[str, float | None] = {}
    for left, right in combinations(TASKS, 2):
        denominator = norms[left] * norms[right] * loss_scale * loss_scale
        key = f"{left}__{right}"
        cosines[key] = (
            _gradient_dot(gradients[left], gradients[right]) / denominator
            if denominator > 0
            else None
        )
    days = sorted({int(value) for value in day.detach().cpu().tolist()})
    result = {
        "batch_index": batch_index,
        "samples": int(x.shape[0]),
        "days": days,
        "valid_positions": int(valid.sum().detach().cpu()),
        "day_valid_samples": int(day_valid.sum().detach().cpu()),
        "tasks": task_rows,
        "day_to_generative_median_gradient_norm": (
            norms["day"] / generative_median if generative_median > 0 else None
        ),
        "cosines": cosines,
    }
    del gradients, components, out, values
    return result


def _distribution(values: list[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "negative_fraction": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(finite.max()),
        "negative_fraction": float((finite < 0).mean()),
    }


def _summarize_batches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_summary = {
        task: {
            field: _distribution([row["tasks"][task][field] for row in rows])
            for field in ("loss", "weighted_loss", "gradient_norm", "gradient_norm_fraction")
        }
        for task in TASKS
    }
    cosine_keys = sorted(rows[0]["cosines"])
    return {
        "batches": len(rows),
        "samples": sum(int(row["samples"]) for row in rows),
        "tasks": task_summary,
        "day_to_generative_median_gradient_norm": _distribution(
            [row["day_to_generative_median_gradient_norm"] for row in rows]
        ),
        "cosines": {
            key: _distribution([row["cosines"][key] for row in rows]) for key in cosine_keys
        },
    }


def _audit_state(
    model: L2FoundationModel,
    batches: list[tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    loss_scale = DEFAULT_LOSS_SCALE if amp else 1.0
    rows = [
        _audit_batch(
            model,
            batch,
            device=device,
            amp=amp,
            loss_scale=loss_scale,
            batch_index=index,
        )
        for index, batch in enumerate(batches)
    ]
    return {"loss_scale": loss_scale, "summary": _summarize_batches(rows), "batches": rows}


def run_gradient_audit(
    config: EventstreamConfig,
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    partition: str,
    batches: int,
    batch_size: int,
    source_revision: str,
    expected_parameter_count: int | None = None,
) -> dict[str, Any]:
    """在同一组固定 batch 上比较初始化和最佳 checkpoint 的任务梯度。"""
    if partition not in {"train", "validation"}:
        raise ValueError("梯度审计只允许读取 train 或 validation 分区")
    if len(expected_checkpoint_sha256) != 64:
        raise ValueError("必须提供 64 位 checkpoint SHA-256")
    if not source_revision or source_revision == "unknown":
        raise ValueError("梯度审计需要有效的源码 revision")
    config.validate()
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"
    root = Path(config.materialized_root)
    manifest = load_materialized_manifest(root)
    assert_materialized_compatible(manifest, config)
    preflight = verify_materialized_dataset(root, partitions=(partition,))
    dataset = MaterializedWindowDataset(root, partition)
    fixed_batches, indices, batch_fingerprint = _load_fixed_batches(
        dataset,
        batch_size=batch_size,
        batches=batches,
    )

    actual_checkpoint_sha256 = file_sha256(checkpoint_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "事件流 checkpoint SHA-256 不匹配："
            f"{actual_checkpoint_sha256} != {expected_checkpoint_sha256}"
        )

    set_seed(config.seed)
    initial_model = build_eventstream_model(config.model).to(device)
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    backbone_parameter_count = sum(
        parameter.numel() for parameter in _backbone_parameters(initial_model)
    )
    if expected_parameter_count is not None and parameter_count != expected_parameter_count:
        raise ValueError(f"事件流参数量不匹配：{parameter_count} != {expected_parameter_count}")
    initial = _audit_state(initial_model, fixed_batches, device=device, amp=use_amp)
    del initial_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_experiment = checkpoint.get("experiment")
    if not isinstance(checkpoint_experiment, dict):
        raise ValueError("事件流 checkpoint 缺少实验身份")
    expected_identity = {
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "model": config.model,
        "seed": config.seed,
    }
    actual_identity = {key: checkpoint_experiment.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise ValueError(
            f"事件流 checkpoint 实验身份不匹配：{actual_identity} != {expected_identity}"
        )
    trained_model = build_eventstream_model(config.model)
    trained_model.load_state_dict(checkpoint["model"], strict=True)
    trained_model = trained_model.to(device)
    trained = _audit_state(trained_model, fixed_batches, device=device, amp=use_amp)
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "source_revision": source_revision,
        "locked_status": "2026_not_accessed",
        "partition": partition,
        "config": {
            "model": config.model,
            "seed": config.seed,
            "seq_len": config.seq_len,
            "batch_size": batch_size,
            "requested_batches": batches,
            "amp": use_amp,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
        "model_parameter_count": parameter_count,
        "backbone_parameter_count": backbone_parameter_count,
        "materialized": {
            "root": str(root),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "preflight": preflight,
            "sample_indices": indices,
            "batch_fingerprint": batch_fingerprint,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": actual_checkpoint_sha256,
            "epoch": checkpoint.get("epoch"),
            "experiment_source_revision": (checkpoint_experiment.get("source_revision")),
            "experiment_identity": actual_identity,
        },
        "loss_weights": LOSS_WEIGHTS,
        "states": {"initialization": initial, "best_checkpoint": trained},
    }
    result["result_fingerprint"] = _canonical_sha256(result)
    return result


def _load_audit(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if (
        not isinstance(value, dict)
        or value.get("mode") != MODE
        or value.get("status") != "complete"
    ):
        raise ValueError(f"无效的梯度审计结果：{path}")
    expected = value.get("result_fingerprint")
    payload = {key: item for key, item in value.items() if key != "result_fingerprint"}
    if expected != _canonical_sha256(payload):
        raise ValueError(f"梯度审计结果指纹不匹配：{path}")
    return value


def decide_gradient_audits(paths: list[Path]) -> dict[str, Any]:
    """按预注册门槛汇总至少两个滚动折的最佳 checkpoint 梯度。"""
    if len(paths) < 2:
        raise ValueError("梯度决策至少需要两个滚动折")
    reports = [_load_audit(path) for path in paths]
    ratios: list[float] = []
    conflicts_by_report: list[set[str]] = []
    evidence = []
    for path, report in zip(paths, reports, strict=True):
        summary = report["states"]["best_checkpoint"]["summary"]
        ratio = summary["day_to_generative_median_gradient_norm"]["median"]
        if ratio is None:
            raise ValueError(f"梯度审计缺少 day 比例：{path}")
        ratios.append(float(ratio))
        conflicts = {
            pair
            for pair, distribution in summary["cosines"].items()
            if "day" in pair.split("__")
            and distribution["median"] is not None
            and distribution["median"] <= CONFLICT_COSINE_THRESHOLD
            and distribution["negative_fraction"] >= CONFLICT_NEGATIVE_FRACTION_THRESHOLD
        }
        conflicts_by_report.append(conflicts)
        evidence.append(
            {
                "path": str(path),
                "result_fingerprint": report["result_fingerprint"],
                "dataset_fingerprint": report["materialized"]["dataset_fingerprint"],
                "day_to_generative_median_gradient_norm": float(ratio),
                "persistent_negative_pairs": sorted(conflicts),
            }
        )

    weak_day = all(ratio <= WEAK_DAY_RATIO_THRESHOLD for ratio in ratios)
    shared_conflicts = set.intersection(*conflicts_by_report)
    if weak_day:
        decision = "day_gradient_weak"
        next_experiment = "EVT-LABEL-SCALE-001"
    elif shared_conflicts:
        decision = "persistent_task_conflict"
        next_experiment = "EVT-SUPERVISION-POSITION-001_WITH_TASK_WEIGHT_REVIEW"
    else:
        decision = "gradient_strength_normal_without_persistent_conflict"
        next_experiment = "EVT-SUPERVISION-POSITION-001"

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "eventstream_gradient_audit_decision",
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "thresholds": {
            "weak_day_ratio": WEAK_DAY_RATIO_THRESHOLD,
            "conflict_cosine": CONFLICT_COSINE_THRESHOLD,
            "conflict_negative_fraction": CONFLICT_NEGATIVE_FRACTION_THRESHOLD,
        },
        "evidence": evidence,
        "decision": decision,
        "shared_persistent_negative_pairs": sorted(shared_conflicts),
        "next_experiment": next_experiment,
    }
    result["result_fingerprint"] = _canonical_sha256(result)
    return result


def _load_config(
    path: Path, *, materialized_root: Path, device: str, amp: bool
) -> EventstreamConfig:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("事件流配置应为 YAML 对象")
    raw["materialized_root"] = str(materialized_root)
    raw["device"] = device
    raw["amp"] = amp
    raw["num_workers"] = 0
    raw["evaluate_test"] = False
    return EventstreamConfig.from_mapping(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="核对事件流多任务对共享主干的梯度强度与方向")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行单个滚动折的梯度审计")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--materialized-root", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--expected-checkpoint-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--partition", choices=("train", "validation"), default="validation")
    run.add_argument("--batches", type=int, default=16)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--expected-parameter-count", type=int)
    decide = commands.add_parser("decide", help="按门槛汇总多个滚动折")
    decide.add_argument("--audit", type=Path, action="append", required=True)
    decide.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        if arguments.batches < 8 or arguments.batches > 16:
            raise ValueError("正式梯度审计的 --batches 应在 8 至 16 之间")
        config = _load_config(
            arguments.config,
            materialized_root=arguments.materialized_root,
            device=arguments.device,
            amp=arguments.amp,
        )
        result = run_gradient_audit(
            config,
            checkpoint_path=arguments.checkpoint,
            expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            partition=arguments.partition,
            batches=arguments.batches,
            batch_size=arguments.batch_size,
            source_revision=arguments.source_revision,
            expected_parameter_count=arguments.expected_parameter_count,
        )
    else:
        result = decide_gradient_audits(arguments.audit)
    _atomic_json(arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
