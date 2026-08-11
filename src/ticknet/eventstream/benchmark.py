"""在真实 eventstream pack 上测量训练吞吐、显存和整轮耗时。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ticknet.eventstream.dataset import L2WindowDataset
from ticknet.eventstream.fingerprint import dataset_fingerprint
from ticknet.eventstream.model import build_eventstream_model, compute_loss
from ticknet.eventstream.train import EventstreamConfig, list_packed_days
from ticknet.train import set_seed


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _train_days(config: EventstreamConfig, root: Path) -> list[int]:
    return (
        sorted({int(day) for day in config.days})
        if config.days
        else list_packed_days(config.train_start, config.train_end, root)
    )


def _step_batch(
    batch: tuple[torch.Tensor, ...],
    *,
    model: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    gradient_accumulation_steps: int,
    use_amp: bool,
) -> tuple[int, float]:
    x, sid, oid, tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid, _day = (
        tensor.to(device, non_blocking=True) for tensor in batch
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        output = model(x, sid, oid)
        loss, _metrics = compute_loss(
            output,
            tgt_sid,
            tgt_oid,
            tgt_reg,
            tgt_day,
            day_valid,
            valid,
        )
    scaler.scale(loss / gradient_accumulation_steps).backward()
    return int(x.shape[0]), float(loss.detach())


def run_benchmark(
    config: EventstreamConfig,
    *,
    batches: int,
    warmup_batches: int,
    output: Path,
    source_revision: str,
    requested_gpu: str,
    expected_parameter_count: int | None,
) -> dict[str, Any]:
    """执行真实前向、反向和 AdamW step，不访问 validation 或 OOS。"""
    config.validate()
    if batches < 1 or warmup_batches < 0:
        raise ValueError("batches 应为正整数，warmup_batches 不能为负数")
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("eventstream benchmark 要求可用 CUDA，且配置 device 必须为 cuda")

    set_seed(config.seed)
    device = torch.device("cuda")
    root = Path(config.pack_root)
    days = _train_days(config, root)
    label_path = Path(config.label_path) if config.label_path else None
    dataset = L2WindowDataset(
        days,
        seq_len=config.seq_len,
        min_events=config.min_events,
        samples_per_day=config.samples_per_day,
        root=root,
        label_path=label_path,
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    required_batches = warmup_batches + batches
    if len(loader) < required_batches:
        raise ValueError(f"数据只有 {len(loader)} 个 batch，少于要求的 {required_batches} 个")

    model = build_eventstream_model(config.model).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameter_count is not None and parameter_count != expected_parameter_count:
        raise ValueError(
            f"模型参数量 {parameter_count:,} 与预期 {expected_parameter_count:,} 不一致"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    use_amp = config.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for index in range(warmup_batches):
        _step_batch(
            next(iterator),
            model=model,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            use_amp=use_amp,
        )
        if (index + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    if warmup_batches % config.gradient_accumulation_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    samples = 0
    loss_total = 0.0
    optimizer_steps = 0
    for index in range(batches):
        batch_samples, loss = _step_batch(
            next(iterator),
            model=model,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            use_amp=use_amp,
        )
        samples += batch_samples
        loss_total += loss * batch_samples
        should_step = (index + 1) % config.gradient_accumulation_steps == 0 or (
            index + 1 == batches
        )
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    properties = torch.cuda.get_device_properties(device)
    throughput = samples / elapsed
    epoch_minutes = len(dataset) / throughput / 60
    fingerprint = dataset_fingerprint(days, root=root, label_path=label_path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "eventstream_capacity_benchmark",
        "source_revision": source_revision,
        "evaluation_status": "validation_and_oos_not_accessed",
        "requested_gpu": requested_gpu,
        "actual_gpu": properties.name,
        "model": {
            "name": config.model,
            "parameter_count": parameter_count,
            "parameter_count_millions": parameter_count / 1_000_000,
            "expected_parameter_count": expected_parameter_count,
        },
        "data": {
            "pack_root": str(root),
            "label_path": str(label_path) if label_path else None,
            "dataset_fingerprint": fingerprint,
            "train_days": len(days),
            "train_samples": len(dataset),
            "seq_len": config.seq_len,
            "features": 80,
        },
        "benchmark": {
            "warmup_batches": warmup_batches,
            "measured_batches": batches,
            "measured_samples": samples,
            "physical_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "optimizer_steps": optimizer_steps,
            "duration_seconds": elapsed,
            "samples_per_second": throughput,
            "mean_loss": loss_total / samples,
            "dataset_epoch_minutes": epoch_minutes,
            "configured_epochs": config.epochs,
            "dataset_seed_hours": epoch_minutes * config.epochs / 60,
        },
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "gpu_total_gib": properties.total_memory / 2**30,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": config.to_dict(),
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="eventstream 训练容量 benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--requested-gpu", default="unspecified")
    parser.add_argument("--expected-parameter-count", type=int)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as file:
        config = EventstreamConfig.from_mapping(dict(yaml.safe_load(file)))
    run_benchmark(
        config,
        batches=args.batches,
        warmup_batches=args.warmup_batches,
        output=args.output.expanduser().resolve(),
        source_revision=args.source_revision,
        requested_gpu=args.requested_gpu,
        expected_parameter_count=args.expected_parameter_count,
    )


if __name__ == "__main__":
    main()
