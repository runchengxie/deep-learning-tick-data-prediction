"""在真实分片上测量次日模型训练吞吐、显存和整轮耗时。"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ticknet.nextday.config import NextDayConfig
from ticknet.nextday.dataset import NextDayShardDataset
from ticknet.nextday.model import build_nextday_model
from ticknet.nextday.train import _class_weights, load_config
from ticknet.train import set_seed


def count_parameters(model: nn.Module) -> int:
    """返回模型全部可训练与不可训练参数的元素数。"""
    return sum(parameter.numel() for parameter in model.parameters())


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _build_model(config: NextDayConfig, dataset: NextDayShardDataset) -> nn.Module:
    return build_nextday_model(
        chunks_per_sample=dataset.chunks_per_sample,
        chunk_size=dataset.chunk_size,
        conv_channels=config.conv_channels,
        inception_channels=config.inception_channels,
        intraday_embedding_size=config.intraday_embedding_size,
        day_hidden_size=config.day_hidden_size,
        day_layers=config.day_layers,
        dropout=config.dropout,
    )


def _step_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    model: nn.Module,
    device: torch.device,
    classification_criterion: nn.Module,
    regression_criterion: nn.Module,
    target_mean: float,
    target_std: float,
    config: NextDayConfig,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    features, labels, target_returns = batch
    features = features.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    target_returns = target_returns.to(device, non_blocking=True)
    normalized_targets = (target_returns - target_mean) / target_std
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.amp):
        output = model(features)
        classification_loss = classification_criterion(output.logits, labels)
        regression_loss = regression_criterion(output.score, normalized_targets)
        loss = (
            config.classification_loss_weight * classification_loss
            + config.regression_loss_weight * regression_loss
        )
    scaler.scale(loss / config.gradient_accumulation_steps).backward()
    return int(features.shape[0]), float(loss.detach())


def run_benchmark(
    config: NextDayConfig,
    *,
    batches: int,
    warmup_batches: int,
    output: Path,
    source_revision: str,
    requested_gpu: str,
    expected_parameter_count: int | None,
    projected_train_samples: int | None,
) -> dict[str, Any]:
    """运行有限批次的真实训练 benchmark，并写出可审计 JSON。"""
    config.validate()
    if batches < 1 or warmup_batches < 0:
        raise ValueError("batches 应为正整数，warmup_batches 不能为负数")
    if projected_train_samples is not None and projected_train_samples < 1:
        raise ValueError("projected_train_samples 应为正整数")
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capacity benchmark 要求可用 CUDA，且配置 device 必须为 cuda")
    if config.evaluate_test:
        raise ValueError("capacity benchmark 禁止访问 locked test")
    if config.manifest_path is None:
        raise ValueError("manifest_path 不能为空")

    set_seed(config.seed)
    device = torch.device("cuda")
    dataset = NextDayShardDataset(
        config.manifest_path,
        date_split=config.date_split(),
        split="train",
        verify_checksums=config.verify_data_checksums,
        target_sidecar_path=config.target_sidecar_path,
        target_horizon=config.target_horizon,
        input_last_chunks=config.input_last_chunks,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    if len(loader) < warmup_batches + batches:
        raise ValueError(
            f"数据只有 {len(loader)} 个 batch，少于 warmup + benchmark 的 "
            f"{warmup_batches + batches} 个"
        )

    model = _build_model(config, dataset).to(device)
    parameter_count = count_parameters(model)
    if expected_parameter_count is not None and parameter_count != expected_parameter_count:
        raise ValueError(
            f"模型参数量 {parameter_count:,} 与预期 {expected_parameter_count:,} 不一致"
        )
    classification_criterion = nn.CrossEntropyLoss(
        weight=_class_weights(dataset, config.class_weighting, device)
    )
    regression_criterion = nn.SmoothL1Loss(beta=0.5)
    target_mean = float(np.mean(dataset.target_returns))
    target_std = float(np.std(dataset.target_returns))
    if not math.isfinite(target_std) or target_std <= 1e-12:
        raise ValueError("训练集目标收益没有有效方差")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for index in range(warmup_batches):
        _step_batch(
            next(iterator),
            model=model,
            device=device,
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            target_mean=target_mean,
            target_std=target_std,
            config=config,
            scaler=scaler,
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
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            target_mean=target_mean,
            target_std=target_std,
            config=config,
            scaler=scaler,
        )
        samples += batch_samples
        loss_total += loss * batch_samples
        should_step = (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == batches
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    properties = torch.cuda.get_device_properties(device)
    samples_per_second = samples / elapsed
    epoch_minutes = len(dataset) / samples_per_second / 60
    projected_epoch_minutes = (
        None
        if projected_train_samples is None
        else projected_train_samples / samples_per_second / 60
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "capacity-benchmark",
        "source_revision": source_revision,
        "test_status": "locked_not_accessed",
        "requested_gpu": requested_gpu,
        "actual_gpu": properties.name,
        "model": {
            "parameter_count": parameter_count,
            "parameter_count_millions": parameter_count / 1_000_000,
            "expected_parameter_count": expected_parameter_count,
            "conv_channels": config.conv_channels,
            "inception_channels": config.inception_channels,
            "intraday_embedding_size": config.intraday_embedding_size,
            "day_hidden_size": config.day_hidden_size,
            "day_layers": config.day_layers,
        },
        "data": {
            "manifest_path": str(config.manifest_path),
            "dataset_fingerprint": dataset.dataset_fingerprint,
            "train_samples": len(dataset),
            "source_chunks_per_sample": dataset.source_chunks_per_sample,
            "input_last_chunks": dataset.input_last_chunks,
            "chunks_per_sample": dataset.chunks_per_sample,
            "chunk_size": dataset.chunk_size,
            "events_per_sample": dataset.chunks_per_sample * dataset.chunk_size,
            "storage_dtype": str(dataset.storage_dtype),
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
            "samples_per_second": samples_per_second,
            "batches_per_second": batches / elapsed,
            "mean_loss": loss_total / samples,
            "dataset_epoch_minutes": epoch_minutes,
            "configured_epochs": config.epochs,
            "dataset_seed_hours": epoch_minutes * config.epochs / 60,
            "projected_train_samples": projected_train_samples,
            "projected_epoch_minutes": projected_epoch_minutes,
            "projected_seed_hours": (
                None
                if projected_epoch_minutes is None
                else projected_epoch_minutes * config.epochs / 60
            ),
            "projected_three_seed_gpu_hours": (
                None
                if projected_epoch_minutes is None
                else projected_epoch_minutes * config.epochs * 3 / 60
            ),
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
        "config": asdict(config),
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--requested-gpu", default="unspecified")
    parser.add_argument("--expected-parameter-count", type=int)
    parser.add_argument("--projected-train-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments, training_arguments = _parser().parse_known_args(argv)
    run_benchmark(
        load_config(training_arguments),
        batches=arguments.batches,
        warmup_batches=arguments.warmup_batches,
        output=arguments.output.expanduser().resolve(),
        source_revision=arguments.source_revision,
        requested_gpu=arguments.requested_gpu,
        expected_parameter_count=arguments.expected_parameter_count,
        projected_train_samples=arguments.projected_train_samples,
    )


if __name__ == "__main__":
    main()
