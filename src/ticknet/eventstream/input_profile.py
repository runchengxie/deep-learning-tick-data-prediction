"""拆分 eventstream 输入流水线与 GPU 计算，并扫描 DataLoader worker 数。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ticknet.eventstream.benchmark import _atomic_json, _step_batch, _train_days, run_benchmark
from ticknet.eventstream.dataset import L2WindowDataset
from ticknet.eventstream.model import build_eventstream_model
from ticknet.eventstream.train import EventstreamConfig
from ticknet.train import set_seed


def build_worker_plan(worker_counts: list[int]) -> list[int]:
    """校验并保留用户指定的 worker 扫描顺序。"""
    if not worker_counts:
        raise ValueError("worker_counts 不能为空")
    if len(set(worker_counts)) != len(worker_counts):
        raise ValueError("worker_counts 不能重复")
    if any(workers < 0 for workers in worker_counts):
        raise ValueError("worker_counts 不能为负数")
    return list(worker_counts)


def classify_bottleneck(data_throughput: float, gpu_throughput: float) -> str:
    """按相差至少 25% 的保守门槛给出瓶颈分类。"""
    if data_throughput <= 0 or gpu_throughput <= 0:
        raise ValueError("吞吐必须为正数")
    if data_throughput < gpu_throughput * 0.75:
        return "input_pipeline"
    if gpu_throughput < data_throughput * 0.75:
        return "gpu_compute"
    return "mixed"


def select_best_workers(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """按真实端到端吞吐选择 worker 数。"""
    if not trials:
        raise ValueError("trials 不能为空")
    return max(
        trials,
        key=lambda trial: (
            float(trial["end_to_end_samples_per_second"]),
            -int(trial["num_workers"]),
        ),
    )


def _config_for_profile(
    config: EventstreamConfig,
    *,
    num_workers: int,
    effective_batch_size: int,
) -> EventstreamConfig:
    if effective_batch_size < 1 or effective_batch_size % config.batch_size:
        raise ValueError("effective_batch_size 必须是 physical batch 的正整数倍")
    mapping = config.to_dict()
    mapping["num_workers"] = num_workers
    mapping["gradient_accumulation_steps"] = effective_batch_size // config.batch_size
    return EventstreamConfig.from_mapping(mapping)


def _dataset(config: EventstreamConfig) -> L2WindowDataset:
    root = Path(config.pack_root)
    label_path = Path(config.label_path) if config.label_path else None
    return L2WindowDataset(
        _train_days(config, root),
        seq_len=config.seq_len,
        min_events=config.min_events,
        samples_per_day=config.samples_per_day,
        root=root,
        label_path=label_path,
        seed=config.seed,
    )


def _loader(config: EventstreamConfig, dataset: L2WindowDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )


def run_dataloader_only(
    config: EventstreamConfig,
    *,
    batches: int,
    warmup_batches: int,
) -> dict[str, Any]:
    """只计时 Dataset、collate 与 pin-memory，不访问 CUDA。"""
    dataset = _dataset(config)
    loader = _loader(config, dataset)
    required_batches = warmup_batches + batches
    if len(loader) < required_batches:
        raise ValueError(f"数据只有 {len(loader)} 个 batch，少于要求的 {required_batches} 个")
    iterator = iter(loader)
    for _ in range(warmup_batches):
        next(iterator)
    started = time.perf_counter()
    samples = 0
    for _ in range(batches):
        batch = next(iterator)
        samples += int(batch[0].shape[0])
    elapsed = time.perf_counter() - started
    del iterator, loader, dataset
    gc.collect()
    return {
        "status": "complete",
        "num_workers": config.num_workers,
        "physical_batch_size": config.batch_size,
        "warmup_batches": warmup_batches,
        "measured_batches": batches,
        "measured_samples": samples,
        "duration_seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


def run_gpu_only(
    config: EventstreamConfig,
    *,
    batches: int,
    warmup_batches: int,
    expected_parameter_count: int | None,
) -> dict[str, Any]:
    """预加载一个 CUDA batch，重复执行前向、反向与 AdamW。"""
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU-only profile 要求可用 CUDA")
    set_seed(config.seed)
    device = torch.device("cuda")
    dataset = _dataset(config)
    loader = _loader(config, dataset)
    batch = tuple(tensor.to(device) for tensor in next(iter(loader)))
    del loader, dataset

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
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def step(index: int, total: int) -> None:
        _step_batch(
            batch,
            model=model,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            use_amp=config.amp,
        )
        if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == total:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    for index in range(warmup_batches):
        step(index, warmup_batches)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index in range(batches):
        step(index, batches)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    samples = batches * int(batch[0].shape[0])
    properties = torch.cuda.get_device_properties(device)
    return {
        "status": "complete",
        "physical_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "warmup_batches": warmup_batches,
        "measured_batches": batches,
        "measured_samples": samples,
        "duration_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "actual_gpu": properties.name,
        "parameter_count": parameter_count,
    }


def run_input_profile(
    config: EventstreamConfig,
    *,
    worker_counts: list[int],
    effective_batch_size: int,
    batches: int,
    warmup_batches: int,
    output_dir: Path,
    source_revision: str,
    requested_gpu: str,
    expected_parameter_count: int | None,
    projected_train_samples: int,
) -> dict[str, Any]:
    """扫描 worker 数，并拆分 DataLoader、GPU-only 和端到端吞吐。"""
    if batches < 1 or warmup_batches < 0:
        raise ValueError("batches 应为正整数，warmup_batches 不能为负数")
    if projected_train_samples < 1:
        raise ValueError("projected_train_samples 应为正整数")
    workers = build_worker_plan(worker_counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_config = _config_for_profile(
        config,
        num_workers=workers[0],
        effective_batch_size=effective_batch_size,
    )
    gpu_only = run_gpu_only(
        reference_config,
        batches=batches,
        warmup_batches=warmup_batches,
        expected_parameter_count=expected_parameter_count,
    )
    _atomic_json(output_dir / "gpu-only.json", gpu_only)

    trials: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for count in workers:
        trial_config = _config_for_profile(
            config,
            num_workers=count,
            effective_batch_size=effective_batch_size,
        )
        data_only = run_dataloader_only(
            trial_config,
            batches=batches,
            warmup_batches=warmup_batches,
        )
        _atomic_json(output_dir / f"workers-{count:02d}-data-only.json", data_only)
        end_to_end = run_benchmark(
            trial_config,
            batches=batches,
            warmup_batches=warmup_batches,
            output=output_dir / f"workers-{count:02d}-end-to-end.json",
            source_revision=source_revision,
            requested_gpu=requested_gpu,
            expected_parameter_count=expected_parameter_count,
        )
        reference = reference or end_to_end
        throughput = float(end_to_end["benchmark"]["samples_per_second"])
        epoch_minutes = projected_train_samples / throughput / 60
        trials.append(
            {
                "status": "complete",
                "num_workers": count,
                "data_only_samples_per_second": data_only["samples_per_second"],
                "end_to_end_samples_per_second": throughput,
                "end_to_end_duration_seconds": end_to_end["benchmark"]["duration_seconds"],
                "projected_epoch_minutes": epoch_minutes,
                "projected_seed_hours": epoch_minutes * config.epochs / 60,
                "peak_allocated_gib": end_to_end["memory"]["peak_allocated_gib"],
                "peak_reserved_gib": end_to_end["memory"]["peak_reserved_gib"],
            }
        )
        gc.collect()
        torch.cuda.empty_cache()

    if reference is None:
        raise RuntimeError("worker sweep 没有生成结果")
    best = select_best_workers(trials)
    bottleneck = classify_bottleneck(
        float(best["data_only_samples_per_second"]),
        float(gpu_only["samples_per_second"]),
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "eventstream_input_profile",
        "source_revision": source_revision,
        "test_status": "validation_and_oos_not_accessed",
        "requested_gpu": requested_gpu,
        "actual_gpu": gpu_only["actual_gpu"],
        "model": reference["model"],
        "data": reference["data"],
        "benchmark": {
            "worker_counts": workers,
            "physical_batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "warmup_batches_per_measurement": warmup_batches,
            "measured_batches_per_measurement": batches,
            "projected_train_samples": projected_train_samples,
            "configured_epochs": config.epochs,
            "gpu_only_samples_per_second": gpu_only["samples_per_second"],
            "selected_num_workers": best["num_workers"],
            "selected_end_to_end_samples_per_second": best["end_to_end_samples_per_second"],
            "selected_projected_seed_hours": best["projected_seed_hours"],
            "bottleneck": bottleneck,
        },
        "trials": trials,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": config.to_dict(),
    }
    if not math.isfinite(float(best["end_to_end_samples_per_second"])):
        raise ValueError("最佳吞吐不是有限数值")
    _atomic_json(output_dir / "input-profile.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="eventstream 输入流水线与 GPU 拆分 benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", nargs="+", type=int, required=True)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--requested-gpu", default="unspecified")
    parser.add_argument("--expected-parameter-count", type=int)
    parser.add_argument("--projected-train-samples", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    with arguments.config.open(encoding="utf-8") as file:
        config = EventstreamConfig.from_mapping(dict(yaml.safe_load(file)))
    run_input_profile(
        config,
        worker_counts=arguments.num_workers,
        effective_batch_size=arguments.effective_batch_size,
        batches=arguments.batches,
        warmup_batches=arguments.warmup_batches,
        output_dir=arguments.output_dir.expanduser().resolve(),
        source_revision=arguments.source_revision,
        requested_gpu=arguments.requested_gpu,
        expected_parameter_count=arguments.expected_parameter_count,
        projected_train_samples=arguments.projected_train_samples,
    )


if __name__ == "__main__":
    main()
