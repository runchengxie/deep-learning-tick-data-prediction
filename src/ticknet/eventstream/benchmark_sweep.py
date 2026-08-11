"""在同一块 GPU 上扫描 eventstream 模型的物理 batch size。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from ticknet.eventstream.benchmark import run_benchmark
from ticknet.eventstream.train import EventstreamConfig


def build_batch_plan(
    batch_sizes: list[int],
    effective_batch_size: int,
) -> list[tuple[int, int]]:
    """校验物理 batch，并返回对应的梯度累积步数。"""
    if effective_batch_size < 1:
        raise ValueError("effective_batch_size 应为正整数")
    if not batch_sizes:
        raise ValueError("batch_sizes 不能为空")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch_sizes 不能重复")

    plan = []
    for batch_size in batch_sizes:
        if batch_size < 1:
            raise ValueError("batch_sizes 都应为正整数")
        if effective_batch_size % batch_size:
            raise ValueError(
                f"effective_batch_size={effective_batch_size} 不能被 batch_size={batch_size} 整除"
            )
        plan.append((batch_size, effective_batch_size // batch_size))
    return plan


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _trial_config(
    config: EventstreamConfig,
    *,
    batch_size: int,
    accumulation_steps: int,
) -> EventstreamConfig:
    mapping = config.to_dict()
    mapping["batch_size"] = batch_size
    mapping["gradient_accumulation_steps"] = accumulation_steps
    return EventstreamConfig.from_mapping(mapping)


def _successful_trial(
    result: dict[str, Any],
    *,
    result_file: str,
    projected_train_samples: int,
    configured_epochs: int,
) -> dict[str, Any]:
    benchmark = result["benchmark"]
    memory = result["memory"]
    throughput = float(benchmark["samples_per_second"])
    projected_epoch_minutes = projected_train_samples / throughput / 60
    return {
        "status": "complete",
        "physical_batch_size": benchmark["physical_batch_size"],
        "gradient_accumulation_steps": benchmark["gradient_accumulation_steps"],
        "effective_batch_size": benchmark["effective_batch_size"],
        "samples_per_second": throughput,
        "duration_seconds": benchmark["duration_seconds"],
        "optimizer_steps": benchmark["optimizer_steps"],
        "peak_allocated_gib": memory["peak_allocated_gib"],
        "peak_reserved_gib": memory["peak_reserved_gib"],
        "projected_train_samples": projected_train_samples,
        "projected_epoch_minutes": projected_epoch_minutes,
        "projected_seed_hours": projected_epoch_minutes * configured_epochs / 60,
        "result_file": result_file,
    }


def select_best_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """按吞吐选择成功且未 OOM 的最佳 trial。"""
    successful = [trial for trial in trials if trial["status"] == "complete"]
    if not successful:
        raise RuntimeError("所有 batch size 均 OOM，没有可用结果")
    return max(
        successful,
        key=lambda trial: (
            float(trial["samples_per_second"]),
            int(trial["physical_batch_size"]),
        ),
    )


def run_sweep(
    config: EventstreamConfig,
    *,
    batch_sizes: list[int],
    effective_batch_size: int,
    batches: int,
    warmup_batches: int,
    output_dir: Path,
    source_revision: str,
    requested_gpu: str,
    expected_parameter_count: int | None,
    projected_train_samples: int,
) -> dict[str, Any]:
    """依次执行 eventstream batch sweep，OOM 后清理显存并继续下一档。"""
    if projected_train_samples < 1:
        raise ValueError("projected_train_samples 应为正整数")
    plan = build_batch_plan(batch_sizes, effective_batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None

    for batch_size, accumulation_steps in plan:
        trial_config = _trial_config(
            config,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
        )
        result_path = output_dir / f"batch-{batch_size:03d}.json"
        try:
            result = run_benchmark(
                trial_config,
                batches=batches,
                warmup_batches=warmup_batches,
                output=result_path,
                source_revision=source_revision,
                requested_gpu=requested_gpu,
                expected_parameter_count=expected_parameter_count,
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            trial = {
                "status": "oom",
                "physical_batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "error": str(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "result_file": result_path.name,
            }
            _atomic_json(result_path, trial)
            trials.append(trial)
        else:
            reference = reference or result
            trials.append(
                _successful_trial(
                    result,
                    result_file=result_path.name,
                    projected_train_samples=projected_train_samples,
                    configured_epochs=config.epochs,
                )
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    best = select_best_trial(trials)
    successful = [trial for trial in trials if trial["status"] == "complete"]
    baseline = min(successful, key=lambda trial: int(trial["physical_batch_size"]))
    baseline_throughput = float(baseline["samples_per_second"])
    for trial in trials:
        throughput = trial.get("samples_per_second")
        trial["speedup_vs_reference"] = (
            None if throughput is None else float(throughput) / baseline_throughput
        )

    if reference is None:
        raise RuntimeError("所有 batch size 均 OOM，没有可用结果")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "eventstream_batch_size_sweep",
        "source_revision": source_revision,
        "test_status": "validation_and_oos_not_accessed",
        "requested_gpu": requested_gpu,
        "actual_gpu": reference["actual_gpu"],
        "model": reference["model"],
        "data": reference["data"],
        "benchmark": {
            "batch_sizes": batch_sizes,
            "effective_batch_size": effective_batch_size,
            "reference_physical_batch_size": baseline["physical_batch_size"],
            "warmup_batches_per_size": warmup_batches,
            "measured_batches_per_size": batches,
            "projected_train_samples": projected_train_samples,
            "configured_epochs": config.epochs,
            "selected_physical_batch_size": best["physical_batch_size"],
            "selected_gradient_accumulation_steps": best["gradient_accumulation_steps"],
            "selected_samples_per_second": best["samples_per_second"],
            "selected_projected_seed_hours": best["projected_seed_hours"],
        },
        "trials": trials,
        "environment": reference["environment"],
        "config": config.to_dict(),
    }
    if not math.isfinite(float(best["samples_per_second"])):
        raise ValueError("最佳吞吐不是有限数值")
    _atomic_json(output_dir / "batch-size-sweep.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="eventstream 模型 batch-size sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
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
    run_sweep(
        config,
        batch_sizes=arguments.batch_sizes,
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
