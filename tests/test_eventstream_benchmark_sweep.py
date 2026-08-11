"""eventstream 100M 模型 batch-size sweep 测试。"""

from pathlib import Path

import pytest

from ticknet.eventstream import benchmark_sweep
from ticknet.eventstream.benchmark_sweep import (
    build_batch_plan,
    run_sweep,
    select_best_trial,
)
from ticknet.eventstream.train import EventstreamConfig


def test_batch_plan_keeps_effective_batch_at_64() -> None:
    assert build_batch_plan([8, 16, 32, 64], 64) == [
        (8, 8),
        (16, 4),
        (32, 2),
        (64, 1),
    ]


@pytest.mark.parametrize(
    ("batch_sizes", "effective_batch_size", "message"),
    [
        ([], 64, "不能为空"),
        ([8, 8], 64, "不能重复"),
        ([0, 8], 64, "正整数"),
        ([24], 64, "不能被"),
        ([8], 0, "正整数"),
    ],
)
def test_batch_plan_rejects_invalid_sweeps(
    batch_sizes: list[int],
    effective_batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_batch_plan(batch_sizes, effective_batch_size)


def test_best_trial_ignores_oom() -> None:
    trials = [
        {"status": "complete", "physical_batch_size": 8, "samples_per_second": 4.0},
        {"status": "oom", "physical_batch_size": 64},
        {"status": "complete", "physical_batch_size": 32, "samples_per_second": 12.0},
    ]

    assert select_best_trial(trials)["physical_batch_size"] == 32


def test_run_sweep_updates_accumulation_and_projects_formal_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, int]] = []

    def fake_benchmark(config: EventstreamConfig, **kwargs: object) -> dict[str, object]:
        observed.append((config.batch_size, config.gradient_accumulation_steps))
        throughput = float(config.batch_size)
        return {
            "actual_gpu": "Fake A100",
            "model": {"parameter_count": 100_604_180},
            "data": {"dataset_fingerprint": "fingerprint"},
            "environment": {"torch": "test"},
            "benchmark": {
                "physical_batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": (config.batch_size * config.gradient_accumulation_steps),
                "samples_per_second": throughput,
                "duration_seconds": 1.0,
                "optimizer_steps": 1,
            },
            "memory": {"peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0},
        }

    monkeypatch.setattr(benchmark_sweep, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(benchmark_sweep.torch.cuda, "empty_cache", lambda: None)
    config = EventstreamConfig(
        days=(20250801,),
        model="smoke",
        device="cuda",
        epochs=20,
    )

    summary = run_sweep(
        config,
        batch_sizes=[8, 16],
        effective_batch_size=64,
        batches=50,
        warmup_batches=5,
        output_dir=tmp_path,
        source_revision="abc123",
        requested_gpu="A100",
        expected_parameter_count=100_604_180,
        projected_train_samples=120_000,
    )

    assert observed == [(8, 8), (16, 4)]
    assert summary["benchmark"]["selected_physical_batch_size"] == 16
    assert summary["benchmark"]["selected_projected_seed_hours"] == pytest.approx(41.6666667)
    assert summary["trials"][1]["speedup_vs_reference"] == 2.0
    assert (tmp_path / "batch-size-sweep.json").is_file()
