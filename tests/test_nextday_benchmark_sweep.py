"""100M 模型 batch-size sweep 的计划与选择测试。"""

from pathlib import Path

import pytest

from ticknet.nextday import benchmark_sweep
from ticknet.nextday.benchmark_sweep import (
    build_batch_plan,
    run_sweep,
    select_best_trial,
)
from ticknet.nextday.config import NextDayConfig


def test_batch_plan_keeps_effective_batch_at_32() -> None:
    assert build_batch_plan([2, 4, 8, 16, 32], 32) == [
        (2, 16),
        (4, 8),
        (8, 4),
        (16, 2),
        (32, 1),
    ]


@pytest.mark.parametrize(
    ("batch_sizes", "effective_batch_size", "message"),
    [
        ([], 32, "不能为空"),
        ([2, 2], 32, "不能重复"),
        ([0, 2], 32, "正整数"),
        ([3], 32, "不能被"),
        ([2], 0, "正整数"),
    ],
)
def test_batch_plan_rejects_invalid_sweeps(
    batch_sizes: list[int],
    effective_batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_batch_plan(batch_sizes, effective_batch_size)


def test_best_trial_ignores_oom_and_selects_highest_throughput() -> None:
    trials = [
        {"status": "complete", "physical_batch_size": 2, "samples_per_second": 80.0},
        {"status": "oom", "physical_batch_size": 32},
        {"status": "complete", "physical_batch_size": 16, "samples_per_second": 215.0},
        {"status": "complete", "physical_batch_size": 8, "samples_per_second": 180.0},
    ]

    assert select_best_trial(trials)["physical_batch_size"] == 16


def test_best_trial_requires_one_success() -> None:
    with pytest.raises(RuntimeError, match="所有 batch size 均 OOM"):
        select_best_trial([{"status": "oom", "physical_batch_size": 2}])


def test_run_sweep_updates_accumulation_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, int]] = []

    def fake_benchmark(config: NextDayConfig, **kwargs: object) -> dict[str, object]:
        observed.append((config.batch_size, config.gradient_accumulation_steps))
        throughput = float(config.batch_size * 10)
        return {
            "actual_gpu": "Fake A100",
            "model": {"parameter_count": 100_817_575},
            "data": {"dataset_fingerprint": "fingerprint"},
            "environment": {"torch": "test"},
            "benchmark": {
                "physical_batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": (config.batch_size * config.gradient_accumulation_steps),
                "samples_per_second": throughput,
                "duration_seconds": 1.0,
                "optimizer_steps": 1,
                "projected_seed_hours": 10.0 / throughput,
            },
            "memory": {"peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0},
        }

    monkeypatch.setattr(benchmark_sweep, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(benchmark_sweep.torch.cuda, "empty_cache", lambda: None)

    summary = run_sweep(
        NextDayConfig(manifest_path="unused", device="cuda"),
        batch_sizes=[2, 4],
        effective_batch_size=8,
        batches=50,
        warmup_batches=5,
        output_dir=tmp_path,
        source_revision="abc123",
        requested_gpu="A100",
        expected_parameter_count=100_817_575,
        projected_train_samples=75_000,
    )

    assert observed == [(2, 4), (4, 2)]
    assert summary["benchmark"]["selected_physical_batch_size"] == 4
    assert summary["trials"][1]["speedup_vs_batch_2"] == 2.0
    assert (tmp_path / "batch-size-sweep.json").is_file()
