"""eventstream 输入流水线 profiling 测试。"""

import pytest

from ticknet.eventstream.input_profile import (
    build_worker_plan,
    classify_bottleneck,
    select_best_workers,
)


def test_worker_plan_preserves_requested_order() -> None:
    assert build_worker_plan([2, 4, 8, 16]) == [2, 4, 8, 16]


@pytest.mark.parametrize(
    ("worker_counts", "message"),
    [
        ([], "不能为空"),
        ([2, 2], "不能重复"),
        ([-1, 2], "不能为负数"),
    ],
)
def test_worker_plan_rejects_invalid_values(
    worker_counts: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_worker_plan(worker_counts)


@pytest.mark.parametrize(
    ("data_throughput", "gpu_throughput", "expected"),
    [
        (5.0, 40.0, "input_pipeline"),
        (40.0, 5.0, "gpu_compute"),
        (8.0, 10.0, "mixed"),
    ],
)
def test_bottleneck_classification(
    data_throughput: float,
    gpu_throughput: float,
    expected: str,
) -> None:
    assert classify_bottleneck(data_throughput, gpu_throughput) == expected


def test_bottleneck_classification_requires_positive_throughput() -> None:
    with pytest.raises(ValueError, match="正数"):
        classify_bottleneck(0.0, 1.0)


def test_best_workers_uses_end_to_end_throughput_and_fewer_workers_on_tie() -> None:
    trials = [
        {"num_workers": 8, "end_to_end_samples_per_second": 12.0},
        {"num_workers": 4, "end_to_end_samples_per_second": 12.0},
        {"num_workers": 2, "end_to_end_samples_per_second": 5.0},
    ]

    assert select_best_workers(trials)["num_workers"] == 4
