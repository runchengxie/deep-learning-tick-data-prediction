"""eventstream 100M 容量配置与 benchmark 前置门槛。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ticknet.eventstream.benchmark import run_benchmark
from ticknet.eventstream.model import build_eventstream_model
from ticknet.eventstream.train import EventstreamConfig


def test_capacity100m_parameter_count() -> None:
    model = build_eventstream_model("capacity100m")
    assert sum(parameter.numel() for parameter in model.parameters()) == 100_604_180


@pytest.mark.parametrize(
    "name",
    [
        "eventstream-h5-recent-capacity100m.yaml",
        "eventstream-h5-recent-capacity100m-colab.yaml",
    ],
)
def test_recent_configs_use_profiled_worker_count(name: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["num_workers"] == 8


def test_benchmark_requires_cuda(tmp_path: Path) -> None:
    config = EventstreamConfig(days=(20210104,), model="smoke", device="cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        run_benchmark(
            config,
            batches=1,
            warmup_batches=0,
            output=tmp_path / "benchmark.json",
            source_revision="test",
            requested_gpu="cpu",
            expected_parameter_count=None,
        )
