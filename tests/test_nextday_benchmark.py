"""100M raw-1000 benchmark 配置与模型规模合同测试。"""

from pathlib import Path

import pytest
import torch

from ticknet.nextday.benchmark import count_parameters
from ticknet.nextday.model import build_nextday_model
from ticknet.nextday.snapshot_cli import load_snapshot_config
from ticknet.nextday.train import load_config

EXPECTED_100M_PARAMETERS = 100_817_575
EXPECTED_1M_PARAMETERS = 1_033_383


def test_raw1000_configs_preserve_event_and_universe_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename, expected_end in (
        ("nextday-raw-1000-preflight.yaml", "2021-01-31"),
        ("nextday-raw-1000-top100.yaml", "2025-12-31"),
    ):
        config = load_snapshot_config(["--config", str(root / "configs" / filename)])
        assert config.chunks_per_sample == 10
        assert config.chunk_size == 100
        assert config.min_valid_events == 1000
        assert config.top_n == 100
        assert config.scan_start_time_ms == 14_400_000
        assert config.end_date == expected_end


@pytest.mark.parametrize(
    "filename",
    [
        "nextday-raw-1000-top100-capacity-100m-benchmark.yaml",
        "nextday-raw-1000-top100-capacity-100m.yaml",
    ],
)
def test_100m_configs_are_exact_and_keep_test_locked(filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        [
            "--config",
            str(root / "configs" / filename),
        ]
    )

    assert config.evaluate_test is False
    if filename.endswith("-benchmark.yaml"):
        assert config.batch_size == 2
        assert config.gradient_accumulation_steps == 16
    else:
        assert config.batch_size == 32
        assert config.gradient_accumulation_steps == 1
        assert config.resume is True
    with torch.device("meta"):
        model = build_nextday_model(
            chunks_per_sample=10,
            chunk_size=100,
            conv_channels=config.conv_channels,
            inception_channels=config.inception_channels,
            intraday_embedding_size=config.intraday_embedding_size,
            day_hidden_size=config.day_hidden_size,
            day_layers=config.day_layers,
            dropout=config.dropout,
        )
    assert count_parameters(model) == EXPECTED_100M_PARAMETERS


@pytest.mark.parametrize(
    ("cell", "expected_chunks", "expected_parameters"),
    [
        ("1m-raw200", 2, EXPECTED_1M_PARAMETERS),
        ("1m-raw1000", 10, EXPECTED_1M_PARAMETERS),
        ("100m-raw200", 2, EXPECTED_100M_PARAMETERS),
    ],
)
def test_capacity_matrix_configs_freeze_shared_contract(
    cell: str,
    expected_chunks: int,
    expected_parameters: int,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        ["--config", str(root / "configs" / f"nextday-capacity-matrix-{cell}.yaml")]
    )

    assert config.manifest_path == (
        "/content/nextday-raw-1000-pilot-2021-2025-top100/manifest.json"
    )
    assert (config.input_last_chunks or 10) == expected_chunks
    assert config.lr == 0.0001
    assert config.batch_size == 32
    assert config.min_symbols_per_day == 50
    assert config.evaluate_test is False
    with torch.device("meta"):
        model = build_nextday_model(
            chunks_per_sample=expected_chunks,
            chunk_size=100,
            conv_channels=config.conv_channels,
            inception_channels=config.inception_channels,
            intraday_embedding_size=config.intraday_embedding_size,
            day_hidden_size=config.day_hidden_size,
            day_layers=config.day_layers,
            dropout=config.dropout,
        )
    assert count_parameters(model) == expected_parameters
