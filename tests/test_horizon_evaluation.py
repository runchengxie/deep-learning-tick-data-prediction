"""2024 validation 多周期固定 checkpoint 评估测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from ticknet.nextday import horizon_evaluation as evaluation_module
from ticknet.nextday.horizon_evaluation import (
    daily_rank_ic,
    evaluate_validation_horizons,
    newey_west_mean,
    summarize_daily_ic,
)
from ticknet.nextday.horizon_labels import HorizonTarget, write_horizon_sidecar
from ticknet.nextday.io import PreparedSample, write_sharded_dataset
from ticknet.nextday.labels import NextDayTarget
from ticknet.nextday.model import NextDayOutput
from ticknet.nextday.train import NextDayConfig, _experiment_signature


def _target(symbol: str, signal_day: int, value: float) -> NextDayTarget:
    label = 0 if value < 0 else 2 if value > 0 else 1
    return NextDayTarget(
        symbol=symbol,
        trading_date=date(2024, 1, signal_day),
        label_date=date(2024, 1, signal_day + 1),
        raw_return=value,
        target_return=value,
        label=label,
    )


def _evaluation_inputs(tmp_path: Path) -> tuple[Path, Path]:
    samples = []
    symbols_and_values = (("A", -0.01), ("B", 0.0), ("C", 0.01))
    for signal_day in (2, 8):
        for feature_value, (symbol, target_return) in enumerate(symbols_and_values):
            samples.append(
                PreparedSample(
                    target=_target(symbol, signal_day, target_return),
                    events=np.full((4, 40), feature_value, dtype=np.float32),
                    last_event_timestamp=datetime(2024, 1, signal_day, 14, 54),
                    signal_timestamp=datetime(2024, 1, signal_day, 14, 55),
                )
            )
    manifest = write_sharded_dataset(
        samples,
        tmp_path / "features",
        chunks_per_sample=1,
        chunk_size=4,
        samples_per_shard=4,
    )
    feature_dataset = evaluation_module.NextDayShardDataset(
        manifest,
        date_split=NextDayConfig(
            manifest_path=str(manifest),
            val_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-12-31",
        ).date_split(),
        split="val",
    )

    targets = []
    for horizon in (1, 3, 5):
        for signal_day in (2, 8):
            for symbol, target_return in symbols_and_values:
                targets.append(
                    HorizonTarget(
                        symbol=symbol,
                        trading_date=date(2024, 1, signal_day),
                        entry_date=date(2024, 1, signal_day + 1),
                        return_end_date=date(2024, 1, signal_day + horizon),
                        horizon=horizon,
                        label=0 if target_return < 0 else 2 if target_return > 0 else 1,
                        raw_return=target_return,
                        benchmark_return=0.0,
                        target_return=target_return,
                    )
                )
    sidecar = write_horizon_sidecar(
        targets,
        tmp_path / "targets",
        source_dataset_fingerprint=feature_dataset.dataset_fingerprint,
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=3,
    )
    return manifest, sidecar


class _TinyNextDayModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classification_head = torch.nn.Linear(1, 3)
        self.score_head = torch.nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> NextDayOutput:
        summary = features.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return NextDayOutput(
            logits=self.classification_head(summary),
            score=self.score_head(summary).squeeze(-1),
        )


def _config(manifest: Path, checkpoint_dir: Path) -> NextDayConfig:
    return NextDayConfig(
        manifest_path=str(manifest),
        train_start="2021-01-01",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        batch_size=3,
        num_workers=0,
        device="cpu",
        resume=True,
        evaluate_test=False,
        verify_data_checksums=True,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="capacity",
        min_symbols_per_day=3,
    )


def _write_checkpoints(config: NextDayConfig, seeds: tuple[int, ...]) -> None:
    assert config.manifest_path is not None
    dataset = evaluation_module.NextDayShardDataset(
        config.manifest_path,
        date_split=config.date_split(),
        split="val",
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        model = _TinyNextDayModel()
        with torch.no_grad():
            model.score_head.weight.fill_(1.0)
            model.score_head.bias.zero_()
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": seed + 1,
                "best_selection_value": 0.01 + seed / 100,
                "experiment": _experiment_signature(
                    seed_config,
                    dataset.dataset_fingerprint,
                ),
            },
            checkpoint_dir / f"capacity.seed{seed}.best.pt",
        )


def test_rank_ic_and_robust_summaries() -> None:
    signal_date = date(2024, 1, 2)
    scores = {
        ("A", signal_date): -1.0,
        ("B", signal_date): 0.0,
        ("C", signal_date): 1.0,
    }
    targets = [
        HorizonTarget(
            symbol=symbol,
            trading_date=signal_date,
            entry_date=date(2024, 1, 3),
            return_end_date=date(2024, 1, 5),
            horizon=3,
            label=index,
            raw_return=float(index - 1),
            benchmark_return=0.0,
            target_return=float(index - 1),
        )
        for index, symbol in enumerate(("A", "B", "C"))
    ]
    rows = daily_rank_ic(scores, targets, min_symbols_per_day=3)
    assert rows[0]["rank_ic"] == pytest.approx(1.0)

    summary = summarize_daily_ic(
        [
            {"signal_date": date(2024, 1, 2), "symbols": 3, "rank_ic": 0.1},
            {"signal_date": date(2024, 1, 3), "symbols": 3, "rank_ic": 0.2},
            {"signal_date": date(2024, 2, 1), "symbols": 3, "rank_ic": -0.1},
            {"signal_date": date(2024, 2, 2), "symbols": 3, "rank_ic": 0.2},
        ],
        horizon=2,
    )
    assert summary["newey_west"]["lag"] == 1
    assert summary["monthly_stability"]["months"] == 2
    assert len(summary["non_overlapping"]["phases"]) == 2
    assert math_is_finite(summary["newey_west"]["standard_error"])


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def test_newey_west_validation() -> None:
    result = newey_west_mean([1.0, 1.0, 1.0], lag=5)
    assert result["lag"] == 2
    assert result["standard_error"] == 0.0
    assert np.isnan(result["t_stat"])
    with pytest.raises(ValueError, match="非空"):
        newey_west_mean([], lag=0)
    with pytest.raises(ValueError, match="不能为负数"):
        newey_west_mean([1.0], lag=-1)


def test_evaluate_validation_horizons_is_checkpoint_only(tmp_path, monkeypatch) -> None:
    manifest, sidecar = _evaluation_inputs(tmp_path)
    config = _config(manifest, tmp_path / "checkpoints")
    _write_checkpoints(config, (0, 1))
    monkeypatch.setattr(
        evaluation_module,
        "build_nextday_model",
        lambda **_kwargs: _TinyNextDayModel(),
    )
    original_dataset = evaluation_module.NextDayShardDataset
    requested_splits = []

    def recording_dataset(*args, **kwargs):
        requested_splits.append(kwargs["split"])
        return original_dataset(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "NextDayShardDataset", recording_dataset)
    output_dir = tmp_path / "output"
    result = evaluate_validation_horizons(
        config,
        sidecar,
        seeds=(0, 1),
        horizons=(1, 3, 5),
        output_dir=output_dir,
        inference_batch_size=6,
        source_revision="test-revision",
    )

    assert requested_splits == ["val"]
    assert result["test_status"] == "locked_not_accessed"
    assert result["training_status"] == "not_run"
    assert result["source_revision"] == "test-revision"
    assert result["results"]["5"]["models"]["seed_0"]["daily_rank_ic_mean"] == 1.0
    assert result["results"]["5"]["roadmap_gate"]["meets_roadmap_gate"] is True
    assert len(result["checkpoints"][0]["sha256"]) == 64
    assert (output_dir / "multi_horizon_validation_2024.json").is_file()
    assert pq.read_table(output_dir / "validation_scores_2024.parquet").num_rows == 6
    assert pq.read_table(output_dir / "daily_rank_ic_2024.parquet").num_rows == 18


def test_validation_only_safety_guards(tmp_path) -> None:
    config = NextDayConfig(
        manifest_path=str(tmp_path / "missing.json"),
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        evaluate_test=True,
    )
    with pytest.raises(ValueError, match="evaluate_test=False"):
        evaluate_validation_horizons(config, tmp_path, output_dir=tmp_path / "output")

    config.evaluate_test = False
    config.val_end = "2024-06-30"
    with pytest.raises(ValueError, match="只允许评估"):
        evaluate_validation_horizons(config, tmp_path, output_dir=tmp_path / "output")
