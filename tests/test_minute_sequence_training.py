"""分钟序列共享训练引擎的入口分派与产物契约测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from tests.test_minute_tcn import _write_minute_manifest
from ticknet.nextday import train_gru, train_tcn
from ticknet.nextday.minute_tcn import MinuteOutput
from ticknet.nextday.train import _load_checkpoint


class _TinyMinuteModel(torch.nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.classification_head = torch.nn.Linear(1, 3)
        self.score_head = torch.nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> MinuteOutput:
        summary = features.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return MinuteOutput(
            logits=self.classification_head(summary),
            score=self.score_head(summary).squeeze(-1),
        )


def _config(module: ModuleType, manifest_path: Path, checkpoint_dir: Path):
    if module is train_tcn:
        return train_tcn.MinuteTCNConfig(
            manifest_path=str(manifest_path),
            train_start="2024-01-01",
            train_end="2024-01-05",
            val_start="2024-01-06",
            val_end="2024-01-08",
            test_start="2024-01-09",
            test_end="2024-01-12",
            epochs=1,
            batch_size=4,
            patience=3,
            num_workers=0,
            device="cpu",
            resume=False,
            evaluate_test=True,
            checkpoint_dir=str(checkpoint_dir),
            min_symbols_per_day=2,
            selection_metric="macro_f1",
            checkpoint_name="shared-tcn",
            hidden_channels=12,
            tcn_layers=2,
            kernel_size=5,
        )
    return train_gru.MinuteGRUConfig(
        manifest_path=str(manifest_path),
        train_start="2024-01-01",
        train_end="2024-01-05",
        val_start="2024-01-06",
        val_end="2024-01-08",
        test_start="2024-01-09",
        test_end="2024-01-12",
        epochs=1,
        batch_size=4,
        patience=3,
        num_workers=0,
        device="cpu",
        resume=False,
        evaluate_test=True,
        checkpoint_dir=str(checkpoint_dir),
        min_symbols_per_day=2,
        selection_metric="macro_f1",
        checkpoint_name="shared-gru",
        gru_hidden_size=10,
        gru_layers=1,
    )


@pytest.mark.parametrize(
    ("module", "builder_name", "expected_model_arguments", "changed_model_field"),
    [
        (
            train_tcn,
            "build_minute_tcn",
            {"hidden_channels": 12, "num_layers": 2, "kernel_size": 5},
            "hidden_channels",
        ),
        (
            train_gru,
            "build_minute_gru",
            {"hidden_size": 10, "num_layers": 1},
            "gru_hidden_size",
        ),
    ],
)
def test_shared_engine_dispatch_resume_and_result_contract(
    tmp_path,
    monkeypatch,
    module,
    builder_name,
    expected_model_arguments,
    changed_model_field,
):
    manifest_path = _write_minute_manifest(tmp_path, n_shards=4, samples_per_shard=3)
    checkpoint_dir = tmp_path / "checkpoints"
    config = _config(module, manifest_path, checkpoint_dir)
    factory_calls: list[dict[str, Any]] = []

    def build_model(**arguments):
        factory_calls.append(arguments)
        return _TinyMinuteModel(arguments["num_features"])

    monkeypatch.setattr(module, builder_name, build_model)
    first = module.train(config)

    expected_result_keys = {
        "config",
        "samples",
        "environment",
        "duration_seconds",
        "dataset_fingerprint",
        "best_selection_value",
        "test",
        "last_checkpoint",
        "best_checkpoint",
        "history",
        "result_file",
    }
    assert set(first) == expected_result_keys
    assert first["test"] is not None
    assert factory_calls[0] == {
        "num_features": factory_calls[0]["num_features"],
        **expected_model_arguments,
        "dropout": config.dropout,
    }

    stem = f"{config.checkpoint_name}.seed0"
    last_path = checkpoint_dir / f"{stem}.last.pt"
    checkpoint = _load_checkpoint(last_path, torch.device("cpu"))
    assert checkpoint["epoch"] == 1
    assert checkpoint["experiment"][changed_model_field] == getattr(config, changed_model_field)
    assert set(checkpoint["target_normalization"]) == {"mean", "std"}
    result_path = checkpoint_dir / f"result.{stem}.json"
    assert set(json.loads(result_path.read_text(encoding="utf-8"))) == expected_result_keys

    locked = module.evaluate_best_checkpoints(config, [0])
    assert locked["mode"] == "best_checkpoint_locked_test"
    assert locked["per_seed"][0]["best_epoch"] == 1

    resumed_config = replace(config, epochs=2, resume=True)
    module.train(resumed_config)
    history_path = checkpoint_dir / f"train_history.{stem}.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]

    mismatched_value = getattr(resumed_config, changed_model_field) + 1
    mismatched_config = replace(
        resumed_config,
        epochs=3,
        **{changed_model_field: mismatched_value},
    )
    with pytest.raises(ValueError, match="实验配置"):
        module.train(mismatched_config)
