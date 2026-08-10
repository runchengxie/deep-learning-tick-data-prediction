"""次日训练配置和端到端链路测试。"""

import json
from datetime import date, datetime

import numpy as np
import pytest
import torch

from ticknet.nextday import train as train_module
from ticknet.nextday.dataset import manifest_fingerprint
from ticknet.nextday.io import PreparedSample, write_sharded_dataset
from ticknet.nextday.labels import NextDayTarget
from ticknet.nextday.model import NextDayOutput
from ticknet.nextday.train import NextDayConfig, load_config


def test_nextday_yaml_and_cli_override(tmp_path):
    config_path = tmp_path / "nextday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "manifest_path: data/manifest.json",
                'train_start: "2024-01-01"',
                'train_end: "2024-01-31"',
                'val_start: "2024-02-01"',
                'val_end: "2024-02-29"',
                'test_start: "2024-03-01"',
                'test_end: "2024-03-31"',
                "batch_size: 16",
                "conv_channels: 32",
                "inception_channels: 64",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(
        [
            "--config",
            str(config_path),
            "--batch-size",
            "8",
            "--no-resume",
            "--evaluate-test",
            "--no-verify-data-checksums",
        ]
    )
    assert config.batch_size == 8
    assert config.resume is False
    assert config.evaluate_test is True
    assert config.verify_data_checksums is False
    assert config.conv_channels == 32
    assert config.inception_channels == 64
    assert config.date_split().train.end.isoformat() == "2024-01-31"


def test_nextday_config_requires_manifest():
    with pytest.raises(ValueError, match="manifest_path"):
        NextDayConfig().validate()


def test_nextday_config_requires_sidecar_for_long_horizon():
    with pytest.raises(ValueError, match="target_sidecar_path"):
        NextDayConfig(manifest_path="manifest.json", target_horizon=5).validate()


def test_nextday_config_rejects_overlapping_dates():
    config = NextDayConfig(
        manifest_path="manifest.json",
        train_start="2024-01-01",
        train_end="2024-02-01",
        val_start="2024-02-01",
        val_end="2024-02-28",
        test_start="2024-03-01",
        test_end="2024-03-31",
    )
    with pytest.raises(ValueError, match="训练区间"):
        config.validate()


@pytest.mark.parametrize("field", ["conv_channels", "inception_channels"])
def test_nextday_config_rejects_non_positive_frontend_width(field):
    config = NextDayConfig(manifest_path="manifest.json")
    setattr(config, field, 0)
    with pytest.raises(ValueError, match="模型隐藏维度"):
        config.validate()


def test_checkpoint_signature_defaults_legacy_frontend_widths():
    config = NextDayConfig(manifest_path="manifest.json")
    expected = train_module._experiment_signature(config, "dataset-fingerprint")
    legacy = dict(expected)
    legacy.pop("conv_channels")
    legacy.pop("inception_channels")
    legacy.pop("target_sidecar_path")
    legacy.pop("target_horizon")
    assert train_module._checkpoint_matches_experiment(
        {"experiment": legacy},
        expected,
    )
    wider = dict(expected, conv_channels=32)
    assert not train_module._checkpoint_matches_experiment(
        {"experiment": legacy},
        wider,
    )


class _TinyNextDayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_head = torch.nn.Linear(1, 3)
        self.score_head = torch.nn.Linear(1, 1)

    def forward(self, features):
        summary = features.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return NextDayOutput(
            logits=self.classification_head(summary),
            score=self.score_head(summary).squeeze(-1),
        )


def _training_manifest(tmp_path):
    samples = []
    for input_day, label_day in ((2, 3), (4, 5), (6, 7)):
        for label, symbol in enumerate(("A", "B", "C")):
            target_return = (label - 1) * 0.01
            target = NextDayTarget(
                symbol=symbol,
                trading_date=date(2024, 1, input_day),
                label_date=date(2024, 1, label_day),
                raw_return=target_return,
                target_return=target_return,
                label=label,
            )
            samples.append(
                PreparedSample(
                    target=target,
                    events=np.full((4, 40), label, dtype=np.float32),
                    last_event_timestamp=datetime(2024, 1, input_day, 14, 54),
                    signal_timestamp=datetime(2024, 1, input_day, 14, 55),
                )
            )
    return write_sharded_dataset(
        samples,
        tmp_path / "data",
        chunks_per_sample=1,
        chunk_size=4,
        samples_per_shard=4,
    )


def test_nextday_training_checkpoint_and_resume(tmp_path, monkeypatch):
    manifest = _training_manifest(tmp_path)
    monkeypatch.setattr(
        train_module,
        "build_nextday_model",
        lambda **kwargs: _TinyNextDayModel(),
    )
    config = NextDayConfig(
        manifest_path=str(manifest),
        train_start="2024-01-02",
        train_end="2024-01-03",
        val_start="2024-01-04",
        val_end="2024-01-05",
        test_start="2024-01-06",
        test_end="2024-01-07",
        epochs=1,
        batch_size=3,
        patience=2,
        min_symbols_per_day=3,
        selection_metric="macro_f1",
        resume=False,
        evaluate_test=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    first = train_module.train(config)
    assert first["samples"] == {"train": 3, "val": 3, "test": 3}
    assert first["test"] is None
    best_path = tmp_path / "checkpoints/chunked-ticknet.seed0.best.pt"
    last_path = tmp_path / "checkpoints/chunked-ticknet.seed0.last.pt"
    assert best_path.is_file()

    best_before = best_path.read_bytes()
    last_before = last_path.read_bytes()
    locked_test = train_module.evaluate_best_checkpoints(config, [0])
    assert locked_test["mode"] == "best_checkpoint_locked_test"
    assert locked_test["samples"] == {"test": 3}
    assert locked_test["per_seed"][0]["best_epoch"] == 1
    assert locked_test["per_seed"][0]["test"]["evaluated_dates"] == 1
    assert locked_test["aggregate"]["daily_rank_ic_mean"]["std"] == 0.0
    assert best_path.read_bytes() == best_before
    assert last_path.read_bytes() == last_before

    with pytest.raises(FileNotFoundError, match="seed 1"):
        train_module.evaluate_best_checkpoints(config, [0, 1])
    assert not (tmp_path / "checkpoints/locked_test.chunked-ticknet.seeds0-1.json").exists()

    config.epochs = 2
    config.resume = True
    config.evaluate_test = True
    second = train_module.train(config)
    assert second["duration_seconds"] > 0
    assert second["test"]["evaluated_dates"] == 1
    history = tmp_path / "checkpoints/train_history.chunked-ticknet.seed0.json"
    assert [row["epoch"] for row in json.loads(history.read_text())] == [1, 2]

    last_checkpoint = train_module._load_checkpoint(last_path, torch.device("cpu"))
    last_checkpoint["epochs_without_improvement"] = config.patience
    torch.save(last_checkpoint, last_path)
    config.epochs = 3
    train_module.train(config)
    assert [row["epoch"] for row in json.loads(history.read_text())] == [1, 2]

    config.conv_channels = 32
    with pytest.raises(ValueError, match="实验配置"):
        train_module.train(config)
    config.conv_channels = 16

    manifest_content = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_content["samples"][0]["target_return"] = 0.123
    manifest_content["dataset_fingerprint"] = manifest_fingerprint(manifest_content)
    manifest.write_text(json.dumps(manifest_content), encoding="utf-8")
    with pytest.raises(ValueError, match="实验配置"):
        train_module.train(config)
