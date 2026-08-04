"""次日训练配置和端到端链路测试。"""

import json
from datetime import date, datetime

import numpy as np
import pytest
import torch

from deeplob.nextday import train as train_module
from deeplob.nextday.dataset import manifest_fingerprint
from deeplob.nextday.io import PreparedSample, write_sharded_dataset
from deeplob.nextday.labels import NextDayTarget
from deeplob.nextday.model import NextDayOutput
from deeplob.nextday.train import NextDayConfig, load_config


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
    assert config.date_split().train.end.isoformat() == "2024-01-31"


def test_nextday_config_requires_manifest():
    with pytest.raises(ValueError, match="manifest_path"):
        NextDayConfig().validate()


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
    assert (tmp_path / "checkpoints/chunked-deeplob.seed0.best.pt").is_file()

    config.epochs = 2
    config.resume = True
    config.evaluate_test = True
    second = train_module.train(config)
    assert second["duration_seconds"] > 0
    assert second["test"]["evaluated_dates"] == 1
    history = tmp_path / "checkpoints/train_history.chunked-deeplob.seed0.json"
    assert [row["epoch"] for row in json.loads(history.read_text())] == [1, 2]

    manifest_content = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_content["samples"][0]["target_return"] = 0.123
    manifest_content["dataset_fingerprint"] = manifest_fingerprint(manifest_content)
    manifest.write_text(json.dumps(manifest_content), encoding="utf-8")
    with pytest.raises(ValueError, match="实验配置"):
        train_module.train(config)
