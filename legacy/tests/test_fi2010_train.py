"""训练配置和调度逻辑测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from legacy import fi2010_train as train_module
from legacy.fi2010_train import Config, load_config, run_setup1
from ticknet.train import f1_metrics, resolve_device


def test_yaml_config_and_cli_override(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset: random\nepochs: 5\nbatch_size: 16\nresume: true\n",
        encoding="utf-8",
    )
    config = load_config(
        [
            "--config",
            str(config_path),
            "--epochs",
            "2",
            "--no-resume",
        ]
    )
    assert config.epochs == 2
    assert config.batch_size == 16
    assert config.resume is False


def test_yaml_rejects_unknown_field(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unknown_option: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="未知字段"):
        load_config(["--config", str(config_path)])


def test_cpu_request_is_honoured_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == torch.device("cpu")


def test_f1_metrics_return_expected_keys():
    labels = np.array([0, 1, 2, 0, 1, 2])
    metrics = f1_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["per_class_precision"] == [1.0, 1.0, 1.0]


def test_random_dataset_runs_single_training_path(monkeypatch):
    calls = []

    def fake_train(config, *, test_cf=None):
        calls.append((config.dataset, test_cf))
        return {}

    monkeypatch.setattr(train_module, "train", fake_train)
    train_module.main(["--dataset", "random", "--protocol", "setup1"])
    assert calls == [("random", None)]


def test_config_rejects_missing_fi2010_paths():
    with pytest.raises(ValueError, match="data_path"):
        Config(dataset="fi2010").validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (Config(k=40), "k 应为"),
        (Config(lr=0), "lr 和 eps"),
        (Config(setup1_cfs=[1, 1]), "重复"),
    ],
)
def test_config_rejects_invalid_values(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_training_checkpoint_and_resume(tmp_path, monkeypatch):
    features = torch.randn(12, 1, 2, 2)
    labels = torch.tensor([0, 1, 2] * 4)
    dataset = TensorDataset(features, labels)

    def fake_dataloaders(config, *, device, test_cf=None):
        del device, test_cf
        loader = DataLoader(dataset, batch_size=config.batch_size)
        return loader, loader, loader

    def fake_model():
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4, 3),
        )

    monkeypatch.setattr(train_module, "make_dataloaders", fake_dataloaders)
    monkeypatch.setattr(train_module, "build_model", fake_model)
    config = Config(
        epochs=1,
        batch_size=4,
        checkpoint_dir=str(tmp_path),
    )
    first = train_module.train(config)
    assert (tmp_path / "ticknet.smoke.seed0.last.pt").is_file()
    assert (tmp_path / "ticknet.smoke.seed0.best.pt").is_file()
    assert (tmp_path / "result.smoke.seed0.json").is_file()

    config.epochs = 2
    second = train_module.train(config)
    history = json.loads((tmp_path / "train_history.smoke.seed0.json").read_text(encoding="utf-8"))
    assert first["run_tag"] == second["run_tag"] == "smoke.seed0"
    assert second["environment"]["python"]
    assert second["duration_seconds"] > 0
    assert [record["epoch"] for record in history] == [1, 2]
    assert all(record["training_seconds"] > 0 for record in history)
    assert all(record["validation_seconds"] > 0 for record in history)
    assert all(record["training_samples_per_second"] > 0 for record in history)

    config.lr = 0.02
    with pytest.raises(ValueError, match="实验配置"):
        train_module.train(config)


def test_setup1_summary_aggregates_selected_cfs(tmp_path, monkeypatch):
    def fake_train(config: Config, *, test_cf: int | None = None):
        del config
        assert test_cf is not None
        return {
            "run_tag": f"cf{test_cf}",
            "test": {
                "accuracy": test_cf / 10,
                "macro_f1": test_cf / 20,
            },
        }

    monkeypatch.setattr(train_module, "train", fake_train)
    config = Config(
        setup1_cfs=[1, 2],
        checkpoint_dir=str(tmp_path),
    )
    summary = run_setup1(config)
    assert summary["mean_accuracy"] == pytest.approx(0.15)
    assert summary["mean_macro_f1"] == pytest.approx(0.075)
    assert (tmp_path / "setup1_summary.k10.json").is_file()
