"""分钟 TCN 训练入口测试。"""

import json
from dataclasses import replace

import pytest

from tests.test_minute_tcn import _write_minute_manifest
from ticknet.nextday.train_tcn import MinuteTCNConfig, evaluate_best_checkpoints, load_config, train


def _small_config(manifest_path, checkpoint_dir) -> MinuteTCNConfig:
    return MinuteTCNConfig(
        manifest_path=str(manifest_path),
        train_start="2024-01-01",
        train_end="2024-01-05",
        val_start="2024-01-06",
        val_end="2024-01-08",
        test_start="2024-01-09",
        test_end="2024-01-12",
        epochs=2,
        batch_size=4,
        patience=2,
        num_workers=0,
        device="cpu",
        resume=False,
        evaluate_test=True,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="tcn-smoke",
        hidden_channels=16,
        tcn_layers=2,
        kernel_size=3,
    )


def test_minute_tcn_config_validate_rejects_bad_hidden_channels(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path, n_shards=4, samples_per_shard=3)
    config = _small_config(manifest_path, tmp_path)
    with pytest.raises(ValueError, match="TCN"):
        replace(config, hidden_channels=0).validate()
    with pytest.raises(ValueError, match="TCN"):
        replace(config, tcn_layers=0).validate()
    with pytest.raises(ValueError, match="TCN"):
        replace(config, kernel_size=0).validate()


def test_minute_tcn_train_runs_and_writes_checkpoints(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path, n_shards=4, samples_per_shard=3)
    checkpoint_dir = tmp_path / "ckpt"
    config = _small_config(manifest_path, checkpoint_dir)
    result = train(config)
    assert result["samples"]["train"] > 0
    assert result["samples"]["val"] > 0
    assert result["samples"]["test"] > 0
    assert result["test"] is not None
    assert "macro_f1" in result["test"]
    assert result["dataset_fingerprint"]
    assert (checkpoint_dir / "tcn-smoke.seed0.best.pt").is_file()
    assert (checkpoint_dir / "tcn-smoke.seed0.last.pt").is_file()
    assert (checkpoint_dir / "result.tcn-smoke.seed0.json").is_file()


def test_minute_tcn_evaluate_best_checkpoints_locked_test(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path, n_shards=4, samples_per_shard=3)
    checkpoint_dir = tmp_path / "ckpt"
    config = _small_config(manifest_path, checkpoint_dir)
    train(config)
    train(replace(config, seed=1))
    locked = evaluate_best_checkpoints(config, seeds=(0, 1))
    assert locked["mode"] == "best_checkpoint_locked_test"
    assert locked["seeds"] == [0, 1]
    assert len(locked["per_seed"]) == 2
    assert "aggregate" in locked
    assert "daily_rank_ic_mean" in locked["aggregate"]


def test_minute_tcn_load_config_unknown_yaml_key_rejected(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path, n_shards=4, samples_per_shard=3)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps({"manifest_path": str(manifest_path), "not_a_field": 1}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="未知字段"):
        load_config(["--config", str(config_path)])


def test_minute_tcn_train_uses_no_workers_with_shared_arrays(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path, n_shards=4, samples_per_shard=3)
    checkpoint_dir = tmp_path / "ckpt2"
    config = replace(
        _small_config(manifest_path, checkpoint_dir),
        num_workers=1,
        epochs=1,
    )
    result = train(config)
    assert result["samples"]["train"] > 0
