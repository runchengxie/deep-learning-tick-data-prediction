"""事件流训练：配置校验、单 epoch 冒烟、resume 签名冲突。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from ticknet.eventstream.config import day_pack_paths
from ticknet.eventstream.fingerprint import dataset_fingerprint
from ticknet.eventstream.train import (
    EventstreamConfig,
    _checkpoint_matches_experiment,
    _experiment_signature,
    list_packed_days,
    train,
)


def _smoke_config(packed_day: dict, checkpoint_dir: Path) -> EventstreamConfig:
    return EventstreamConfig(
        pack_root=str(packed_day["pack_root"]),
        label_path=str(packed_day["label_path"]),
        days=(packed_day["day"],),
        model="smoke",
        seq_len=8,
        min_events=2,
        samples_per_day=8,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        patience=1,
        device="cpu",
        amp=False,
        resume=False,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="smoke",
        val_start=packed_day["day"],
        val_end=packed_day["day"],
        eval_tickers=1,
        min_symbols_per_day=2,
    )


class TestConfig:
    def test_validate_and_round_trip(self, tmp_path):
        cfg = _smoke_config(
            {"pack_root": str(tmp_path), "label_path": str(tmp_path), "day": 20210104},
            tmp_path / "ckpt",
        )
        cfg.validate()
        yaml_text = yaml.safe_dump(cfg.to_dict())
        loaded = EventstreamConfig.from_mapping(yaml.safe_load(yaml_text))
        assert loaded.to_dict() == cfg.to_dict()
        assert loaded.day_supervision_mode == "all"

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="未知字段"):
            EventstreamConfig.from_mapping({"mystery": 1})

    def test_rejects_bad_model(self):
        with pytest.raises(ValueError, match="model"):
            EventstreamConfig(model="nope").validate()

    def test_requires_days_or_range(self):
        with pytest.raises(ValueError, match="days"):
            EventstreamConfig().validate()

    def test_monitor_label_and_name_are_paired(self):
        with pytest.raises(ValueError, match="必须提供"):
            EventstreamConfig(
                days=(20210104,), monitor_label_path="h3.parquet", monitor_name=""
            ).validate()

    def test_target_overlay_requires_materialized_dataset(self):
        with pytest.raises(ValueError, match="materialized_root"):
            EventstreamConfig(days=(20210104,), target_overlay_root="target-overlay").validate()

    def test_rejects_unknown_day_supervision_mode(self):
        with pytest.raises(ValueError, match="day_supervision_mode"):
            EventstreamConfig(days=(20210104,), day_supervision_mode="middle").validate()


class TestTrain:
    def test_smoke_train_writes_checkpoints(self, packed_day, tmp_path):
        cfg = _smoke_config(packed_day, tmp_path / "ckpt")
        cfg.monitor_label_path = cfg.label_path
        cfg.monitor_name = "h3"
        result = train(cfg)
        ckpt = tmp_path / "ckpt"
        assert (ckpt / "smoke.seed0.best.pt").exists()
        assert (ckpt / "smoke.seed0.last.pt").exists()
        assert (ckpt / "train_history.smoke.seed0.json").exists()
        assert result["mode"] == "eventstream_train"
        assert result["dataset_fingerprint"]
        assert "val" in result
        assert "test" in result
        assert result["monitor"]["name"] == "h3"

    def test_resume_rejects_different_signature(self, packed_day, tmp_path):
        cfg = _smoke_config(packed_day, tmp_path / "ckpt")
        train(cfg)
        # 修改训练区间使签名不一致后 resume 必须拒绝
        other = EventstreamConfig(
            pack_root=cfg.pack_root,
            label_path=cfg.label_path,
            days=(packed_day["day"],),
            model="smoke",
            seq_len=8,
            min_events=2,
            epochs=2,
            batch_size=2,
            lr=2e-3,
            patience=1,
            device="cpu",
            amp=False,
            resume=True,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name="smoke",
            min_symbols_per_day=2,
        )
        with pytest.raises(ValueError, match="实验配置与本次运行不同"):
            train(other)


class TestFingerprint:
    def test_deterministic_and_sensitive(self, packed_day, tmp_path):
        my_pack = tmp_path / "pack"
        shutil.copytree(packed_day["pack_root"], my_pack)
        fp1 = dataset_fingerprint([packed_day["day"]], root=my_pack)
        assert fp1 == dataset_fingerprint([packed_day["day"]], root=my_pack)
        paths = day_pack_paths(packed_day["day"], my_pack)
        paths["order"].write_bytes(b"tampered")
        assert dataset_fingerprint([packed_day["day"]], root=my_pack) != fp1

    def test_label_change_flips_fingerprint(self, packed_day, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        labels = tmp_path / "labels.parquet"
        pq.write_table(pa.Table.from_pylist([{"value": 20210104, "600000": 0.7}]), labels)
        fp_a = dataset_fingerprint(
            [packed_day["day"]], root=packed_day["pack_root"], label_path=packed_day["label_path"]
        )
        fp_b = dataset_fingerprint(
            [packed_day["day"]], root=packed_day["pack_root"], label_path=labels
        )
        assert fp_a != fp_b

    def test_signature_includes_fingerprint(self, packed_day):
        cfg = _smoke_config(packed_day, Path("/tmp/ckpt"))
        sig = _experiment_signature(cfg, "fp-123")
        assert sig["dataset_fingerprint"] == "fp-123"
        assert "monitor_label_path" not in sig
        assert "monitor_label_fingerprint" not in sig
        assert "target_overlay_root" not in sig
        assert "materialized_source_revision" not in sig
        assert "epochs" not in sig
        assert "device" not in sig
        assert sig["day_supervision_mode"] == "all"
        assert sig["day_supervision_weight_version"] == "linear-v1"

    def test_checkpoint_signature_binds_supervision_mode(self, packed_day):
        cfg = _smoke_config(packed_day, Path("/tmp/ckpt"))
        all_signature = _experiment_signature(cfg, "fp-123")
        cfg.day_supervision_mode = "last"
        last_signature = _experiment_signature(cfg, "fp-123")

        assert not _checkpoint_matches_experiment(
            {"experiment": all_signature},
            last_signature,
        )

    def test_legacy_all_checkpoint_signature_remains_compatible(self, packed_day):
        cfg = _smoke_config(packed_day, Path("/tmp/ckpt"))
        expected = _experiment_signature(cfg, "fp-123")
        legacy = dict(expected)
        legacy.pop("day_supervision_mode")
        legacy.pop("day_supervision_weight_version")

        assert _checkpoint_matches_experiment({"experiment": legacy}, expected)


def test_list_packed_days(packed_day):
    days = list_packed_days(20210101, 20210131, Path(packed_day["pack_root"]))
    assert days == [20210104]
