"""事件流与分钟特征联合微调测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import ticknet.eventstream.joint as joint_module
import ticknet.eventstream.joint_cache as joint_cache_module
from ticknet.eventstream.dataset import N_FEATURES
from ticknet.eventstream.fingerprint import file_sha256
from ticknet.eventstream.joint import (
    JointConfig,
    JointDataset,
    JointEventstreamModel,
    _ranking_metrics,
    load_joint_config,
    load_pretrained_backbone,
    train_joint,
)
from ticknet.eventstream.joint_cache import (
    MODE,
    SCHEMA_VERSION,
    _canonical_sha256,
    _manifest_payload,
    build_joint_cache,
    load_joint_cache_manifest,
)
from ticknet.eventstream.model import build_eventstream_model
from ticknet.nextday.embedding_comparison import ComparisonConfig
from ticknet.nextday.formal_targets import FORMAL_TARGET_RETURN_CONTRACT, FormalNextOpenTarget
from ticknet.nextday.minute_baseline import MinuteBaselineConfig, MinuteSample


def _fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
    )


def _cache_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    cache_root = tmp_path / "joint-cache"
    close_root = tmp_path / "close-cache"
    artifacts = []
    feature_count = 3
    specifications = {
        "train": ("train-202508", 20250801, (0.01, -0.01)),
        "val": ("validation-202511", 20251103, (0.02, -0.02)),
        "test": ("oos-202512", 20251201, (0.03, -0.03)),
    }
    for partition, (shard, day, returns) in specifications.items():
        close_shard = close_root / "shards" / shard
        close_shard.mkdir(parents=True)
        rng = np.random.default_rng(day)
        np.save(
            close_shard / "x.npy",
            rng.normal(size=(2, 8, N_FEATURES)).astype(np.float32),
        )
        np.save(close_shard / "sid.npy", np.ones((2, 8), dtype=np.int16))
        np.save(close_shard / "oid.npy", np.ones((2, 8), dtype=np.int16))
        features = np.asarray([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        table = pa.table(
            {
                "trading_day": pa.array([day, day], type=pa.int32()),
                "symbol": ["600000", "000001"],
                "label_date": pa.array([date(2025, 12, 2), date(2025, 12, 2)], type=pa.date32()),
                "return_end_date": pa.array(
                    [date(2025, 12, 3), date(2025, 12, 3)], type=pa.date32()
                ),
                "label": pa.array([2, 0], type=pa.int8()),
                "ranking_target_return": pa.array(returns, type=pa.float64()),
                "feature_available": [True, True],
                "minute_features": _fixed_list(features),
                "close_shard": [f"shards/{shard}", f"shards/{shard}"],
                "close_row": pa.array([0, 1], type=pa.int32()),
            }
        )
        path = cache_root / "shards" / f"{partition}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        artifacts.append(
            {
                "partition": partition,
                "path": path.relative_to(cache_root).as_posix(),
                "rows": 2,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    targets_path = cache_root / "portfolio-targets.parquet"
    target_days = [date(2025, 8, 1), date(2025, 11, 3), date(2025, 12, 1)]
    target_rows = [
        {
            "trading_date": trading_date,
            "symbol": symbol,
            "label_date": trading_date,
            "portfolio_return": value,
            "can_buy": True,
            "can_sell": True,
            "in_universe": True,
        }
        for trading_date in target_days
        for symbol, value in (("600000", 0.01), ("000001", -0.01))
    ]
    pq.write_table(pa.Table.from_pylist(target_rows), targets_path)
    target_record = {
        "partition": "portfolio_targets",
        "path": targets_path.name,
        "rows": len(target_rows),
        "bytes": targets_path.stat().st_size,
        "sha256": file_sha256(targets_path),
    }
    contract = {
        "feature_count": feature_count,
        "close_cache_fingerprint": "a" * 64,
        "comparison_config": {"oos_end": "2025-12-31"},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "complete",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "artifacts": artifacts,
        "portfolio_targets": target_record,
        "totals": {
            "rows": 6,
            "bytes": sum(int(row["bytes"]) for row in artifacts) + int(target_record["bytes"]),
            "partitions": {"train": 2, "val": 2, "test": 2},
        },
    }
    manifest["dataset_fingerprint"] = _canonical_sha256(_manifest_payload(manifest))
    (cache_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache_root, close_root, manifest


def test_joint_cache_dataset_model_and_metrics(tmp_path: Path) -> None:
    cache_root, close_root, manifest = _cache_fixture(tmp_path)

    assert (
        load_joint_cache_manifest(cache_root)["dataset_fingerprint"]
        == manifest["dataset_fingerprint"]
    )
    means = np.asarray([1.5, 4.0, 4.5], dtype=np.float32)
    scales = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    dataset = JointDataset(cache_root, close_root, "test", means, scales)
    batch = [value.unsqueeze(0) for value in dataset[0][:5]]
    config = JointConfig(
        model="smoke",
        batch_size=1,
        minute_hidden=8,
        fusion_hidden=8,
        min_symbols_per_day=2,
        top_ks=(1,),
        device="cpu",
        amp=False,
        num_workers=0,
    )
    model = JointEventstreamModel(config, feature_count=3)
    logits = model(batch[0], batch[1], batch[2], batch[3])
    torch.nn.functional.cross_entropy(logits, batch[4]).backward()

    assert logits.shape == (1, 3)
    assert model.eventstream.feat_proj[0].weight.grad is not None
    metrics = _ranking_metrics(dataset, np.asarray([1.0, -1.0]), config)
    assert metrics["summary"]["daily_rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["summary"]["precision_at_1"] == pytest.approx(1.0)


def test_joint_cache_detects_tamper(tmp_path: Path) -> None:
    cache_root, _close_root, _manifest = _cache_fixture(tmp_path)
    path = cache_root / "shards" / "val.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="大小漂移"):
        load_joint_cache_manifest(cache_root)


def test_joint_checkpoint_and_config_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "joint.yaml"
    config_path.write_text(
        "model: smoke\nseed: 0\nmin_symbols_per_day: 2\ntop_ks: [1]\ndevice: cpu\n",
        encoding="utf-8",
    )
    config = load_joint_config(config_path)
    model = JointEventstreamModel(config, feature_count=3)
    pretrained = build_eventstream_model("smoke")
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(
        {
            "model": pretrained.state_dict(),
            "epoch": 2,
            "best_selection_value": 0.1,
            "experiment": {
                "model": "smoke",
                "seed": 0,
                "source_revision": "abcdef123456",
            },
        },
        checkpoint,
    )
    sha256 = file_sha256(checkpoint)

    contract = load_pretrained_backbone(
        model,
        checkpoint,
        model_name="smoke",
        seed=0,
        expected_sha256=sha256,
    )

    assert contract["epoch"] == 2
    with pytest.raises(ValueError, match="SHA-256"):
        load_pretrained_backbone(
            model,
            checkpoint,
            model_name="smoke",
            seed=0,
            expected_sha256="0" * 64,
        )


def test_joint_cli_overrides_seed_and_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "joint.yaml"
    config_path.write_text(
        "model: smoke\nseed: 0\nepochs: 5\nmin_symbols_per_day: 2\ntop_ks: [1]\ndevice: cpu\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def capture(config: JointConfig, **kwargs: object) -> dict[str, object]:
        captured["config"] = config
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(joint_module, "train_joint", capture)
    joint_module.main(
        [
            "--config",
            str(config_path),
            "--cache",
            str(tmp_path / "cache"),
            "--close-cache",
            str(tmp_path / "close-cache"),
            "--pretrained-checkpoint",
            str(tmp_path / "pretrained.pt"),
            "--expected-pretrained-sha256",
            "a" * 64,
            "--output",
            str(tmp_path / "output"),
            "--source-revision",
            "abcdef123456",
            "--allow-oos",
            "--seed",
            "2",
            "--epochs",
            "3",
        ]
    )

    config = captured["config"]
    assert isinstance(config, JointConfig)
    assert config.seed == 2
    assert config.epochs == 3


def test_joint_training_runs_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, close_root, _manifest = _cache_fixture(tmp_path)
    monkeypatch.setattr(
        joint_module,
        "load_close_cache_manifest",
        lambda _root: {"dataset_fingerprint": "a" * 64},
    )
    monkeypatch.setattr(joint_module, "verify_close_cache", lambda _root: {})
    pretrained = build_eventstream_model("smoke")
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(
        {
            "model": pretrained.state_dict(),
            "epoch": 1,
            "best_selection_value": 0.1,
            "experiment": {
                "model": "smoke",
                "seed": 0,
                "source_revision": "abcdef123456",
            },
        },
        checkpoint,
    )
    sha256 = file_sha256(checkpoint)
    output = tmp_path / "output"
    base = JointConfig(
        model="smoke",
        seed=0,
        epochs=1,
        batch_size=2,
        backbone_lr=1e-4,
        head_lr=1e-3,
        patience=2,
        freeze_backbone_epochs=1,
        minute_hidden=8,
        fusion_hidden=8,
        dropout=0.0,
        num_workers=0,
        device="cpu",
        amp=False,
        min_symbols_per_day=2,
        top_ks=(1,),
    )

    first = train_joint(
        base,
        cache_root=cache_root,
        close_root=close_root,
        pretrained_checkpoint=checkpoint,
        expected_pretrained_sha256=sha256,
        output_root=output,
        source_revision="abcdef123456",
        allow_oos=True,
    )
    resumed = train_joint(
        replace(base, epochs=2),
        cache_root=cache_root,
        close_root=close_root,
        pretrained_checkpoint=checkpoint,
        expected_pretrained_sha256=sha256,
        output_root=output,
        source_revision="abcdef123456",
        allow_oos=True,
    )

    assert first["best_epoch"] == 1
    assert resumed["samples"] == {"train": 2, "val": 2, "test": 2}
    assert resumed["locked_status"] == "2026_not_accessed"
    assert (output / "predictions.eventstream-joint-smoke.seed0.test.parquet").is_file()


def test_joint_cache_builds_exact_frozen_e2_intersection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbols = ("600000", "000001")
    days = {
        "train": date(2025, 8, 1),
        "val": date(2025, 11, 3),
        "test": date(2025, 12, 1),
    }
    samples: list[MinuteSample] = []
    targets: list[FormalNextOpenTarget] = []
    locations: dict[tuple[date, str], tuple[str, str, int]] = {}
    close_partition = {"train": "train", "val": "validation", "test": "oos"}
    for partition, trading_date in days.items():
        for row, symbol in enumerate(symbols):
            label_date = date(2025, trading_date.month, trading_date.day + 1)
            return_end_date = date(2025, trading_date.month, trading_date.day + 2)
            target_return = 0.01 if row == 0 else -0.01
            samples.append(
                MinuteSample(
                    trading_date=trading_date,
                    symbol=symbol,
                    label_date=label_date,
                    label=2 if row == 0 else 0,
                    target_return=target_return,
                    features=np.asarray([row, 1.0, np.nan], dtype=np.float32),
                    return_end_date=return_end_date,
                    feature_available=True,
                )
            )
            targets.append(
                FormalNextOpenTarget(
                    symbol=symbol,
                    trading_date=trading_date,
                    label_date=label_date,
                    raw_return=target_return,
                    target_return=target_return,
                    label=2 if row == 0 else 0,
                    return_end_date=return_end_date,
                    portfolio_return=target_return,
                )
            )
            shard = f"shards/{close_partition[partition]}-2025{trading_date.month:02d}"
            locations[(trading_date, symbol)] = (close_partition[partition], shard, row)
    close_manifest = {
        "dataset_fingerprint": "a" * 64,
        "contract_sha256": "b" * 64,
        "shards": [],
    }
    monkeypatch.setattr(joint_cache_module, "verify_close_cache", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        joint_cache_module,
        "load_close_cache_manifest",
        lambda _root: close_manifest,
    )
    monkeypatch.setattr(
        joint_cache_module,
        "_close_locations",
        lambda _root, _manifest: locations,
    )
    monkeypatch.setattr(
        joint_cache_module,
        "build_target_bundle",
        lambda _config: SimpleNamespace(targets=targets),
    )
    monkeypatch.setattr(
        joint_cache_module,
        "load_materialized_minute_features",
        lambda *_args, **_kwargs: SimpleNamespace(
            samples=samples,
            materialization_identity="c" * 64,
            manifest_fingerprint="d" * 64,
        ),
    )
    minute_config = MinuteBaselineConfig(
        basic_root="unused",
        benchmark_path="unused",
        start_date="2025-08-01",
        end_date="2025-12-31",
        top_n=2,
        min_history_days=1,
        liquidity_lookback_days=1,
        min_liquidity_observations=1,
        lower_quantile=0.3,
        upper_quantile=0.7,
        min_cross_section=2,
        train_start="2025-08-01",
        train_end="2025-10-31",
        val_start="2025-11-01",
        val_end="2025-11-30",
        test_start="2025-12-01",
        test_end="2025-12-31",
        target_return_contract=FORMAL_TARGET_RETURN_CONTRACT,
    )
    comparison = ComparisonConfig(
        train_start="2025-08-01",
        train_end="2025-10-31",
        validation_start="2025-11-01",
        validation_end="2025-11-30",
        oos_start="2025-12-01",
        oos_end="2025-12-31",
        min_symbols_per_day=2,
        top_ks=(1,),
    )
    output = tmp_path / "built-cache"

    manifest = build_joint_cache(
        minute_config=minute_config,
        minute_features_root=tmp_path / "minute",
        comparison_config=comparison,
        close_cache_root=tmp_path / "close",
        output_root=output,
        source_revision="abcdef123456",
    )
    resumed = build_joint_cache(
        minute_config=minute_config,
        minute_features_root=tmp_path / "minute",
        comparison_config=comparison,
        close_cache_root=tmp_path / "close",
        output_root=output,
        source_revision="abcdef123456",
    )

    assert manifest["totals"]["partitions"] == {"train": 2, "val": 2, "test": 2}
    assert manifest["contract"]["feature_count"] == 3
    assert resumed["dataset_fingerprint"] == manifest["dataset_fingerprint"]
