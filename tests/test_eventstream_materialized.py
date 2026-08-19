"""事件流固定窗口物化、校验和训练读取。"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from ticknet.eventstream.config import day_pack_paths
from ticknet.eventstream.dataset import L2WindowDataset
from ticknet.eventstream.fingerprint import file_sha256
from ticknet.eventstream.gradient_audit import TASKS, run_gradient_audit
from ticknet.eventstream.materialized import (
    MaterializedWindowDataset,
    build_materialized_dataset,
    load_materialized_manifest,
    verify_materialized_dataset,
)
from ticknet.eventstream.materialized_predictions import (
    export_materialized_predictions,
    export_materialized_sample_keys,
)
from ticknet.eventstream.model import build_eventstream_model
from ticknet.eventstream.storage_readiness import build_storage_manifest
from ticknet.eventstream.train import EventstreamConfig, train
from ticknet.train import set_seed

SOURCE_REVISION = "abcdef1234567890"
DAYS = (20210104, 20210105, 20210106, 20210107)


def _prepare_formal_fixture(packed_day: dict, root: Path) -> tuple[EventstreamConfig, Path]:
    pack_root = root / "pack"
    pack_root.mkdir()
    source_paths = day_pack_paths(packed_day["day"], Path(packed_day["pack_root"]))
    for day in DAYS:
        for name, destination in day_pack_paths(day, pack_root).items():
            shutil.copy2(source_paths[name], destination)

    labels = [{"value": day, "600000": float(index + 1)} for index, day in enumerate(DAYS)]
    h5 = root / "h5.parquet"
    h3 = root / "h3.parquet"
    pq.write_table(pa.Table.from_pylist(labels), h5)
    pq.write_table(pa.Table.from_pylist(labels), h3)
    fold_manifest = root / "fold-manifest.json"
    fold_manifest.write_text('{"status": "complete"}', encoding="utf-8")

    config = EventstreamConfig(
        pack_root=str(pack_root),
        label_path=str(h5),
        monitor_label_path=str(h3),
        monitor_name="h3",
        train_start=DAYS[0],
        train_end=DAYS[1],
        val_start=DAYS[2],
        val_end=DAYS[2],
        test_start=DAYS[3],
        test_end=DAYS[3],
        model="smoke",
        seq_len=2,
        min_events=2,
        samples_per_day=2,
        eval_tickers=1,
        epochs=1,
        batch_size=2,
        patience=1,
        device="cpu",
        amp=False,
        resume=False,
        min_symbols_per_day=2,
    )
    storage_config = root / "storage-config.yaml"
    storage_config.write_text(
        yaml.safe_dump(
            {
                "train_start": config.train_start,
                "train_end": config.train_end,
                "val_start": config.val_start,
                "val_end": config.val_end,
                "test_start": config.test_start,
                "test_end": config.test_end,
            }
        ),
        encoding="utf-8",
    )
    universe = root / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "source_dataset_fingerprint": "a" * 64,
                "days": len(DAYS),
                "universes": {str(day): ["600000"] for day in DAYS},
            }
        ),
        encoding="utf-8",
    )
    storage = build_storage_manifest(
        config_path=storage_config,
        pack_root=pack_root,
        universe_paths=[universe],
        artifacts={
            "fold-labels/manifest.json": fold_manifest,
            "fold-labels/h3.parquet": h3,
            "fold-labels/h5.parquet": h5,
        },
    )
    storage_path = root / "storage-manifest.json"
    storage_path.write_text(json.dumps(storage), encoding="utf-8")
    return config, storage_path


@pytest.fixture
def materialized_fixture(packed_day: dict, tmp_path: Path) -> tuple[EventstreamConfig, Path]:
    config, storage_path = _prepare_formal_fixture(packed_day, tmp_path)
    output = tmp_path / "materialized"
    manifest = build_materialized_dataset(
        config,
        storage_manifest_path=storage_path,
        output_root=output,
        source_revision=SOURCE_REVISION,
    )
    assert manifest["status"] == "complete"
    assert manifest["contract"]["sampling_policy"] == "seeded_fixed_window_v1"
    return config, output


def test_materialized_samples_match_canonical_windows(
    materialized_fixture: tuple[EventstreamConfig, Path],
) -> None:
    config, output = materialized_fixture
    source = L2WindowDataset(
        list(DAYS[:2]),
        seq_len=config.seq_len,
        min_events=config.min_events,
        samples_per_day=config.samples_per_day,
        root=Path(config.pack_root),
        label_path=Path(config.label_path),
        seed=config.seed,
        fixed_windows=True,
    )
    materialized = MaterializedWindowDataset(output, "train")

    assert len(materialized) == len(source)
    assert all(start >= 0 for _day, _ticker, start in source.entries)
    for row in range(len(source)):
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(materialized[row], source[row], strict=True)
        )
    report = verify_materialized_dataset(output)
    assert report["samples"] == 8
    assert report["shards"] == 5


def test_gradient_audit_uses_fixed_validation_batches_and_checkpoint_identity(
    materialized_fixture: tuple[EventstreamConfig, Path],
    tmp_path: Path,
) -> None:
    config, output = materialized_fixture
    config.materialized_root = str(output)
    config.batch_size = 2
    config.device = "cpu"
    config.amp = False
    set_seed(config.seed)
    model = build_eventstream_model(config.model)
    checkpoint = tmp_path / "smoke.best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 1,
            "experiment": {
                "source_revision": SOURCE_REVISION,
                "dataset_fingerprint": load_materialized_manifest(output)["dataset_fingerprint"],
                "model": config.model,
                "seed": config.seed,
            },
        },
        checkpoint,
    )
    checkpoint_sha256 = file_sha256(checkpoint)

    result = run_gradient_audit(
        config,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=checkpoint_sha256,
        partition="validation",
        batches=2,
        batch_size=2,
        source_revision="audit123",
    )

    assert result["status"] == "complete"
    assert result["locked_status"] == "2026_not_accessed"
    assert result["materialized"]["preflight"]["partitions"] == ["validation"]
    assert len(result["materialized"]["batch_fingerprint"]) == 64
    assert result["checkpoint"]["sha256"] == checkpoint_sha256
    assert set(result["states"]) == {"initialization", "best_checkpoint"}
    assert result["states"]["initialization"] == result["states"]["best_checkpoint"]
    for state in result["states"].values():
        assert set(state["summary"]["tasks"]) == set(TASKS)
        assert state["summary"]["batches"] == 1

    with pytest.raises(ValueError, match="SHA-256 不匹配"):
        run_gradient_audit(
            config,
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            partition="validation",
            batches=1,
            batch_size=2,
            source_revision="audit123",
        )

    wrong_identity = tmp_path / "wrong-identity.best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["experiment"]["dataset_fingerprint"] = "0" * 64
    torch.save(state, wrong_identity)
    with pytest.raises(ValueError, match="实验身份不匹配"):
        run_gradient_audit(
            config,
            checkpoint_path=wrong_identity,
            expected_checkpoint_sha256=file_sha256(wrong_identity),
            partition="validation",
            batches=1,
            batch_size=2,
            source_revision="audit123",
        )


def test_materialization_resume_and_tamper_detection(
    materialized_fixture: tuple[EventstreamConfig, Path],
    tmp_path: Path,
) -> None:
    config, output = materialized_fixture
    resumed = build_materialized_dataset(
        config,
        storage_manifest_path=tmp_path / "storage-manifest.json",
        output_root=output,
        source_revision=SOURCE_REVISION,
    )
    assert resumed["status"] == "complete"

    path = next((output / "shards").glob("train-*/x.npy"))
    with path.open("r+b") as file:
        file.seek(-1, 2)
        byte = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="内容漂移"):
        verify_materialized_dataset(output)


def test_partial_preflight_does_not_read_oos(
    materialized_fixture: tuple[EventstreamConfig, Path],
) -> None:
    _config, output = materialized_fixture
    oos_path = next((output / "shards").glob("oos-*/x.npy"))
    with oos_path.open("r+b") as file:
        file.seek(-1, 2)
        byte = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([byte[0] ^ 1]))

    report = verify_materialized_dataset(
        output,
        partitions=("train", "validation", "monitor_validation"),
    )
    assert report["partitions"] == ["train", "validation", "monitor_validation"]
    with pytest.raises(ValueError, match="内容漂移"):
        verify_materialized_dataset(output)


def test_materialization_registers_atomic_orphan_shard(
    packed_day: dict,
    tmp_path: Path,
) -> None:
    config, storage_path = _prepare_formal_fixture(packed_day, tmp_path)
    complete = tmp_path / "complete"
    output = tmp_path / "resumed"
    build_materialized_dataset(
        config,
        storage_manifest_path=storage_path,
        output_root=complete,
        source_revision=SOURCE_REVISION,
    )
    shutil.copytree(complete, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    orphan = manifest["shards"].pop(0)
    manifest["status"] = "in_progress"
    manifest.pop("dataset_fingerprint")
    manifest["totals"] = {
        "shards": len(manifest["shards"]),
        "samples": sum(row["samples"] for row in manifest["shards"]),
        "bytes": sum(item["bytes"] for row in manifest["shards"] for item in row["files"]),
        "partitions": {
            partition: sum(
                row["samples"] for row in manifest["shards"] if row["partition"] == partition
            )
            for partition in ("train", "validation", "oos", "monitor_validation", "monitor_oos")
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = build_materialized_dataset(
        config,
        storage_manifest_path=storage_path,
        output_root=output,
        source_revision=SOURCE_REVISION,
    )

    assert resumed["status"] == "complete"
    assert any(
        row["partition"] == orphan["partition"] and row["month"] == orphan["month"]
        for row in resumed["shards"]
    )
    verify_materialized_dataset(output)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_materialization_supports_multiple_workers(
    packed_day: dict,
    tmp_path: Path,
) -> None:
    config, storage_path = _prepare_formal_fixture(packed_day, tmp_path)
    config.num_workers = 2
    manifest = build_materialized_dataset(
        config,
        storage_manifest_path=storage_path,
        output_root=tmp_path / "workers",
        source_revision=SOURCE_REVISION,
    )

    assert manifest["status"] == "complete"
    assert manifest["totals"]["samples"] == 8


def test_materialized_prediction_export_recovers_symbols_and_scores(
    materialized_fixture: tuple[EventstreamConfig, Path],
    tmp_path: Path,
) -> None:
    config, output = materialized_fixture
    with pytest.raises(ValueError, match="显式"):
        export_materialized_sample_keys(
            config=config,
            storage_manifest_path=tmp_path / "storage-manifest.json",
            materialized_root=output,
            output_dir=tmp_path / "keys-rejected",
        )

    key_report = export_materialized_sample_keys(
        config=config,
        storage_manifest_path=tmp_path / "storage-manifest.json",
        materialized_root=output,
        output_dir=tmp_path / "keys",
        allow_oos=True,
    )
    keys = pq.read_table(tmp_path / "keys" / "sample-keys.parquet").to_pylist()
    assert key_report["artifact"]["rows_by_partition"] == {"validation": 1, "oos": 1}
    assert [(row["partition"], row["trading_day"], row["symbol"]) for row in keys] == [
        ("validation", DAYS[2], "600000"),
        ("oos", DAYS[3], "600000"),
    ]

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = tmp_path / "smoke.best.pt"
    torch.save(
        {
            "epoch": 1,
            "model": build_eventstream_model("smoke").state_dict(),
            "experiment": {
                "model": "smoke",
                "source_revision": SOURCE_REVISION,
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
        },
        checkpoint,
    )
    prediction_report = export_materialized_predictions(
        checkpoint_path=checkpoint,
        materialized_root=output,
        model_name="smoke",
        output_dir=tmp_path / "predictions",
        batch_size=2,
        partitions=("validation", "oos"),
        allow_oos=True,
        source_revision="fedcba9876543210",
    )
    assert prediction_report["totals"]["rows"] == 2
    for partition, expected_day in (("validation", DAYS[2]), ("oos", DAYS[3])):
        rows = pq.read_table(tmp_path / "predictions" / f"{partition}.parquet").to_pylist()
        assert len(rows) == 1
        assert rows[0]["row_index"] == 0
        assert rows[0]["trading_day"] == expected_day
        assert math.isfinite(rows[0]["score"])


def test_materialized_prediction_export_rejects_checkpoint_fingerprint(
    materialized_fixture: tuple[EventstreamConfig, Path],
    tmp_path: Path,
) -> None:
    _config, output = materialized_fixture
    checkpoint = tmp_path / "wrong.best.pt"
    torch.save(
        {
            "epoch": 1,
            "model": build_eventstream_model("smoke").state_dict(),
            "experiment": {
                "model": "smoke",
                "source_revision": SOURCE_REVISION,
                "dataset_fingerprint": "wrong",
            },
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="数据指纹"):
        export_materialized_predictions(
            checkpoint_path=checkpoint,
            materialized_root=output,
            model_name="smoke",
            output_dir=tmp_path / "predictions",
            partitions=("validation",),
            source_revision="fedcba9876543210",
        )


def test_training_reads_materialized_windows_without_oos(
    materialized_fixture: tuple[EventstreamConfig, Path],
    tmp_path: Path,
) -> None:
    source, output = materialized_fixture
    config = EventstreamConfig(
        materialized_root=str(output),
        monitor_name="h3",
        train_start=source.train_start,
        train_end=source.train_end,
        val_start=source.val_start,
        val_end=source.val_end,
        test_start=source.test_start,
        test_end=source.test_end,
        model="smoke",
        seq_len=source.seq_len,
        min_events=source.min_events,
        samples_per_day=source.samples_per_day,
        eval_tickers=source.eval_tickers,
        evaluate_test=False,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        patience=1,
        seed=source.seed,
        device="cpu",
        amp=False,
        resume=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_name="materialized-smoke",
        source_revision=SOURCE_REVISION,
        min_symbols_per_day=2,
    )

    with pytest.raises(ValueError, match="参数量不匹配"):
        train(config, expected_parameter_count=1)

    result = train(config)

    assert result["test_status"] == "not_evaluated"
    assert result["test"] is None
    assert result["parameter_count"] > 0
    assert result["samples"] == {"train": 4, "val": 1, "test": 0}
    assert result["dataset_fingerprint"]

    resumed_values = config.to_dict()
    resumed_values.update({"epochs": 2, "resume": True, "evaluate_test": True})
    resumed = train(EventstreamConfig.from_mapping(resumed_values))
    history_path = Path(config.checkpoint_dir) / "train_history.materialized-smoke.seed0.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(history) == 2
    assert resumed["test_status"] == "evaluated"
    assert resumed["samples"]["test"] == 1
