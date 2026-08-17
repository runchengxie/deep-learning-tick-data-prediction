"""尾盘窗口缓存与冻结事件流 embedding 测试。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from ticknet.eventstream.close_cache import build_close_cache, verify_close_cache
from ticknet.eventstream.config import day_pack_paths
from ticknet.eventstream.embedding import export_frozen_embeddings, load_embedding_manifest
from ticknet.eventstream.materialized import build_materialized_dataset
from ticknet.eventstream.model import build_eventstream_model
from ticknet.eventstream.storage_readiness import build_storage_manifest
from ticknet.eventstream.train import EventstreamConfig

SOURCE_REVISION = "abcdef1234567890"
DAYS = (20210104, 20210105, 20210106, 20210107)


def _fixture(packed_day: dict, root: Path) -> tuple[EventstreamConfig, Path]:
    pack_root = root / "pack"
    pack_root.mkdir()
    source = day_pack_paths(packed_day["day"], Path(packed_day["pack_root"]))
    for day in DAYS:
        for name, destination in day_pack_paths(day, pack_root).items():
            shutil.copy2(source[name], destination)
    labels = [{"value": day, "600000": float(index)} for index, day in enumerate(DAYS)]
    h5 = root / "h5.parquet"
    h3 = root / "h3.parquet"
    pq.write_table(pa.Table.from_pylist(labels), h5)
    pq.write_table(pa.Table.from_pylist(labels), h3)
    fold_manifest = root / "fold-manifest.json"
    fold_manifest.write_text('{"status":"complete"}', encoding="utf-8")
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
        samples_per_day=1,
        eval_tickers=0,
        epochs=1,
        batch_size=2,
        device="cpu",
        amp=False,
        resume=False,
        min_symbols_per_day=2,
    )
    storage_config = root / "storage.yaml"
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


def test_close_cache_and_frozen_embedding_round_trip(packed_day: dict, tmp_path: Path) -> None:
    config, storage_path = _fixture(packed_day, tmp_path)
    training_root = tmp_path / "training-cache"
    training_manifest = build_materialized_dataset(
        config,
        storage_manifest_path=storage_path,
        output_root=training_root,
        source_revision=SOURCE_REVISION,
    )
    close_root = tmp_path / "close-cache"
    close_manifest = build_close_cache(
        storage_manifest_path=storage_path,
        pack_root=Path(config.pack_root),
        output_root=close_root,
        seq_len=config.seq_len,
        min_events=config.min_events,
        batch_size=2,
        num_workers=0,
        source_revision=SOURCE_REVISION,
    )
    assert close_manifest["totals"]["samples"] == 4
    assert verify_close_cache(close_root)["samples"] == 4

    model = build_eventstream_model("smoke")
    checkpoint = tmp_path / "smoke.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 1,
            "best_selection_value": 0.1,
            "experiment": {
                "model": "smoke",
                "seed": 0,
                "seq_len": config.seq_len,
                "min_events": config.min_events,
                "train_start": config.train_start,
                "train_end": config.train_end,
                "val_start": config.val_start,
                "val_end": config.val_end,
                "test_start": config.test_start,
                "test_end": config.test_end,
                "source_revision": SOURCE_REVISION,
                "dataset_fingerprint": training_manifest["dataset_fingerprint"],
            },
        },
        checkpoint,
    )
    output = tmp_path / "embeddings"
    manifest = export_frozen_embeddings(
        close_cache_root=close_root,
        checkpoint_path=checkpoint,
        training_manifest_root=training_root,
        model_name="smoke",
        output_root=output,
        device="cpu",
        batch_size=2,
        num_workers=0,
        allow_oos=True,
        source_revision=SOURCE_REVISION,
    )

    assert manifest["totals"]["rows"] == 4
    assert manifest["contract"]["encoder"]["embedding_dimension"] == 64
    loaded = load_embedding_manifest(output)
    assert loaded["dataset_fingerprint"] == manifest["dataset_fingerprint"]
