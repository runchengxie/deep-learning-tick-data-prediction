"""分钟 TCN 模型与分钟分片数据集测试。"""

import json

import numpy as np
import pytest
import torch

from ticknet.nextday.minute_tcn import (
    MinuteShardDataset,
    build_minute_tcn,
)
from ticknet.nextday.splits import WalkForwardSplit


def _write_minute_manifest(tmp_path, n_shards=2, samples_per_shard=3):
    manifest = {
        "format_version": 1,
        "dtype": "float32",
        "layout": "samples_time_features",
        "window_minutes": 8,
        "feature_count": 6,
        "shards": [],
        "samples": [],
    }
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    sample_index = 0
    for shard_index in range(n_shards):
        rows = np.stack(
            [
                np.random.RandomState(i).uniform(-1, 1, (8, 6)).astype(np.float32)
                for i in range(samples_per_shard)
            ]
        )
        path = shard_dir / f"part-{shard_index:05d}.npy"
        np.save(path, rows)
        import hashlib

        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["shards"].append(
            {
                "path": f"shards/part-{shard_index:05d}.npy",
                "samples": samples_per_shard,
                "bytes": path.stat().st_size,
                "sha256": checksum,
            }
        )
        for row in range(samples_per_shard):
            day = sample_index + 1
            manifest["samples"].append(
                {
                    "symbol": f"60000{sample_index}",
                    "trading_date": f"2024-01-{day:02d}",
                    "label_date": f"2024-01-{day + 1:02d}",
                    "label": sample_index % 3,
                    "raw_return": 0.01,
                    "target_return": 0.01 + 0.1 * (sample_index % 3),
                    "minutes": 8,
                    "shard": shard_index,
                    "row": row,
                }
            )
            sample_index += 1
    from ticknet.nextday.dataset import manifest_fingerprint

    manifest["dataset_fingerprint"] = manifest_fingerprint(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def test_minute_tcn_forward_and_backprop():
    model = build_minute_tcn(num_features=6, hidden_channels=16, num_layers=2)
    x = torch.randn(3, 8, 6)
    output = model(x)
    assert output.logits.shape == (3, 3)
    assert output.score.shape == (3,)
    (output.logits.sum() + output.score.sum()).backward()
    assert model.input_projection.weight.grad is not None


def test_minute_tcn_encode_sequence_takes_last_step():
    model = build_minute_tcn(num_features=6, hidden_channels=16, num_layers=2)
    x = torch.randn(2, 8, 6)
    assert model.encode_sequence(x).shape == (2, 16)


def test_minute_tcn_rejects_wrong_input_rank():
    model = build_minute_tcn(num_features=6)
    with pytest.raises(ValueError, match="输入应为"):
        model(torch.randn(2, 8, 6, 6))


def test_minute_shard_dataset_reads_and_splits(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path)
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-01-05",
        val_start="2024-01-06",
        val_end="2024-01-08",
        test_start="2024-01-09",
        test_end="2024-01-12",
    )
    dataset = MinuteShardDataset(
        manifest_path,
        date_split=split,
        split="train",
        verify_checksums=True,
    )
    assert len(dataset) > 0
    features, label, target_return = dataset[0]
    assert features.shape == (8, 6)
    assert features.dtype == np.float32
    assert label in (0, 1, 2)
    assert np.isfinite(target_return)


def test_minute_shard_dataset_verifies_checksums(tmp_path):
    manifest_path = _write_minute_manifest(tmp_path)
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-01",
        train_end="2024-01-05",
        val_start="2024-01-06",
        val_end="2024-01-08",
        test_start="2024-01-09",
        test_end="2024-01-12",
    )
    shard_path = tmp_path / "shards" / "part-00000.npy"
    array = np.load(shard_path)
    array[0, 0, 0] += 0.5
    np.save(shard_path, array)
    with pytest.raises(ValueError, match="sha256"):
        MinuteShardDataset(
            manifest_path,
            date_split=split,
            split="train",
            verify_checksums=True,
        )
