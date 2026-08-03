"""次日预测分片写入和读取测试。"""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pytest

from deeplob.nextday.dataset import NextDayShardDataset
from deeplob.nextday.io import PreparedSample, pack_events, write_sharded_dataset
from deeplob.nextday.labels import NextDayTarget
from deeplob.nextday.splits import WalkForwardSplit


def _target(input_day: int, label_day: int, symbol: str = "000001.SZ"):
    return NextDayTarget(
        symbol=symbol,
        trading_date=date(2024, 1, input_day),
        label_date=date(2024, 1, label_day),
        raw_return=0.01,
        target_return=0.008,
        label=2,
    )


def _sample(input_day: int, label_day: int, *, rows: int = 7):
    target = _target(input_day, label_day)
    events = np.arange(rows * 40, dtype=np.float32).reshape(rows, 40)
    return PreparedSample(
        target=target,
        events=events,
        last_event_timestamp=datetime(2024, 1, input_day, 14, 54, 59),
        signal_timestamp=datetime(2024, 1, input_day, 14, 55),
    )


def _split():
    return WalkForwardSplit.from_strings(
        train_start="2024-01-02",
        train_end="2024-01-03",
        val_start="2024-01-04",
        val_end="2024-01-05",
        test_start="2024-01-06",
        test_end="2024-01-07",
    )


def test_pack_events_left_pads_and_keeps_latest_events():
    short = np.arange(3 * 40, dtype=np.float32).reshape(3, 40)
    packed, valid = pack_events(short, chunks_per_sample=2, chunk_size=2)
    assert packed.shape == (2, 2, 40)
    assert valid == 3
    assert np.array_equal(packed[0, 0], short[0])
    assert np.array_equal(packed.reshape(4, 40)[-3:], short)

    long = np.arange(6 * 40, dtype=np.float32).reshape(6, 40)
    packed, valid = pack_events(long, chunks_per_sample=2, chunk_size=2)
    assert valid == 4
    assert np.array_equal(packed.reshape(4, 40), long[-4:])


def test_dataset_reads_shards_and_purges_cross_boundary_samples(tmp_path):
    samples = [
        _sample(2, 3),
        _sample(3, 4),
        _sample(4, 5),
        _sample(6, 7),
    ]
    manifest = write_sharded_dataset(
        samples,
        tmp_path / "prepared",
        chunks_per_sample=2,
        chunk_size=3,
        samples_per_shard=2,
    )
    train = NextDayShardDataset(manifest, date_split=_split(), split="train")
    validation = NextDayShardDataset(manifest, date_split=_split(), split="val")
    test = NextDayShardDataset(manifest, date_split=_split(), split="test")

    features, label, target_return = train[0]
    assert features.shape == (2, 1, 3, 40)
    assert features.dtype == np.float32
    assert label == 2
    assert target_return == pytest.approx(0.008)
    assert train.purged_samples == 1
    assert len(train) == len(validation) == len(test) == 1
    assert train.label_dates == [date(2024, 1, 3)]
    assert train.symbols == ["000001.SZ"]
    assert train.target_returns.tolist() == pytest.approx([0.008])


def test_writer_rejects_events_after_signal(tmp_path):
    sample = _sample(2, 3)
    invalid = PreparedSample(
        target=sample.target,
        events=sample.events,
        last_event_timestamp=datetime(2024, 1, 2, 14, 56),
        signal_timestamp=sample.signal_timestamp,
    )
    with pytest.raises(ValueError, match="信号时点之后"):
        write_sharded_dataset([invalid], tmp_path)


def test_float16_storage_is_cast_back_and_can_return_continuous_target(tmp_path):
    manifest = write_sharded_dataset(
        [_sample(2, 3)],
        tmp_path,
        chunks_per_sample=1,
        chunk_size=4,
        storage_dtype="float16",
    )
    dataset = NextDayShardDataset(
        manifest,
        date_split=_split(),
        split="train",
    )
    features, label, target_return = dataset[0]
    assert dataset.storage_dtype == np.dtype("float16")
    assert features.dtype == np.float32
    assert label == 2
    assert target_return == pytest.approx(0.008)


def test_dataset_rejects_wrong_shard_shape(tmp_path):
    manifest_path = write_sharded_dataset(
        [_sample(2, 3)],
        tmp_path,
        chunks_per_sample=2,
        chunk_size=3,
    )
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / metadata["shards"][0]["path"]
    np.save(shard_path, np.zeros((1, 2, 2, 40), dtype=np.float32))
    with pytest.raises(ValueError, match="形状应为"):
        NextDayShardDataset(manifest_path, date_split=_split(), split="train")
