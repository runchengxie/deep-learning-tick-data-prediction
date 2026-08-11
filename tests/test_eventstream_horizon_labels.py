"""eventstream H3/H5 标签适配与 fold 边界 purge。"""

from __future__ import annotations

import json
from datetime import date

import pyarrow.parquet as pq
import pytest

from ticknet.eventstream.horizon_labels import prepare_eventstream_fold_labels
from ticknet.nextday.dataset import manifest_fingerprint
from ticknet.nextday.horizon_labels import HorizonTarget, write_horizon_sidecar
from ticknet.nextday.splits import WalkForwardSplit


def _target(
    *,
    trading: date,
    entry: date,
    end: date,
    horizon: int,
    symbol: str = "600000",
) -> HorizonTarget:
    return HorizonTarget(
        symbol=symbol,
        trading_date=trading,
        entry_date=entry,
        return_end_date=end,
        horizon=horizon,
        label=2,
        raw_return=0.03,
        benchmark_return=0.01,
        target_return=0.02,
    )


def test_prepare_fold_labels_purges_cross_boundary_targets(tmp_path) -> None:
    manifest = {"samples": [{"symbol": "600000", "trading_date": "2021-01-02"}]}
    manifest["dataset_fingerprint"] = manifest_fingerprint(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    targets = []
    for horizon in (3, 5):
        targets.extend(
            [
                _target(
                    trading=date(2021, 1, 2),
                    entry=date(2021, 1, 3),
                    end=date(2021, 1, 5),
                    horizon=horizon,
                ),
                _target(
                    trading=date(2021, 1, 8),
                    entry=date(2021, 1, 9),
                    end=date(2021, 1, 12),
                    horizon=horizon,
                ),
                _target(
                    trading=date(2021, 1, 11),
                    entry=date(2021, 1, 12),
                    end=date(2021, 1, 15),
                    horizon=horizon,
                ),
                _target(
                    trading=date(2021, 1, 16),
                    entry=date(2021, 1, 17),
                    end=date(2021, 1, 20),
                    horizon=horizon,
                ),
            ]
        )
    sidecar = write_horizon_sidecar(
        targets,
        tmp_path / "sidecar",
        source_dataset_fingerprint=manifest["dataset_fingerprint"],
        lower_quantile=0.2,
        upper_quantile=0.8,
        min_cross_section=2,
    )
    split = WalkForwardSplit.from_strings(
        train_start="2021-01-01",
        train_end="2021-01-10",
        val_start="2021-01-11",
        val_end="2021-01-15",
        test_start="2021-01-16",
        test_end="2021-01-20",
    )

    report_path = prepare_eventstream_fold_labels(
        sidecar_manifest=sidecar,
        feature_manifest=manifest_path,
        output_dir=tmp_path / "fold",
        split=split,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["purge_contract"] == "signal_entry_return_end_must_share_split"
    assert report["artifacts"]["5"]["accepted_by_split"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert report["artifacts"]["5"]["purged_by_split"]["train"] == 1
    table = pq.read_table(tmp_path / "fold" / "h5.parquet")
    assert table["value"].to_pylist() == [20210102, 20210111, 20210116]
    assert table["600000"].to_pylist() == pytest.approx([0.02, 0.02, 0.02])
