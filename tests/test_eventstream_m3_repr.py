"""M3-inspired 事件流表征的因果性、兼容性与配置测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ticknet.eventstream import dataset as eventstream_dataset
from ticknet.eventstream.config import ORDER_DTYPE, SNAP_DTYPE, TRADE_DTYPE
from ticknet.eventstream.materialized import assert_materialized_compatible
from ticknet.eventstream.train import (
    DAY_SUPERVISION_WEIGHT_VERSION,
    EventstreamConfig,
    _checkpoint_matches_experiment,
)

DAY = 20210104


def _snapshots() -> np.ndarray:
    snaps = np.zeros(2, dtype=SNAP_DTYPE)
    for row, (time_ms, last, bid1, ask1) in enumerate(
        ((100, 1000, 999, 1001), (200, 1020, 1019, 1021))
    ):
        snaps[row]["time_ms"] = time_ms
        snaps[row]["last"] = last
        snaps[row]["total_bidvol"] = 1000
        snaps[row]["total_askvol"] = 1200
        snaps[row]["wbid"] = bid1
        snaps[row]["wask"] = ask1
        for level in range(10):
            snaps[row]["bid_px"][level] = bid1 - level
            snaps[row]["ask_px"][level] = ask1 + level
            snaps[row]["bid_vol"][level] = 100 * (level + 1)
            snaps[row]["ask_vol"][level] = 120 * (level + 1)
            snaps[row]["bid_cnt"][level] = level + 1
            snaps[row]["ask_cnt"][level] = level + 2
    return snaps


def test_lob_prefix_preserves_public_shapes_and_target_alignment(packed_day: dict) -> None:
    dataset = eventstream_dataset.L2WindowDataset(
        [int(packed_day["day"])],
        seq_len=4,
        min_events=2,
        samples_per_day=1,
        root=Path(packed_day["pack_root"]),
        label_path=Path(packed_day["label_path"]),
        eval_mode=True,
        use_lob_prefix=True,
        use_session_anchors=True,
    )

    x, sid, oid, tgt_sid, _tgt_oid, tgt_reg, _tgt_day, _day_valid, valid, _day = dataset[0]

    assert x.shape == (4, 80)
    assert sid.shape == oid.shape == tgt_sid.shape == valid.shape == (4,)
    assert int(sid[0]) == 0
    assert int(oid[0]) == eventstream_dataset.ORDER_TYPE_LOB_PREFIX == 11
    assert valid.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert int(tgt_sid[0]) == int(sid[1])
    assert tgt_reg[0, 0].item() == pytest.approx(x[1, 1].item())
    assert tgt_reg[0, 1].item() == pytest.approx(x[1, 0].item())
    assert x[0, 8].item() == 1.0


def test_lob_prefix_uses_strictly_prior_snapshot_and_fixed_causal_anchor() -> None:
    orders = np.zeros(0, dtype=ORDER_DTYPE)
    trades = np.zeros(1, dtype=TRADE_DTYPE)
    trades[0]["time_ms"] = 150
    trades[0]["price"] = 1010
    snaps = _snapshots()

    before_market_state = eventstream_dataset._lob_prefix_features(
        orders,
        trades,
        snaps,
        positions=(0, 0, 0),
        prev_close_cent=980.0,
        use_session_anchors=True,
    )
    assert before_market_state[7] == 0.0
    assert before_market_state[8] == 0.0

    after_first_snapshot = eventstream_dataset._lob_prefix_features(
        orders,
        trades,
        snaps,
        positions=(0, 0, 1),
        prev_close_cent=980.0,
        use_session_anchors=True,
    )
    assert after_first_snapshot[7] == 1.0
    assert after_first_snapshot[8] == 1.0
    assert after_first_snapshot[5] == pytest.approx(0.0, abs=1e-7)

    after_trade_and_second_snapshot = eventstream_dataset._lob_prefix_features(
        orders,
        trades,
        snaps,
        positions=(0, 1, 2),
        prev_close_cent=980.0,
        use_session_anchors=True,
    )
    expected_session = (1020.0 / 1000.0 - 1.0) * 1e4 / eventstream_dataset.BPS_SCALE
    expected_previous_close = (1020.0 / 980.0 - 1.0) * 1e4 / eventstream_dataset.BPS_SCALE
    assert after_trade_and_second_snapshot[5] == pytest.approx(expected_session)
    assert after_trade_and_second_snapshot[6] == pytest.approx(expected_previous_close)
    assert after_trade_and_second_snapshot[9] == pytest.approx(
        (1021.0 - 1019.0) / 1020.0 * 1e4 / eventstream_dataset.BPS_SCALE
    )


def test_session_anchors_require_lob_prefix() -> None:
    config = EventstreamConfig(days=(DAY,), use_session_anchors=True)

    with pytest.raises(ValueError, match="use_session_anchors"):
        config.validate()


def test_materialized_contract_rejects_representation_mismatch() -> None:
    config = EventstreamConfig(
        days=(DAY,),
        train_start=20210101,
        train_end=20210102,
        val_start=20210103,
        val_end=20210103,
        test_start=20210104,
        test_end=20210104,
        seq_len=4,
        min_events=2,
        samples_per_day=2,
        eval_tickers=1,
        monitor_name="h3",
        use_lob_prefix=True,
        use_session_anchors=True,
    )
    manifest = {
        "contract": {
            "seed": config.seed,
            "seq_len": config.seq_len,
            "min_events": config.min_events,
            "samples_per_day": config.samples_per_day,
            "eval_tickers": config.eval_tickers,
            "monitor_name": config.monitor_name,
            "use_lob_prefix": False,
            "use_session_anchors": False,
            "ranges": {
                "train": {"start": config.train_start, "end": config.train_end},
                "validation": {"start": config.val_start, "end": config.val_end},
                "oos": {"start": config.test_start, "end": config.test_end},
            },
        }
    }

    with pytest.raises(ValueError, match="use_lob_prefix"):
        assert_materialized_compatible(manifest, config)


def test_legacy_checkpoint_identity_gets_new_disabled_defaults() -> None:
    checkpoint = {
        "experiment": {
            "model": "smoke",
            "day_supervision_mode": "all",
            "day_supervision_weight_version": DAY_SUPERVISION_WEIGHT_VERSION,
        }
    }
    expected = {
        "model": "smoke",
        "day_supervision_mode": "all",
        "day_supervision_weight_version": DAY_SUPERVISION_WEIGHT_VERSION,
        "use_lob_prefix": False,
        "use_session_anchors": False,
        "use_vq": False,
        "vq_codebook_size": 1024,
        "vq_dim": 64,
        "vq_loss_weight": 0.25,
    }

    assert _checkpoint_matches_experiment(checkpoint, expected)
