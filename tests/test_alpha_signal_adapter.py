from __future__ import annotations

import pyarrow as pa
import pytest

from ticknet.research.alpha_signal_adapter import (
    ALPHA_SIGNAL_COLUMNS,
    build_alpha_signal_table,
)


def _formal_table() -> pa.Table:
    return pa.table(
        {
            "symbol": ["BBB", "AAA", "CCC", "DDD"],
            "trading_date": ["2026-01-05"] * 4,
            "label_date": ["2026-01-06"] * 4,
            "return_end_date": ["2026-01-07"] * 4,
            "target_return": [0.01, 0.02, -0.01, 0.03],
            "score": [0.5, 0.9, 0.1, 0.9],
            "can_buy": [True, True, False, True],
            "can_sell": [True, True, True, True],
            "in_universe": [True, True, True, False],
        }
    )


def test_build_alpha_signal_table_uses_prediction_date_and_deterministic_rank() -> None:
    signals = build_alpha_signal_table(
        _formal_table(),
        model_version="eventstream-v1",
        feature_set_id="l2-clean-v1",
    )

    assert signals.column_names == list(ALPHA_SIGNAL_COLUMNS)
    assert signals["signal_date"].to_pylist() == ["20260105"] * 4
    assert signals["symbol"].to_pylist() == ["AAA", "DDD", "BBB", "CCC"]
    assert signals["rank"].to_pylist() == [1, 2, 3, 4]
    assert signals["eligible_for_backtest"].to_pylist() == [True, False, True, False]
    assert signals["eligible_for_live"].to_pylist() == [False] * 4
    assert signals["model_version"].to_pylist() == ["eventstream-v1"] * 4


def test_build_alpha_signal_table_rejects_missing_formal_fields() -> None:
    table = _formal_table().drop(["label_date"])

    with pytest.raises(ValueError, match="label_date"):
        build_alpha_signal_table(table)
