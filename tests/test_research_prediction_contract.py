"""正式 prediction artifact 契约测试。"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.research.prediction_contract import (
    PredictionContractError,
    attach_formal_prediction_metadata,
    validate_formal_prediction_artifact,
)


def _write_formal_predictions(
    path: Path,
    *,
    fingerprint: str = "dataset-fingerprint",
    candidate_count: int = 2,
) -> None:
    rows = []
    for offset, (trading_date, label_date) in enumerate(
        [("2025-01-02", "2025-01-03"), ("2025-01-03", "2025-01-06")]
    ):
        for index in range(candidate_count):
            rows.append(
                {
                    "symbol": f"{600000 + index:06d}",
                    "trading_date": trading_date,
                    "label_date": label_date,
                    "target_return": 0.01 * (index + 1),
                    "score": float(candidate_count - index),
                    "can_buy": index != 0,
                    "can_sell": True,
                    "in_universe": True,
                }
            )
        rows.append(
            {
                "symbol": f"STATUS{offset}",
                "trading_date": trading_date,
                "label_date": label_date,
                "target_return": -0.01,
                "score": -100.0,
                "can_buy": False,
                "can_sell": offset == 0,
                "in_universe": False,
            }
        )
    table = attach_formal_prediction_metadata(
        pa.Table.from_pylist(rows), dataset_fingerprint=fingerprint
    )
    pq.write_table(table, path)


def test_formal_prediction_contract_accepts_candidates_and_status_rows(tmp_path) -> None:
    path = tmp_path / "formal.parquet"
    _write_formal_predictions(path)
    report = validate_formal_prediction_artifact(
        path,
        expected_universe_size=2,
        expected_dataset_fingerprint="dataset-fingerprint",
    )
    assert report.row_count == 6
    assert report.candidate_row_count == 4
    assert report.status_only_row_count == 2
    assert report.label_date_count == 2
    assert report.candidate_count_min == report.candidate_count_max == 2
    assert report.cannot_buy_count == 4
    assert report.cannot_sell_count == 1


def test_formal_prediction_contract_rejects_missing_metadata(tmp_path) -> None:
    path = tmp_path / "missing-metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "symbol": ["600000"],
                "trading_date": ["2025-01-02"],
                "label_date": ["2025-01-03"],
                "target_return": [0.01],
                "score": [1.0],
                "can_buy": [True],
                "can_sell": [True],
                "in_universe": [True],
            }
        ),
        path,
    )
    with pytest.raises(PredictionContractError, match="metadata 缺少"):
        validate_formal_prediction_artifact(path, expected_universe_size=1)


def test_formal_prediction_contract_rejects_wrong_universe_size(tmp_path) -> None:
    path = tmp_path / "wrong-size.parquet"
    _write_formal_predictions(path)
    with pytest.raises(PredictionContractError, match="恰有 3 个"):
        validate_formal_prediction_artifact(path, expected_universe_size=3)


def test_formal_prediction_contract_binds_dataset_fingerprint(tmp_path) -> None:
    path = tmp_path / "wrong-fingerprint.parquet"
    _write_formal_predictions(path, fingerprint="actual")
    with pytest.raises(PredictionContractError, match="与 Registry 不一致"):
        validate_formal_prediction_artifact(
            path,
            expected_universe_size=2,
            expected_dataset_fingerprint="expected",
        )


def test_formal_prediction_contract_rejects_duplicate_stock_day(tmp_path) -> None:
    path = tmp_path / "duplicate.parquet"
    _write_formal_predictions(path)
    table = pq.read_table(path)
    duplicate = pa.concat_tables([table, table.slice(0, 1)])
    pq.write_table(duplicate, path)
    with pytest.raises(PredictionContractError, match="重复股票日"):
        validate_formal_prediction_artifact(path, expected_universe_size=2)
