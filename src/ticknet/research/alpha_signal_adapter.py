"""将 formal prediction artifact 转为 alpha-research 的标准 signal 表。"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.research.prediction_contract import validate_formal_prediction_artifact

ALPHA_SIGNAL_COLUMNS = (
    "signal_date",
    "symbol",
    "raw_pred",
    "signal_eval",
    "signal_backtest",
    "signal_direction",
    "rank",
    "model_version",
    "feature_set_id",
    "eligible_for_backtest",
    "eligible_for_live",
)
_FORMAL_COLUMNS = {
    "symbol",
    "trading_date",
    "label_date",
    "return_end_date",
    "target_return",
    "score",
    "can_buy",
    "can_sell",
    "in_universe",
}


def _as_date(value: object, *, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{field} 不是 ISO 日期: {value}") from error


def _as_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} 必须为 boolean")
    return value


def build_alpha_signal_table(
    predictions: pa.Table,
    *,
    model_version: str = "unknown",
    feature_set_id: str = "unknown",
    eligible_for_live: bool = False,
) -> pa.Table:
    """转换为 alpha-research canonical signal 表。

    ``trading_date`` 是模型可见信息的日期，因此作为 ``signal_date``；
    ``label_date`` 和 ``return_end_date`` 只保留在源 prediction artifact 中。
    """
    missing = _FORMAL_COLUMNS - set(predictions.column_names)
    if missing:
        raise ValueError(f"formal prediction artifact 缺少字段: {sorted(missing)}")
    if predictions.num_rows == 0:
        raise ValueError("formal prediction artifact 不能为空")
    if not model_version.strip() or not feature_set_id.strip():
        raise ValueError("model_version 和 feature_set_id 不能为空")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(predictions.to_pylist()):
        trading_date = _as_date(row["trading_date"], field="trading_date")
        label_date = _as_date(row["label_date"], field="label_date")
        return_end_date = _as_date(row["return_end_date"], field="return_end_date")
        if label_date <= trading_date:
            raise ValueError(f"row {index} label_date 必须晚于 trading_date")
        if return_end_date <= label_date:
            raise ValueError(f"row {index} return_end_date 必须晚于 label_date")
        symbol = str(row["symbol"]).strip()
        if not symbol:
            raise ValueError(f"row {index} symbol 不能为空")
        score = float(row["score"])
        can_buy = _as_bool(row["can_buy"], field="can_buy")
        in_universe = _as_bool(row["in_universe"], field="in_universe")
        rows.append(
            {
                "signal_date": trading_date.strftime("%Y%m%d"),
                "symbol": symbol,
                "raw_pred": score,
                "signal_eval": score,
                "signal_backtest": score,
                "signal_direction": 1.0,
                "model_version": model_version,
                "feature_set_id": feature_set_id,
                "eligible_for_backtest": can_buy and in_universe,
                "eligible_for_live": eligible_for_live and can_buy and in_universe,
            }
        )

    rows.sort(key=lambda item: (item["signal_date"], -item["signal_backtest"], item["symbol"]))
    current_date: str | None = None
    rank = 0
    for row in rows:
        if row["signal_date"] != current_date:
            current_date = row["signal_date"]
            rank = 0
        rank += 1
        row["rank"] = rank

    schema = pa.schema(
        [
            pa.field("signal_date", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("raw_pred", pa.float64()),
            pa.field("signal_eval", pa.float64()),
            pa.field("signal_backtest", pa.float64()),
            pa.field("signal_direction", pa.float64()),
            pa.field("rank", pa.int64()),
            pa.field("model_version", pa.string()),
            pa.field("feature_set_id", pa.string()),
            pa.field("eligible_for_backtest", pa.bool_()),
            pa.field("eligible_for_live", pa.bool_()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    metadata = dict(predictions.schema.metadata or {})
    metadata.update(
        {
            b"ticknet.alpha_signal_contract": b"alpha_research.signals.v1",
            b"ticknet.alpha_signal_date_semantics": b"trading_date",
            b"ticknet.source_prediction_label_semantics": b"label_date_to_return_end_date",
        }
    )
    return table.replace_schema_metadata(metadata)


def export_alpha_signal_artifact(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = "unknown",
    feature_set_id: str = "unknown",
    eligible_for_live: bool = False,
    expected_universe_size: int = 400,
) -> dict[str, Any]:
    """校验 formal artifact 并写出 alpha-research signal parquet。"""
    report = validate_formal_prediction_artifact(
        predictions_path, expected_universe_size=expected_universe_size
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = build_alpha_signal_table(
        pq.read_table(predictions_path),
        model_version=model_version,
        feature_set_id=feature_set_id,
        eligible_for_live=eligible_for_live,
    )
    pq.write_table(table, output)
    return {
        "artifact_type": "alpha_research.signals",
        "schema_version": 1,
        "path": str(output.resolve()),
        "rows": table.num_rows,
        "source_prediction": report.to_dict(),
        "signal_date_semantics": "trading_date",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 alpha-research signal artifact")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--feature-set-id", default="unknown")
    parser.add_argument("--eligible-for-live", action="store_true")
    parser.add_argument("--expected-universe-size", type=int, default=400)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_alpha_signal_artifact(
                args.predictions,
                args.output,
                model_version=args.model_version,
                feature_set_id=args.feature_set_id,
                eligible_for_live=args.eligible_for_live,
                expected_universe_size=args.expected_universe_size,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


__all__ = ["ALPHA_SIGNAL_COLUMNS", "build_alpha_signal_table", "export_alpha_signal_artifact"]
