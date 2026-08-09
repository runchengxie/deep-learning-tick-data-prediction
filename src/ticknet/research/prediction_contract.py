"""正式 Top-K prediction artifact 的内容与 Parquet metadata 契约。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.research.registry import file_sha256

FORMAL_TARGET_RETURN_CONTRACT = "next_open_to_following_open"
FORMAL_UNIVERSE_CONTRACT = "lagged_turnover_top_n"
FORMAL_TRADABILITY_CONTRACT = "next_open_suspension_one_price_limit"
FORMAL_SUSPENDED_MARK_POLICY = "previous_close"

METADATA_KEYS = {
    "dataset_fingerprint": b"ticknet.dataset_fingerprint",
    "target_return_contract": b"ticknet.target_return_contract",
    "universe_contract": b"ticknet.universe_contract",
    "tradability_contract": b"ticknet.tradability_contract",
    "suspended_mark_policy": b"ticknet.suspended_mark_policy",
}


class PredictionContractError(ValueError):
    """prediction artifact 不满足正式 Top-K 契约。"""


@dataclass(frozen=True)
class FormalPredictionReport:
    """可进入 Registry 和 M3 诊断的正式预测摘要。"""

    path: str
    sha256: str
    dataset_fingerprint: str
    target_return_contract: str
    universe_contract: str
    tradability_contract: str
    suspended_mark_policy: str
    row_count: int
    candidate_row_count: int
    status_only_row_count: int
    trading_date_count: int
    label_date_count: int
    date_range: tuple[str, str]
    expected_universe_size: int
    candidate_count_min: int
    candidate_count_max: int
    cannot_buy_count: int
    cannot_sell_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_text(metadata: dict[bytes, bytes], name: str) -> str:
    key = METADATA_KEYS[name]
    raw = metadata.get(key)
    if raw is None:
        raise PredictionContractError(f"Parquet metadata 缺少 {key.decode()}")
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise PredictionContractError(f"Parquet metadata {key.decode()} 不是 UTF-8") from error
    if not value:
        raise PredictionContractError(f"Parquet metadata {key.decode()} 不能为空")
    return value


def _as_date(value: object, *, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise PredictionContractError(f"{field} 不是 ISO 日期: {value}") from error


def _as_contract_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise PredictionContractError(f"{field} 必须为 Parquet boolean")
    return value


def _as_finite_float(value: Any, *, field: str, row_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PredictionContractError(f"row {row_index} {field} 必须为有限数值") from error
    if not math.isfinite(number):
        raise PredictionContractError(f"row {row_index} {field} 必须为有限数值")
    return number


def attach_formal_prediction_metadata(
    table: pa.Table,
    *,
    dataset_fingerprint: str,
    target_return_contract: str = FORMAL_TARGET_RETURN_CONTRACT,
    universe_contract: str = FORMAL_UNIVERSE_CONTRACT,
    tradability_contract: str = FORMAL_TRADABILITY_CONTRACT,
    suspended_mark_policy: str = FORMAL_SUSPENDED_MARK_POLICY,
) -> pa.Table:
    """给 prediction Table 写入不可依赖文件名推断的正式语义。"""
    values = {
        "dataset_fingerprint": dataset_fingerprint,
        "target_return_contract": target_return_contract,
        "universe_contract": universe_contract,
        "tradability_contract": tradability_contract,
        "suspended_mark_policy": suspended_mark_policy,
    }
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise PredictionContractError("正式 prediction metadata 只能包含非空字符串")
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {METADATA_KEYS[name]: value.strip().encode("utf-8") for name, value in values.items()}
    )
    return table.replace_schema_metadata(metadata)


def _validate_contract_values(
    metadata: dict[bytes, bytes],
    *,
    expected_dataset_fingerprint: str | None,
    expected_target_return_contract: str,
) -> dict[str, str]:
    values = {name: _metadata_text(metadata, name) for name in METADATA_KEYS}
    expected = {
        "target_return_contract": expected_target_return_contract,
        "universe_contract": FORMAL_UNIVERSE_CONTRACT,
        "tradability_contract": FORMAL_TRADABILITY_CONTRACT,
        "suspended_mark_policy": FORMAL_SUSPENDED_MARK_POLICY,
    }
    for name, expected_value in expected.items():
        if values[name] != expected_value:
            raise PredictionContractError(
                f"Parquet metadata {METADATA_KEYS[name].decode()} 应为 {expected_value}，"
                f"实际为 {values[name]}"
            )
    if (
        expected_dataset_fingerprint is not None
        and values["dataset_fingerprint"] != expected_dataset_fingerprint
    ):
        raise PredictionContractError("prediction dataset_fingerprint 与 Registry 不一致")
    return values


def validate_formal_prediction_artifact(
    path: str | Path,
    *,
    expected_universe_size: int = 400,
    expected_dataset_fingerprint: str | None = None,
    expected_target_return_contract: str = FORMAL_TARGET_RETURN_CONTRACT,
) -> FormalPredictionReport:
    """校验正式预测列、metadata、股票日唯一性和动态股票池支持行。"""
    if expected_universe_size <= 0:
        raise PredictionContractError("expected_universe_size 必须为正整数")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PredictionContractError(f"prediction artifact 不存在: {source}")
    parquet = pq.ParquetFile(source)
    required = {
        "symbol",
        "trading_date",
        "label_date",
        "target_return",
        "score",
        "can_buy",
        "can_sell",
        "in_universe",
    }
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise PredictionContractError(f"正式 prediction artifact 缺少字段: {sorted(missing)}")
    metadata = dict(parquet.schema_arrow.metadata or {})
    contracts = _validate_contract_values(
        metadata,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_target_return_contract=expected_target_return_contract,
    )
    table = parquet.read(columns=sorted(required))
    if table.num_rows == 0:
        raise PredictionContractError("prediction artifact 不能为空")

    seen: set[tuple[date, str]] = set()
    signal_dates: dict[date, set[date]] = defaultdict(set)
    candidate_counts: Counter[date] = Counter()
    candidate_rows = cannot_buy = cannot_sell = 0
    label_dates: list[date] = []
    for row_index, row in enumerate(table.to_pylist()):
        raw_symbol = row["symbol"]
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise PredictionContractError(f"row {row_index} symbol 必须为非空字符串")
        symbol = raw_symbol.strip()
        trading_date = _as_date(row["trading_date"], field="trading_date")
        label_date = _as_date(row["label_date"], field="label_date")
        if label_date <= trading_date:
            raise PredictionContractError("label_date 必须晚于 trading_date")
        key = (label_date, symbol)
        if key in seen:
            raise PredictionContractError(f"prediction artifact 存在重复股票日: {key}")
        seen.add(key)
        signal_dates[label_date].add(trading_date)
        label_dates.append(label_date)

        _as_finite_float(row["score"], field="score", row_index=row_index)
        _as_finite_float(row["target_return"], field="target_return", row_index=row_index)
        can_buy = _as_contract_bool(row["can_buy"], field="can_buy")
        can_sell = _as_contract_bool(row["can_sell"], field="can_sell")
        in_universe = _as_contract_bool(row["in_universe"], field="in_universe")
        cannot_buy += not can_buy
        cannot_sell += not can_sell
        if in_universe:
            candidate_rows += 1
            candidate_counts[label_date] += 1

    if any(len(dates) != 1 for dates in signal_dates.values()):
        raise PredictionContractError("同一 label_date 必须对应唯一 trading_date")
    if set(candidate_counts) != set(signal_dates):
        raise PredictionContractError("每个 label_date 都必须包含候选股票")
    invalid_counts = {
        label_date.isoformat(): count
        for label_date, count in candidate_counts.items()
        if count != expected_universe_size
    }
    if invalid_counts:
        preview = dict(sorted(invalid_counts.items())[:5])
        raise PredictionContractError(
            f"每个 label_date 必须恰有 {expected_universe_size} 个 in_universe 候选: {preview}"
        )

    counts = list(candidate_counts.values())
    return FormalPredictionReport(
        path=str(source),
        sha256=file_sha256(source),
        dataset_fingerprint=contracts["dataset_fingerprint"],
        target_return_contract=contracts["target_return_contract"],
        universe_contract=contracts["universe_contract"],
        tradability_contract=contracts["tradability_contract"],
        suspended_mark_policy=contracts["suspended_mark_policy"],
        row_count=table.num_rows,
        candidate_row_count=candidate_rows,
        status_only_row_count=table.num_rows - candidate_rows,
        trading_date_count=len({next(iter(dates)) for dates in signal_dates.values()}),
        label_date_count=len(signal_dates),
        date_range=(min(label_dates).isoformat(), max(label_dates).isoformat()),
        expected_universe_size=expected_universe_size,
        candidate_count_min=min(counts),
        candidate_count_max=max(counts),
        cannot_buy_count=cannot_buy,
        cannot_sell_count=cannot_sell,
    )
