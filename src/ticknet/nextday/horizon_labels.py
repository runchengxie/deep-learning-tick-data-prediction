"""为既有盘口特征分片生成可版本化的多周期收益标签侧车。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.nextday.dataset import manifest_fingerprint
from ticknet.nextday.snapshot_config import DailyPanel, _yyyymmdd
from ticknet.nextday.snapshot_io import read_wide_daily_panel
from ticknet.nextday.splits import parse_date

HORIZON_LABEL_FORMAT_VERSION = 1
HORIZON_RETURN_CONTRACT = "next_open_to_horizon_close_excess_benchmark"
LABELS_FILENAME = "labels.parquet"
SIDECAR_FILENAME = "horizon-labels.json"
LABEL_COLUMNS = (
    "symbol",
    "trading_date",
    "entry_date",
    "return_end_date",
    "horizon",
    "label",
    "raw_return",
    "benchmark_return",
    "target_return",
)


@dataclass(frozen=True)
class HorizonTarget:
    """一个特征样本在指定交易日跨度下的监督目标。"""

    symbol: str
    trading_date: date
    entry_date: date
    return_end_date: date
    horizon: int
    label: int
    raw_return: float
    benchmark_return: float
    target_return: float


@dataclass(frozen=True)
class LoadedHorizonSidecar:
    """已校验并按股票交易日索引的单个 horizon 标签。"""

    horizon: int
    return_contract: str
    sidecar_fingerprint: str
    source_dataset_fingerprint: str
    records: dict[tuple[str, date], HorizonTarget]


@dataclass(frozen=True)
class _LegacyH1Target:
    symbol: str
    trading_date: date
    label_date: date
    label: int
    raw_return: float
    target_return: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sidecar_fingerprint(manifest: dict[str, Any]) -> str:
    content = {key: value for key, value in manifest.items() if key != "sidecar_fingerprint"}
    payload = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons 必须包含至少一个正整数")
    if len(set(values)) != len(values):
        raise ValueError("horizons 不能重复")
    return tuple(sorted(values))


def _feature_sample_keys(
    manifest_path: str | Path,
) -> tuple[dict[date, tuple[str, ...]], str, list[_LegacyH1Target]]:
    source = Path(manifest_path).expanduser().resolve()
    with source.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("特征数据清单根节点应为对象")
    computed = manifest_fingerprint(manifest)
    stored = manifest.get("dataset_fingerprint")
    if stored is not None and stored != computed:
        raise ValueError("特征数据清单 dataset_fingerprint 与内容不一致")
    fingerprint = str(stored or computed)
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("特征数据清单缺少非空的 samples 列表")

    by_date: dict[date, set[str]] = defaultdict(set)
    seen: set[tuple[str, date]] = set()
    legacy_targets: list[_LegacyH1Target] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"samples[{index}] 应为对象")
        values = cast(dict[str, Any], raw)
        try:
            symbol = str(values["symbol"])
            trading_date = parse_date(str(values["trading_date"]))
            label_date = parse_date(str(values["label_date"]))
            label = int(values["label"])
            raw_return = float(values["raw_return"])
            target_return = float(values["target_return"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"samples[{index}] 缺少有效 H=1 目标字段") from error
        key = (symbol, trading_date)
        if (
            not symbol
            or key in seen
            or label_date <= trading_date
            or label not in {0, 1, 2}
            or not math.isfinite(raw_return)
            or not math.isfinite(target_return)
        ):
            raise ValueError(f"特征样本股票交易日无效或重复：{key}")
        seen.add(key)
        by_date[trading_date].add(symbol)
        legacy_targets.append(
            _LegacyH1Target(
                symbol=symbol,
                trading_date=trading_date,
                label_date=label_date,
                label=label,
                raw_return=raw_return,
                target_return=target_return,
            )
        )
    return (
        {day: tuple(sorted(symbols)) for day, symbols in by_date.items()},
        fingerprint,
        legacy_targets,
    )


def _read_benchmark_prices(path: str | Path) -> dict[date, tuple[float, float]]:
    source = Path(path).expanduser().resolve()
    table = pq.read_table(source, columns=["trade_date", "open", "close"])
    prices: dict[date, tuple[float, float]] = {}
    for raw_date, raw_open, raw_close in zip(
        table["trade_date"].to_pylist(),
        table["open"].to_pylist(),
        table["close"].to_pylist(),
        strict=True,
    ):
        trading_date = _yyyymmdd(raw_date)
        open_price = float(raw_open)
        close_price = float(raw_close)
        if (
            open_price > 0
            and close_price > 0
            and math.isfinite(open_price)
            and math.isfinite(close_price)
        ):
            prices[trading_date] = (open_price, close_price)
    if not prices:
        raise ValueError(f"{source} 没有有效基准开收盘价格")
    return prices


def _cross_sectional_labels(
    target_returns: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    lower = float(np.quantile(target_returns, lower_quantile))
    upper = float(np.quantile(target_returns, upper_quantile))
    labels = np.ones(target_returns.shape, dtype=np.int8)
    labels[target_returns < lower] = 0
    labels[target_returns > upper] = 2
    return labels


def build_horizon_targets(
    sample_symbols_by_date: dict[date, tuple[str, ...]],
    open_panel: DailyPanel,
    close_panel: DailyPanel,
    benchmark_prices: dict[date, tuple[float, float]],
    *,
    horizons: Sequence[int] = (1, 3, 5),
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
    min_cross_section: int = 20,
) -> list[HorizonTarget]:
    """按 `T+1 open → T+H close` 构造超额收益与横截面标签。"""
    selected_horizons = _validate_horizons(horizons)
    if not 0 < lower_quantile < upper_quantile < 1:
        raise ValueError("横截面分位点必须满足 0 < lower < upper < 1")
    if min_cross_section < 2:
        raise ValueError("min_cross_section 至少为 2")
    if open_panel.dates != close_panel.dates or open_panel.symbols != close_panel.symbols:
        raise ValueError("open 和 close 日线轴不一致")

    calendar = open_panel.dates
    date_index = {trading_date: index for index, trading_date in enumerate(calendar)}
    symbol_index = {symbol: index for index, symbol in enumerate(open_panel.symbols)}
    targets: list[HorizonTarget] = []
    for trading_date in sorted(sample_symbols_by_date):
        signal_row = date_index.get(trading_date)
        if signal_row is None or signal_row + 1 >= len(calendar):
            continue
        entry_date = calendar[signal_row + 1]
        columns_and_symbols = [
            (symbol_index[symbol], symbol)
            for symbol in sample_symbols_by_date[trading_date]
            if symbol in symbol_index
        ]
        if not columns_and_symbols:
            continue
        columns = np.asarray([column for column, _symbol in columns_and_symbols], dtype=np.int64)
        symbols = [symbol for _column, symbol in columns_and_symbols]
        entry_open = open_panel.values[signal_row + 1, columns]

        for horizon in selected_horizons:
            end_row = signal_row + horizon
            if end_row >= len(calendar):
                continue
            return_end_date = calendar[end_row]
            benchmark_entry = benchmark_prices.get(entry_date)
            benchmark_end = benchmark_prices.get(return_end_date)
            if benchmark_entry is None or benchmark_end is None:
                continue
            benchmark_return = benchmark_end[1] / benchmark_entry[0] - 1.0
            exit_close = close_panel.values[end_row, columns]
            valid = (
                np.isfinite(entry_open)
                & (entry_open > 0)
                & np.isfinite(exit_close)
                & (exit_close > 0)
            )
            if int(valid.sum()) < min_cross_section:
                continue
            valid_columns = np.flatnonzero(valid)
            raw_returns = exit_close[valid] / entry_open[valid] - 1.0
            target_returns = raw_returns - benchmark_return
            labels = _cross_sectional_labels(
                target_returns,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            for offset, raw_return, target_return, label in zip(
                valid_columns,
                raw_returns,
                target_returns,
                labels,
                strict=True,
            ):
                targets.append(
                    HorizonTarget(
                        symbol=symbols[int(offset)],
                        trading_date=trading_date,
                        entry_date=entry_date,
                        return_end_date=return_end_date,
                        horizon=horizon,
                        label=int(label),
                        raw_return=float(raw_return),
                        benchmark_return=float(benchmark_return),
                        target_return=float(target_return),
                    )
                )
    if not targets:
        raise ValueError("没有生成任何多周期标签")
    return sorted(targets, key=lambda row: (row.horizon, row.trading_date, row.symbol))


def write_horizon_sidecar(
    targets: Sequence[HorizonTarget],
    output_dir: str | Path,
    *,
    source_dataset_fingerprint: str,
    lower_quantile: float,
    upper_quantile: float,
    min_cross_section: int,
) -> Path:
    """原子写入 Parquet 标签和包含来源绑定的 JSON 合同。"""
    if not targets:
        raise ValueError("没有可写入的多周期标签")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    labels_path = root / LABELS_FILENAME
    temporary = labels_path.with_suffix(labels_path.suffix + ".tmp")
    table = pa.table(
        {
            "symbol": [row.symbol for row in targets],
            "trading_date": pa.array([row.trading_date for row in targets], type=pa.date32()),
            "entry_date": pa.array([row.entry_date for row in targets], type=pa.date32()),
            "return_end_date": pa.array([row.return_end_date for row in targets], type=pa.date32()),
            "horizon": pa.array([row.horizon for row in targets], type=pa.int16()),
            "label": pa.array([row.label for row in targets], type=pa.int8()),
            "raw_return": pa.array([row.raw_return for row in targets], type=pa.float64()),
            "benchmark_return": pa.array(
                [row.benchmark_return for row in targets], type=pa.float64()
            ),
            "target_return": pa.array([row.target_return for row in targets], type=pa.float64()),
        }
    )
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, labels_path)
    rows_by_horizon: dict[str, int] = defaultdict(int)
    for target in targets:
        rows_by_horizon[str(target.horizon)] += 1
    manifest: dict[str, Any] = {
        "format_version": HORIZON_LABEL_FORMAT_VERSION,
        "return_contract": HORIZON_RETURN_CONTRACT,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "horizons": sorted(int(value) for value in rows_by_horizon),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "min_cross_section": min_cross_section,
        "columns": list(LABEL_COLUMNS),
        "labels": {
            "path": LABELS_FILENAME,
            "rows": len(targets),
            "rows_by_horizon": dict(rows_by_horizon),
            "bytes": labels_path.stat().st_size,
            "sha256": _file_sha256(labels_path),
        },
    }
    manifest["sidecar_fingerprint"] = _sidecar_fingerprint(manifest)
    manifest_path = root / SIDECAR_FILENAME
    _atomic_json(manifest_path, manifest)
    return manifest_path


def prepare_horizon_sidecar(
    feature_manifest_path: str | Path,
    basic_root: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path,
    *,
    horizons: Sequence[int] = (1, 3, 5),
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
    min_cross_section: int = 20,
) -> Path:
    """只读特征 manifest 和日线，生成不复制特征的标签侧车。"""
    selected_horizons = _validate_horizons(horizons)
    symbols_by_date, source_fingerprint, legacy_h1 = _feature_sample_keys(feature_manifest_path)
    symbols = tuple(sorted({symbol for values in symbols_by_date.values() for symbol in values}))
    root = Path(basic_root).expanduser().resolve()
    open_panel = read_wide_daily_panel(root / "open_data.parquet", symbols=symbols)
    close_panel = read_wide_daily_panel(root / "close_data.parquet", symbols=symbols)
    extended_horizons = tuple(value for value in selected_horizons if value != 1)
    targets = (
        build_horizon_targets(
            symbols_by_date,
            open_panel,
            close_panel,
            _read_benchmark_prices(benchmark_path),
            horizons=extended_horizons,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            min_cross_section=min_cross_section,
        )
        if extended_horizons
        else []
    )
    if 1 in selected_horizons:
        targets.extend(
            HorizonTarget(
                symbol=row.symbol,
                trading_date=row.trading_date,
                entry_date=row.label_date,
                return_end_date=row.label_date,
                horizon=1,
                label=row.label,
                raw_return=row.raw_return,
                benchmark_return=row.raw_return - row.target_return,
                target_return=row.target_return,
            )
            for row in legacy_h1
        )
    targets.sort(key=lambda row: (row.horizon, row.trading_date, row.symbol))
    return write_horizon_sidecar(
        targets,
        output_dir,
        source_dataset_fingerprint=source_fingerprint,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        min_cross_section=min_cross_section,
    )


def load_horizon_sidecar(
    manifest_path: str | Path,
    *,
    horizon: int,
    source_dataset_fingerprint: str,
    verify_checksum: bool = True,
) -> LoadedHorizonSidecar:
    """校验来源、合同和文件摘要后读取一个 horizon。"""
    source = Path(manifest_path).expanduser().resolve()
    with source.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("标签侧车清单根节点应为对象")
    if manifest.get("format_version") != HORIZON_LABEL_FORMAT_VERSION:
        raise ValueError(f"标签侧车 format_version 应为 {HORIZON_LABEL_FORMAT_VERSION}")
    stored_fingerprint = manifest.get("sidecar_fingerprint")
    if not isinstance(stored_fingerprint, str) or stored_fingerprint != _sidecar_fingerprint(
        manifest
    ):
        raise ValueError("标签侧车 fingerprint 与内容不一致")
    bound_fingerprint = manifest.get("source_dataset_fingerprint")
    if bound_fingerprint != source_dataset_fingerprint:
        raise ValueError("标签侧车绑定的特征数据指纹与当前 manifest 不一致")
    horizons = manifest.get("horizons")
    if not isinstance(horizons, list) or horizon not in horizons:
        raise ValueError(f"标签侧车不包含 horizon={horizon}")
    return_contract = manifest.get("return_contract")
    if return_contract != HORIZON_RETURN_CONTRACT:
        raise ValueError("标签侧车收益合同不受支持")
    labels = manifest.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("标签侧车缺少 labels 文件信息")
    labels_path = source.parent / str(labels.get("path", ""))
    if not labels_path.is_file():
        raise ValueError(f"标签文件不存在：{labels_path}")
    if labels_path.stat().st_size != int(labels.get("bytes", -1)):
        raise ValueError("标签文件大小与侧车清单不一致")
    if verify_checksum and _file_sha256(labels_path) != labels.get("sha256"):
        raise ValueError("标签文件 sha256 与侧车清单不一致")

    table = pq.read_table(
        labels_path,
        columns=list(LABEL_COLUMNS),
        filters=[("horizon", "=", horizon)],
    )
    records: dict[tuple[str, date], HorizonTarget] = {}
    for row in table.to_pylist():
        target = HorizonTarget(
            symbol=str(row["symbol"]),
            trading_date=row["trading_date"],
            entry_date=row["entry_date"],
            return_end_date=row["return_end_date"],
            horizon=int(row["horizon"]),
            label=int(row["label"]),
            raw_return=float(row["raw_return"]),
            benchmark_return=float(row["benchmark_return"]),
            target_return=float(row["target_return"]),
        )
        key = (target.symbol, target.trading_date)
        if (
            key in records
            or target.horizon != horizon
            or target.entry_date <= target.trading_date
            or target.return_end_date < target.entry_date
            or target.label not in {0, 1, 2}
            or not math.isfinite(target.target_return)
        ):
            raise ValueError(f"标签侧车记录无效或重复：{key}")
        records[key] = target
    if not records:
        raise ValueError(f"标签侧车 horizon={horizon} 没有记录")
    return LoadedHorizonSidecar(
        horizon=horizon,
        return_contract=return_contract,
        sidecar_fingerprint=stored_fingerprint,
        source_dataset_fingerprint=bound_fingerprint,
        records=records,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="为既有盘口分片生成 1/3/5 日标签侧车")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--basic-root", required=True)
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--lower-quantile", type=float, default=0.2)
    parser.add_argument("--upper-quantile", type=float, default=0.8)
    parser.add_argument("--min-cross-section", type=int, default=100)
    args = parser.parse_args(argv)
    path = prepare_horizon_sidecar(
        args.manifest,
        args.basic_root,
        args.benchmark_path,
        args.output_dir,
        horizons=args.horizons,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        min_cross_section=args.min_cross_section,
    )
    print(f"多周期标签侧车：{path}")


if __name__ == "__main__":
    main()
