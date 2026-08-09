"""正式分钟聚合特征的按月物化、断点续跑与完整性校验。"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.nextday.dataset import file_sha256, manifest_fingerprint
from ticknet.nextday.minute_baseline import (
    L2_MODALITIES,
    MinuteBaselineConfig,
    MinuteExtractionReport,
    MinuteSample,
    _feature_columns,
    build_samples,
    read_l2_minute_rows,
)
from ticknet.nextday.splits import parse_date

MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZED_FEATURE_CONTRACT = "l2_trailing_minute_aggregate_v1"


@dataclass(frozen=True)
class MaterializedFeatureLoad:
    """通过 manifest 校验后加载的正式分钟样本。"""

    samples: list[MinuteSample]
    manifest_path: Path
    materialization_identity: str
    manifest_fingerprint: str
    shard_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "materialization_identity": self.materialization_identity,
            "manifest_fingerprint": self.manifest_fingerprint,
            "shard_count": self.shard_count,
            "row_count": len(self.samples),
        }


def complete_formal_samples(
    targets: Sequence[Any],
    samples: Sequence[MinuteSample],
    *,
    feature_count: int,
    report: MinuteExtractionReport,
) -> list[MinuteSample]:
    """保留完整正式股票池，缺失分钟窗口写成固定维度的全 NaN 特征。"""
    if feature_count < 1:
        raise ValueError("feature_count 必须为正整数")
    if any(sample.features.size != feature_count for sample in samples):
        raise ValueError("分钟样本特征维度不一致")
    indexed: dict[tuple[Any, str], MinuteSample] = {}
    for sample in samples:
        key = (sample.trading_date, sample.symbol)
        if key in indexed:
            raise ValueError(f"分钟样本存在重复股票日：{key}")
        indexed[key] = sample

    completed: list[MinuteSample] = []
    for target in sorted(targets, key=lambda item: (item.trading_date, item.symbol)):
        key = (target.trading_date, target.symbol)
        sample = indexed.get(key)
        if sample is not None:
            completed.append(sample)
            continue
        completed.append(
            MinuteSample(
                trading_date=target.trading_date,
                symbol=target.symbol,
                label_date=target.label_date,
                label=target.label,
                target_return=target.target_return,
                features=np.full(feature_count, np.nan, dtype=np.float32),
                return_end_date=target.return_end_date,
                feature_available=False,
            )
        )
        report.imputed_missing_samples += 1
    return completed


def _json_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _period_for(target: Any) -> str:
    return target.trading_date.strftime("%Y-%m")


def _target_key_records(targets: Sequence[Any]) -> list[tuple[str, str]]:
    records = sorted((target.trading_date.isoformat(), target.symbol) for target in targets)
    if len(set(records)) != len(records):
        raise ValueError("正式特征目标存在重复股票日")
    if not records:
        raise ValueError("正式特征目标不能为空")
    return records


def _source_identity(
    config: MinuteBaselineConfig,
    targets: Sequence[Any],
) -> tuple[dict[str, Any], int]:
    root = Path(config.l2_root).expanduser().resolve()
    years = sorted({target.trading_date.year for target in targets})
    layouts: dict[str, tuple[str, ...]] = {}
    source_files: list[dict[str, Any]] = []
    for year in years:
        for modality in L2_MODALITIES:
            path = root / "yearly" / modality / f"{year}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"缺少 {modality} 分钟缓存文件：{path}")
            columns = _feature_columns(pq.ParquetFile(path).schema_arrow.names, modality)
            if not columns:
                raise ValueError(f"{path} 缺少 {modality} 特征列")
            existing = layouts.get(modality)
            if existing is not None and existing != columns:
                raise ValueError(f"{modality} 在不同年份的特征列不一致")
            layouts[modality] = columns
            stat = path.stat()
            source_files.append(
                {
                    "year": year,
                    "modality": modality,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    feature_count = 4 * sum(len(layouts[modality]) for modality in L2_MODALITIES)
    keys = _target_key_records(targets)
    period_counts: dict[str, int] = {}
    for target in targets:
        period = _period_for(target)
        period_counts[period] = period_counts.get(period, 0) + 1
    identity = {
        "feature_contract": MATERIALIZED_FEATURE_CONTRACT,
        "window_minutes": config.window_minutes,
        "min_window_minutes": config.min_window_minutes,
        "feature_layout": {name: list(layouts[name]) for name in L2_MODALITIES},
        "feature_count": feature_count,
        "target_count": len(keys),
        "target_keys_sha256": _json_sha256(keys),
        "period_counts": dict(sorted(period_counts.items())),
        "source_files": source_files,
    }
    return identity, feature_count


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["dataset_fingerprint"] = manifest_fingerprint(manifest)
    _atomic_json(path, manifest)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取分钟特征 manifest：{path}") from error
    if not isinstance(raw, dict):
        raise ValueError("分钟特征 manifest 根节点应为对象")
    manifest = cast(dict[str, Any], raw)
    if manifest.get("format_version") != MATERIALIZATION_FORMAT_VERSION:
        raise ValueError("分钟特征 manifest format_version 不受支持")
    if manifest.get("dataset_fingerprint") != manifest_fingerprint(manifest):
        raise ValueError("分钟特征 manifest dataset_fingerprint 与内容不一致")
    return manifest


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def _feature_table(
    samples: Sequence[MinuteSample],
    *,
    feature_count: int,
    identity: str,
    period: str,
) -> pa.Table:
    matrix = np.stack([sample.features for sample in samples]).astype(np.float32, copy=False)
    if matrix.shape != (len(samples), feature_count):
        raise ValueError(f"分钟特征矩阵形状异常：{matrix.shape}")
    if np.isinf(matrix).any():
        raise ValueError("分钟特征包含无穷值")
    values = pa.array(matrix.reshape(-1), type=pa.float32())
    features = pa.FixedSizeListArray.from_arrays(values, feature_count)
    table = pa.table(
        {
            "trading_date": [sample.trading_date.isoformat() for sample in samples],
            "symbol": [sample.symbol for sample in samples],
            "feature_available": [sample.feature_available for sample in samples],
            "features": features,
        }
    )
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"ticknet.feature_contract": MATERIALIZED_FEATURE_CONTRACT.encode(),
            b"ticknet.materialization_identity": identity.encode(),
            b"ticknet.period": period.encode(),
        }
    )
    return table.replace_schema_metadata(metadata)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _update_summary(manifest: dict[str, Any]) -> None:
    shards = cast(list[dict[str, Any]], manifest["shards"])
    expected = set(cast(dict[str, int], manifest["identity"]["period_counts"]))
    completed = {str(shard["period"]) for shard in shards}
    manifest["status"] = "complete" if completed == expected else "in_progress"
    manifest["summary"] = {
        "expected_periods": len(expected),
        "completed_periods": len(completed),
        "row_count": sum(int(shard["row_count"]) for shard in shards),
        "available_feature_rows": sum(int(shard["available_feature_rows"]) for shard in shards),
        "imputed_feature_rows": sum(int(shard["imputed_feature_rows"]) for shard in shards),
        "elapsed_seconds": round(sum(float(shard["elapsed_seconds"]) for shard in shards), 6),
        "peak_rss_mb": max((float(shard["peak_rss_mb"]) for shard in shards), default=0.0),
    }


def _validate_shards(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_counts = cast(dict[str, int], manifest["identity"]["period_counts"])
    identity = str(manifest["materialization_identity"])
    feature_count = int(manifest["identity"]["feature_count"])
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("分钟特征 manifest 的 shards 应为数组")
    validated: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, dict):
            raise ValueError(f"shards[{index}] 应为对象")
        shard = cast(dict[str, Any], raw)
        period = str(shard.get("period", ""))
        if period not in expected_counts or period in validated:
            raise ValueError(f"shards[{index}] 的 period 无效或重复：{period}")
        relative = Path(str(shard.get("path", "")))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"shards[{index}] 路径越出物化目录")
        if not path.is_file():
            raise ValueError(f"分钟特征分片不存在：{path}")
        if file_sha256(path) != shard.get("sha256"):
            raise ValueError(f"分钟特征分片 SHA-256 不一致：{path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_counts[period]:
            raise ValueError(f"{period} 分片行数与目标数不一致")
        schema = parquet.schema_arrow
        required = {"trading_date", "symbol", "feature_available", "features"}
        if required - set(schema.names):
            raise ValueError(f"{period} 分片缺少字段：{sorted(required - set(schema.names))}")
        feature_type = schema.field("features").type
        if not pa.types.is_fixed_size_list(feature_type) or feature_type.list_size != feature_count:
            raise ValueError(f"{period} 分片特征维度不一致")
        metadata = schema.metadata or {}
        if metadata.get(b"ticknet.materialization_identity", b"").decode() != identity:
            raise ValueError(f"{period} 分片物化身份不一致")
        validated[period] = shard
    return validated


def materialize_minute_features(
    config: MinuteBaselineConfig,
    targets: Sequence[Any],
    output_dir: str | Path,
    *,
    resume: bool = True,
    periods: Sequence[str] | None = None,
    on_period: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """按月原子写入正式 L2 聚合特征，已有完整分片经校验后跳过。"""
    if not config.formal:
        raise ValueError("分钟特征物化只接受正式 open-to-following-open 配置")
    if config.feature_source != "l2_cache" or not config.l2_root:
        raise ValueError("分钟特征物化要求 feature_source=l2_cache 且 l2_root 非空")
    identity_payload, feature_count = _source_identity(config, targets)
    identity = _json_sha256(identity_payload)
    expected_periods = set(cast(dict[str, int], identity_payload["period_counts"]))
    selected_periods = expected_periods if periods is None else set(periods)
    unknown = selected_periods - expected_periods
    if unknown:
        raise ValueError(f"请求了目标范围外的月份：{sorted(unknown)}")

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    abandoned_manifest = manifest_path.with_suffix(".json.tmp")
    if not manifest_path.exists() and abandoned_manifest.exists():
        abandoned_manifest.unlink()
    if root.exists() and not manifest_path.exists() and any(root.iterdir()):
        raise ValueError(f"物化目录非空但缺少 manifest：{root}")
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"物化 manifest 已存在，不能以 --no-resume 覆盖：{manifest_path}")
        manifest = _read_manifest(manifest_path)
        if manifest.get("materialization_identity") != identity:
            raise ValueError("现有分钟特征 manifest 与本次目标或源文件身份不一致")
    else:
        manifest = {
            "format_version": MATERIALIZATION_FORMAT_VERSION,
            "feature_contract": MATERIALIZED_FEATURE_CONTRACT,
            "materialization_identity": identity,
            "identity": identity_payload,
            "status": "in_progress",
            "shards": [],
            "summary": {},
        }
        _update_summary(manifest)
        _write_manifest(manifest_path, manifest)

    existing = _validate_shards(root, manifest)
    targets_by_period: dict[str, list[Any]] = {}
    for target in targets:
        targets_by_period.setdefault(_period_for(target), []).append(target)

    for period in sorted(selected_periods):
        if period in existing:
            if on_period is not None:
                on_period({**existing[period], "resumed": True})
            continue
        started = time.perf_counter()
        period_targets = targets_by_period[period]
        report = MinuteExtractionReport()
        rows = read_l2_minute_rows(
            config.l2_root,
            period_targets,
            keep_minutes=config.window_minutes,
            report=report,
        )
        samples = build_samples(
            rows,
            period_targets,
            window_minutes=config.window_minutes,
            min_window_minutes=config.min_window_minutes,
            report=report,
        )
        del rows
        completed = complete_formal_samples(
            period_targets,
            samples,
            feature_count=feature_count,
            report=report,
        )
        relative = Path("shards") / f"features-{period}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(
            path,
            _feature_table(
                completed,
                feature_count=feature_count,
                identity=identity,
                period=period,
            ),
        )
        shard = {
            "period": period,
            "path": relative.as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": len(completed),
            "available_feature_rows": sum(sample.feature_available for sample in completed),
            "imputed_feature_rows": sum(not sample.feature_available for sample in completed),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "peak_rss_mb": round(_peak_rss_mb(), 3),
            "extraction": asdict(report),
        }
        cast(list[dict[str, Any]], manifest["shards"]).append(shard)
        cast(list[dict[str, Any]], manifest["shards"]).sort(key=lambda item: item["period"])
        _update_summary(manifest)
        _write_manifest(manifest_path, manifest)
        existing[period] = shard
        if on_period is not None:
            on_period({**shard, "resumed": False})
    return manifest


def load_materialized_minute_features(
    config: MinuteBaselineConfig,
    targets: Sequence[Any],
    source: str | Path,
    report: MinuteExtractionReport,
) -> MaterializedFeatureLoad:
    """校验完整 manifest、全部分片及目标键后，重建带当前标签的样本。"""
    if not config.formal or config.feature_source != "l2_cache":
        raise ValueError("已物化分钟特征只用于正式 l2_cache 配置")
    resolved = Path(source).expanduser().resolve()
    manifest_path = resolved if resolved.name == "manifest.json" else resolved / "manifest.json"
    root = manifest_path.parent
    manifest = _read_manifest(manifest_path)
    identity_payload, feature_count = _source_identity(config, targets)
    identity = _json_sha256(identity_payload)
    if manifest.get("materialization_identity") != identity:
        raise ValueError("分钟特征 manifest 与当前目标或源文件身份不一致")
    if manifest.get("status") != "complete":
        expected = set(cast(dict[str, int], identity_payload["period_counts"]))
        completed = {str(item["period"]) for item in cast(list[dict[str, Any]], manifest["shards"])}
        raise ValueError(f"分钟特征物化尚未完成，缺少月份：{sorted(expected - completed)}")
    shards = _validate_shards(root, manifest)
    expected_periods = set(cast(dict[str, int], identity_payload["period_counts"]))
    if set(shards) != expected_periods:
        raise ValueError("分钟特征 manifest 未覆盖全部目标月份")

    feature_rows: dict[tuple[Any, str], tuple[np.ndarray, bool]] = {}
    for period in sorted(shards):
        path = root / str(shards[period]["path"])
        table = pq.read_table(
            path, columns=["trading_date", "symbol", "feature_available", "features"]
        )
        feature_array = table["features"].combine_chunks()
        matrix = np.asarray(
            feature_array.values.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(table.num_rows, feature_count)
        if np.isinf(matrix).any():
            raise ValueError(f"{period} 分片包含无穷值")
        for index, (raw_date, symbol, available) in enumerate(
            zip(
                table["trading_date"].to_pylist(),
                table["symbol"].to_pylist(),
                table["feature_available"].to_pylist(),
                strict=True,
            )
        ):
            key = (parse_date(str(raw_date)), str(symbol))
            if key in feature_rows:
                raise ValueError(f"分钟特征分片存在重复股票日：{key}")
            feature_rows[key] = (matrix[index], bool(available))

    expected_keys = {(target.trading_date, target.symbol) for target in targets}
    if set(feature_rows) != expected_keys:
        missing = sorted(expected_keys - set(feature_rows))[:5]
        extra = sorted(set(feature_rows) - expected_keys)[:5]
        raise ValueError(f"分钟特征股票日与正式目标不一致：missing={missing} extra={extra}")
    samples = [
        MinuteSample(
            trading_date=target.trading_date,
            symbol=target.symbol,
            label_date=target.label_date,
            label=target.label,
            target_return=target.target_return,
            features=feature_rows[(target.trading_date, target.symbol)][0],
            return_end_date=target.return_end_date,
            feature_available=feature_rows[(target.trading_date, target.symbol)][1],
        )
        for target in sorted(targets, key=lambda item: (item.trading_date, item.symbol))
    ]
    report.requested_targets = len(samples)
    report.written_samples = sum(sample.feature_available for sample in samples)
    report.imputed_missing_samples = sum(not sample.feature_available for sample in samples)
    report.materialized_shards = len(shards)
    report.materialized_rows = len(samples)
    return MaterializedFeatureLoad(
        samples=samples,
        manifest_path=manifest_path,
        materialization_identity=identity,
        manifest_fingerprint=str(manifest["dataset_fingerprint"]),
        shard_count=len(shards),
    )
