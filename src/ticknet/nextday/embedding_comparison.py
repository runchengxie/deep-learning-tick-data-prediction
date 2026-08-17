"""比较分钟特征、冻结事件流 embedding 及二者组合的下游增量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import ndcg_score

from ticknet.eventstream.embedding import load_embedding_manifest
from ticknet.nextday.minute_baseline import (
    MinuteBaselineConfig,
    MinuteExtractionReport,
    MinuteSample,
    build_target_bundle,
    load_minute_baseline_config,
)
from ticknet.nextday.minute_materialization import load_materialized_minute_features
from ticknet.nextday.splits import WalkForwardSplit, parse_date
from ticknet.research.portfolio import (
    CostModel,
    PortfolioPolicy,
    PortfolioPrediction,
    evaluate_topk_portfolio,
)

FEATURE_SETS = ("minute", "embedding", "combined")
DOWNSTREAM_MODELS = ("hgb", "lambdamart")
EXPOSURE_COLUMNS = ("size", "liquidity", "volatility")


@dataclass(frozen=True)
class ComparisonConfig:
    """最近折冻结表征比较的切分和评估口径。"""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    oos_start: str
    oos_end: str
    downstream_seed: int = 0
    min_symbols_per_day: int = 350
    relevance_levels: int = 5
    top_ks: tuple[int, ...] = (50, 100)
    cost_bps: float = 10.0
    sell_stamp_tax_bps: float = 5.0
    hgb_max_iter: int = 500
    lambdamart_estimators: int = 500
    lambdamart_early_stopping_rounds: int = 50

    def split(self) -> WalkForwardSplit:
        return WalkForwardSplit.from_strings(
            train_start=self.train_start,
            train_end=self.train_end,
            val_start=self.validation_start,
            val_end=self.validation_end,
            test_start=self.oos_start,
            test_end=self.oos_end,
        )

    def validate(self) -> None:
        split = self.split()
        if split.test.end >= date(2026, 1, 1):
            raise ValueError("冻结 embedding 最近折不能进入 2026 locked 区间")
        if self.min_symbols_per_day < max(self.top_ks, default=1):
            raise ValueError("min_symbols_per_day 不能小于最大的 Top-K")
        if self.relevance_levels < 2:
            raise ValueError("relevance_levels 至少为 2")
        if not self.top_ks or any(value < 1 for value in self.top_ks):
            raise ValueError("top_ks 必须包含正整数")
        if self.cost_bps < 0 or self.sell_stamp_tax_bps < 0:
            raise ValueError("交易成本不能为负数")


@dataclass(frozen=True)
class _Rows:
    keys: tuple[tuple[date, str], ...]
    samples: tuple[MinuteSample, ...]
    minute: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    target_returns: np.ndarray

    @property
    def dates(self) -> tuple[date, ...]:
        return tuple(key[0] for key in self.keys)

    def features(self, feature_set: str) -> np.ndarray:
        if feature_set == "minute":
            return self.minute
        if feature_set == "embedding":
            return self.embedding
        if feature_set == "combined":
            return np.concatenate((self.minute, self.embedding), axis=1)
        raise ValueError(f"未知输入组合：{feature_set}")


def load_comparison_config(path: str | Path) -> ComparisonConfig:
    with Path(path).open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("冻结 embedding 比较配置应为 YAML 对象")
    known: set[str] = set(ComparisonConfig.__dataclass_fields__)
    unknown = {str(key) for key in raw} - known
    if unknown:
        raise ValueError(f"冻结 embedding 比较配置包含未知字段：{sorted(unknown)}")
    values = dict(raw)
    if "top_ks" in values:
        values["top_ks"] = tuple(int(value) for value in values["top_ks"])
    config = ComparisonConfig(**values)
    config.validate()
    return config


def _fixed_list_matrix(column: pa.ChunkedArray, dimension: int) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_fixed_size_list(array.type) or array.type.list_size != dimension:
        raise ValueError("固定向量列维度与 manifest 不一致")
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
        len(array), dimension
    )


def _load_embeddings(root: Path) -> tuple[dict[tuple[date, str], np.ndarray], dict[str, Any]]:
    manifest = load_embedding_manifest(root)
    dimension = int(manifest["contract"]["encoder"]["embedding_dimension"])
    result: dict[tuple[date, str], np.ndarray] = {}
    for artifact in manifest["artifacts"]:
        table = pq.read_table(
            root / str(artifact["path"]),
            columns=["trading_day", "symbol", "embedding"],
        )
        matrix = _fixed_list_matrix(table["embedding"], dimension)
        for row, (raw_day, raw_symbol) in enumerate(
            zip(table["trading_day"].to_pylist(), table["symbol"].to_pylist(), strict=True)
        ):
            text = str(int(raw_day))
            key = (date(int(text[:4]), int(text[4:6]), int(text[6:])), str(raw_symbol))
            if key in result:
                raise ValueError(f"冻结 embedding 股票日重复：{key}")
            result[key] = matrix[row]
    if len(result) != int(manifest["totals"]["rows"]):
        raise ValueError("冻结 embedding 加载行数与 manifest 不一致")
    return result, manifest


def _assigned_partition(sample: MinuteSample, split: WalkForwardSplit) -> str | None:
    assigned = split.assign(sample.trading_date, sample.label_date)
    if (
        assigned is not None
        and sample.return_end_date is not None
        and not split.range_for(assigned).contains(sample.return_end_date)
    ):
        return None
    return assigned


def _make_rows(
    samples: list[MinuteSample],
    embeddings: dict[tuple[date, str], np.ndarray],
    *,
    split: WalkForwardSplit,
) -> dict[str, _Rows]:
    selected: dict[str, list[tuple[MinuteSample, np.ndarray]]] = defaultdict(list)
    for sample in samples:
        partition = _assigned_partition(sample, split)
        key = (sample.trading_date, sample.symbol)
        if partition is not None and key in embeddings:
            selected[partition].append((sample, embeddings[key]))
    result: dict[str, _Rows] = {}
    for partition in ("train", "val", "test"):
        items = sorted(selected[partition], key=lambda row: (row[0].trading_date, row[0].symbol))
        if not items:
            raise ValueError(f"冻结 embedding 比较分区为空：{partition}")
        sample_rows = tuple(row[0] for row in items)
        result[partition] = _Rows(
            keys=tuple((row.trading_date, row.symbol) for row in sample_rows),
            samples=sample_rows,
            minute=np.stack([row.features for row in sample_rows]).astype(np.float32, copy=False),
            embedding=np.stack([row[1] for row in items]).astype(np.float32, copy=False),
            labels=np.asarray([row.label for row in sample_rows], dtype=np.int64),
            target_returns=np.asarray([row.target_return for row in sample_rows], dtype=np.float64),
        )
    return result


def _group_counts(dates: tuple[date, ...]) -> list[int]:
    counts: list[int] = []
    previous: date | None = None
    for trading_date in dates:
        if trading_date != previous:
            counts.append(0)
            previous = trading_date
        counts[-1] += 1
    return counts


def _relevance_by_day(rows: _Rows, levels: int) -> np.ndarray:
    relevance = np.zeros(len(rows.keys), dtype=np.int32)
    offset = 0
    for count in _group_counts(rows.dates):
        returns = rows.target_returns[offset : offset + count]
        order = np.argsort(returns, kind="mergesort")
        rank = np.empty(count, dtype=np.int64)
        rank[order] = np.arange(count)
        relevance[offset : offset + count] = np.minimum(levels - 1, rank * levels // count)
        offset += count
    return relevance


def _fit_scores(
    model_name: str,
    feature_set: str,
    rows: dict[str, _Rows],
    config: ComparisonConfig,
) -> dict[str, np.ndarray]:
    train = rows["train"]
    validation = rows["val"]
    train_x = train.features(feature_set)
    if model_name == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=config.hgb_max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            random_state=config.downstream_seed,
        )
        model.fit(train_x, train.labels)
        if not np.array_equal(model.classes_, np.arange(3)):
            raise ValueError("HGB 训练分区没有覆盖三个类别")
        result: dict[str, np.ndarray] = {}
        for name, part in rows.items():
            probabilities = model.predict_proba(part.features(feature_set))
            result[name] = probabilities[:, 2] - probabilities[:, 0]
        return result
    if model_name != "lambdamart":
        raise ValueError(f"未知下游模型：{model_name}")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=config.lambdamart_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        random_state=config.downstream_seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
        lambdarank_truncation_level=max(config.top_ks),
    )
    model.fit(
        train_x,
        _relevance_by_day(train, config.relevance_levels),
        group=_group_counts(train.dates),
        eval_X=validation.features(feature_set),
        eval_y=_relevance_by_day(validation, config.relevance_levels),
        eval_group=[_group_counts(validation.dates)],
        eval_at=list(config.top_ks),
        callbacks=[lgb.early_stopping(config.lambdamart_early_stopping_rounds, verbose=False)],
    )
    return {
        name: np.asarray(model.predict(part.features(feature_set)), dtype=np.float64)
        for name, part in rows.items()
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _rank_correlation(scores: np.ndarray, returns: np.ndarray) -> float:
    score_ranks = _average_ranks(scores)
    return_ranks = _average_ranks(returns)
    if np.std(score_ranks) == 0 or np.std(return_ranks) == 0:
        return math.nan
    return float(np.corrcoef(score_ranks, return_ranks)[0, 1])


def _ranking_metrics(
    rows: _Rows,
    scores: np.ndarray,
    config: ComparisonConfig,
) -> dict[str, Any]:
    daily: list[dict[str, Any]] = []
    offset = 0
    for count in _group_counts(rows.dates):
        daily_scores = scores[offset : offset + count]
        returns = rows.target_returns[offset : offset + count]
        trading_date = rows.dates[offset]
        relevance = _relevance_by_day(
            _Rows(
                keys=rows.keys[offset : offset + count],
                samples=rows.samples[offset : offset + count],
                minute=rows.minute[offset : offset + count],
                embedding=rows.embedding[offset : offset + count],
                labels=rows.labels[offset : offset + count],
                target_returns=returns,
            ),
            config.relevance_levels,
        )
        row: dict[str, Any] = {
            "trading_date": trading_date.isoformat(),
            "symbols": count,
            "rank_ic": _rank_correlation(daily_scores, returns),
        }
        predicted_order = np.argsort(-daily_scores, kind="mergesort")
        realized_order = np.argsort(-returns, kind="mergesort")
        for top_k in config.top_ks:
            k = min(top_k, count)
            row[f"ndcg_at_{top_k}"] = float(
                ndcg_score(relevance[None, :], daily_scores[None, :], k=k)
            )
            row[f"precision_at_{top_k}"] = float(
                len(set(predicted_order[:k]) & set(realized_order[:k])) / k
            )
        daily.append(row)
        offset += count
    if any(row["symbols"] < config.min_symbols_per_day for row in daily):
        low = min(row["symbols"] for row in daily)
        raise ValueError(f"比较股票池低于 min_symbols_per_day：{low}")
    metrics: dict[str, Any] = {
        "dates": len(daily),
        "symbols_per_day_min": min(row["symbols"] for row in daily),
        "symbols_per_day_max": max(row["symbols"] for row in daily),
        "daily_rank_ic_mean": float(np.mean([row["rank_ic"] for row in daily])),
        "daily_rank_ic_std": float(np.std([row["rank_ic"] for row in daily], ddof=1)),
        "monthly_rank_ic": {},
    }
    by_month: dict[str, list[float]] = defaultdict(list)
    for row in daily:
        by_month[row["trading_date"][:7]].append(float(row["rank_ic"]))
    metrics["monthly_rank_ic"] = {
        month: float(np.mean(values)) for month, values in sorted(by_month.items())
    }
    for top_k in config.top_ks:
        metrics[f"ndcg_at_{top_k}"] = float(np.mean([row[f"ndcg_at_{top_k}"] for row in daily]))
        metrics[f"precision_at_{top_k}"] = float(
            np.mean([row[f"precision_at_{top_k}"] for row in daily])
        )
    return metrics


def _comparison_universe_predictions(
    rows: _Rows,
    scores: np.ndarray,
    all_targets: list[Any],
) -> list[PortfolioPrediction]:
    score_by_key = dict(zip(rows.keys, scores, strict=True))
    selected_by_day: dict[date, set[str]] = defaultdict(set)
    for trading_date, symbol in rows.keys:
        selected_by_day[trading_date].add(symbol)
    targets_by_day: dict[date, dict[str, Any]] = defaultdict(dict)
    for target in all_targets:
        if target.trading_date in selected_by_day:
            targets_by_day[target.trading_date][target.symbol] = target

    predictions: list[PortfolioPrediction] = []
    previous: set[str] = set()
    for trading_date in sorted(selected_by_day):
        current = selected_by_day[trading_date]
        day_targets = targets_by_day[trading_date]
        for symbol in sorted(current | (previous - current)):
            target = day_targets.get(symbol)
            if target is None:
                continue
            in_universe = symbol in current
            predictions.append(
                PortfolioPrediction(
                    symbol=symbol,
                    trading_date=trading_date,
                    label_date=target.label_date,
                    score=float(score_by_key.get((trading_date, symbol), 0.0)),
                    target_return=float(target.portfolio_return),
                    can_buy=bool(target.can_buy),
                    can_sell=bool(target.can_sell),
                    tradability_known=True,
                    in_universe=in_universe,
                    universe_membership_known=True,
                )
            )
        previous = current
    return predictions


def _portfolio_metrics(
    rows: _Rows,
    scores: np.ndarray,
    all_targets: list[Any],
    config: ComparisonConfig,
) -> dict[str, Any]:
    predictions = _comparison_universe_predictions(rows, scores, all_targets)
    result: dict[str, Any] = {}
    for top_k in config.top_ks:
        evaluation = evaluate_topk_portfolio(
            predictions,
            policy=PortfolioPolicy(
                top_k=top_k,
                min_symbols_per_day=config.min_symbols_per_day,
                require_tradability=True,
                require_universe_membership=True,
            ),
            cost_model=CostModel(
                per_side_bps=config.cost_bps,
                sell_stamp_tax_bps=config.sell_stamp_tax_bps,
            ),
        )
        summary = dict(evaluation.summary)
        summary["ranking"] = dict(summary["ranking"])
        summary["ranking"]["mean_net_active_return"] = float(
            np.mean([row["net_active_return"] for row in evaluation.daily])
        )
        result[str(top_k)] = summary
    return result


def _load_exposures(path: Path | None) -> dict[tuple[date, str], dict[str, Any]]:
    if path is None:
        return {}
    table = pq.read_table(path)
    required = {"trading_date", "symbol", *EXPOSURE_COLUMNS, "industry"}
    if missing := required - set(table.column_names):
        raise ValueError(f"风险暴露文件缺少字段：{sorted(missing)}")
    result: dict[tuple[date, str], dict[str, Any]] = {}
    for row in table.select(sorted(required)).to_pylist():
        key = (parse_date(str(row["trading_date"])), str(row["symbol"]))
        if key in result:
            raise ValueError(f"风险暴露文件股票日重复：{key}")
        result[key] = row
    return result


def _exposure_metrics(
    rows: _Rows,
    scores: np.ndarray,
    exposures: dict[tuple[date, str], dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    if not exposures:
        return {
            "status": "unavailable",
            "required_columns": ["trading_date", "symbol", *EXPOSURE_COLUMNS, "industry"],
        }
    correlations: dict[str, list[float]] = defaultdict(list)
    selected_z: dict[str, list[float]] = defaultdict(list)
    industry_distances: list[float] = []
    offset = 0
    evaluated = 0
    for count in _group_counts(rows.dates):
        keys = rows.keys[offset : offset + count]
        if any(key not in exposures for key in keys):
            offset += count
            continue
        day_scores = scores[offset : offset + count]
        top = np.argsort(-day_scores, kind="mergesort")[: min(top_k, count)]
        for name in EXPOSURE_COLUMNS:
            values = np.asarray([float(exposures[key][name]) for key in keys], dtype=np.float64)
            if np.all(np.isfinite(values)) and np.std(values) > 0:
                correlations[name].append(float(np.corrcoef(day_scores, values)[0, 1]))
                selected_z[name].append(
                    float(np.mean((values[top] - values.mean()) / values.std()))
                )
        industries = [str(exposures[key]["industry"]) for key in keys]
        names = sorted(set(industries))
        universe_share = np.asarray([industries.count(name) / count for name in names])
        selected_names = [industries[index] for index in top]
        selected_share = np.asarray(
            [selected_names.count(name) / len(selected_names) for name in names]
        )
        industry_distances.append(float(np.abs(selected_share - universe_share).sum() / 2))
        evaluated += 1
        offset += count
    return {
        "status": "complete" if evaluated else "unavailable",
        "evaluated_dates": evaluated,
        "score_correlation": {
            name: float(np.mean(values)) for name, values in sorted(correlations.items())
        },
        "top_k_mean_zscore": {
            name: float(np.mean(values)) for name, values in sorted(selected_z.items())
        },
        "industry_total_variation": (
            float(np.mean(industry_distances)) if industry_distances else None
        ),
    }


def _write_predictions(
    path: Path,
    rows: _Rows,
    scores: np.ndarray,
    *,
    metadata: dict[str, str],
) -> None:
    table = pa.table(
        {
            "trading_date": pa.array([key[0] for key in rows.keys], type=pa.date32()),
            "symbol": [key[1] for key in rows.keys],
            "label_date": pa.array(
                [sample.label_date for sample in rows.samples], type=pa.date32()
            ),
            "return_end_date": pa.array(
                [sample.return_end_date for sample in rows.samples], type=pa.date32()
            ),
            "target_return": rows.target_returns,
            "score": scores,
        }
    )
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update(
        {f"ticknet.{key}".encode(): value.encode() for key, value in metadata.items()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table.replace_schema_metadata(schema_metadata), temporary, compression="zstd")
    os.replace(temporary, path)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    return value


EmbeddingSet = tuple[str, dict[tuple[date, str], np.ndarray], dict[str, Any]]


def _prepare_embedding_sets(
    roots: list[Path],
) -> tuple[list[EmbeddingSet], set[tuple[date, str]], str]:
    embedding_sets: list[EmbeddingSet] = []
    common_keys: set[tuple[date, str]] | None = None
    close_fingerprint = ""
    for root in roots:
        values, manifest = _load_embeddings(root)
        encoder_id = f"seed{int(manifest['contract']['encoder']['seed'])}"
        if any(existing[0] == encoder_id for existing in embedding_sets):
            raise ValueError(f"冻结 embedding seed 重复：{encoder_id}")
        current_close = str(manifest["contract"]["close_cache_fingerprint"])
        if close_fingerprint and current_close != close_fingerprint:
            raise ValueError("多组冻结 embedding 没有使用同一尾盘窗口缓存")
        close_fingerprint = current_close
        common_keys = set(values) if common_keys is None else common_keys & set(values)
        embedding_sets.append((encoder_id, values, manifest))
    if not common_keys:
        raise ValueError("多组冻结 embedding 没有共同股票日")
    for encoder_id, values, _manifest in embedding_sets:
        if set(values) != common_keys:
            raise ValueError(f"{encoder_id} 的冻结 embedding 股票日集合与其他 seed 不一致")
    return embedding_sets, common_keys, close_fingerprint


def _evaluate_and_write(
    *,
    cell_id: str,
    rows: dict[str, _Rows],
    scores: dict[str, np.ndarray],
    targets: list[Any],
    config: ComparisonConfig,
    exposures: dict[tuple[date, str], dict[str, Any]],
    output_root: Path,
    encoder_id: str,
    model_name: str,
    feature_set: str,
    encoder_fingerprint: str = "",
) -> tuple[str, dict[str, Any]]:
    result = _evaluate_cell(rows, scores, targets, config, exposures)
    metadata = {"encoder": encoder_id, "model": model_name, "features": feature_set}
    if encoder_fingerprint:
        metadata["encoder_fingerprint"] = encoder_fingerprint
    for partition, values in scores.items():
        _write_predictions(
            output_root
            / "predictions"
            / encoder_id
            / model_name
            / feature_set
            / f"{partition}.parquet",
            rows[partition],
            values,
            metadata=metadata,
        )
    return cell_id, result


def _run_baseline_cells(
    rows: dict[str, _Rows],
    *,
    targets: list[Any],
    config: ComparisonConfig,
    exposures: dict[tuple[date, str], dict[str, Any]],
    output_root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, np.ndarray]]]:
    results: dict[str, Any] = {}
    store: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for model_name in DOWNSTREAM_MODELS:
        scores = _fit_scores(model_name, "minute", rows, config)
        store[(model_name, "minute", "baseline")] = scores
        cell_id, result = _evaluate_and_write(
            cell_id=f"{model_name}/minute",
            rows=rows,
            scores=scores,
            targets=targets,
            config=config,
            exposures=exposures,
            output_root=output_root,
            encoder_id="baseline",
            model_name=model_name,
            feature_set="minute",
        )
        results[cell_id] = result
    return results, store


def _run_encoder_cells(
    embedding_sets: list[EmbeddingSet],
    minute_samples: list[MinuteSample],
    baseline_rows: dict[str, _Rows],
    *,
    split: WalkForwardSplit,
    targets: list[Any],
    config: ComparisonConfig,
    exposures: dict[tuple[date, str], dict[str, Any]],
    output_root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, np.ndarray]]]:
    results: dict[str, Any] = {}
    store: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for encoder_id, embeddings, manifest in embedding_sets:
        rows = _make_rows(minute_samples, embeddings, split=split)
        if any(rows[name].keys != baseline_rows[name].keys for name in rows):
            raise ValueError(f"{encoder_id} 的下游股票日与基线不一致")
        for model_name in DOWNSTREAM_MODELS:
            for feature_set in ("embedding", "combined"):
                scores = _fit_scores(model_name, feature_set, rows, config)
                store[(model_name, feature_set, encoder_id)] = scores
                cell_id, result = _evaluate_and_write(
                    cell_id=f"{model_name}/{feature_set}/{encoder_id}",
                    rows=rows,
                    scores=scores,
                    targets=targets,
                    config=config,
                    exposures=exposures,
                    output_root=output_root,
                    encoder_id=encoder_id,
                    model_name=model_name,
                    feature_set=feature_set,
                    encoder_fingerprint=str(manifest["dataset_fingerprint"]),
                )
                results[cell_id] = result
    return results, store


def _run_prediction_ensembles(
    embedding_sets: list[EmbeddingSet],
    rows: dict[str, _Rows],
    score_store: dict[tuple[str, str, str], dict[str, np.ndarray]],
    *,
    targets: list[Any],
    config: ComparisonConfig,
    exposures: dict[tuple[date, str], dict[str, Any]],
) -> dict[str, Any]:
    if len(embedding_sets) < 2:
        return {}
    results: dict[str, Any] = {}
    for model_name in DOWNSTREAM_MODELS:
        for feature_set in ("embedding", "combined"):
            ensemble = {
                partition: np.mean(
                    [
                        score_store[(model_name, feature_set, encoder_id)][partition]
                        for encoder_id, _values, _manifest in embedding_sets
                    ],
                    axis=0,
                )
                for partition in ("train", "val", "test")
            }
            cell_id = f"{model_name}/{feature_set}/prediction_ensemble"
            results[cell_id] = _evaluate_cell(rows, ensemble, targets, config, exposures)
    return results


def run_comparison(
    *,
    minute_config: MinuteBaselineConfig,
    minute_features_root: Path,
    embedding_roots: list[Path],
    comparison_config: ComparisonConfig,
    output_root: Path,
    exposure_path: Path | None = None,
) -> dict[str, Any]:
    """分别训练三组下游模型，并组合预测分数，不混合 embedding 坐标。"""
    if not minute_config.formal:
        raise ValueError("冻结 embedding 比较要求正式 open-to-following-open 分钟配置")
    if len(embedding_roots) < 1:
        raise ValueError("至少需要一个冻结 embedding 目录")
    comparison_config.validate()
    split = comparison_config.split()
    bundle = build_target_bundle(minute_config)
    candidates = [target for target in bundle.targets if target.in_universe]
    report = MinuteExtractionReport()
    materialized = load_materialized_minute_features(
        minute_config,
        candidates,
        minute_features_root,
        report,
    )
    embedding_sets, common_keys, close_fingerprint = _prepare_embedding_sets(embedding_roots)

    candidate_keys = {(sample.trading_date, sample.symbol) for sample in materialized.samples}
    recent_candidate_keys = {
        key for key in candidate_keys if split.train.start <= key[0] <= split.test.end
    }
    coverage = len(common_keys & recent_candidate_keys) / max(len(recent_candidate_keys), 1)
    if coverage < 0.9:
        raise ValueError(f"事件流与分钟候选的股票日覆盖率过低：{coverage:.2%}")
    exposures = _load_exposures(exposure_path)
    output_root.mkdir(parents=True, exist_ok=True)
    first_rows = _make_rows(materialized.samples, embedding_sets[0][1], split=split)
    baseline_results, score_store = _run_baseline_cells(
        first_rows,
        targets=bundle.targets,
        config=comparison_config,
        exposures=exposures,
        output_root=output_root,
    )
    encoder_results, encoder_scores = _run_encoder_cells(
        embedding_sets,
        materialized.samples,
        first_rows,
        split=split,
        targets=bundle.targets,
        config=comparison_config,
        exposures=exposures,
        output_root=output_root,
    )
    score_store.update(encoder_scores)
    results = {**baseline_results, **encoder_results}
    results.update(
        _run_prediction_ensembles(
            embedding_sets,
            first_rows,
            score_store,
            targets=bundle.targets,
            config=comparison_config,
            exposures=exposures,
        )
    )

    identity = {
        "minute_materialization_identity": materialized.materialization_identity,
        "minute_manifest_fingerprint": materialized.manifest_fingerprint,
        "close_cache_fingerprint": close_fingerprint,
        "embedding_fingerprints": {
            encoder_id: manifest["dataset_fingerprint"]
            for encoder_id, _values, manifest in embedding_sets
        },
        "comparison_config": comparison_config.__dict__,
        "common_rows": {name: len(rows.keys) for name, rows in first_rows.items()},
        "minute_eventstream_coverage": coverage,
        "embedding_combination": "independent_downstream_models_then_prediction_average",
    }
    output = _safe_json(
        {
            "status": "complete",
            "experiment": "FEAT-EMB-FROZEN-001",
            "identity": identity,
            "dataset_fingerprint": _fingerprint(identity),
            "results": results,
            "decision": _decision_summary(results, embedding_sets),
        }
    )
    path = output_root / "comparison.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return output


def _evaluate_cell(
    rows: dict[str, _Rows],
    scores: dict[str, np.ndarray],
    targets: list[Any],
    config: ComparisonConfig,
    exposures: dict[tuple[date, str], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition in ("val", "test"):
        part = rows[partition]
        result[partition] = {
            "ranking": _ranking_metrics(part, scores[partition], config),
            "portfolio": _portfolio_metrics(part, scores[partition], targets, config),
            "exposures": _exposure_metrics(
                part,
                scores[partition],
                exposures,
                top_k=max(config.top_ks),
            ),
        }
    return result


def _decision_summary(
    results: dict[str, Any],
    embedding_sets: list[tuple[str, dict[tuple[date, str], np.ndarray], dict[str, Any]]],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for model_name in DOWNSTREAM_MODELS:
        baseline = results[f"{model_name}/minute"]["test"]
        for encoder_id, _values, _manifest in embedding_sets:
            combined = results[f"{model_name}/combined/{encoder_id}"]["test"]
            key = f"{model_name}/{encoder_id}"
            deltas[key] = {
                "daily_rank_ic_mean": combined["ranking"]["daily_rank_ic_mean"]
                - baseline["ranking"]["daily_rank_ic_mean"],
                "ndcg_at_100": combined["ranking"].get("ndcg_at_100", math.nan)
                - baseline["ranking"].get("ndcg_at_100", math.nan),
                "precision_at_100": combined["ranking"].get("precision_at_100", math.nan)
                - baseline["ranking"].get("precision_at_100", math.nan),
                "top100_net_active_mean": combined["portfolio"]
                .get("100", {})
                .get("ranking", {})
                .get("mean_net_active_return", math.nan)
                - baseline["portfolio"]
                .get("100", {})
                .get("ranking", {})
                .get("mean_net_active_return", math.nan),
            }
    return {
        "paired_oos_deltas_combined_minus_minute": deltas,
        "150m_status": "deferred_until_embedding_increment_is_reviewed",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="比较分钟特征与冻结事件流 embedding")
    parser.add_argument("--minute-config", type=Path, required=True)
    parser.add_argument("--minute-features", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, action="append", required=True)
    parser.add_argument("--exposures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run_comparison(
        minute_config=load_minute_baseline_config(arguments.minute_config),
        minute_features_root=arguments.minute_features.expanduser().resolve(),
        embedding_roots=[path.expanduser().resolve() for path in arguments.embedding],
        comparison_config=load_comparison_config(arguments.comparison_config),
        output_root=arguments.output.expanduser().resolve(),
        exposure_path=(arguments.exposures.expanduser().resolve() if arguments.exposures else None),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
