"""用分钟级特征训练低成本树模型基线（内部对照 B）。

特征源可切换：
- ``l2_cache``：snapshot + order + trade 三模态合并的 33 维微观结构分钟特征
- ``tushare``：tushare 分钟 OHLCV

两个源共用同一套标签、日期切分、窗口聚合和横截面评估，便于直接对比
微观结构信息相对普通量价 bar 的增量价值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ticknet.nextday.metrics import evaluate_predictions
from ticknet.nextday.minute_baseline import (
    MINUTE_FEATURE_SOURCES,
    DayRows,
    MinuteBaselineConfig,
    MinuteExtractionReport,
    MinuteSample,
    build_samples,
    build_target_bundle,
    load_minute_baseline_config,
    read_l2_minute_rows,
    read_tushare_minute_rows,
)
from ticknet.nextday.minute_materialization import (
    complete_formal_samples,
    load_materialized_minute_features,
)
from ticknet.nextday.splits import WalkForwardSplit, parse_date
from ticknet.research.prediction_contract import (
    attach_formal_prediction_metadata,
    validate_formal_prediction_artifact,
)
from ticknet.research.protocol import ResearchProtocol

_KNOWN_SOURCES = MINUTE_FEATURE_SOURCES


def _load_config(path: str | Path) -> MinuteBaselineConfig:
    try:
        return load_minute_baseline_config(path)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


def _read_rows(
    config: MinuteBaselineConfig,
    targets: list[Any],
    report: MinuteExtractionReport,
) -> DayRows:
    if config.feature_source == "l2_cache":
        if not config.l2_root:
            raise SystemExit("feature_source 为 l2_cache 时必须提供 l2_root")
        return read_l2_minute_rows(
            config.l2_root,
            targets,
            keep_minutes=config.window_minutes,
            report=report,
        )
    if not config.tushare_root:
        raise SystemExit("feature_source 为 tushare 时必须提供 tushare_root")
    return read_tushare_minute_rows(
        config.tushare_root,
        targets,
        keep_minutes=config.window_minutes,
        report=report,
    )


def _build_samples_by_year(
    config: MinuteBaselineConfig,
    targets: list[Any],
    report: MinuteExtractionReport,
) -> list[MinuteSample]:
    """按年份流式读取分钟行并构建样本，控制峰值内存。

    一次性读取多年 L2 数据会把全部股票日的分钟序列常驻内存，多年滚动验证时
    容易触发 OOM。这里按 targets 的年份分块：每年只读当年 parquet、构建当年
    样本后立即释放该年行数据，峰值内存约为单年规模。
    """
    by_year: dict[int, list[Any]] = {}
    for target in targets:
        by_year.setdefault(target.trading_date.year, []).append(target)

    samples: list[MinuteSample] = []
    for year in sorted(by_year):
        year_targets = by_year[year]
        rows = _read_rows(config, year_targets, report)
        samples.extend(
            build_samples(
                rows,
                year_targets,
                window_minutes=config.window_minutes,
                min_window_minutes=config.min_window_minutes,
                report=report,
            )
        )
        del rows
    return samples


def _split_samples(
    samples: list[MinuteSample],
    split: WalkForwardSplit,
) -> dict[str, list[MinuteSample]]:
    parts: dict[str, list[MinuteSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        assigned = split.assign(sample.trading_date, sample.label_date)
        if (
            assigned is not None
            and sample.return_end_date is not None
            and not split.range_for(assigned).contains(sample.return_end_date)
        ):
            assigned = None
        if assigned is not None:
            parts[assigned].append(sample)
    return parts


def _complete_formal_samples(
    targets: list[Any],
    samples: list[MinuteSample],
    report: MinuteExtractionReport,
) -> list[MinuteSample]:
    """保留完整股票池；缺失分钟窗口使用 HGB 原生支持的全 NaN 特征。"""
    if not samples:
        raise ValueError("正式 HGB 没有任何可用于确定特征维度的分钟样本")
    return complete_formal_samples(
        targets,
        samples,
        feature_count=samples[0].features.size,
        report=report,
    )


def _formal_dataset_fingerprint(
    config: MinuteBaselineConfig,
    samples: list[MinuteSample],
    targets: list[Any],
) -> str:
    """按实际入模特征、正式标签、状态和数据语义计算稳定 SHA-256。"""
    digest = hashlib.sha256()
    semantics = {
        name: getattr(config, name)
        for name in (
            "start_date",
            "end_date",
            "top_n",
            "min_history_days",
            "liquidity_lookback_days",
            "min_liquidity_observations",
            "lower_quantile",
            "upper_quantile",
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
            "feature_source",
            "window_minutes",
            "min_window_minutes",
            "target_return_contract",
        )
    }
    digest.update(json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for sample in sorted(samples, key=lambda item: (item.trading_date, item.symbol)):
        record = (
            sample.trading_date.isoformat(),
            sample.symbol,
            sample.label_date.isoformat(),
            sample.return_end_date.isoformat() if sample.return_end_date else None,
            sample.label,
            sample.target_return,
            sample.feature_available,
        )
        digest.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
        features = np.asarray(sample.features, dtype="<f4")
        digest.update(np.asarray(features.shape, dtype="<i8").tobytes())
        digest.update(features.tobytes(order="C"))
    for target in sorted(
        targets,
        key=lambda item: (item.trading_date, not item.in_universe, item.symbol),
    ):
        record = (
            target.trading_date.isoformat(),
            target.symbol,
            target.label_date.isoformat(),
            target.return_end_date.isoformat(),
            target.portfolio_return,
            target.benchmark_return,
            target.can_buy,
            target.can_sell,
            target.in_universe,
            target.execution_status,
        )
        digest.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _save_test_predictions(
    items: list[MinuteSample],
    model: Any,
    path: Path,
) -> None:
    """把 test 集每个样本的预测明细写成 parquet，供成本后回测使用。

    每行一个股票日样本：symbol、输入日、标签日、目标收益、三分类概率、
    以及用于排序的连续分数（上涨概率减下跌概率）。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    features = np.stack([item.features for item in items])
    returns = np.asarray([item.target_return for item in items], dtype=np.float64)
    dates = [item.label_date for item in items]
    probabilities = model.predict_proba(features)
    scores = probabilities[:, 2] - probabilities[:, 0]
    table = pa.table(
        {
            "symbol": [item.symbol for item in items],
            "trading_date": [item.trading_date.isoformat() for item in items],
            "label_date": [date_value.isoformat() for date_value in dates],
            "target_return": returns,
            "score": scores,
            "prob_up": probabilities[:, 2],
            "prob_neutral": probabilities[:, 1],
            "prob_down": probabilities[:, 0],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)
    print(f"已保存 test 预测明细：{path}")


def _save_formal_test_predictions(
    items: list[MinuteSample],
    targets: list[Any],
    model: Any,
    path: Path,
    *,
    dataset_fingerprint: str,
    expected_universe_size: int,
) -> dict[str, Any]:
    """输出带正式 metadata、交易状态和动态股票池状态行的预测。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    features = np.stack([item.features for item in items])
    probabilities = model.predict_proba(features)
    scores = probabilities[:, 2] - probabilities[:, 0]
    target_index = {
        (target.label_date, target.symbol): target for target in targets if target.in_universe
    }
    output_label_dates = {item.label_date for item in items}
    rows: list[dict[str, Any]] = []
    for item, probability, score in zip(items, probabilities, scores, strict=True):
        target = target_index[(item.label_date, item.symbol)]
        rows.append(
            {
                "symbol": item.symbol,
                "trading_date": item.trading_date.isoformat(),
                "label_date": item.label_date.isoformat(),
                "return_end_date": target.return_end_date.isoformat(),
                "target_return": target.portfolio_return,
                "model_target_return": target.target_return,
                "benchmark_return": target.benchmark_return,
                "score": float(score),
                "prob_up": float(probability[2]),
                "prob_neutral": float(probability[1]),
                "prob_down": float(probability[0]),
                "can_buy": target.can_buy,
                "can_sell": target.can_sell,
                "in_universe": True,
                "execution_status": target.execution_status,
                "feature_available": item.feature_available,
            }
        )
    for target in targets:
        if target.in_universe or target.label_date not in output_label_dates:
            continue
        rows.append(
            {
                "symbol": target.symbol,
                "trading_date": target.trading_date.isoformat(),
                "label_date": target.label_date.isoformat(),
                "return_end_date": target.return_end_date.isoformat(),
                "target_return": target.portfolio_return,
                "model_target_return": target.target_return,
                "benchmark_return": target.benchmark_return,
                "score": 0.0,
                "prob_up": 0.0,
                "prob_neutral": 1.0,
                "prob_down": 0.0,
                "can_buy": target.can_buy,
                "can_sell": target.can_sell,
                "in_universe": False,
                "execution_status": target.execution_status,
                "feature_available": False,
            }
        )
    rows.sort(key=lambda row: (row["label_date"], not row["in_universe"], row["symbol"]))
    table = attach_formal_prediction_metadata(
        pa.Table.from_pylist(rows),
        dataset_fingerprint=dataset_fingerprint,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)
    report = validate_formal_prediction_artifact(
        path,
        expected_universe_size=expected_universe_size,
        expected_dataset_fingerprint=dataset_fingerprint,
    )
    print(f"已保存正式 test 预测明细：{path}")
    return report.to_dict()


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    return value


def _validate_formal_run(config: MinuteBaselineConfig, args: argparse.Namespace) -> None:
    if not config.formal:
        return
    if config.top_n != 400:
        raise ValueError("正式 prediction export 必须使用 top_n=400")
    if config.min_symbols_per_day != 400:
        raise ValueError("正式 prediction export 必须使用 min_symbols_per_day=400")
    if args.evaluate_test is not True:
        raise ValueError("正式 prediction export 必须显式设置 --evaluate-test")
    if args.save_predictions is None:
        raise ValueError("正式 prediction export 必须设置 --save-predictions")
    if parse_date(config.end_date) > ResearchProtocol().validation_end_date:
        raise ValueError("研究阶段正式 prediction export 不能进入 2026 locked 区间")
    if (
        getattr(args, "materialized_features", None) is not None
        and config.feature_source != "l2_cache"
    ):
        raise ValueError("正式已物化特征要求 feature_source=l2_cache")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行分钟级特征低成本基线")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=sorted(_KNOWN_SOURCES), default=None)
    parser.add_argument("--output", type=Path, default=Path("results/minute-baseline.json"))
    parser.add_argument(
        "--materialized-features",
        type=Path,
        default=None,
        help="读取已完整物化并校验的分钟聚合特征目录，跳过原始 L2 扫描",
    )
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="把 test 集每日成分与预测明细存为 parquet，供成本后回测",
    )
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.source is not None:
        config = replace(config, feature_source=args.source)

    _validate_formal_run(config, args)

    report = MinuteExtractionReport()
    bundle = build_target_bundle(config)
    targets = [target for target in bundle.targets if getattr(target, "in_universe", True)]
    materialized_feature_summary = None
    if args.materialized_features is not None:
        materialized = load_materialized_minute_features(
            config,
            targets,
            args.materialized_features,
            report,
        )
        samples = materialized.samples
        materialized_feature_summary = materialized.summary()
    else:
        samples = _build_samples_by_year(config, targets, report)
        if config.formal:
            samples = _complete_formal_samples(targets, samples, report)
    parts = _split_samples(samples, config.date_split())
    dataset_fingerprint = (
        _formal_dataset_fingerprint(config, samples, bundle.targets) if config.formal else None
    )

    def arrays(items: list[MinuteSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
        features = np.stack([item.features for item in items])
        labels = np.asarray([item.label for item in items], dtype=np.int64)
        returns = np.asarray([item.target_return for item in items], dtype=np.float64)
        dates = [item.label_date for item in items]
        return features, labels, returns, dates

    if not parts["train"] or not parts["val"]:
        raise ValueError(f"训练或验证样本为空：train={len(parts['train'])} val={len(parts['val'])}")

    train_x, train_y, _train_r, _train_d = arrays(parts["train"])
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        random_state=config.seed,
    )
    model.fit(train_x, train_y)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError(f"训练集应包含三个类别，实际为 {model.classes_.tolist()}")

    def metrics(items: list[MinuteSample]) -> dict[str, Any]:
        features, labels, returns, dates = arrays(items)
        probabilities = model.predict_proba(features)
        return evaluate_predictions(
            labels,
            probabilities,
            returns,
            dates,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )

    test_metrics = None
    formal_prediction_report = None
    if (config.feature_source == "tushare" or args.evaluate_test) and parts["test"]:
        test_metrics = metrics(parts["test"])
        if args.save_predictions is not None:
            if config.formal:
                formal_prediction_report = _save_formal_test_predictions(
                    parts["test"],
                    bundle.targets,
                    model,
                    args.save_predictions,
                    dataset_fingerprint=str(dataset_fingerprint),
                    expected_universe_size=config.top_n,
                )
            else:
                _save_test_predictions(
                    parts["test"],
                    model,
                    args.save_predictions,
                )
    if config.formal and formal_prediction_report is None:
        raise ValueError("正式 prediction export 没有生成可登记的 test 预测")

    result = _safe_json(
        {
            "model": f"minute_hist_gradient_boosting_{config.feature_source}",
            "feature_source": config.feature_source,
            "target_return_contract": config.target_return_contract,
            "window_minutes": config.window_minutes,
            "feature_count": int(train_x.shape[1]),
            "samples": {key: len(value) for key, value in parts.items()},
            "extraction": report.__dict__,
            "materialized_features": materialized_feature_summary,
            "formal_target_build": (
                asdict(bundle.formal_report) if bundle.formal_report is not None else None
            ),
            "dataset_fingerprint": dataset_fingerprint,
            "validation": metrics(parts["val"]),
            "test": test_metrics,
            "predictions_path": (
                str(args.save_predictions.resolve())
                if formal_prediction_report is not None and args.save_predictions is not None
                else None
            ),
            "formal_prediction_report": formal_prediction_report,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
