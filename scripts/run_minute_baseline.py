"""用分钟级特征训练低成本树模型基线（内部对照 B）。

特征源可切换：
- ``l2_cache``：snapshot + order + trade 三模态合并的 33 维微观结构分钟特征
- ``tushare``：tushare 分钟 OHLCV

两个源共用同一套标签、日期切分、窗口聚合和横截面评估，便于直接对比
微观结构信息相对普通量价 bar 的增量价值。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ticknet.nextday.metrics import evaluate_predictions
from ticknet.nextday.minute_baseline import (
    MinuteBaselineConfig,
    MinuteExtractionReport,
    MinuteSample,
    build_samples,
    build_targets,
    read_l2_minute_rows,
    read_tushare_minute_rows,
)
from ticknet.nextday.splits import WalkForwardSplit

_KNOWN_SOURCES = {"l2_cache", "tushare"}


def _load_config(path: str | Path) -> MinuteBaselineConfig:
    import yaml

    values: dict[str, Any] = {}
    if path:
        with Path(path).open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise SystemExit("minute YAML 根节点应为对象")
        values.update(loaded)
    defaults = MinuteBaselineConfig(
        basic_root=values.get("basic_root", ""),
        benchmark_path=values.get("benchmark_path", ""),
        start_date=values.get("start_date", ""),
        end_date=values.get("end_date", ""),
        top_n=int(values.get("top_n", 400)),
        min_history_days=int(values.get("min_history_days", 120)),
        liquidity_lookback_days=int(values.get("liquidity_lookback_days", 20)),
        min_liquidity_observations=int(values.get("min_liquidity_observations", 15)),
        lower_quantile=float(values.get("lower_quantile", 0.2)),
        upper_quantile=float(values.get("upper_quantile", 0.8)),
        min_cross_section=int(values.get("min_cross_section", 20)),
        train_start=values.get("train_start", ""),
        train_end=values.get("train_end", ""),
        val_start=values.get("val_start", ""),
        val_end=values.get("val_end", ""),
        test_start=values.get("test_start", ""),
        test_end=values.get("test_end", ""),
        feature_source=str(values.get("feature_source", "l2_cache")),
        l2_root=str(values.get("l2_root", "")),
        tushare_root=str(values.get("tushare_root", "")),
        window_minutes=int(values.get("window_minutes", 60)),
        min_window_minutes=int(values.get("min_window_minutes", 30)),
        min_symbols_per_day=int(values.get("min_symbols_per_day", 20)),
        portfolio_quantile=float(values.get("portfolio_quantile", 0.1)),
        seed=int(values.get("seed", 0)),
    )
    if defaults.feature_source not in _KNOWN_SOURCES:
        raise SystemExit(f"feature_source 应为 {sorted(_KNOWN_SOURCES)} 之一")
    if not defaults.basic_root or not defaults.benchmark_path:
        raise SystemExit("basic_root 和 benchmark_path 不能为空")
    if not defaults.start_date or not defaults.end_date:
        raise SystemExit("start_date 和 end_date 不能为空")
    for name in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
        if not getattr(defaults, name):
            raise SystemExit(f"{name} 不能为空")
    return defaults


def _read_rows(
    config: MinuteBaselineConfig,
    targets: list[Any],
    report: MinuteExtractionReport,
) -> dict[tuple[int, str], list[tuple[int, np.ndarray]]]:
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


def _split_samples(
    samples: list[MinuteSample],
    split: WalkForwardSplit,
) -> dict[str, list[MinuteSample]]:
    parts: dict[str, list[MinuteSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        assigned = split.assign(sample.trading_date, sample.label_date)
        if assigned is not None:
            parts[assigned].append(sample)
    return parts


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行分钟级特征低成本基线")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=sorted(_KNOWN_SOURCES), default=None)
    parser.add_argument("--output", type=Path, default=Path("results/minute-baseline.json"))
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.source is not None:
        config = replace(config, feature_source=args.source)

    report = MinuteExtractionReport()
    targets = build_targets(config)
    rows = _read_rows(config, targets, report)
    samples = build_samples(
        rows,
        targets,
        window_minutes=config.window_minutes,
        min_window_minutes=config.min_window_minutes,
        report=report,
    )
    parts = _split_samples(samples, config.date_split())

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
    if (config.feature_source == "tushare" or args.evaluate_test) and parts["test"]:
        test_metrics = metrics(parts["test"])

    result = _safe_json(
        {
            "model": f"minute_hist_gradient_boosting_{config.feature_source}",
            "feature_source": config.feature_source,
            "window_minutes": config.window_minutes,
            "feature_count": int(train_x.shape[1]),
            "samples": {key: len(value) for key, value in parts.items()},
            "extraction": report.__dict__,
            "validation": metrics(parts["val"]),
            "test": test_metrics,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
