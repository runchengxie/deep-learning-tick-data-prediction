"""用日内聚合特征训练 Logistic Regression 基线。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deeplob.nextday.dataset import NextDayShardDataset
from deeplob.nextday.metrics import evaluate_predictions
from deeplob.nextday.train import load_config


def _aggregate(dataset: NextDayShardDataset) -> tuple[np.ndarray, np.ndarray]:
    """把原始日内序列变为均值、标准差、末值和首尾变化。"""
    features = np.empty((len(dataset), dataset.num_features * 4), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int64)
    for index in range(len(dataset)):
        chunks, label, _target_return = dataset[index]
        events = chunks[:, 0].reshape(-1, dataset.num_features)
        features[index] = np.concatenate(
            (
                events.mean(axis=0),
                events.std(axis=0),
                events[-1],
                events[-1] - events[0],
            )
        )
        labels[index] = label
    return features, labels


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行次日方向 Logistic Regression 基线")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("results/nextday-baseline.json"))
    args = parser.parse_args(argv)
    config = load_config(["--config", args.config])
    if config.manifest_path is None:
        raise ValueError("manifest_path 不能为空")
    date_split = config.date_split()
    train = NextDayShardDataset(config.manifest_path, date_split=date_split, split="train")
    validation = NextDayShardDataset(config.manifest_path, date_split=date_split, split="val")
    test = NextDayShardDataset(config.manifest_path, date_split=date_split, split="test")
    train_x, train_y = _aggregate(train)
    val_x, val_y = _aggregate(validation)
    test_x, test_y = _aggregate(test)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=config.seed,
        ),
    )
    model.fit(train_x, train_y)
    classifier = model[-1]
    if not np.array_equal(classifier.classes_, np.arange(3)):
        raise ValueError(f"训练集应包含三个类别，实际为 {classifier.classes_.tolist()}")

    def metrics(
        dataset: NextDayShardDataset,
        labels: np.ndarray,
        features: np.ndarray,
    ) -> dict[str, Any]:
        return evaluate_predictions(
            labels,
            model.predict_proba(features),
            dataset.target_returns,
            dataset.label_dates,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )

    result = _safe_json(
        {
            "model": "aggregate_lob_logistic_regression",
            "feature_count": int(train_x.shape[1]),
            "samples": {"train": len(train), "val": len(validation), "test": len(test)},
            "validation": metrics(validation, val_y, val_x),
            "test": metrics(test, test_y, test_x),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
