"""次日预测 Logistic Regression 基线脚本测试。

仓库此前缺少该脚本的测试，其余脚本均有对应测试文件。这里补齐：
- ``_aggregate`` 的日内降维逻辑（均值/标准差/末值/首尾变化）；
- ``_safe_json`` 对非有限浮点值的处理；
- ``main`` 端到端能在合成数据集上产出结构化结果，且默认不评估测试集。
"""

import json
import math

from scripts.run_nextday_baseline import _aggregate, _safe_json, main
from tests.test_nextday_train import _training_manifest
from ticknet.nextday.dataset import NextDayShardDataset
from ticknet.nextday.train import NextDayConfig


def _date_split_for(manifest: str):
    config = NextDayConfig(
        manifest_path=str(manifest),
        train_start="2024-01-02",
        train_end="2024-01-03",
        val_start="2024-01-04",
        val_end="2024-01-05",
        test_start="2024-01-06",
        test_end="2024-01-07",
    )
    return config.date_split()


def _make_manifest(tmp_path):
    """复用训练测试的 fixture 构造器，独立暴露给本模块。"""
    return _training_manifest(tmp_path)


def test_aggregate_flattens_intraday_events(tmp_path):
    manifest = _make_manifest(tmp_path)
    dataset = NextDayShardDataset(
        str(manifest),
        date_split=_date_split_for(manifest),
        split="train",
    )
    features, labels = _aggregate(dataset)
    # 40 原始特征 × (均值, 标准差, 末值, 首尾变化) = 160 维。
    assert features.shape[1] == dataset.num_features * 4
    assert len(labels) == len(dataset)


def test_safe_json_nullifies_non_finite_floats():
    payload = {"a": float("nan"), "b": float("inf"), "c": 1.5, "d": {"e": float("-inf")}}
    cleaned = _safe_json(payload)
    assert cleaned["a"] is None
    assert cleaned["b"] is None
    assert cleaned["c"] == 1.5
    assert cleaned["d"]["e"] is None
    assert math.isfinite(cleaned["c"])


def test_main_writes_baseline_json_without_test_metrics(tmp_path):
    manifest = _make_manifest(tmp_path)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"manifest_path: {manifest}",
                "train_start: '2024-01-02'",
                "train_end: '2024-01-03'",
                "val_start: '2024-01-04'",
                "val_end: '2024-01-05'",
                "test_start: '2024-01-06'",
                "test_end: '2024-01-07'",
                "seed: 0",
                "evaluate_test: false",
                "verify_data_checksums: false",
                "min_symbols_per_day: 2",
                "portfolio_quantile: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results" / "nextday-baseline.json"
    main(["--config", str(config_path), "--output", str(output), "--no-evaluate-test"])

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["model"] == "aggregate_lob_logistic_regression"
    assert result["feature_count"] == 160
    assert result["validation"] is not None
    assert result["test"] is None
    # 训练集三个类别都应出现，类别一致性已在 main 内校验。
    assert result["samples"]["train"] > 0
