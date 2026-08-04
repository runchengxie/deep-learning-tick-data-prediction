"""次日聚合特征基线测试。"""

import json

from deeplob.nextday.dataset import NextDayShardDataset
from deeplob.nextday.splits import WalkForwardSplit
from scripts.run_nextday_baseline import _aggregate, main
from tests.test_nextday_train import _training_manifest


def test_aggregate_baseline_uses_fixed_size_features(tmp_path):
    manifest = _training_manifest(tmp_path)
    split = WalkForwardSplit.from_strings(
        train_start="2024-01-02",
        train_end="2024-01-03",
        val_start="2024-01-04",
        val_end="2024-01-05",
        test_start="2024-01-06",
        test_end="2024-01-07",
    )
    dataset = NextDayShardDataset(manifest, date_split=split, split="train")
    features, labels = _aggregate(dataset)
    assert features.shape == (3, 160)
    assert labels.tolist() == [0, 1, 2]


def test_baseline_keeps_test_metrics_locked_by_default(tmp_path):
    manifest = _training_manifest(tmp_path)
    config = tmp_path / "baseline.yaml"
    config.write_text(
        "\n".join(
            [
                f"manifest_path: {manifest}",
                'train_start: "2024-01-02"',
                'train_end: "2024-01-03"',
                'val_start: "2024-01-04"',
                'val_end: "2024-01-05"',
                'test_start: "2024-01-06"',
                'test_end: "2024-01-07"',
                "min_symbols_per_day: 3",
                "evaluate_test: false",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "baseline.json"
    main(["--config", str(config), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["validation"]["evaluated_dates"] == 1
    assert result["test"] is None

    main(["--config", str(config), "--output", str(output), "--evaluate-test"])
    unlocked = json.loads(output.read_text(encoding="utf-8"))
    assert unlocked["test"]["evaluated_dates"] == 1
