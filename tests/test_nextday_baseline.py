"""次日聚合特征基线测试。"""

from deeplob.nextday.dataset import NextDayShardDataset
from deeplob.nextday.splits import WalkForwardSplit
from scripts.run_nextday_baseline import _aggregate
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
