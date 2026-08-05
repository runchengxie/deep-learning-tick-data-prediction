"""FI-2010 数据集的本地冒烟检查（已归档，不再纳入主链路门禁）。

运行方式：

    python legacy/scripts/fi2010_smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from legacy.fi2010_core import (
    K_TO_LABEL_COLUMN,
    NUM_CLASSES,
    NUM_FEATURES,
    TOTAL_COLUMNS,
    WINDOW_SIZE,
    FI2010WindowDataset,
    get_dummy_batch,
)


def check_forward_pass() -> None:
    import torch
    import torch.nn.functional as F

    from ticknet.model import build_model

    model = build_model()
    features, _ = get_dummy_batch(batch_size=8)
    logits = model(features)
    probabilities = F.softmax(logits, dim=1)
    assert features.shape == (8, 1, WINDOW_SIZE, NUM_FEATURES)
    assert logits.shape == (8, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-5)
    print("通过：前向传播形状和 softmax 概率")


def check_fi2010_dataset() -> None:
    segment_length = 500
    segments = [
        {"cf": 7, "role": "train", "start": 0, "end": segment_length},
        {
            "cf": 7,
            "role": "test",
            "start": segment_length,
            "end": segment_length * 2,
        },
        {
            "cf": 8,
            "role": "test",
            "start": segment_length * 2,
            "end": segment_length * 3,
        },
        {
            "cf": 9,
            "role": "test",
            "start": segment_length * 3,
            "end": segment_length * 4,
        },
    ]
    rows = segment_length * 4
    rng = np.random.default_rng(1)
    data = np.zeros((rows, TOTAL_COLUMNS), dtype=np.float32)
    data[:, :NUM_FEATURES] = rng.standard_normal(
        (rows, NUM_FEATURES),
        dtype=np.float32,
    )
    labels = np.resize(np.array([1, 2, 3], dtype=np.float32), rows)
    for column in K_TO_LABEL_COLUMN.values():
        data[:, column] = labels

    with tempfile.TemporaryDirectory() as directory:
        data_path = Path(directory) / "fi2010.npy"
        meta_path = Path(directory) / "fi2010_meta.json"
        np.save(data_path, data)
        meta_path.write_text(
            json.dumps({"rows": rows, "segments": segments}),
            encoding="utf-8",
        )
        for horizon in K_TO_LABEL_COLUMN:
            with FI2010WindowDataset(
                str(data_path),
                str(meta_path),
                k=horizon,
                split="train",
                protocol="setup2",
            ) as dataset:
                features, label = dataset[0]
                assert features.shape == (1, WINDOW_SIZE, NUM_FEATURES)
                assert label in {0, 1, 2}
    print("通过：五个预测跨度的数据窗口和标签")


def main() -> None:
    check_forward_pass()
    check_fi2010_dataset()
    print("全部 FI-2010 冒烟检查通过。")


if __name__ == "__main__":
    main()
