"""无需真实数据的本地冒烟检查。

运行方式：

    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ticknet.dataset import (
    K_TO_LABEL_COLUMN,
    NUM_CLASSES,
    NUM_FEATURES,
    TOTAL_COLUMNS,
    WINDOW_SIZE,
    FI2010WindowDataset,
    get_dummy_batch,
)
from ticknet.model import build_model


def check_forward_pass() -> None:
    model = build_model()
    features, _ = get_dummy_batch(batch_size=8)
    logits = model(features)
    probabilities = F.softmax(logits, dim=1)
    assert features.shape == (8, 1, WINDOW_SIZE, NUM_FEATURES)
    assert logits.shape == (8, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-5)
    print("通过：前向传播形状和 softmax 概率")


def check_gradient_flow() -> None:
    model = build_model()
    features, labels = get_dummy_batch(batch_size=16)
    loss = torch.nn.CrossEntropyLoss()(model(features), labels)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    print(f"通过：梯度存在且为有限值，损失为 {loss.item():.4f}")


def check_parameter_count() -> None:
    parameter_count = sum(parameter.numel() for parameter in build_model().parameters())
    assert 55_000 < parameter_count < 70_000
    print(f"通过：模型参数量为 {parameter_count:,}，符合论文约 60k 的规模")


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
    check_gradient_flow()
    check_parameter_count()
    check_fi2010_dataset()
    print("全部冒烟检查通过。")


if __name__ == "__main__":
    main()
