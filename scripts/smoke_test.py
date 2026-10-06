"""Run lightweight synthetic-data model and FI-2010 checks.

运行方式：

    python scripts/smoke_test.py

"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ticknet.dataset import NUM_CLASSES, NUM_FEATURES, WINDOW_SIZE, get_dummy_batch
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


def main() -> None:
    check_forward_pass()
    check_gradient_flow()
    check_parameter_count()
    from ticknet.fi2010.scripts.fi2010_smoke_test import check_fi2010_dataset

    check_fi2010_dataset()
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
