"""DeepLOB 模型使用的共享张量形状常量与合成数据工具。

FI-2010 数据集类（``RandomLOBDataset`` / ``FI2010WindowDataset``）已归档到
``legacy/fi2010_core.py``。这里只保留主链路与冒烟检查仍依赖的常量和工具函数。
"""

from __future__ import annotations

import torch

WINDOW_SIZE = 100
NUM_FEATURES = 40
NUM_CLASSES = 3


def get_dummy_batch(
    batch_size: int = 8,
    window_size: int = WINDOW_SIZE,
    num_features: int = NUM_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一批可直接交给模型的合成数据。"""
    x = torch.randn(batch_size, 1, window_size, num_features)
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    return x, y
