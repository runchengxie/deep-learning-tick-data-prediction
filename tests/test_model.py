"""模型结构测试，用随机输入，不需要真实数据。"""

import torch
import torch.nn.functional as F

from src.dataset import NUM_CLASSES, NUM_FEATURES, WINDOW_SIZE
from src.model import build_model


def test_forward_shape():
    model = build_model()
    x = torch.randn(8, 1, WINDOW_SIZE, NUM_FEATURES)
    logits = model(x)
    assert logits.shape == (8, NUM_CLASSES)


def test_softmax_sums_to_one():
    model = build_model()
    x = torch.randn(8, 1, WINDOW_SIZE, NUM_FEATURES)
    probs = F.softmax(model(x), dim=1)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)


def test_param_count_in_sane_range():
    total = sum(p.numel() for p in build_model().parameters())
    # 骨架模型，参数量远小于论文全量，设一个宽松区间防回归
    assert 20_000 < total < 200_000
