"""DeepLOB 网络结构测试。"""

import pytest
import torch
import torch.nn.functional as F

from deeplob.dataset import NUM_CLASSES, NUM_FEATURES, WINDOW_SIZE
from deeplob.model import InceptionModule, build_model


def test_forward_shape():
    model = build_model()
    features = torch.randn(8, 1, WINDOW_SIZE, NUM_FEATURES)
    logits = model(features)
    assert logits.shape == (8, NUM_CLASSES)


def test_softmax_sums_to_one():
    model = build_model()
    features = torch.randn(8, 1, WINDOW_SIZE, NUM_FEATURES)
    probabilities = F.softmax(model(features), dim=1)
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(8),
        atol=1e-5,
    )


def test_architecture_matches_paper_scale():
    model = build_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 55_000 < parameter_count < 70_000
    assert isinstance(model.inception, InceptionModule)
    assert model.inception.output_channels == 96
    assert model.lstm.input_size == 96
    assert model.lstm.hidden_size == 64


def test_rejects_non_paper_feature_shape():
    model = build_model()
    with pytest.raises(ValueError, match="40"):
        model(torch.randn(2, 1, WINDOW_SIZE, 144))
