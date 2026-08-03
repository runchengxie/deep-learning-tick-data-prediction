"""分块 DeepLOB 模型测试。"""

import pytest
import torch

from deeplob.nextday.model import build_nextday_model


def test_chunked_model_forward_and_day_embedding_shape():
    model = build_nextday_model(
        chunks_per_sample=3,
        chunk_size=20,
        intraday_embedding_size=16,
        day_hidden_size=12,
    )
    features = torch.randn(2, 3, 1, 20, 40)
    assert model.encode_day(features).shape == (2, 12)
    output = model(features)
    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)


def test_chunked_model_backpropagates_into_intraday_encoder():
    model = build_nextday_model(chunks_per_sample=2, chunk_size=20)
    features = torch.randn(2, 2, 1, 20, 40)
    output = model(features)
    (output.logits.sum() + output.score.sum()).backward()
    assert model.intraday_encoder.conv1[0].weight.grad is not None


def test_chunked_model_rejects_wrong_number_of_chunks():
    model = build_nextday_model(chunks_per_sample=2, chunk_size=20)
    with pytest.raises(ValueError, match="输入应为"):
        model(torch.randn(2, 3, 1, 20, 40))
