"""分钟 GRU 模型测试。"""

from __future__ import annotations

import pytest
import torch

from ticknet.dataset import NUM_CLASSES
from ticknet.nextday.minute_gru import MinuteGRU, build_minute_gru


class TestMinuteGRU:
    def test_forward_shapes(self):
        model = build_minute_gru(num_features=8, hidden_size=16, num_layers=2)
        model.eval()
        batch, minutes = 4, 20
        x = torch.randn(batch, minutes, 8)
        with torch.no_grad():
            output = model(x)
        assert output.logits.shape == (batch, NUM_CLASSES)
        assert output.score.shape == (batch,)
        encoded = model.encode_sequence(x)
        assert encoded.shape == (batch, 16)

    def test_backward_updates_weights(self):
        model = MinuteGRU(num_features=8, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(2, 10, 8)
        output = model(x)
        loss = (
            torch.nn.functional.cross_entropy(output.logits, torch.zeros(2, dtype=torch.long))
            + output.score.mean()
        )
        loss.backward()
        assert model.gru.weight_ih_l0.grad is not None

    def test_rejects_bad_input_rank(self):
        model = build_minute_gru(num_features=8)
        with pytest.raises(ValueError, match="MinuteGRU"):
            model.encode_sequence(torch.randn(2, 8))

    def test_rejects_bad_hyperparams(self):
        with pytest.raises(ValueError, match="hidden_size"):
            build_minute_gru(num_features=8, hidden_size=0)
        with pytest.raises(ValueError, match="num_layers"):
            build_minute_gru(num_features=8, num_layers=0)
        with pytest.raises(ValueError, match="dropout"):
            build_minute_gru(num_features=8, dropout=1.0)
