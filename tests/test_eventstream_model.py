"""事件流因果 Transformer 模型与多任务损失测试。"""

from __future__ import annotations

import pytest
import torch

from ticknet.eventstream.dataset import N_FEATURES, N_ORDER_TYPES
from ticknet.eventstream.model import build_eventstream_model, compute_loss


class TestModel:
    def test_forward_shapes(self):
        model = build_eventstream_model("smoke")
        model.eval()
        batch, length = 2, 16
        x = torch.randn(batch, length, N_FEATURES)
        sid = torch.randint(1, 4, (batch, length))
        oid = torch.randint(0, N_ORDER_TYPES, (batch, length))
        with torch.no_grad():
            out = model(x, sid, oid)
        assert out["stream"].shape == (batch, length, 4)
        assert out["otype"].shape == (batch, length, N_ORDER_TYPES)
        assert out["reg"].shape == (batch, length, 3)
        assert out["day"].shape == (batch, length)
        assert out["hidden"].shape == (batch, length, model.cfg.d_model)

    def test_backward_updates_weights(self):
        model = build_eventstream_model("smoke")
        batch, length = 2, 16
        x = torch.randn(batch, length, N_FEATURES)
        sid = torch.randint(1, 4, (batch, length))
        oid = torch.randint(0, N_ORDER_TYPES, (batch, length))
        tgt_sid = torch.randint(0, 4, (batch, length))
        tgt_oid = torch.randint(0, N_ORDER_TYPES, (batch, length))
        tgt_reg = torch.randn(batch, length, 3)
        tgt_day = torch.randn(batch)
        day_valid = torch.ones(batch)
        valid = torch.ones(batch, length)
        out = model(x, sid, oid)
        loss, metrics = compute_loss(out, tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid)
        assert loss.ndim == 0
        assert set(metrics) == {"loss", "ce_stream", "ce_otype", "reg", "day"}
        loss.backward()
        qkv = model.blocks[0].qkv
        assert isinstance(qkv, torch.nn.Linear)
        assert qkv.weight.grad is not None

    def test_unknown_model_name(self):
        with pytest.raises(ValueError, match="未知模型配置"):
            build_eventstream_model("nope")
