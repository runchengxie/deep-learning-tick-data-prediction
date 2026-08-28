"""事件流因果 Transformer 模型与多任务损失测试。"""

from __future__ import annotations

import pytest
import torch

from ticknet.eventstream.dataset import N_FEATURES, N_ORDER_TYPES
from ticknet.eventstream.model import (
    DAY_SUPERVISION_WEIGHT_VERSION,
    LOSS_WEIGHTS,
    build_eventstream_model,
    compute_loss,
    compute_loss_components,
    day_supervision_weights,
)


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
        components = compute_loss_components(
            out, tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid
        )
        loss, metrics = compute_loss(out, tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid)
        assert loss.ndim == 0
        assert set(components) == set(LOSS_WEIGHTS)
        expected = components["stream"] * LOSS_WEIGHTS["stream"]
        for name in ("otype", "reg", "day"):
            expected = expected + components[name] * LOSS_WEIGHTS[name]
        assert torch.equal(loss, expected)
        assert set(metrics) == {"loss", "ce_stream", "ce_otype", "reg", "day"}
        loss.backward()
        qkv = model.blocks[0].qkv
        assert isinstance(qkv, torch.nn.Linear)
        assert qkv.weight.grad is not None

    def test_unknown_model_name(self):
        with pytest.raises(ValueError, match="未知模型配置"):
            build_eventstream_model("nope")

    def test_day_loss_weight_scales_only_day_component(self):
        model = build_eventstream_model("smoke")
        batch, length = 2, 4
        x = torch.randn(batch, length, N_FEATURES)
        sid = torch.randint(1, 4, (batch, length))
        oid = torch.randint(0, N_ORDER_TYPES, (batch, length))
        targets = (
            torch.randint(0, 4, (batch, length)),
            torch.randint(0, N_ORDER_TYPES, (batch, length)),
            torch.randn(batch, length, 3),
            torch.randn(batch),
            torch.ones(batch),
            torch.ones(batch, length),
        )
        out = model(x, sid, oid)
        components = compute_loss_components(out, *targets)
        weighted, _ = compute_loss(out, *targets, day_loss_weight=2.0)
        expected = components["stream"] + components["otype"] * 0.5 + components["reg"]
        assert torch.allclose(weighted, expected + components["day"] * 2.0)

    def test_rejects_negative_day_loss_weight(self):
        with pytest.raises(ValueError, match="day_loss_weight"):
            compute_loss(
                {
                    "stream": torch.zeros(1, 1, 4),
                    "otype": torch.zeros(1, 1, N_ORDER_TYPES),
                    "reg": torch.zeros(1, 1, 3),
                    "day": torch.zeros(1, 1),
                },
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1, 3),
                torch.zeros(1),
                torch.ones(1),
                torch.ones(1, 1),
                day_loss_weight=-1.0,
            )


class TestDaySupervision:
    def test_all_preserves_every_valid_position(self):
        valid = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        day_valid = torch.tensor([1.0, 0.0])

        weights = day_supervision_weights(valid, day_valid, mode="all")

        assert torch.equal(weights, torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0] * 4]))

    def test_last_uses_each_samples_final_valid_position(self):
        valid = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        day_valid = torch.ones(3)

        weights = day_supervision_weights(valid, day_valid, mode="last")

        assert torch.equal(
            weights,
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        )

    def test_tail_weighted_uses_linear_position_weights(self):
        valid = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        day_valid = torch.tensor([1.0, 1.0])

        weights = day_supervision_weights(valid, day_valid, mode="tail_weighted")

        expected = torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0], [1.0 / 2.0, 1.0, 0.0, 0.0]])
        assert torch.allclose(weights, expected)
        assert DAY_SUPERVISION_WEIGHT_VERSION == "linear-v1"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="day_supervision_mode"):
            day_supervision_weights(torch.ones(1, 2), torch.ones(1), mode="middle")
