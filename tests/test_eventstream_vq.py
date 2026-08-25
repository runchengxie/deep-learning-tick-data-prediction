"""事件流 Hybrid VQ 残差分支测试。"""

from __future__ import annotations

import torch

from ticknet.eventstream.dataset import N_FEATURES
from ticknet.eventstream.model import (
    build_eventstream_model,
    compute_loss,
    compute_loss_components,
)


def _targets(length: int = 3):
    tgt_sid = torch.tensor([[2, 3, 1][:length]], dtype=torch.long)
    tgt_oid = torch.zeros((1, length), dtype=torch.long)
    tgt_reg = torch.zeros((1, length, 3), dtype=torch.float32)
    tgt_day = torch.tensor([0.25], dtype=torch.float32)
    day_valid = torch.tensor([1.0], dtype=torch.float32)
    valid = torch.ones((1, length), dtype=torch.float32)
    return tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid


def test_disabled_vq_preserves_legacy_model_state_shapes() -> None:
    legacy = build_eventstream_model("smoke")
    disabled = build_eventstream_model("smoke", use_vq=False)

    legacy_count = sum(parameter.numel() for parameter in legacy.parameters())
    disabled_count = sum(parameter.numel() for parameter in disabled.parameters())
    assert legacy_count == disabled_count
    assert {
        name: tuple(value.shape) for name, value in legacy.state_dict().items()
    } == {name: tuple(value.shape) for name, value in disabled.state_dict().items()}


def test_vq_returns_codes_and_ignores_prefix_and_padding() -> None:
    torch.manual_seed(0)
    model = build_eventstream_model(
        "smoke",
        use_vq=True,
        vq_codebook_size=8,
        vq_dim=4,
    )
    x = torch.randn(1, 5, N_FEATURES)
    sid = torch.tensor([[0, 2, 3, 1, 0]], dtype=torch.long)
    oid = torch.tensor([[11, 1, 0, 0, 0]], dtype=torch.long)

    out = model(x, sid, oid)

    assert out["vq_codes"].shape == sid.shape
    assert out["vq_codes"].dtype == torch.long
    assert out["vq_codes"][0, 0].item() == -1
    assert out["vq_codes"][0, 4].item() == -1
    assert torch.all(out["vq_codes"][sid != 0] >= 0)
    assert torch.isfinite(out["vq_loss"])
    assert out["vq_loss"].item() >= 0.0


def test_vq_regularizer_changes_total_loss_without_changing_task_components() -> None:
    length = 3
    out = {
        "stream": torch.zeros((1, length, 4), requires_grad=True),
        "otype": torch.zeros((1, length, 12), requires_grad=True),
        "reg": torch.zeros((1, length, 3), requires_grad=True),
        "day": torch.zeros((1, length), requires_grad=True),
        "vq_loss": torch.tensor(2.0, requires_grad=True),
    }
    targets = _targets(length)

    components = compute_loss_components(out, *targets)
    base, base_metrics = compute_loss(out, *targets, vq_loss_weight=0.0)
    regularized, regularized_metrics = compute_loss(out, *targets, vq_loss_weight=0.5)

    assert set(components) == {"stream", "otype", "reg", "day"}
    assert torch.allclose(regularized - base, torch.tensor(1.0))
    assert base_metrics["vq"] == 2.0
    assert regularized_metrics["vq"] == 2.0
