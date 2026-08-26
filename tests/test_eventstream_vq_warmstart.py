"""eventstream 热启动：非 VQ checkpoint 作为启用 VQ 实验的初始化。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ticknet.eventstream.model import build_eventstream_model
from ticknet.eventstream.train import EventstreamConfig, _warm_start_from_checkpoint


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    torch.save({"model": model.state_dict()}, path)


def test_warm_start_fills_only_vq_modules(tmp_path: Path) -> None:
    base = build_eventstream_model("smoke")
    ckpt = tmp_path / "base.pt"
    _save_checkpoint(base, ckpt)

    vq_model = build_eventstream_model("smoke", use_vq=True, vq_dim=16)
    _warm_start_from_checkpoint(vq_model, ckpt, torch.device("cpu"), use_vq=True)

    # 主干权重与源一致
    src = base.state_dict()
    dst = vq_model.state_dict()
    for key in ("feat_proj.0.weight", "blocks.0.norm1.weight", "head_reg.weight"):
        assert torch.equal(src[key], dst[key])
    # VQ 模块保持随机初始化（与另一个新实例不同）
    other = build_eventstream_model("smoke", use_vq=True, vq_dim=16)
    codebook = dst["vector_quantizer.codebook.weight"]
    assert not torch.equal(codebook, other.state_dict()["vector_quantizer.codebook.weight"])


def test_warm_start_rejects_unexpected_keys(tmp_path: Path) -> None:
    vq_source = build_eventstream_model("smoke", use_vq=True, vq_dim=16)
    ckpt = tmp_path / "vq.pt"
    _save_checkpoint(vq_source, ckpt)

    plain = build_eventstream_model("smoke")
    with pytest.raises(ValueError, match="无法对齐"):
        _warm_start_from_checkpoint(plain, ckpt, torch.device("cpu"), use_vq=False)


def test_warm_start_rejects_missing_backbone(tmp_path: Path) -> None:
    base = build_eventstream_model("smoke")
    state = base.state_dict()
    state.pop("feat_proj.0.weight")
    ckpt = tmp_path / "broken.pt"
    torch.save({"model": state}, ckpt)

    vq_model = build_eventstream_model("smoke", use_vq=True, vq_dim=16)
    with pytest.raises(ValueError, match="非 VQ 主干"):
        _warm_start_from_checkpoint(vq_model, ckpt, torch.device("cpu"), use_vq=True)


def test_config_accepts_init_checkpoint() -> None:
    config = EventstreamConfig.from_mapping(
        {"days": (20210104,), "model": "smoke", "device": "cpu", "init_checkpoint": "/tmp/x.pt"}
    )
    assert config.init_checkpoint == "/tmp/x.pt"
    assert "init_checkpoint" in config.to_dict()
