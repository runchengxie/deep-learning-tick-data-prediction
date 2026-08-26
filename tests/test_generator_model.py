"""RED: generator 真实模型接入。

加载 eventstream 100M 权重到 L2FoundationModel，前向生成下一事件，
转为 BackgroundOrder。无 GPU 时用 CPU 加载（权重约 400MB，可用）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ticknet.eventstream.model import L2FoundationModel, ModelConfig
from ticknet.simulator.generator import GenerationContext, OrderGenerator
from ticknet.simulator.replay import BackgroundOrder

CHECKPOINT = (
    "artifacts/eventstream-h5-recent-fold/training/seed0/"
    "eventstream-top400-h5-capacity100m-recent.seed0.last.pt"
)

if not Path(CHECKPOINT).exists():
    pytest.skip("需要本地 eventstream 100M checkpoint，CI 环境不提供", allow_module_level=True)


def _dummy_batch(seq_len: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造随机但合法的输入张量（仅用于验证前向通路，不代表真实分布）。"""
    x = torch.randn(1, seq_len, 80, dtype=torch.float32)
    sid = torch.randint(0, 3, (1, seq_len))
    oid = torch.randint(0, 6, (1, seq_len))
    return x, sid, oid


def test_load_real_checkpoint_and_forward():
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = L2FoundationModel(ModelConfig(**_cfg_from_state(sd["model"])))
    model.load_state_dict(sd["model"], strict=True)
    model.eval()

    x, sid, oid = _dummy_batch()
    with torch.no_grad():
        out = model(x, sid, oid)
    assert "reg" in out
    assert out["reg"].shape == (1, 8, 3)


def test_generator_model_mode_produces_orders():
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = L2FoundationModel(ModelConfig(**_cfg_from_state(sd["model"])))
    model.load_state_dict(sd["model"], strict=True)
    model.eval()

    gen = OrderGenerator(model=model, stub=False)
    ctx = GenerationContext(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        history=[],
        n_orders=2,
        features=torch.randn(1, 8, 80, dtype=torch.float32),
        stream_ids=torch.randint(0, 3, (1, 8)),
        order_ids=torch.randint(0, 6, (1, 8)),
    )
    orders = gen.generate(ctx)
    assert len(orders) == 2
    for o in orders:
        assert isinstance(o, BackgroundOrder)
        assert o.volume > 0


def _cfg_from_state(state: dict) -> dict:
    """从 state_dict key 推断容量配置（blocks 数 / d_model）。"""
    n_layers = sum(1 for k in state if k.startswith("blocks.") and k.endswith("norm1.weight"))
    d_model = state["feat_proj.0.weight"].shape[0]
    n_heads = d_model // 64  # capacity100m 用 d_head=64
    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_model * 4,
        "max_seq": 512,
    }
