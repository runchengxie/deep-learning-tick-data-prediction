"""RED: generator VQ token 解码路径。

验证 generator 能检测模型是否启用 VQ：启用时走 token 解码分支，
未启用时降级到连续特征通路。无真实 VQ 权重时用构造的 mock 模型验证
代码路径，不要求语义正确。
"""
from __future__ import annotations

import torch

from ticknet.eventstream.model import L2FoundationModel, ModelConfig, VectorQuantizer
from ticknet.simulator.generator import OrderGenerator, GenerationContext
from ticknet.simulator.replay import BackgroundOrder


def _make_vq_model(d_model: int = 64, n_layers: int = 1) -> L2FoundationModel:
    """构造启用 VQ 的 mock 模型（随机权重，仅验证代码通路）。"""
    cfg = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=2, d_ff=d_model * 2, max_seq=64)
    model = L2FoundationModel(cfg, use_vq=True, vq_codebook_size=16, vq_dim=8)
    return model


def _ctx() -> GenerationContext:
    return GenerationContext(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        history=[
            BackgroundOrder(time_ms=1000, side=1, price=1000, volume=500),
            BackgroundOrder(time_ms=1010, side=-1, price=1001, volume=300),
        ],
        n_orders=2,
        features=torch.randn(1, 8, 80, dtype=torch.float32),
        stream_ids=torch.randint(0, 3, (1, 8)),
        order_ids=torch.randint(0, 6, (1, 8)),
    )


def test_vq_model_generates_orders():
    model = _make_vq_model()
    assert model.use_vq is True
    gen = OrderGenerator(model=model, stub=False)
    orders = gen.generate(_ctx())
    assert len(orders) == 2
    for o in orders:
        assert isinstance(o, BackgroundOrder)
        assert o.volume > 0


def test_non_vq_model_still_works():
    # 连续通路（与 #101 行为一致）：无 VQ 权重时降级
    cfg = ModelConfig(d_model=64, n_layers=1, n_heads=2, d_ff=128, max_seq=64)
    model = L2FoundationModel(cfg, use_vq=False)
    gen = OrderGenerator(model=model, stub=False)
    orders = gen.generate(_ctx())
    assert len(orders) == 2
