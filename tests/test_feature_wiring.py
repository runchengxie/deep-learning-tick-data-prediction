"""RED: generator 从历史事件自主构造 80 维特征（端到端自回归）。

验证 model 模式不再依赖外部手塞 features，而是从 history + 初始盘口
自动构造模型输入并生成订单。
"""

from __future__ import annotations

import torch

from ticknet.eventstream.model import L2FoundationModel, ModelConfig
from ticknet.simulator.generator import GenerationContext, OrderGenerator
from ticknet.simulator.replay import BackgroundOrder

CHECKPOINT = (
    "artifacts/eventstream-h5-recent-fold/training/seed0/"
    "eventstream-top400-h5-capacity100m-recent.seed0.last.pt"
)


def _load_model():
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ModelConfig(
        d_model=sd["model"]["feat_proj.0.weight"].shape[0],
        n_layers=sum(
            1 for k in sd["model"] if k.startswith("blocks.") and k.endswith("norm1.weight")
        ),
        n_heads=sd["model"]["feat_proj.0.weight"].shape[0] // 64,
        d_ff=sd["model"]["feat_proj.0.weight"].shape[0] * 4,
        max_seq=512,
    )
    model = L2FoundationModel(cfg)
    model.load_state_dict(sd["model"], strict=True)
    model.eval()
    return model


def test_events_to_features_shape():
    from ticknet.simulator.generator import events_to_features

    history = [
        BackgroundOrder(time_ms=1000, side=1, price=1000, volume=500, order_id="B1"),
        BackgroundOrder(time_ms=1010, side=-1, price=1001, volume=300, order_id="B2"),
    ]
    feats, sids, oids = events_to_features(
        history, initial_bid=(1000, 500), initial_ask=(1001, 300)
    )
    assert feats.shape == (3, 80)  # 初始快照 + 2 笔
    assert sids.shape == (3,)
    assert oids.shape == (3,)


def test_generator_autonomous_from_history():
    model = _load_model()
    gen = OrderGenerator(model=model, stub=False)
    ctx = GenerationContext(
        initial_bid=(1000, 500),
        initial_ask=(1001, 300),
        history=[
            BackgroundOrder(time_ms=1000, side=1, price=1000, volume=500),
            BackgroundOrder(time_ms=1010, side=-1, price=1001, volume=300),
        ],
        n_orders=2,
        # 注意：不再传 features，应从 history 自动构造
    )
    orders = gen.generate(ctx)
    assert len(orders) == 2
    for o in orders:
        assert isinstance(o, BackgroundOrder)
        assert o.volume > 0
