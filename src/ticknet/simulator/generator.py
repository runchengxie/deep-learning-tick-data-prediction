"""生成式背景订单流。

加载 eventstream Transformer（L2FoundationModel + VectorQuantizer），
以初始盘口 prefix + 历史事件为条件，自回归生成下一批背景订单。
测试与无模型环境下可用 stub 模式产出确定性样本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from .replay import BackgroundOrder


@dataclass
class GenerationContext:
    initial_bid: tuple[int, int]
    initial_ask: tuple[int, int]
    history: list[BackgroundOrder] = field(default_factory=list)
    n_orders: int = 1
    seed: int = 0
    # 模型模式输入：由 dataset 构造的 80 维特征与 ID 序列
    features: object | None = None  # (seq_len, 80) array-like
    stream_ids: object | None = None  # (seq_len,) int
    order_ids: object | None = None  # (seq_len,) int


class OrderGenerator:
    """从 replay 上下文生成背景订单。"""

    def __init__(self, model: Optional[nn.Module] = None, stub: bool = False) -> None:
        self.model = model
        self.stub = stub or model is None

    def generate(self, ctx: GenerationContext) -> list[BackgroundOrder]:
        if self.stub:
            return self._stub_generate(ctx)
        return self._model_generate(ctx)

    def _stub_generate(self, ctx: GenerationContext) -> list[BackgroundOrder]:
        """确定性占位生成：在买一/卖一附近交替出小单，便于测试与管线接通。"""
        bid_px, _ = ctx.initial_bid
        ask_px, _ = ctx.initial_ask
        orders: list[BackgroundOrder] = []
        for i in range(ctx.n_orders):
            side = 1 if i % 2 == 0 else -1
            price = bid_px if side == 1 else ask_px
            volume = 100 * (i + 1)
            orders.append(
                BackgroundOrder(time_ms=1000 + i * 10, side=side, price=price, volume=volume)
            )
        return orders

    def _model_generate(self, ctx: GenerationContext) -> list[BackgroundOrder]:
        """真实模型生成：加载的 L2FoundationModel 前向预测下一事件，解码为订单。

        输入：ctx.features 为 (seq_len, 80) 浮点张量，ctx.stream_ids / ctx.order_ids
        为对应 (seq_len,) 长整型张量。若未提供，则退化为 stub 行为。
        模型输出 reg 头 [price_bps, dt_log, qty_log]，转为价格/量/时间。
        """
        import torch

        if ctx.features is None or self.model is None:
            return self._stub_generate(ctx)

        model = self.model
        assert model is not None
        model.eval()
        x = torch.as_tensor(ctx.features, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        sid = torch.as_tensor(
            ctx.stream_ids
            if ctx.stream_ids is not None
            else torch.zeros(x.shape[1], dtype=torch.long)
        )
        oid = torch.as_tensor(
            ctx.order_ids
            if ctx.order_ids is not None
            else torch.zeros(x.shape[1], dtype=torch.long)
        )
        if sid.dim() == 1:
            sid = sid.unsqueeze(0)
        if oid.dim() == 1:
            oid = oid.unsqueeze(0)
        with torch.no_grad():
            out = model(x, sid, oid)
        reg = out["reg"]
        # reg: (batch, seq, 3) -> [price_bps, dt_log, qty_log]
        price_bps = float(reg[0, -1, 0].item())
        dt_log = float(reg[0, -1, 1].item())
        qty_log = float(reg[0, -1, 2].item())

        # 相对初始盘口中间价解码价格
        mid = (ctx.initial_bid[0] + ctx.initial_ask[0]) / 2.0
        price = int(mid * (1.0 + price_bps / 10000.0))
        side = 1 if price_bps >= 0 else -1
        import math

        dt_ms = max(1, round(math.exp(dt_log)))
        volume = max(1, round(math.exp(qty_log)))
        return [
            BackgroundOrder(
                time_ms=1000 + i * dt_ms,
                side=side,
                price=price,
                volume=volume,
            )
            for i in range(ctx.n_orders)
        ]
