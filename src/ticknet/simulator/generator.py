"""生成式背景订单流。

加载 eventstream Transformer（L2FoundationModel + VectorQuantizer），
以初始盘口 prefix + 历史事件为条件，自回归生成下一批背景订单。
测试与无模型环境下可用 stub 模式产出确定性样本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .replay import BackgroundOrder


@dataclass
class GenerationContext:
    initial_bid: tuple[int, int]
    initial_ask: tuple[int, int]
    history: list[BackgroundOrder] = field(default_factory=list)
    n_orders: int = 1
    seed: int = 0


class OrderGenerator:
    """从 replay 上下文生成背景订单。"""

    def __init__(self, model: object | None = None, stub: bool = False) -> None:
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
        """真实模型生成：调用 L2FoundationModel 自回归解码 VQ token。

        需要 model 为 ticknet.eventstream.model.L2FoundationModel 实例，
        并通过 VectorQuantizer 解码为订单字段。此处接口预留，
        真实接入在模型权重可用后启用。
        """
        raise NotImplementedError("model-based generation requires trained weights")
