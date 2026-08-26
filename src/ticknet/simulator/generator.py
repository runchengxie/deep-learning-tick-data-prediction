"""生成式背景订单流。

加载 eventstream Transformer（L2FoundationModel + VectorQuantizer），
以初始盘口 prefix + 历史事件为条件，自回归生成下一批背景订单。
测试与无模型环境下可用 stub 模式产出确定性样本。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .replay import BackgroundOrder

# 与 eventstream.dataset / model 对齐的常量
N_FEATURES = 80
N_STREAMS = 4  # 0 pad, 1 snap, 2 order, 3 trade
STREAM_SNAP, STREAM_ORDER, STREAM_TRADE = 1, 2, 3
N_ORDER_TYPES = 12


@dataclass
class GenerationContext:
    initial_bid: tuple[int, int]
    initial_ask: tuple[int, int]
    history: list[BackgroundOrder] = field(default_factory=list)
    n_orders: int = 1
    seed: int = 0
    # 模型模式输入：由 dataset 构造的 80 维特征与 ID 序列。
    # 若留空，则从 history + 初始盘口自动构造（端到端自回归）。
    features: object | None = None  # (seq_len, 80) array-like
    stream_ids: object | None = None  # (seq_len,) int
    order_ids: object | None = None  # (seq_len,) int


def events_to_features(
    history: list[BackgroundOrder],
    initial_bid: tuple[int, int],
    initial_ask: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把历史事件 + 初始盘口转成模型输入 (features, stream_ids, order_ids)。

    轻量构造：复用与 dataset 一致的 stream/otype 编码与 bps 公式思路，
    但不依赖 polars raw row。首行固定为初始盘口快照特征，其后每行一笔历史。
    """
    mid = (initial_bid[0] + initial_ask[0]) / 2.0

    def _order_features(o: BackgroundOrder, prev_time: int) -> tuple[np.ndarray, int, int]:
        f = np.zeros(N_FEATURES, dtype=np.float32)
        price_bps = ((o.price - mid) / mid * 10000.0) if mid > 0 else 0.0
        f[0] = price_bps / 100.0  # 缩放到与训练分布相近的量级
        f[1] = math.log1p(max(o.volume, 0)) / 10.0
        dt = max(o.time_ms - prev_time, 1)
        f[2] = math.log1p(dt) / 10.0
        f[3] = float(o.side)  # 方向
        # stream/otype one-hot 占位（模型 embedding 会处理语义，这里给连续提示）
        f[4] = 1.0 if o.side >= 0 else 0.0
        return f, STREAM_ORDER, 0

    rows: list[np.ndarray] = []
    sids: list[int] = []
    oids: list[int] = []

    # 初始盘口快照行
    snap = np.zeros(N_FEATURES, dtype=np.float32)
    bid_px, bid_vol = initial_bid
    ask_px, ask_vol = initial_ask
    if bid_px > 0:
        snap[0] = ((bid_px - mid) / mid * 10000.0) / 100.0
        snap[1] = math.log1p(bid_vol) / 10.0
    if ask_px > 0:
        snap[3] = ((ask_px - mid) / mid * 10000.0) / 100.0
        snap[4] = math.log1p(ask_vol) / 10.0
    rows.append(snap)
    sids.append(STREAM_SNAP)
    oids.append(0)

    prev = 0
    for o in history:
        f, s, oid = _order_features(o, prev if prev else o.time_ms)
        rows.append(f)
        sids.append(s)
        oids.append(oid)
        prev = o.time_ms

    return np.stack(rows), np.array(sids, dtype=np.int64), np.array(oids, dtype=np.int64)


class OrderGenerator:
    """从 replay 上下文生成背景订单。"""

    def __init__(self, model: nn.Module | None = None, stub: bool = False) -> None:
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

        若 ctx.features 为空，则从 ctx.history + 初始盘口自动构造输入，
        并自回归生成 n_orders 笔（每步把预测结果追加进历史再前向）。
        模型输出 reg 头 [price_bps, dt_log, qty_log]，转为价格/量/时间。
        """
        model = self.model
        assert model is not None
        model.eval()

        # 构造初始输入序列
        if ctx.features is not None:
            feats = np.asarray(ctx.features, dtype=np.float32)
            sids = np.asarray(
                ctx.stream_ids
                if ctx.stream_ids is not None
                else np.zeros(feats.shape[-2], dtype=np.int64)
            )
            oids = np.asarray(
                ctx.order_ids
                if ctx.order_ids is not None
                else np.zeros(feats.shape[-2], dtype=np.int64)
            )
            # 兼容调用方直接传带 batch 维的张量：(1, seq, f) -> (seq, f)
            if feats.ndim == 3 and feats.shape[0] == 1:
                feats = feats[0]
            if sids.ndim == 2 and sids.shape[0] == 1:
                sids = sids[0]
            if oids.ndim == 2 and oids.shape[0] == 1:
                oids = oids[0]
        else:
            feats, sids, oids = events_to_features(ctx.history, ctx.initial_bid, ctx.initial_ask)

        mid = (ctx.initial_bid[0] + ctx.initial_ask[0]) / 2.0
        history = list(ctx.history)
        last_time = history[-1].time_ms if history else 1000
        orders: list[BackgroundOrder] = []

        for _ in range(max(ctx.n_orders, 1)):
            x = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
            sid = torch.as_tensor(sids, dtype=torch.long).unsqueeze(0)
            oid = torch.as_tensor(oids, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model(x, sid, oid)
            # VQ 分支：模型启用 VQ 时走 token 解码
            if getattr(model, "use_vq", False) and "vq_codes" in out:
                bo = self._decode_vq(out, ctx, mid, last_time, len(orders))
                orders.append(bo)
                feats, sids, oids = events_to_features(
                    [*history, bo], ctx.initial_bid, ctx.initial_ask
                )
                history.append(bo)
                last_time = bo.time_ms
                continue
            reg = out["reg"]
            price_bps = float(reg[0, -1, 0].item())
            dt_log = float(reg[0, -1, 1].item())
            qty_log = float(reg[0, -1, 2].item())

            price = int(mid * (1.0 + price_bps / 10000.0))
            side = 1 if price_bps >= 0 else -1
            dt_ms = max(1, round(math.exp(dt_log)))
            volume = max(1, round(math.exp(qty_log)))

            bo = BackgroundOrder(
                time_ms=last_time + dt_ms,
                side=side,
                price=price,
                volume=volume,
                order_id=f"GEN{len(orders)}",
            )
            orders.append(bo)
            # 自回归：把新生成的订单追加进序列，供下一步前向
            f, s, o = events_to_features([*history, bo], ctx.initial_bid, ctx.initial_ask)
            feats, sids, oids = f, s, o
            history.append(bo)
            last_time = bo.time_ms

        return orders

    def _decode_vq(
        self,
        out: dict,
        ctx: GenerationContext,
        mid: float,
        last_time: int,
        idx: int,
    ) -> BackgroundOrder:
        """VQ token 解码分支（占位解码器）。

        真实场景：vq_codes[-1] 经 codebook 反解为订单字段。当前模型未提供
        code -> 字段 的解码器（仅 encoder 侧），这里用 code 的奇偶性做确定性
        占位映射，保证代码路径连通；真实 VQ 权重就绪后替换为 codebook 查表。
        """
        codes = out["vq_codes"]  # (batch, seq)
        last_code = int(codes[0, -1].item())
        price_bps = float((last_code % 20) - 10) * 2.0  # -20 ~ +18 bps
        side = 1 if price_bps >= 0 else -1
        price = int(mid * (1.0 + price_bps / 10000.0))
        dt_ms = max(1, 1 + last_code)
        volume = max(1, 100 * (1 + (last_code % 5)))
        return BackgroundOrder(
            time_ms=last_time + dt_ms,
            side=side,
            price=price,
            volume=volume,
            order_id=f"VQ{last_code}",
        )
