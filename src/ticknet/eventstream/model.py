"""因果 Transformer：归并后的 L2 事件流下一事件预测 + 日级信号头。

多任务头：
    - stream：下一事件流类型（0 pad, 1 snap, 2 order, 3 trade）
    - otype：下一订单类型（vocab）
    - reg：下一事件 price_bps / dt_log / qty_log（Smooth L1）
    - day：外部标签表提供的日级连续信号（每个有效位置监督同一标量）

尺寸：
    probe25m : d=512,  12 层  -> ~25M 主干参数
    capacity100m: d=960, 9 层 -> 100.6M 全部参数
    probe150m: d=1024, 12 层  -> ~150M 主干参数
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ticknet.eventstream.dataset import N_FEATURES, N_ORDER_TYPES

N_STREAMS = 4  # 0 pad, 1 snap, 2 order, 3 trade
LOSS_WEIGHTS = {"stream": 1.0, "otype": 0.5, "reg": 1.0, "day": 1.0}


@dataclass
class ModelConfig:
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.0
    max_seq: int = 8192


CONFIGS: dict[str, ModelConfig] = {
    "smoke": ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq=64),
    "probe25m": ModelConfig(d_model=512, n_layers=12, n_heads=8, d_ff=2048),
    "probe50m": ModelConfig(d_model=768, n_layers=10, n_heads=12, d_ff=3072),
    "capacity100m": ModelConfig(d_model=960, n_layers=9, n_heads=15, d_ff=3840),
    "probe150m": ModelConfig(d_model=1024, n_layers=12, n_heads=16, d_ff=4096),
}


class _RotaryCache(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        inverse: torch.Tensor = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        position: torch.Tensor = torch.arange(max_seq, dtype=torch.float32)
        freqs: torch.Tensor = position[:, None] * inverse[None, :]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:length], self.sin[:length]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, Dh)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos.view(1, 1, -1, cos.shape[-1])
    s = sin.view(1, 1, -1, sin.shape[-1])
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class _Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def shape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        att = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        att = att.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.proj(att)
        x = x + self.mlp(self.norm2(x))
        return x


class L2FoundationModel(nn.Module):
    """事件流因果 Transformer，输出多任务头与 day 头。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat_proj = nn.Sequential(
            nn.Linear(N_FEATURES, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.stream_emb = nn.Embedding(N_STREAMS, cfg.d_model)
        self.otype_emb = nn.Embedding(N_ORDER_TYPES, cfg.d_model)
        self.rope = _RotaryCache(cfg.d_model // cfg.n_heads, cfg.max_seq)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.LayerNorm(cfg.d_model)

        self.head_stream = nn.Linear(cfg.d_model, N_STREAMS)
        self.head_otype = nn.Linear(cfg.d_model, N_ORDER_TYPES)
        self.head_reg = nn.Linear(cfg.d_model, 3)  # next price_bps, dt_log, qty_log
        self.head_day = nn.Linear(cfg.d_model, 1)  # 日级信号

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def backbone(self, x: torch.Tensor, sid: torch.Tensor, oid: torch.Tensor) -> torch.Tensor:
        h = self.feat_proj(x) + self.stream_emb(sid) + self.otype_emb(oid)
        cos, sin = self.rope(x.shape[1])
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.norm_f(h)

    def forward(
        self, x: torch.Tensor, sid: torch.Tensor, oid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.backbone(x, sid, oid)
        return {
            "stream": self.head_stream(h),
            "otype": self.head_otype(h),
            "reg": self.head_reg(h),
            "day": self.head_day(h).squeeze(-1),
            "hidden": h,
        }


def compute_loss_components(
    out: dict[str, torch.Tensor],
    tgt_sid: torch.Tensor,
    tgt_oid: torch.Tensor,
    tgt_reg: torch.Tensor,
    tgt_day: torch.Tensor,
    day_valid: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """返回事件流训练的四项未加权损失，供训练和梯度审计共用。

    day 头在每个有效位置预测同一个 (ticker, day) 日级标签，day_valid 屏蔽无标签样本。
    """
    mask = valid > 0
    n = mask.sum().clamp(min=1)

    ce_stream = F.cross_entropy(
        out["stream"].flatten(0, 1), tgt_sid.flatten(), reduction="none"
    ).view_as(tgt_sid)
    ce_stream = (ce_stream * valid).sum() / n

    is_order_next = (tgt_sid == 2) & mask
    if is_order_next.any():
        ce_otype = F.cross_entropy(out["otype"][is_order_next], tgt_oid[is_order_next])
    else:
        ce_otype = out["otype"].sum() * 0.0

    reg_err = F.smooth_l1_loss(out["reg"], tgt_reg, reduction="none").mean(-1)
    reg_loss = (reg_err * valid).sum() / n

    day_mask = valid * day_valid[:, None]
    day_err = F.smooth_l1_loss(out["day"], tgt_day[:, None].expand_as(out["day"]), reduction="none")
    day_loss = (day_err * day_mask).sum() / day_mask.sum().clamp(min=1)

    return {
        "stream": ce_stream,
        "otype": ce_otype,
        "reg": reg_loss,
        "day": day_loss,
    }


def compute_loss(
    out: dict[str, torch.Tensor],
    tgt_sid: torch.Tensor,
    tgt_oid: torch.Tensor,
    tgt_reg: torch.Tensor,
    tgt_day: torch.Tensor,
    day_valid: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """多任务损失：stream CE + otype CE + reg SmoothL1 + day SmoothL1。"""
    components = compute_loss_components(
        out,
        tgt_sid,
        tgt_oid,
        tgt_reg,
        tgt_day,
        day_valid,
        valid,
    )
    total = components["stream"] * LOSS_WEIGHTS["stream"]
    for name in ("otype", "reg", "day"):
        total = total + components[name] * LOSS_WEIGHTS[name]
    return total, {
        "loss": float(total.detach()),
        "ce_stream": float(components["stream"].detach()),
        "ce_otype": float(components["otype"].detach()),
        "reg": float(components["reg"].detach()),
        "day": float(components["day"].detach()),
    }


def build_eventstream_model(name: str) -> L2FoundationModel:
    if name not in CONFIGS:
        raise ValueError(f"未知模型配置：{name}，可用 {sorted(CONFIGS)}")
    model = L2FoundationModel(CONFIGS[name])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {name}: {n_params / 1e6:.1f}M params")
    return model
