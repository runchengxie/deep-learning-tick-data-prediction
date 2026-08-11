"""分钟级序列的 GRU 基线（与 MinuteTCN 同接口的受控对比）。

与 ``minute_tcn.py`` 消费同一套分钟分片数据（``MinuteShardDataset``）与标签，
输入是未聚合的 ``T x features`` 分钟序列，模型换成 GRU，用于和聚合特征 HGB
基线及 TCN 做三路同口径对比。序列编码取自最后一层最后时间步的隐状态，
与 ``MinuteTCN`` 取最后时间步的表示方式一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ticknet.dataset import NUM_CLASSES
from ticknet.nextday.minute_tcn import MinuteOutput, MinuteRecord, MinuteShardDataset

__all__ = ["MinuteGRU", "MinuteRecord", "MinuteShardDataset", "build_minute_gru"]


class MinuteGRU(nn.Module):
    """分钟序列 GRU：输入 ``B x time x features``，输出分类和分数。"""

    def __init__(
        self,
        *,
        num_features: int,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features 应为正整数")
        if hidden_size < 1:
            raise ValueError("hidden_size 应为正整数")
        if num_layers < 1:
            raise ValueError("num_layers 应为正整数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 应在 [0, 1) 内")
        self.num_features = num_features
        self.input_projection = nn.Linear(num_features, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.score_head = nn.Linear(hidden_size, 1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``B x hidden_size`` 的序列级表示（最后一层最后时间步）。"""
        if x.ndim != 3 or x.shape[2] != self.num_features:
            raise ValueError(
                f"MinuteGRU 输入应为 (B, T, {self.num_features})，实际为 {tuple(x.shape)}"
            )
        h = self.input_projection(x)
        _output, last = self.gru(h)
        return last[-1]

    def forward(self, x: torch.Tensor) -> MinuteOutput:
        representation = self.dropout(self.encode_sequence(x))
        return MinuteOutput(
            logits=self.classification_head(representation),
            score=self.score_head(representation).squeeze(-1),
        )


def build_minute_gru(
    *,
    num_features: int,
    num_classes: int = NUM_CLASSES,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> MinuteGRU:
    return MinuteGRU(
        num_features=num_features,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
