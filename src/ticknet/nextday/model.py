"""分块编码日内订单簿序列的次日方向模型。"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from ticknet.dataset import NUM_CLASSES, NUM_FEATURES
from ticknet.model import DeepLOB


class NextDayOutput(NamedTuple):
    """端到端模型同时输出方向分类和连续横截面分数。"""

    logits: torch.Tensor
    score: torch.Tensor


class ChunkedDeepLOB(nn.Module):
    """先编码固定长度事件块，再汇总为一个股票日级向量。

    输入形状是 ``B × chunks × 1 × chunk_size × 40``。每个事件块复用论文结构的
    DeepLOB 编码器，块级 GRU 只处理几十个向量，显存和序列长度不会随原始 tick 数量
    线性压在一个 LSTM 上。
    """

    def __init__(
        self,
        *,
        chunks_per_sample: int = 10,
        chunk_size: int = 100,
        num_classes: int = NUM_CLASSES,
        intraday_embedding_size: int = 64,
        day_hidden_size: int = 64,
        day_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if chunks_per_sample < 1 or chunk_size < 1:
            raise ValueError("chunks_per_sample 和 chunk_size 应为正整数")
        if day_layers < 1:
            raise ValueError("day_layers 应为正整数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 应在 [0, 1) 内")
        self.chunks_per_sample = chunks_per_sample
        self.chunk_size = chunk_size
        self.intraday_encoder = DeepLOB(
            num_classes=num_classes,
            window_size=chunk_size,
            lstm_units=intraday_embedding_size,
        )
        self.chunk_sequence = nn.GRU(
            input_size=intraday_embedding_size,
            hidden_size=day_hidden_size,
            num_layers=day_layers,
            batch_first=True,
            dropout=dropout if day_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(day_hidden_size, num_classes)
        self.score_head = nn.Linear(day_hidden_size, 1)

    def encode_day(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``B × day_hidden_size`` 的股票日级表示。"""
        expected = (self.chunks_per_sample, 1, self.chunk_size, NUM_FEATURES)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "ChunkedDeepLOB 输入应为 "
                f"(B, {self.chunks_per_sample}, 1, {self.chunk_size}, {NUM_FEATURES})，"
                f"实际为 {tuple(x.shape)}"
            )
        batch_size = x.shape[0]
        chunks = x.reshape(-1, 1, self.chunk_size, NUM_FEATURES)
        embeddings = self.intraday_encoder.encode(chunks)
        embeddings = embeddings.reshape(batch_size, self.chunks_per_sample, -1)
        _, hidden = self.chunk_sequence(embeddings)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> NextDayOutput:
        """返回三分类 logits 和形状为 ``B`` 的连续交易分数。"""
        representation = self.dropout(self.encode_day(x))
        return NextDayOutput(
            logits=self.classification_head(representation),
            score=self.score_head(representation).squeeze(-1),
        )


def build_nextday_model(
    *,
    chunks_per_sample: int = 10,
    chunk_size: int = 100,
    num_classes: int = NUM_CLASSES,
    intraday_embedding_size: int = 64,
    day_hidden_size: int = 64,
    day_layers: int = 1,
    dropout: float = 0.0,
) -> ChunkedDeepLOB:
    return ChunkedDeepLOB(
        chunks_per_sample=chunks_per_sample,
        chunk_size=chunk_size,
        num_classes=num_classes,
        intraday_embedding_size=intraday_embedding_size,
        day_hidden_size=day_hidden_size,
        day_layers=day_layers,
        dropout=dropout,
    )
