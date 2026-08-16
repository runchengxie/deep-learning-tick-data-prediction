"""分钟序列 GRU 的次日横截面方向训练入口。

输入是未聚合的 ``T x features`` 分钟序列，模型使用 ``MinuteGRU``。数据切分、
训练、恢复和评估流程由分钟序列模型共享训练模块统一执行。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ticknet.nextday.config import NextDayConfig
from ticknet.nextday.minute_gru import build_minute_gru
from ticknet.nextday.minute_sequence_training import (
    evaluate_best_minute_checkpoints,
    load_minute_config,
    train_minute_sequence_model,
)


@dataclass
class MinuteGRUConfig(NextDayConfig):
    """分钟 GRU 实验配置，继承通用字段并追加 GRU 结构超参。"""

    gru_hidden_size: int = 64
    gru_layers: int = 2

    def validate(self) -> None:
        super().validate()
        if self.gru_hidden_size < 1 or self.gru_layers < 1:
            raise ValueError("GRU 隐藏维度和层数应为正整数")


def _build_model(config: MinuteGRUConfig, num_features: int) -> torch.nn.Module:
    return build_minute_gru(
        num_features=num_features,
        hidden_size=config.gru_hidden_size,
        num_layers=config.gru_layers,
        dropout=config.dropout,
    )


def train(config: MinuteGRUConfig) -> dict[str, Any]:
    """训练分钟 GRU，并按配置决定是否评估完整测试区间。"""
    return train_minute_sequence_model(
        config,
        lambda num_features: _build_model(config, num_features),
    )


def load_config(argv: list[str] | None = None) -> MinuteGRUConfig:
    return load_minute_config(
        MinuteGRUConfig,
        description="训练分钟序列 GRU 到次日横截面方向模型",
        argv=argv,
    )


def evaluate_best_checkpoints(
    config: MinuteGRUConfig,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """只读取固定的最佳 checkpoint，并一次性评估 locked test。"""
    return evaluate_best_minute_checkpoints(
        config,
        seeds,
        lambda num_features: _build_model(config, num_features),
    )


def main(argv: list[str] | None = None) -> None:
    train(load_config(argv))


def evaluate_main(argv: list[str] | None = None) -> None:
    probe = argparse.ArgumentParser(
        description="用固定 best checkpoint 一次性评估分钟 GRU locked test",
    )
    probe.add_argument("--seeds", nargs="+", type=int, required=True)
    arguments, remaining = probe.parse_known_args(argv)
    evaluate_best_checkpoints(load_config(remaining), arguments.seeds)


if __name__ == "__main__":
    main()
