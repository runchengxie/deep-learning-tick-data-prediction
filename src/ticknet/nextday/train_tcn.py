"""分钟序列 TCN 的次日横截面方向训练入口。

输入是未聚合的 ``T x features`` 分钟序列，模型使用 ``MinuteTCN``。数据切分、
训练、恢复和评估流程由分钟序列模型共享训练模块统一执行。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from ticknet.nextday.config import NextDayConfig
from ticknet.nextday.minute_sequence_training import (
    evaluate_best_minute_checkpoints,
    evaluate_minute_model,
    load_minute_config,
    make_minute_dataloaders,
    train_minute_sequence_model,
)
from ticknet.nextday.minute_tcn import build_minute_tcn


@dataclass
class MinuteTCNConfig(NextDayConfig):
    """分钟 TCN 实验配置，继承通用字段并追加 TCN 结构超参。"""

    hidden_channels: int = 64
    tcn_layers: int = 4
    kernel_size: int = 3

    def validate(self) -> None:
        super().validate()
        if self.hidden_channels < 1 or self.tcn_layers < 1 or self.kernel_size < 1:
            raise ValueError("TCN 隐藏维度、层数和核大小应为正整数")


def make_tcn_dataloaders(
    config: NextDayConfig,
    *,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """保留原入口名称，构造分钟序列的三段数据加载器。"""
    return make_minute_dataloaders(config, device=device)


evaluate = evaluate_minute_model


def _build_model(config: MinuteTCNConfig, num_features: int) -> torch.nn.Module:
    return build_minute_tcn(
        num_features=num_features,
        hidden_channels=config.hidden_channels,
        num_layers=config.tcn_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )


def train(config: MinuteTCNConfig) -> dict[str, Any]:
    """训练分钟 TCN，并按配置决定是否评估完整测试区间。"""
    return train_minute_sequence_model(
        config,
        lambda num_features: _build_model(config, num_features),
    )


def load_config(argv: list[str] | None = None) -> MinuteTCNConfig:
    return load_minute_config(
        MinuteTCNConfig,
        description="训练分钟序列 TCN 到次日横截面方向模型",
        argv=argv,
    )


def evaluate_best_checkpoints(
    config: MinuteTCNConfig,
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
        description="用固定 best checkpoint 一次性评估分钟 TCN locked test",
    )
    probe.add_argument("--seeds", nargs="+", type=int, required=True)
    arguments, remaining = probe.parse_known_args(argv)
    evaluate_best_checkpoints(load_config(remaining), arguments.seeds)


if __name__ == "__main__":
    main()
