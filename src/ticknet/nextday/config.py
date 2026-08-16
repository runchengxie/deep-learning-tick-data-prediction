"""次日横截面实验的配置定义与校验。

``NextDayConfig`` 聚合了一次固定日期区间实验所需的全部参数：数据/日期区间、
训练超参与模型结构。把配置从 ``train.py`` 拆到本模块，可以让训练/评估逻辑专注于
流程，也方便测试与脚本在不触碰训练代码的前提下构造配置。

配置同时被 ``ticknet.nextday.train`` 重新导出，因此下游仍可从
``ticknet.nextday.train`` 导入 ``NextDayConfig``。
"""

from __future__ import annotations

from dataclasses import dataclass

from ticknet.nextday.splits import WalkForwardSplit

SELECTION_METRICS = {"daily_rank_ic_mean", "macro_f1", "balanced_accuracy", "mcc"}
DEFAULT_CONV_CHANNELS = 16
DEFAULT_INCEPTION_CHANNELS = 32
LOCKED_TEST_AGGREGATE_METRICS = (
    "daily_rank_ic_mean",
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "brier_score",
    "daily_long_short_return_mean",
)


@dataclass
class NextDayConfig:
    """一次固定日期区间实验的配置。"""

    manifest_path: str | None = None
    target_sidecar_path: str | None = None
    target_horizon: int = 1
    input_last_chunks: int = 0
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str = "2024-12-31"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0
    patience: int = 8
    seed: int = 0
    num_workers: int = 0
    device: str = "cpu"
    resume: bool = True
    evaluate_test: bool = False
    verify_data_checksums: bool = True
    checkpoint_dir: str = "./checkpoints-nextday"
    checkpoint_name: str = "chunked-ticknet"
    conv_channels: int = DEFAULT_CONV_CHANNELS
    inception_channels: int = DEFAULT_INCEPTION_CHANNELS
    intraday_embedding_size: int = 64
    day_hidden_size: int = 64
    day_layers: int = 1
    dropout: float = 0.1
    class_weighting: str = "balanced"
    selection_metric: str = "daily_rank_ic_mean"
    min_symbols_per_day: int = 20
    portfolio_quantile: float = 0.1
    classification_loss_weight: float = 1.0
    regression_loss_weight: float = 0.5
    gradient_accumulation_steps: int = 1
    amp: bool = True

    def _validate_target(self) -> None:
        if self.target_horizon < 1:
            raise ValueError("target_horizon 应为正整数")
        if self.target_horizon != 1 and not self.target_sidecar_path:
            raise ValueError("target_horizon 大于 1 时必须提供 target_sidecar_path")

    def _validate_input(self) -> None:
        if self.input_last_chunks < 0:
            raise ValueError("input_last_chunks 不能为负数")

    def validate(self) -> None:
        if not self.manifest_path:
            raise ValueError("manifest_path 不能为空")
        self._validate_target()
        self._validate_input()
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs、batch_size 和 patience 应为正整数")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr 应为正数，weight_decay 不能为负数")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")
        model_dimensions = (
            self.conv_channels,
            self.inception_channels,
            self.intraday_embedding_size,
            self.day_hidden_size,
            self.day_layers,
        )
        if any(value < 1 for value in model_dimensions):
            raise ValueError("模型隐藏维度和层数应为正整数")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 应在 [0, 1) 内")
        if self.class_weighting not in {"none", "balanced"}:
            raise ValueError("class_weighting 应为 none 或 balanced")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric 应为 {sorted(SELECTION_METRICS)} 中的一个")
        if self.min_symbols_per_day < 2:
            raise ValueError("min_symbols_per_day 至少为 2")
        if not 0 < self.portfolio_quantile <= 0.5:
            raise ValueError("portfolio_quantile 应在 (0, 0.5] 内")
        if self.classification_loss_weight < 0 or self.regression_loss_weight < 0:
            raise ValueError("两个损失权重不能为负数")
        if self.classification_loss_weight + self.regression_loss_weight <= 0:
            raise ValueError("至少一个损失权重必须为正数")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps 应为正整数")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device 应为 cpu 或 cuda")
        self.date_split()

    def date_split(self) -> WalkForwardSplit:
        return WalkForwardSplit.from_strings(
            train_start=self.train_start,
            train_end=self.train_end,
            val_start=self.val_start,
            val_end=self.val_end,
            test_start=self.test_start,
            test_end=self.test_end,
        )
