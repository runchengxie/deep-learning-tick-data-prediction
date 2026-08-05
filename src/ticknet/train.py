"""主链路与 FI-2010 复现共用的训练工具。

这里只保留被次日预测主链路（``ticknet.nextday``）复用的公共工具：
``set_seed``、``resolve_device`` 和 ``f1_metrics``。FI-2010 论文复现的完整
训练/评估/实验调度已归档到 ``legacy/fi2010_train.py``。
"""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch

NUM_CLASSES = 3


class Metrics(TypedDict):
    """分类评估指标。"""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: list[float]
    per_class_recall: list[float]


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """解析运行设备，并在 CUDA 不可用时给出清楚提示。"""
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"device 应为 cpu 或 cuda，收到 {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        print("未检测到 CUDA，将使用 CPU。")
        return torch.device("cpu")
    return torch.device(requested)


def f1_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Metrics:
    """计算准确率、F1，以及各类别的精确率和召回率。"""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in recall],
    }
