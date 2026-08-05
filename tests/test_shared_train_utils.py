"""主链路与 FI-2010 复现共用的训练工具测试。

这些工具保留在 ``ticknet.train`` 中，被次日预测主链路（``ticknet.nextday``）
复用。FI-2010 专属的训练/调度逻辑测试已归档到 ``legacy/tests``。
"""

from __future__ import annotations

import numpy as np
import torch

from ticknet.train import f1_metrics, resolve_device


def test_f1_metrics_return_expected_keys():
    labels = np.array([0, 1, 2, 0, 1, 2])
    metrics = f1_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["per_class_precision"] == [1.0, 1.0, 1.0]


def test_resolve_device_accepts_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError, match="cpu 或 cuda"):
        resolve_device("tpu")
