"""DeepLOB 模型。

结构对应论文图 3 和图 4：

* 输入形状为 ``(B, 1, 100, 40)``
* 三个卷积块提取订单簿空间特征和局部时间特征
* 三分支 Inception 模块提取多时间尺度特征
* 64 单元 LSTM 汇总时间信息
* 线性层输出三个类别的 logits

训练时直接把 logits 交给 ``CrossEntropyLoss``。该损失内部包含
``log_softmax``，与论文使用 softmax 和分类交叉熵的目标一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_FEATURES = 40
WINDOW_SIZE = 100
NUM_CLASSES = 3


def _temporal_layer(channels: int, leak: float) -> nn.Sequential:
    """构造保持时间长度不变的 ``4×1`` 卷积层。"""
    return nn.Sequential(
        nn.ZeroPad2d((0, 0, 1, 2)),
        nn.Conv2d(channels, channels, kernel_size=(4, 1)),
        nn.LeakyReLU(negative_slope=leak),
        nn.BatchNorm2d(channels),
    )


class InceptionModule(nn.Module):
    """论文图 4 的三分支 Inception 模块。"""

    def __init__(self, in_channels: int = 16, branch_channels: int = 32, leak: float = 0.01):
        super().__init__()
        self.output_channels = branch_channels * 3

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(5, 1),
                padding=(2, 0),
            ),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(branch_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (self.branch_3(x), self.branch_5(x), self.branch_pool(x)),
            dim=1,
        )


class DeepLOB(nn.Module):
    """论文中的 DeepLOB 网络。"""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        window_size: int = WINDOW_SIZE,
        conv_channels: int = 16,
        inception_channels: int = 32,
        lstm_units: int = 64,
        leak: float = 0.01,
    ):
        super().__init__()
        self.window_size = window_size
        self.embedding_size = lstm_units

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(conv_channels),
            _temporal_layer(conv_channels, leak),
            _temporal_layer(conv_channels, leak),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(conv_channels),
            _temporal_layer(conv_channels, leak),
            _temporal_layer(conv_channels, leak),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=leak),
            nn.BatchNorm2d(conv_channels),
            _temporal_layer(conv_channels, leak),
            _temporal_layer(conv_channels, leak),
        )

        self.inception = InceptionModule(
            in_channels=conv_channels,
            branch_channels=inception_channels,
            leak=leak,
        )
        self.lstm = nn.LSTM(
            input_size=self.inception.output_channels,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.output = nn.Linear(lstm_units, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回最后一个 LSTM 状态，供下游任务复用日内编码器。"""
        expected = (self.window_size, NUM_FEATURES)
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[2:]) != expected:
            raise ValueError(
                "DeepLOB 输入应为 "
                f"(B, 1, {self.window_size}, {NUM_FEATURES})，实际为 {tuple(x.shape)}"
            )

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.inception(x)
        x = x.squeeze(-1).transpose(1, 2)
        x, _ = self.lstm(x)
        return x[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``(B, 3)`` 的 logits。"""
        return self.output(self.encode(x))


def build_model(
    num_classes: int = NUM_CLASSES,
    window_size: int = WINDOW_SIZE,
) -> DeepLOB:
    """创建论文配置下的 DeepLOB 模型。"""
    return DeepLOB(num_classes=num_classes, window_size=window_size)


if __name__ == "__main__":
    model = build_model()
    sample = torch.randn(2, 1, WINDOW_SIZE, NUM_FEATURES)
    print(model)
    print(f"参数量：{sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"输出形状：{tuple(model(sample).shape)}")
