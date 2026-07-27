"""DeepLOB model (PyTorch implementation).

Architecture mapping to the paper (Zhang, Zohren, Roberts 2020, arXiv:1808.03668):

  Input  X ∈ R^{B, 1, T=100, D=40}
    │  40 features = [pa(i), va(i), pb(i), vb(i)] for i=1..10  (Eq.6)
    │
    ├─ Conv2d(1×2, stride 1×2) @16   ← pairs {p(i), v(i)} per level   (Sec.IV-B.a)
    ├─ Conv2d(1×2, stride 1×2) @16   ← integrates across levels → micro-price-like (Eq.7)
    ├─ Conv2d(1×10)            @16   ← integrates all levels → (T, 1) feature map
    │
    ├─ Inception Module @32          ← multi-timescale (3×1, 5×1, 1×1, maxpool) (Sec.IV-B.b, Fig.4)
    │
    ├─ LSTM @64                      ← temporal dependency (Sec.IV-B.c)
    │
    └─ Linear(64 → 3) + LogSoftmax   ← P(Down)/P(Stationary)/P(Up)

Channels evolve:
  40 -> 16 (after first conv, T kept via padding) -> 16 -> 1 (after 1×10 conv)
  -> 32 (inception) -> LSTM 64 -> 3.

NOTE on the 1×2 stride: it prevents price(p) and volume(v) from sharing
convolution weights with the neighbouring {v(i), p(i+1)} pair, which would be
semantically wrong (Sec.IV-B.a).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Default feature dim for official FI-2010 (144 features + 4 labels).
# Kept in sync with dataset.NUM_FEATURES.
NUM_FEATURES = 144


class InceptionModule(nn.Module):
    """Inception block for LOB feature maps (paper Fig.4).

    Operates on a single 'column' axis (the feature/level axis is already
    reduced to 1 by the time data reaches here). The module mixes:
      - 1×1 conv  (Network-in-Network, dimensionality reduction)
      - 3×1 conv  (short-timescale interaction)
      - 5×1 conv  (long-timescale interaction)
      - 3×1 max-pool + 1×1 conv
    and concatenates the four branches along the channel axis.
    """

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        # split the output channels evenly across the 4 branches
        c = out_channels // 4  # paper uses 32 -> 8 per branch

        self.b1_conv1x1 = nn.Conv2d(in_channels, c, kernel_size=(1, 1))

        self.b2_conv1x1 = nn.Conv2d(in_channels, c, kernel_size=(1, 1))
        self.b2_conv3x1 = nn.Conv2d(c, c, kernel_size=(3, 1), padding=(1, 0))

        self.b3_conv1x1 = nn.Conv2d(in_channels, c, kernel_size=(1, 1))
        self.b3_conv5x1 = nn.Conv2d(c, c, kernel_size=(5, 1), padding=(2, 0))

        self.b4_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.b4_conv1x1 = nn.Conv2d(in_channels, c, kernel_size=(1, 1))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.relu(self.b1_conv1x1(x))

        b2 = self.relu(self.b2_conv1x1(x))
        b2 = self.relu(self.b2_conv3x1(b2))

        b3 = self.relu(self.b3_conv1x1(x))
        b3 = self.relu(self.b3_conv5x1(b3))

        b4 = self.b4_pool(x)
        b4 = self.relu(self.b4_conv1x1(b4))

        return torch.cat([b1, b2, b3, b4], dim=1)


class DeepLOB(nn.Module):
    """DeepLOB: CNN + Inception + LSTM for limit-order-book forecasting."""

    def __init__(
        self,
        num_classes: int = 3,
        window_size: int = 100,
        conv_channels: int = 16,
        inception_channels: int = 32,
        lstm_units: int = 64,
        leak: float = 0.01,
        num_features: int = NUM_FEATURES,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_classes = num_classes

        # --- Convolutional front-end (Sec.IV-B.a) ---
        # All conv layers use zero-padding on the time axis so T stays 100.
        # Leaky-ReLU with negative slope 0.01 (grid-searched on validation set).
        self.leaky = nn.LeakyReLU(negative_slope=leak)

        # Two (1×2) strided convs halve the feature axis twice:
        #   D -> D/2 -> D/4   (requires D divisible by 4; 144 -> 36, 40 -> 10)
        # A final (1 x D/4) conv collapses the remaining axis to 1.
        assert num_features % 4 == 0, f"num_features must be divisible by 4, got {num_features}"
        feat_after_2 = num_features // 4

        self.conv1 = nn.Conv2d(
            1, conv_channels, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )
        # A (1 x feat_after_2) conv collapses feat_after_2 -> 1, leaving (T, 1) map.
        self.conv3 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=(1, feat_after_2), stride=(1, 1), padding=(0, 0)
        )

        # --- Inception module (Sec.IV-B.b) ---
        # Input channels into inception = conv_channels (16).
        self.inception = InceptionModule(conv_channels, inception_channels)

        # --- LSTM (Sec.IV-B.c) ---
        # Inception output has 1 'column' left; squeeze it before LSTM.
        # LSTM input size = inception_channels (32).
        self.lstm = nn.LSTM(
            input_size=inception_channels,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )

        # --- Output (softmax over 3 classes) ---
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape (B, 1, T, D)  with T=window_size, D=40.
        Returns:
            log-probabilities of shape (B, num_classes).
        """
        # Conv front-end. Convs operate on (H=T, W=D); we keep H fixed via the
        # natural geometry: (1×2) stride halves W, padding keeps H.
        x = self.leaky(self.conv1(x))  # (B,16,T,20)
        x = self.leaky(self.conv2(x))  # (B,16,T,10)
        x = self.leaky(self.conv3(x))  # (B,16,T,1)

        # Inception: (B,16,T,1) -> (B,32,T,1)
        x = self.inception(x)

        # Squeeze the singleton column axis -> (B,32,T) then transpose to
        # (B,T,32) for the LSTM (batch_first=True).
        x = x.squeeze(-1)            # (B,32,T)
        x = x.transpose(1, 2)        # (B,T,32)

        # LSTM over the time axis. We only need the last time-step output.
        lstm_out, _ = self.lstm(x)           # (B,T,64)
        last = lstm_out[:, -1, :]            # (B,64)

        logits = self.fc(last)               # (B,3)
        return logits


def build_model(num_classes: int = 3, window_size: int = 100, num_features: int = NUM_FEATURES) -> DeepLOB:
    """Factory used by train.py / smoke_test.py."""
    return DeepLOB(num_classes=num_classes, window_size=window_size, num_features=num_features)


if __name__ == "__main__":
    # Quick structural sanity check (no training, no data needed).
    model = build_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"DeepLOB total parameters: {total:,}")
    print(f"  conv1  out: (B,16,T,{NUM_FEATURES//2})")
    print(f"  conv2  out: (B,16,T,10)")
    print(f"  conv3  out: (B,16,T,1)")
    print(f"  incept out: (B,32,T,1)")
    print(f"  lstm  out: (B,T,64) -> last (B,64) -> fc (B,3)")
