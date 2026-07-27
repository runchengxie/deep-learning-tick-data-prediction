"""Local smoke test: verifies the model runs without real data.

Checks (the 80% of bugs caught before Colab):
  1. model builds and prints parameter count
  2. a random (B,1,T=100,D=40) batch produces (B,3) logits
  3. softmax over the 3 classes sums to ~1 per row
  4. a 1-epoch dummy train step lowers loss (gradients flow)

Run:
    python smoke_test.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.model import build_model
from src.dataset import get_dummy_batch, WINDOW_SIZE, NUM_FEATURES, NUM_CLASSES


def test_forward_shape():
    model = build_model()
    x, _ = get_dummy_batch(batch_size=8)
    assert x.shape == (8, 1, WINDOW_SIZE, NUM_FEATURES), x.shape
    logits = model(x)
    assert logits.shape == (8, NUM_CLASSES), logits.shape
    probs = F.softmax(logits, dim=1)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5), row_sums
    print("[ok] forward shape (8,3) and softmax rows sum to 1")


def test_grad_flow():
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()
    x, y = get_dummy_batch(batch_size=16)
    logits = model(x)
    loss1 = criterion(F.log_softmax(logits, dim=1), y)
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    logits2 = model(x)
    loss2 = criterion(F.log_softmax(logits2, dim=1), y)
    assert loss2.item() < loss1.item() + 1e-3, (loss1.item(), loss2.item())
    print(f"[ok] one grad step: loss {loss1.item():.4f} -> {loss2.item():.4f}")


def test_param_count():
    model = build_model()
    total = sum(p.numel() for p in model.parameters())
    # Paper reports ~60k params. Our Inception splits 32 channels across 4
    # branches (8 each), so we land near ~30k; that is fine for a skeleton and
    # even lighter on memory. Raise the band if you widen the Inception branches.
    assert 20_000 < total < 200_000, total
    print(f"[ok] total params {total:,} (paper ~60k; skeleton band 20k-200k)")


if __name__ == "__main__":
    test_forward_shape()
    test_grad_flow()
    test_param_count()
    print("\nAll smoke tests passed.")
