"""Training & evaluation skeleton for DeepLOB.

Designed so the SAME code runs:
  - locally for a quick smoke test (small data, few epochs), and
  - on Colab with a real GPU and the real FI-2010 dataset.

The only thing that changes between the two is the `config` (dataset source,
epochs, batch size, device). Wiring the real dataset is left to Colab (see
dataset.FI2010WindowDataset). Here we default to the synthetic dataset so the
module is executable end-to-end without any download.

What this skeleton already provides (the "verify, don't skip" layer):
  - fixed random seed
  - categorical cross-entropy (LogSoftmax + NLLLoss, matches paper)
  - Adam optimiser (paper: lr=0.01, eps=1)
  - macro / weighted F1 + per-class precision-recall
  - early stopping on validation accuracy
  - checkpoint save/restore hook (writes to a path you control, e.g. Drive)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import RandomLOBDataset, WINDOW_SIZE, NUM_FEATURES, NUM_CLASSES


@dataclass
class Config:
    # data
    dataset: str = "random"          # "random" (smoke) or "fi2010" (Colab)
    data_path: Optional[str] = None  # used when dataset == "fi2010"
    k: int = 10                       # prediction horizon (FI-2010 label column)
    # training
    epochs: int = 3                  # smoke default; raise on Colab
    batch_size: int = 32
    lr: float = 0.01
    eps: float = 1.0
    patience: int = 20               # early-stop patience (paper uses 20)
    seed: int = 0
    # io
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "best.keras"  # .pt really; name kept for Drive habit
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def f1_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES):
    """Compute macro/weighted F1 and per-class precision/recall."""
    from sklearn.metrics import (
        f1_score,
        precision_recall_fscore_support,
        accuracy_score,
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": [float(p) for p in prec],
        "per_class_recall": [float(r) for r in rec],
    }


def make_dataloaders(cfg: Config):
    if cfg.dataset == "fi2010" and not cfg.data_path:
        raise ValueError(
            "dataset='fi2010' requires --data_path (path to FI-2010 .npy/.csv mirror)"
        )
    if cfg.dataset == "random":
        train_ds = RandomLOBDataset(num_samples=2000, seed=cfg.seed)
        val_ds = RandomLOBDataset(num_samples=400, seed=cfg.seed + 1)
        test_ds = RandomLOBDataset(num_samples=400, seed=cfg.seed + 2)
    else:
        # Colab: import and use the real dataset here.
        from dataset import FI2010WindowDataset

        train_ds = FI2010WindowDataset(cfg.data_path, k=cfg.k, split="train")
        val_ds = FI2010WindowDataset(cfg.data_path, k=cfg.k, split="val")
        test_ds = FI2010WindowDataset(cfg.data_path, k=cfg.k, split="test")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_dl, val_dl, test_dl


def evaluate(model: nn.Module, dl: DataLoader, device: str):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return f1_metrics(y_true, y_pred)


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = cfg.device if torch.cuda.is_available() or cfg.device != "cuda" else "cuda"
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu.")

    model = build_model(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, num_features=NUM_FEATURES).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.eps)

    train_dl, val_dl, test_dl = make_dataloaders(cfg)

    best_val_acc = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # build_model returns raw logits; NLLLoss wants log-probabilities,
            # so apply log_softmax here (paper uses a softmax output layer).
            logits = model(x)
            loss = criterion(torch.log_softmax(logits, dim=1), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_dl.dataset)
        val_metrics = evaluate(model, val_dl, device)
        print(
            f"epoch {epoch+1:03d} | train_loss {train_loss:.4f} | "
            f"val_acc {val_metrics['accuracy']:.4f} | "
            f"val_macroF1 {val_metrics['macro_f1']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        # early stopping on validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_no_improve = 0
            ckpt = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
            torch.save(model.state_dict(), ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    # final test evaluation using the best checkpoint
    ckpt = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    test_metrics = evaluate(model, test_dl, device)
    print("TEST:", json.dumps(test_metrics, indent=2))

    return {"history": history, "test": test_metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["random", "fi2010"], default="random")
    p.add_argument("--data_path", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    args = p.parse_args()

    cfg = Config(
        dataset=args.dataset,
        data_path=args.data_path,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
