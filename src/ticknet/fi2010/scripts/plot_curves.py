"""绘制训练曲线。

读取训练目录中的 ``train_history.*.json``，绘制训练损失和验证集
macro F1。Setup 1 会为每个 ``CF`` 生成一条曲线，并额外显示平均曲线。

运行方式：

    python scripts/plot_curves.py --checkpoint-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无显示环境（如 Colab、CI）也能保存图片
import matplotlib.pyplot as plt


def load_histories(
    checkpoint_dir: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """按文件名读取训练历史。"""
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "train_history.*.json")))
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        name = os.path.basename(p).removeprefix("train_history.").removesuffix(".json")
        out.append((name, data))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="./checkpoints")
    ap.add_argument("--out", default="results/train_curves.png")
    args = ap.parse_args()

    histories = load_histories(args.checkpoint_dir)
    if not histories:
        raise SystemExit(
            f"在 {args.checkpoint_dir} 下没有找到 train_history.*.json，请先运行训练。"
        )

    n = len(histories)
    # 每折一个子图，最后一格留给平均曲线
    cols = 2
    rows = (n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    all_epochs: list[list[int]] = []
    all_loss: list[list[float]] = []
    all_f1: list[list[float]] = []

    for i, (name, hist) in enumerate(histories):
        epochs = [h["epoch"] for h in hist]
        loss = [h["train_loss"] for h in hist]
        f1 = [h["val_macro_f1"] for h in hist]
        all_epochs.append(epochs)
        all_loss.append(loss)
        all_f1.append(f1)

        ax = flat[i]
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.plot(epochs, loss, label="train_loss")
        ax.plot(epochs, f1, label="val_macro_f1")
        ax.legend()
        ax.grid(True)

    # 平均曲线子图：把所有折按 epoch 对齐求平均
    max_len = max(len(e) for e in all_epochs)
    avg_loss, avg_f1 = [], []
    for t in range(max_len):
        ls = [all_loss[k][t] for k in range(n) if t < len(all_loss[k])]
        fs = [all_f1[k][t] for k in range(n) if t < len(all_f1[k])]
        avg_loss.append(sum(ls) / len(ls))
        avg_f1.append(sum(fs) / len(fs))

    ax_avg = flat[n]
    ax_avg.set_title("average" if n > 1 else "curve")
    ax_avg.set_xlabel("epoch")
    ax_avg.plot(range(1, max_len + 1), avg_loss, label="avg_train_loss")
    ax_avg.plot(range(1, max_len + 1), avg_f1, label="avg_val_macro_f1")
    ax_avg.legend()
    ax_avg.grid(True)

    # 多余的子图隐藏
    for j in range(n + 1, len(flat)):
        flat[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved curves -> {args.out}")


if __name__ == "__main__":
    main()
