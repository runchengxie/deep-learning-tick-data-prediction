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
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:

    import yaml

except ImportError:  # yaml 仅在使用 --config 时需要

    yaml: Any = None



from dataset import NUM_CLASSES, NUM_FEATURES, WINDOW_SIZE, RandomLOBDataset
from model import build_model


@dataclass

class Config:

    # data

    dataset: str = "random"          # "random" (smoke) or "fi2010" (Colab)

    data_path: str | None = None  # used when dataset == "fi2010"

    k: int = 10                       # prediction horizon (FI-2010 label column)

    folds_path: str | None = None  # fold-id .npy for 9-fold CV

    # training

    epochs: int = 3                  # smoke default; raise on Colab

    batch_size: int = 32

    lr: float = 0.01

    eps: float = 1.0

    patience: int = 20               # early-stop patience (paper uses 20)

    seed: int = 0

    val_frac: float = 0.1            # fraction of train held out for early stopping

    resume: bool = True              # resume from checkpoint if present

    cv: bool = False                 # 9-fold cross-validation over folds_path

    num_folds: int = 9               # number of folds for CV

    # io

    checkpoint_dir: str = "./checkpoints"

    checkpoint_name: str = "best"  # 实际保存为 <name>.pt / <name>.fold<N>.pt

    device: str = "cpu"





def set_seed(seed: int) -> None:

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)





def f1_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES):

    """Compute macro/weighted F1 and per-class precision/recall."""

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
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





def make_dataloaders(cfg: Config, test_fold: int | None = None):

    if cfg.dataset == "fi2010" and not cfg.data_path:

        raise ValueError(

            "dataset='fi2010' requires --data_path (path to FI-2010 .npy mirror)"

        )

    if cfg.dataset == "random":

        train_ds = RandomLOBDataset(num_samples=2000, seed=cfg.seed)

        val_ds = RandomLOBDataset(num_samples=400, seed=cfg.seed + 1)

        test_ds = RandomLOBDataset(num_samples=400, seed=cfg.seed + 2)

    else:

        # Colab: import and use the real dataset here.

        from dataset import FI2010WindowDataset



        # 前面已校验 fi2010 必须带 data_path，这里收窄类型供静态检查。

        assert cfg.data_path is not None

        data_path: str = cfg.data_path



        train_ds = FI2010WindowDataset(

            data_path,

            split="train",

            k=cfg.k,

            window_size=WINDOW_SIZE,

            folds_path=cfg.folds_path,

            val_frac=cfg.val_frac,

            seed=cfg.seed,

            test_fold=test_fold,

        )

        val_ds = FI2010WindowDataset(

            data_path,

            split="val",

            k=cfg.k,

            window_size=WINDOW_SIZE,

            folds_path=cfg.folds_path,

            val_frac=cfg.val_frac,

            seed=cfg.seed,

            test_fold=test_fold,

        )

        test_ds = FI2010WindowDataset(

            data_path,

            split="test",

            k=cfg.k,

            window_size=WINDOW_SIZE,

            folds_path=cfg.folds_path,

            val_frac=cfg.val_frac,

            seed=cfg.seed,

            test_fold=test_fold,

        )



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





def train(cfg: Config, test_fold: int | None = None) -> dict:

    set_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.device == "cuda" and device == "cpu":

        print("[warn] cuda requested but unavailable; falling back to cpu.")



    model = build_model(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, num_features=NUM_FEATURES).to(device)

    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.eps)



    train_dl, val_dl, test_dl = make_dataloaders(cfg, test_fold=test_fold)



    # --- resume support: restore model + optimizer + epoch counter ---

    # checkpoint file holds {state_dict, optimizer, epoch, best_val_acc}.

    # For CV we namespace per fold so folds don't clobber each other.

    fold_tag = f".fold{test_fold}" if test_fold is not None else ""

    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name + fold_tag + ".pt")

    start_epoch = 0

    best_val_acc = -1.0

    if cfg.resume and os.path.exists(ckpt_path):

        ck = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ck["model"])

        optimizer.load_state_dict(ck["optimizer"])

        start_epoch = ck["epoch"]

        best_val_acc = ck["best_val_acc"]

        print(f"[resume] loaded {ckpt_path} @ epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")



    epochs_no_improve = 0

    history = []



    for epoch in range(start_epoch, cfg.epochs):

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

            # save full training state for resume

            torch.save(

                {

                    "model": model.state_dict(),

                    "optimizer": optimizer.state_dict(),

                    "epoch": epoch + 1,

                    "best_val_acc": best_val_acc,

                },

                ckpt_path,

            )

        else:

            epochs_no_improve += 1

            if epochs_no_improve >= cfg.patience:

                print(f"early stopping at epoch {epoch+1}")

                break



    # final test evaluation using the best checkpoint

    if os.path.exists(ckpt_path):

        ck = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ck["model"])

    test_metrics = evaluate(model, test_dl, device)

    print("TEST:", json.dumps(test_metrics, indent=2))



    # 把每个 epoch 的训练历史落盘，供 plot_curves.py 画图。

    history_path = os.path.join(

        cfg.checkpoint_dir, f"train_history{fold_tag}.json"

    )

    with open(history_path, "w", encoding="utf-8") as fh:

        json.dump(history, fh, indent=2)



    return {"history": history, "test": test_metrics, "best_val_acc": best_val_acc}





def run_cv(cfg: Config) -> dict:

    """9-fold anchored cross-validation (paper Setup 2).



    For each fold i: train on all OTHER folds, test on fold i. Average the

    per-fold test macro-F1 (with std) so the result is comparable to the

    paper's Table II. Each fold's training is resumable via its namespaced

    checkpoint.

    """

    if not cfg.folds_path:

        raise ValueError("--cv requires --folds_path (the FI2010_folds.npy)")

    all_f1, all_acc = [], []

    for i in range(cfg.num_folds):

        print(f"\n===== FOLD {i+1}/{cfg.num_folds} (test fold {i}) =====")

        res = train(cfg, test_fold=i)

        all_f1.append(res["test"]["macro_f1"])

        all_acc.append(res["test"]["accuracy"])



    mean_f1, std_f1 = np.mean(all_f1), np.std(all_f1)

    mean_acc, std_acc = np.mean(all_acc), np.std(all_acc)

    summary = {

        "per_fold_macro_f1": all_f1,

        "per_fold_acc": all_acc,

        "mean_macro_f1": float(mean_f1),

        "std_macro_f1": float(std_f1),

        "mean_acc": float(mean_acc),

        "std_acc": float(std_acc),

    }

    print("\n===== 9-FOLD CV SUMMARY =====")

    print(json.dumps(summary, indent=2))

    return summary





def _build_arg_parser(file_cfg: dict | None = None) -> argparse.ArgumentParser:

    """从 Config 字段自动生成命令行参数。



    file_cfg 是 yaml 里读到的字段（默认 None）。有 --config 时，yaml 的值

    作为 argparse 的默认值，命令行显式传的参数会自然覆盖它。

    """

    file_cfg = file_cfg or {}

    p = argparse.ArgumentParser(description="DeepLOB training / evaluation")

    p.add_argument(

        "--config",

        default=None,

        help="yaml 配置文件路径；其中的字段作为默认值，命令行可覆盖",

    )

    for f in fields(Config):

        default = file_cfg.get(f.name, f.default)

        if f.name == "dataset":

            p.add_argument("--dataset", choices=["random", "fi2010"], default=default)

            continue

        if f.default is True or f.default is False:

            # bool 字段提供 --flag / --no-flag 两种写法

            p.add_argument(

                f"--{f.name}",

                action="store_true",

                default=default,

                help=f"disable with --no-{f.name}",

            )

            p.add_argument(

                f"--no-{f.name}", dest=f.name, action="store_false", default=default

            )

            continue

        type_ = str if f.default is None else type(f.default)

        p.add_argument(f"--{f.name}", type=type_, default=default)

    return p





def _load_config_from_args() -> Config:

    # 先解析一次，只为拿到 --config 路径（此时用 Config 原默认值）

    probe = argparse.ArgumentParser(add_help=False)

    probe.add_argument("--config", default=None)

    probe_args, _ = probe.parse_known_args()



    file_cfg = None

    if probe_args.config:

        if yaml is None:

            raise SystemExit("读取 --config 需要 PyYAML，请先 pip install pyyaml")

        with open(probe_args.config, encoding="utf-8") as fh:

            file_cfg = yaml.safe_load(fh) or {}

        for name in file_cfg:

            if name not in {f.name for f in fields(Config)}:

                raise SystemExit(f"yaml 中含未知字段：{name}")



    # yaml 值作为 argparse 默认值，命令行显式传的覆盖它

    p = _build_arg_parser(file_cfg)

    args = p.parse_args()

    values = {k: v for k, v in vars(args).items() if k != "config"}

    return Config(**values)





def main():

    cfg = _load_config_from_args()

    if cfg.cv:

        run_cv(cfg)

    else:

        train(cfg)





if __name__ == "__main__":

    main()

