# DeepLOB Reproduction

PyTorch reproduction skeleton of **DeepLOB: Deep Convolutional Neural Networks
for Limit Order Books** (Zhang, Zohren, Roberts 2020).

- Paper: https://arxiv.org/abs/1808.03668
- Original code (reference): https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

## Scope of this skeleton

This repository is a **lightweight, verifiable training/eval framework** built
around the official model architecture. It currently provides:

- `src/model.py` — DeepLOB (CNN → Inception → LSTM → softmax), PyTorch.
- `src/dataset.py` — `RandomLOBDataset` (smoke test, no data needed) + the
  `FI2010WindowDataset` *interface* (not yet wired to disk; see labelling note
  below).
- `src/train.py` — training loop with fixed seed, Adam (lr=0.01, eps=1),
  macro/weighted F1, early stopping, checkpoint hook.
- `smoke_test.py` — local checks: forward shape, softmax sum=1, grad flow.
- `configs/base.yaml`, `notebooks/colab.ipynb` — Colab GPU entry point.

## Quick start (local, no data)

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/model.py        # prints param count + shape trace
python smoke_test.py       # forward + grad smoke checks
python src/train.py        # 3-epoch smoke training on synthetic data
```

## CRITICAL labelling note (easy to get wrong)

FI-2010's prediction horizons `k = 10, 20, 50, 100` are **label-column indices**
in the normalised file, NOT raw event counts. The last 4 columns of the
normalised data hold the 3-class labels for these k values (in that order).
Always select the column via `dataset.K_TO_LABEL_COLUMN[k]`. Picking the wrong
column silently reproduces a *different* task.

## Workflow (local dev → Colab GPU)

```
local edit → GitHub push → Colab git pull → GPU train
```

- Code on GitHub; data + model weights on Google Drive.
- Do NOT commit multi-GB FI-2010 data to git.
- On Colab, implement `FI2010WindowDataset.__init__` using the HF mirror
  (`shanehans/FI2010`), window with `float32` + stride tricks, and pick the
  label column per `k`.

## What is intentionally NOT included yet

- Real data wiring (left for Colab).
- Crypto data / Transformer baselines / backtest (phase 2, after baseline works).
- Multi-seed runs (add after the single-seed pipeline is verified).
