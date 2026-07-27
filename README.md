# DeepLOB Reproduction

PyTorch reproduction skeleton of **DeepLOB: Deep Convolutional Neural Networks
for Limit Order Books** (Zhang, Zohren, Roberts 2020).

- Paper: https://arxiv.org/abs/1808.03668
- Original code (reference): https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

## Scope of this skeleton

This repository is a **lightweight, verifiable training/eval framework** built
around the official model architecture. It currently provides:

- `src/model.py` — DeepLOB (CNN → Inception → LSTM → softmax), PyTorch.
  Input feature dim is configurable (default 144 for official FI-2010).
- `src/dataset.py` — `RandomLOBDataset` (smoke test, no data needed) +
  `FI2010WindowDataset` (reads a prepared `(N,148)` float32 .npy: 144 features
  + 4 labels for k=10/20/50/100).
- `src/train.py` — training loop with fixed seed, Adam (lr=0.01, eps=1),
  macro/weighted F1, early stopping, checkpoint hook.
- `convert_fi2010.py` — **local** script: convert official `.mat` → clean `.npy`.
- `smoke_test.py` — local checks: forward shape, softmax sum=1, grad flow.
- `configs/base.yaml`, `run_colab.py` — Colab GPU entry point.

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
in the normalised file, NOT raw event counts. In the OFFICIAL data, the last 4
columns hold the 3-class labels for these k values (cols 144/145/146/147).
Always select the column via `dataset.K_TO_LABEL_COLUMN[k]`. Picking the wrong
column silently reproduces a *different* task.

> WARNING: third-party mirrors such as `shanehans/FI2010` (Hugging Face CSV)
> do NOT follow the official layout. That CSV is 150 columns = 130 features +
> 15 junk/empty columns + 4 labels with dirty values (row numbers leaked into
> label columns). Do NOT use it for a faithful reproduction.

## Data preparation (local → Drive)

The official FI-2010 data is distributed as a `.mat` file (Ntakaris et al. 2017,
arXiv:1705.03233):

- Zenodo: https://zenodo.org/records/5603905
  (look for the `FI-2010` / `FI2010` `.mat` file under "Files")
- Download it manually to your computer (a few hundred MB).

Then convert it locally (your machine has RAM + a real Python env to inspect
the structure; Colab does not need to touch `.mat`):

```bash
pip install scipy numpy
python convert_fi2010.py --mat path/to/FI-2010.mat --out FI2010_normalised.npy
# inspect-only first if unsure about .mat variable names:
python convert_fi2010.py --mat path/to/FI-2010.mat --inspect-only
```

This produces `FI2010_normalised.npy` of shape `(N, 148)` float32:
- cols `[0:144]`  → LOB features
- cols `[144:148]` → labels for k = 10, 20, 50, 100

Upload `FI2010_normalised.npy` to your Google Drive at:

```
MyDrive/DeepLOB/data/FI2010_normalised.npy
```

Colab's `run_colab.py` will find it there and skip any download.

## Workflow (local dev → Colab GPU)

```
local edit → GitHub push → Colab pull → GPU train
```

- Code on GitHub; data + model weights on Google Drive.
- Do NOT commit multi-GB FI-2010 data to git.

On Colab, after `git pull` (or re-downloading the zip), run:

```python
!python deeplob-reproduction/run_colab.py
```

It mounts Drive, finds the prepared `.npy`, installs deps, and trains.

## What is intentionally NOT included yet

- Crypto data / Transformer baselines / backtest (phase 2, after baseline works).
- Multi-seed runs (add after the single-seed pipeline is verified).
