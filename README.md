# DeepLOB Reproduction

PyTorch reproduction skeleton of **DeepLOB: Deep Convolutional Neural Networks
for Limit Order Books** (Zhang, Zohren, Roberts 2018).

- Paper: https://arxiv.org/abs/1808.03668
- Original code (reference): https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

## Scope of this skeleton

This repository is a **lightweight, verifiable training/eval framework** built
around the official model architecture. It currently provides:

- `src/model.py` — DeepLOB (CNN → Inception → LSTM → softmax), PyTorch.
  Input feature dim is configurable (default 144 for official FI-2010).
- `src/dataset.py` — `RandomLOBDataset` (smoke test, no data needed) +
  `FI2010WindowDataset` (reads a prepared `(N,149)` float32 .npy: 144 features
  + 5 label columns for k=10/20/50/100 + one extra horizon).
- `src/train.py` — training loop with fixed seed, Adam (lr=0.01, eps=1),
  macro/weighted F1, early stopping, checkpoint hook.
- `convert_fi2010.py` — **local** script: convert official `.txt` → clean `.npy`.
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
in the normalised file, NOT raw event counts. In the OFFICIAL data, the last 5
columns hold the 3-class labels for these k values (cols 144/145/146/147; the
5th column is an additional horizon). Always select the column via
`dataset.K_TO_LABEL_COLUMN[k]`. Picking the wrong column silently reproduces a
*different* task.

> WARNING: third-party mirrors such as `shanehans/FI2010` (Hugging Face CSV)
> do NOT follow the official layout. That CSV is 150 columns = 130 features +
> 15 junk/empty columns + 4 labels with dirty values (row numbers leaked into
> label columns). Do NOT use it for a faithful reproduction.

## Data preparation (local → Drive)

The official FI-2010 data is distributed as `.txt` files (Ntakaris et al. 2017,
arXiv:1705.03233), hosted on the Finnish FAIR data platform:

- https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  (download `BenchmarkDatasets.zip`, 1.74 GB; unzip twice to reach
  `BenchmarkDatasets/BenchmarkDatasets/`)

**IMPORTANT file-format notes (verified on the real download):**
- Inside the zip the layout is
  `BenchmarkDatasets/{NoAuction,Auction}/<i>.<Auction>_<Norm>/<Auction>_<Norm>_{Training,Testing}/<Split>_Dst_<Auction>_<Norm>_CF_<N>.txt`
  e.g. `NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt`.
  `<i>` is the norm index (1=Zscore, 2=MinMax, 3=DecPre), NOT a fold number;
  each norm folder holds all 9 fold files (CF_1..CF_9).
- Each `.txt` is a matrix stored as **(rows=channels) x (cols=samples)**:
  149 rows total (144 feature rows + 5 label rows), values {1,2,3} for labels.
  `convert_fi2010.py` TRANSPOSES on load so the .npy is (N_samples, 149).

Then convert it locally (your machine has RAM + a real Python env to inspect
the structure; Colab does not need to touch the raw .txt):

```bash
pip install numpy
python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets/BenchmarkDatasets \
    --auction NoAuction --norm z-score --folds 1 2 3 4 5 6 7 8 9 \
    --out FI2010_normalised.npy
# inspect-only first to verify the real shape (expect (N, 149)) on a sample:
python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets/BenchmarkDatasets --inspect-only
```

This produces `FI2010_normalised.npy` of shape `(N, 149)` float32 (N ~ 2.04M
for all 9 folds of NoAuction Z-score):
- cols `[0:144]`  → LOB features
- cols `[144:149]` → 5 label columns; k=10/20/50/100 map to cols 144/145/146/147

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

## Reproducing the paper numbers (read before comparing to Table II)

The DeepLOB paper reports F1 on the FI-2010 **9-fold anchored
cross-validation** protocol (their "Setup 2"): for each fold, train on the
other 8 folds' Training data and test on that fold's data, then average the 9
folds. The current `FI2010WindowDataset` / `run_colab.py` instead naively
splits the concatenated `.npy` into 70/15/15 train/val/test by row. That is
fine for a **pipeline smoke test** (confirm the model learns a signal) but its
numbers are NOT directly comparable to the paper's Table II.

To faithfully reproduce Table II you must:
1. Convert each fold's Training/Testing `.txt` separately (or split the merged
   `.npy` back into the 9 folds by row ranges — each fold has a known number of
   samples), and
2. Train 9 models, one per fold, evaluating on that fold's held-out data.

This 9-fold wiring is the next step after the single-seed pipeline is verified
on Colab.

## What is intentionally NOT included yet

- Crypto data / Transformer baselines / backtest (phase 2, after baseline works).
- Multi-seed runs (add after the single-seed pipeline is verified).
- 9-fold anchored cross-validation wiring (see above; needed for Table II).
