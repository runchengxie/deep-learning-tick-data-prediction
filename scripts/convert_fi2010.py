"""Convert official FI-2010 .txt files to clean float32 .npy for training.

Run LOCALLY (not on Colab). The official data (Ntakaris et al. 2017,
arXiv:1705.03233) is distributed as nested .txt files from:
  https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  (file: BenchmarkDatasets.zip, 1.74 GB)

Directory layout inside the zip (VERIFIED on the real download):
  BenchmarkDatasets/
    NoAuction/                          # or Auction/
      1.NoAuction_Zscore/               # norm: Zscore | MinMax | DecPre
        NoAuction_Zscore_Training/      # or ..._Testing/
          Train_Dst_NoAuction_ZScore_CF_1.txt ... _9.txt   (9 folds)
          Test_Dst_NoAuction_ZScore_CF_1.txt  ... _9.txt

ACTUAL FILE FORMAT (critical, easy to get wrong):
  Each .txt file is a matrix stored as (rows=channels) x (cols=samples):
    - 149 rows total
    - rows   0-143  -> 144 LOB features
    - rows 144-148  -> 5 label columns (3-class, values {1,2,3})
  We therefore TRANSPOSE on load, ending with (N_samples, 149):
    cols [0:144] = features, cols [144:149] = 5 labels.
  Label encoding: 1=up, 2=stationary, 3=down (dataset maps via sorted unique
  so the exact ordering does not matter). The first label column (col 144,
  0-indexed) is the k=10 horizon used in the paper's main Table II.

Usage:
  # Inspect one file's real shape/columns first (cheap, no full conversion):
  python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets --inspect-only

  # Full conversion of all 9 folds, NoAuction + Zscore (paper's Setup 2 uses
  # the 9-fold anchored cross-validation; we concatenate train+test per fold):
  python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets \
      --auction NoAuction --norm Zscore --folds 1 2 3 4 5 6 7 8 9 \
      --out FI2010_normalised.npy
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# FI-2010 label encoding is {1,2,3}; we keep all 5 label rows.
NUM_FEATURES = 144
NUM_LABEL_COLS = 5
EXPECTED_ROWS = NUM_FEATURES + NUM_LABEL_COLS  # 149


def _norm_token(norm: str) -> str:
    """Map our --norm flag to the substring used in the real directory name."""
    table = {
        "z-score": "Zscore",
        "zscore": "Zscore",
        "Zscore": "Zscore",
        "min-max": "MinMax",
        "minmax": "MinMax",
        "MinMax": "MinMax",
        "decimal": "DecPre",
        "decpre": "DecPre",
        "DecPre": "DecPre",
    }
    return table[norm]


# 目录名用 Zscore，但文件名里写成 ZScore（首字母大写）。下面把目录 token
# 映射到文件名 token，find_files 用它拼出真实文件名。
FILE_NORM_TOKEN = {"Zscore": "ZScore", "MinMax": "MinMax", "DecPre": "DecPre"}


def find_files(base_dir: str, norm: str, auction: str, folds) -> list[str]:
    """Locate FI-2010 .txt files by the VERIFIED real naming convention.

    <base>/<Auction>/<i>.<Auction>_<Norm>/
        <Auction>_<Norm>_Training|Testing/<Split>_Dst_<Auction>_<Norm>_CF_<N>.txt
    """
    norm_tok = _norm_token(norm)
    # The REAL folder name binds a numeric prefix to the norm method:
    #   1.<Auction>_Zscore, 2.<Auction>_MinMax, 3.<Auction>_DecPre
    # The prefix is the norm INDEX, NOT a fold number; each norm folder holds
    # ALL 9 fold files (CF_1 .. CF_9).
    norm_index = {"Zscore": 1, "MinMax": 2, "DecPre": 3}[norm_tok]
    folder = f"{norm_index}.{auction}_{norm_tok}"
    found = []
    for split in ("Training", "Testing"):
        for f in folds:
            subdir = f"{auction}_{norm_tok}_{split}"
            # e.g. Train_Dst_NoAuction_ZScore_CF_1.txt
            file_norm_tok = FILE_NORM_TOKEN[norm_tok]
            split_prefix = "Train" if split == "Training" else "Test"
            fname = f"{split_prefix}_Dst_{auction}_{file_norm_tok}_CF_{f}.txt"
            pattern = os.path.join(base_dir, auction, folder, subdir, fname)
            matches = sorted(glob.glob(pattern))
            if not matches:
                print(f"WARNING: no file for {pattern}")
                continue
            found.extend(matches)
    return found


def _read_txt(path: str) -> np.ndarray:
    """Read one FI-2010 .txt and return it TRANSPOSED to (N, 149).

    The raw file is (149 channels x N samples) with multi-space separators and
    occasional leading whitespace. numpy.loadtxt with its DEFAULT delimiter
    (any run of whitespace) handles both robustly and is ~100x faster than a
    regex-separated pandas read.
    """
    arr = np.loadtxt(path, dtype=np.float32)  # (149, N)
    if arr.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            f"{path}: expected {EXPECTED_ROWS} rows (channels), got {arr.shape[0]}. "
            "This file does not look like the official FI-2010 layout."
        )
    return arr.T.astype(np.float32)  # (N, 149)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        required=True,
        help="path to BenchmarkDatasets/ (contains NoAuction/ and Auction/)",
    )
    ap.add_argument(
        "--norm",
        default="z-score",
        choices=["z-score", "min-max", "decimal"],
        help="normalisation: z-score (paper default) | min-max | decimal",
    )
    ap.add_argument(
        "--auction",
        default="NoAuction",
        choices=["NoAuction", "Auction"],
        help="NoAuction is the version used in the DeepLOB paper",
    )
    ap.add_argument(
        "--folds", type=int, nargs="+", default=list(range(1, 10)),
        help="fold numbers to include (1-9)",
    )
    ap.add_argument("--out", help="output .npy path (not needed with --inspect-only)")
    ap.add_argument(
        "--out_folds",
        default=None,
        help="output fold-id array (.npy). Defaults to <out> with '_folds' infix, "
             "e.g. FI2010_normalised.npy -> FI2010_folds.npy. Needed for 9-fold CV.",
    )
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    files = find_files(args.base_dir, args.norm, args.auction, args.folds)
    if not files:
        raise SystemExit(
            "No .txt files matched. Check --base_dir / --auction / --norm / --folds."
        )
    print(f"Found {len(files)} files:")
    for f in files[:10]:
        print("  ", os.path.basename(f))
    if len(files) > 10:
        print(f"  ... (+{len(files)-10} more)")

    if args.inspect_only:
        sample = _read_txt(files[0])
        print(f"transposed sample shape (one file): {sample.shape}")
        print("feature row0 head:", sample[0, :5])
        print("label col144 head (k=10):", sample[:10, 144])
        print("unique labels in col144-148:", np.unique(sample[:, 144:149]))
        print("=> expect shape (N, 149); features cols 0-143, labels 144-148.")
        return

    if not args.out:
        raise SystemExit("--out is required for full conversion.")

    # Build the fold-id array. Each CF_<N> file (Training OR Testing) gets the
    # same 0-based fold id = N-1, so "fold i" = all samples from the i-th
    # dataset (its Training + Testing), matching the paper's Setup 2.
    import re

    rows = []
    fold_ids = []
    for f in files:
        arr = _read_txt(f)  # (N_f, 149)
        m = re.search(r"CF_(\d+)\.txt$", os.path.basename(f))
        n = int(m.group(1)) if m else len(rows) + 1
        rows.append(arr)
        fold_ids.append(np.full(arr.shape[0], n - 1, dtype=np.int16))

    data = np.vstack(rows).astype(np.float32)
    folds = np.concatenate(fold_ids)
    print("combined shape:", data.shape, "| folds:", np.unique(folds))

    n_cols = data.shape[1]
    if n_cols != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} columns after transpose, got {n_cols}."
        )
    np.save(args.out, data)
    out_folds = args.out_folds or re.sub(r"\.npy$", "_folds.npy", args.out)
    np.save(out_folds, folds)
    print(f"saved {args.out} shape={data.shape} dtype={data.dtype}")
    print(f"saved {out_folds} shape={folds.shape} dtype={folds.dtype}")
    print("  features: cols 0-143 (144), labels k=10/20/50/100: cols 144-147")


if __name__ == "__main__":
    main()
