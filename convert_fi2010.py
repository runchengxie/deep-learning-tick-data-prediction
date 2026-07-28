"""Convert official FI-2010 .txt files to clean float32 .npy for training.

Run LOCALLY (not on Colab). The official data (Ntakaris et al. 2017,
arXiv:1705.03233) is distributed as .txt files from:
  https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  (file: BenchmarkDatasets.zip, 1.74 GB)

Layout per the authors' description:
  columns 1-144  -> 144 LOB features
  columns 145-149 -> 5 label columns (classification problems).
  Label encoding: 1 = up, 2 = stationary, 3 = down.

We keep all 5 label columns in the saved .npy so train.py can pick one via k.
By convention the first label column (row 145) corresponds to k=10, the
horizon used in the paper's main Table II comparison.

Usage:
  # Convert a single fold's training file (example):
  python convert_fi2010.py --txt_dir /path/to/BenchmarkDatasets \\
      --norm z-score --auction without --folds 1 2 3 4 5 6 7 8 9 \\
      --out FI2010_normalised.npy

  # Or if you already unzipped and have specific file names, use --files.
"""

from __future__ import annotations
import argparse
import glob
import os
import numpy as np


def find_files(txt_dir: str, norm: str, auction: str, folds) -> list[str]:
    """Locate FI-2010 .txt files by the naming convention:
    <train|test>_with|without_auction_<norm>_fold<N>.txt
    """
    found = []
    for split in ("training", "testing"):
        for f in folds:
            pattern = os.path.join(
                txt_dir, f"{split}_{auction}_auction_{norm}_fold{f}.txt"
            )
            matches = sorted(glob.glob(pattern))
            if not matches:
                # try alternate naming seen in some mirrors
                pattern2 = os.path.join(
                    txt_dir, f"{split}_{auction}auction_{norm}_fold{f}.txt"
                )
                matches = sorted(glob.glob(pattern2))
            if not matches:
                print(f"WARNING: no file for {pattern}")
                continue
            found.extend(matches)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", help="directory with unzipped FI-2010 .txt files")
    ap.add_argument("--norm", default="z-score", choices=["z-score", "min-max", "decimal"])
    ap.add_argument("--auction", default="without", choices=["with", "without"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 10)),
                    help="fold numbers to include (1-9)")
    ap.add_argument("--out", help="output .npy path (not needed with --inspect-only)")
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    files = find_files(args.txt_dir, args.norm, args.auction, args.folds)
    if not files:
        raise SystemExit("No .txt files matched. Check --txt_dir / --norm / --auction.")
    print(f"Found {len(files)} files:")
    for f in files[:10]:
        print("  ", os.path.basename(f))
    if len(files) > 10:
        print(f"  ... (+{len(files)-10} more)")

    if args.inspect_only:
        if not args.out:
            args.out = "inspect_only_dummy.npy"  # not actually written
        # load just the first file's head to verify column count
        sample = np.loadtxt(files[0], delimiter=" ", max_rows=5)
        print(f"sample shape (5 rows): {sample.shape}")
        print("First row head:", sample[0, :5])
        print("Last 6 columns of row0:", sample[0, -6:])
        return

    rows = []
    for f in files:
        # delimiter is whitespace; dtype float32 to save RAM
        arr = np.loadtxt(f, delimiter=" ", dtype=np.float32)
        rows.append(arr)
    data = np.vstack(rows).astype(np.float32)
    print("combined shape:", data.shape)

    n_cols = data.shape[1]
    if n_cols < 149:
        raise ValueError(
            f"expected >=149 columns (144 features + 5 labels), got {n_cols}. "
            "Verify the .txt format."
        )
    # keep 144 features + 5 labels = 149 columns
    out = data[:, :149].astype(np.float32)
    np.save(args.out, out)
    print(f"saved {args.out} shape={out.shape} dtype={out.dtype}")
    print("  features: cols 0-143 (144), labels k-problems: cols 144-148")


if __name__ == "__main__":
    main()
