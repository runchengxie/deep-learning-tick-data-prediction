"""Convert official FI-2010 .mat to a clean float32 .npy for training.

Run LOCALLY (not on Colab): your machine has the RAM and a real Python
env to inspect the .mat structure. The resulting .npy goes to Google Drive
so Colab only trains.

Official data: Ntakaris et al. 2017, arXiv:1705.03233
  Zenodo: https://zenodo.org/records/5603905  (file: FI-2010.mat or similar)

Expected layout after flattening:
  columns [0:144]   -> 144 LOB features
  columns [144:148] -> 3-class labels for k = 10, 20, 50, 100

Usage:
  python convert_fi2010.py --mat path/to/FI-2010.mat --out FI2010_normalised.npy
"""

from __future__ import annotations
import argparse
import numpy as np
import scipy.io


def inspect(mat_path: str) -> dict:
    """Print the .mat top-level structure so we know the variable name."""
    mat = scipy.io.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f".mat top-level variables ({len(keys)}):")
    info = {}
    for k in keys:
        v = mat[k]
        print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',None)}")
        info[k] = v
    return info


def flatten_to_2d(info: dict) -> np.ndarray:
    """Heuristic: find the big 2D array (samples x features+labels).

    FI-2010 .mat typically stores data as a cell array of days, or one big
    matrix. We try common patterns:
      1. a single 2D variable whose width is ~148
      2. a cell array of day-matrices -> vstack
    """
    # case 1: direct 2D matrix
    for k, v in info.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] in (144, 148, 150, 152):
            print(f"using variable '{k}' as direct 2D matrix, shape={v.shape}")
            return v.astype(np.float32)

    # case 2: cell array of days (object dtype, each element a 2D day matrix)
    for k, v in info.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            rows = []
            for i in range(v.shape[0]):
                day = v[i, 0] if v.shape[1] == 1 else v[0, i]
                day = np.asarray(day, dtype=np.float32)
                if day.ndim == 2:
                    rows.append(day)
            if rows:
                out = np.vstack(rows).astype(np.float32)
                print(f"stacked {len(rows)} days from cell '{k}', shape={out.shape}")
                return out

    raise RuntimeError(
        "Could not auto-detect the data matrix. Inspect printed keys and "
        "adjust flatten_to_2d(), or print mat[name].shape manually."
    )


def clean_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary label encoding to {0,1,2}.

    FI-2010 labels are usually 1/2/3 (down/stationary/up) or 0/1/2.
    We map by sorted unique values so any consistent encoding works.
    Rows with unexpected values are coerced to the majority (index 1).
    """
    uniq = np.unique(labels)
    print(f"label raw unique values: {uniq[:10]} (n={len(uniq)})")
    if len(uniq) > 3:
        print(f"WARNING: >3 unique label values; coercing extras to class 1")
    # build map: smallest->0, middle->1, largest->2 (sorted)
    sorted_u = np.sort(uniq)
    # if more than 3, map all beyond the first 3 to class 1
    mapper = {v: i for i, v in enumerate(sorted_u[:3])}
    out = np.zeros(labels.shape, dtype=np.int64)
    for v in sorted_u[:3]:
        out[labels == v] = mapper[v]
    # any leftover (unexpected) -> class 1
    out[~np.isin(labels, sorted_u[:3])] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="path to FI-2010 .mat")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--inspect-only", action="store_true",
                    help="only print structure, do not convert")
    args = ap.parse_args()

    mat = scipy.io.loadmat(args.mat)
    info = inspect(args.mat)
    if args.inspect_only:
        return

    arr = flatten_to_2d(info)
    print("flattened shape:", arr.shape)

    n_cols = arr.shape[1]
    if n_cols < 148:
        raise ValueError(
            f"expected >=148 columns (144 features + 4 labels), got {n_cols}. "
            "The .mat layout differs; check inspect output."
        )
    features = arr[:, :144].astype(np.float32)
    # last 4 columns are k=10,20,50,100 labels
    labels_all = arr[:, -4:].astype(np.int64)

    # save a combined (N, 148) npy: 144 features + 4 labels, matching dataset.py
    combined = np.hstack([features, labels_all.astype(np.float32)]).astype(np.float32)
    np.save(args.out, combined)
    print(f"saved {args.out} shape={combined.shape} dtype={combined.dtype}")
    print("  features: cols 0-143 (144), labels k=10/20/50/100: cols 144-147")


if __name__ == "__main__":
    main()
