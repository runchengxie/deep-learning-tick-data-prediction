"""DeepLOB on Colab - single entry point.

Run it from a Colab code cell. IMPORTANT: because google.colab.drive.mount()
only works inside the notebook's MAIN kernel (not in a `!python` subprocess),
the mount must happen in the notebook cell BEFORE this script runs. The
recommended notebook cell is:

    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://github.com/runchengxie/deeplob-reproduction.git /content/deeplob-reproduction
    %cd /content/deeplob-reproduction
    !python scripts/run_colab.py

This script assumes /content/drive is already mounted and simply checks it.
It also assumes the repo was cloned to /content/deeplob-reproduction (so
`src/`, `configs/`, and this file live together), which is why the notebook
cell above uses `git clone` rather than downloading run_colab.py in isolation.

Default protocol is `light_setup2` (train CF_7, test CF_7/8/9): a cheaper run
to confirm the pipeline and measure GPU time before the full 9-fold
`standard9` sweep. Switch to standard9 by editing the protocol line in this
script (or pass --protocol standard9 --num_folds 9).
"""
import os
import subprocess
from pathlib import Path

# When cloned to /content/deeplob-reproduction, REPO_ROOT is that directory.
REPO_ROOT = Path("/content/deeplob-reproduction")

DATA_ROOT = "/content/drive/MyDrive"
if not os.path.isdir(DATA_ROOT):
    raise SystemExit(
        "Drive is not mounted. In the Colab cell, run BEFORE this script:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "then re-run the notebook cell (git clone + python scripts/run_colab.py)"
    )

# 1. Drive paths (already mounted by the notebook cell above).
DATA_DIR = os.path.join(DATA_ROOT, "DeepLOB", "data")
CKPT_DIR = os.path.join(DATA_ROOT, "DeepLOB", "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 2. Use FI-2010 data. Expected format: (N, 149) float32 npy at npy_path,
#    with cols [0:144] = features and [144:149] = 5 label columns
#    (k=10/20/50/100 map to cols 144/145/146/147). Prepare it LOCALLY with
#    convert_fi2010.py (reads the official .txt files) and upload the .npy to
#    MyDrive/DeepLOB/data/. The segment meta (FI2010_meta.json) must be
#    uploaded alongside it so dataset.py can build non-leaking windows.
npy_path = os.path.join(DATA_DIR, "FI2010_normalised.npy")
folds_path = os.path.join(DATA_DIR, "FI2010_folds.npy")
meta_path = os.path.join(DATA_DIR, "FI2010_meta.json")
if not os.path.exists(npy_path):
    raise FileNotFoundError(
        f"Missing {npy_path}. Prepare it locally:\n"
        "  python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets \\\n"
        "      --auction NoAuction --norm z-score --folds 1 2 3 4 5 6 7 8 9 \\\n"
        "      --out FI2010_normalised.npy\n"
        "then upload BOTH FI2010_normalised.npy AND FI2010_meta.json to\n"
        "MyDrive/DeepLOB/data/ on Drive.\n"
        "Do NOT use the shanehans/FI2010 CSV mirror: it has 130 features + 15 junk "
        "columns + dirty labels (150 cols), not the official 144+5 layout."
    )
else:
    print("reusing", npy_path)

# 3. Install third-party deps. Colab already ships torch / numpy / pyyaml;
#    scikit-learn is the one usually missing. We avoid `pip install -e .`
#    (this repo's package layout is loose) and just ensure the deps exist.
subprocess.run(["pip", "install", "scikit-learn", "pyyaml"], check=True)

# 4. Decide device. If the Colab runtime has no GPU enabled, fall back to CPU
#     (slow, but it runs) instead of crashing on .to('cuda').
import torch  # available after the pip install above
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("=" * 60)
    print("WARNING: CUDA not available -> training on CPU (very slow).")
    print("To use the T4 GPU: Runtime -> Change runtime type -> Hardware")
    print("accelerator = GPU -> Save, then re-run this cell.")
    print("=" * 60)

# 5. Train. Use configs/colab.yaml as the base config, then override the
#    runtime-only values. Prefer the segment-aware protocol (standard9) when
#    FI2010_meta.json is present; otherwise fall back to the legacy fold-id
#    mode (deprecated, numbers NOT directly comparable to the paper).
train_cmd = [
    "python", str(REPO_ROOT / "src" / "train.py"),
    "--config", str(REPO_ROOT / "configs" / "colab.yaml"),
    "--dataset", "fi2010",
    "--data_path", npy_path,
    "--device", device,
    "--checkpoint_dir", CKPT_DIR,
]
if os.path.exists(meta_path):
    # Default FIRST run = light_setup2: train on CF_7, test on CF_7/8/9.
    # Cheaper (~20w train / 14w test) so you can confirm the whole pipeline
    # and measure real per-epoch GPU time before committing to the full
    # 9-fold standard9 sweep. To run the paper-Table-II protocol instead,
    # change "light_setup2" below to "standard9" (or pass
    # --protocol standard9 --num_folds 9 on the command line).
    print("FI2010_meta.json found -> protocol=light_setup2 (CF_7 train, CF_7/8/9 test)")
    train_cmd += ["--protocol", "light_setup2", "--meta_path", meta_path,
                  "--light_test_cf", "7,8,9"]
elif os.path.exists(folds_path):
    print("WARNING: only FI2010_folds.npy found -> legacy fold-id CV (deprecated).")
    train_cmd += ["--cv", "--folds_path", folds_path, "--num_folds", "9"]
else:
    print("WARNING: neither FI2010_meta.json nor FI2010_folds.npy found -> "
          "using 70/15/15 split (numbers NOT comparable to the paper's Table II).")

subprocess.run(train_cmd, check=True)

print("DONE. Checkpoints at:", CKPT_DIR)
