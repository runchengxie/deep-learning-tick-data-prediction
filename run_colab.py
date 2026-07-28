"""DeepLOB on Colab - single entry point.

Run it from a Colab code cell. IMPORTANT: because google.colab.drive.mount()
only works inside the notebook's MAIN kernel (not in a `!python` subprocess),
the mount must happen in the notebook cell BEFORE this script runs. The
recommended notebook cell is:

    from google.colab import drive
    drive.mount('/content/drive')
    !curl -L -o /content/run_colab.py https://raw.githubusercontent.com/runchengxie/deeplob-reproduction/main/run_colab.py
    !python /content/run_colab.py

This script assumes /content/drive is already mounted and simply checks it.
"""
import os
import subprocess

REPO = "https://github.com/runchengxie/deeplob-reproduction.git"

# 0. Make sure Drive is already mounted (done in the notebook cell, not here).
DATA_ROOT = "/content/drive/MyDrive"
if not os.path.isdir(DATA_ROOT):
    raise SystemExit(
        "Drive is not mounted. In the Colab cell, run BEFORE this script:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "then re-run !python /content/run_colab.py"
    )

# 1. Fetch code via GitHub ZIP download (avoids git HTTPS credential
#    prompts that fail on Colab: "could not read Username").
os.chdir("/content")
local = "deeplob-reproduction"
ZIP_URL = "https://github.com/runchengxie/deeplob-reproduction/archive/refs/heads/main.zip"
# Always re-fetch latest (cheap, and avoids stale state).
subprocess.run(["rm", "-rf", local, local + "-main"], check=True)
r = subprocess.run(["curl", "-L", "-o", "repo.zip", ZIP_URL], capture_output=True, text=True)
if r.returncode != 0:
    print("DOWNLOAD STDERR:", r.stderr[-800:])
    raise RuntimeError("failed to download repo zip")
subprocess.run(["unzip", "-q", "repo.zip"], check=True)
os.rename(local + "-main", local)
os.chdir(local)

# 2. Drive paths (already mounted by the notebook cell above).
DATA_DIR = os.path.join(DATA_ROOT, "DeepLOB", "data")
CKPT_DIR = os.path.join(DATA_ROOT, "DeepLOB", "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 3. Use FI-2010 data. Expected format: (N, 149) float32 npy at npy_path,
#    with cols [0:144] = features and [144:149] = 5 label columns
#    (k=10/20/50/100 map to cols 144/145/146/147). Prepare it LOCALLY with
#    convert_fi2010.py (reads the official .txt files) and upload the .npy to
#    MyDrive/DeepLOB/data/. If absent, training cannot start.
npy_path = os.path.join(DATA_DIR, "FI2010_normalised.npy")
if not os.path.exists(npy_path):
    raise FileNotFoundError(
        f"Missing {npy_path}. Prepare it locally:\n"
        "  python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets \\\n"
        "      --auction NoAuction --norm z-score --folds 1 2 3 4 5 6 7 8 9 \\\n"
        "      --out FI2010_normalised.npy\n"
        "then upload FI2010_normalised.npy to MyDrive/DeepLOB/data/ on Drive.\n"
        "Do NOT use the shanehans/FI2010 CSV mirror: it has 130 features + 15 junk "
        "columns + dirty labels (150 cols), not the official 144+5 layout."
    )
else:
    print("reusing", npy_path)

# 4. Install deps
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# 5. Train (Setup 2: 70/15/15 train/val/test; k selects label column)
subprocess.run([
    "python", "src/train.py",
    "--dataset", "fi2010",
    "--data_path", os.path.join(DATA_DIR, "FI2010_normalised.npy"),
    "--k", "10",
    "--epochs", "100",
    "--batch_size", "32",
    "--device", "cuda",
    "--checkpoint_dir", CKPT_DIR,
], check=True)

print("DONE. Best weights at:", os.path.join(CKPT_DIR, "best.pt"))
