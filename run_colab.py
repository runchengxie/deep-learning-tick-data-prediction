"""DeepLOB on Colab - single entry point.

Run this from a Colab cell with:  !python run_colab.py
All logic lives in this .py file so notebook line-wrapping cannot
turn it into one commented-out line.
"""
import os
import subprocess
import numpy as np
import pandas as pd
from google.colab import drive

REPO = "https://github.com/runchengxie/deeplob-reproduction.git"

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

# 2. Mount Drive
drive.mount("/content/drive")
DATA_DIR = "/content/drive/MyDrive/DeepLOB/data"
CKPT_DIR = "/content/drive/MyDrive/DeepLOB/checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 3. Download FI-2010 CSV and convert to float32 .npy (only once)
npy_path = os.path.join(DATA_DIR, "FI2010_normalised.npy")
if not os.path.exists(npy_path):
    subprocess.run([
        "wget", "-O", "fi2010_train.csv",
        "https://huggingface.co/datasets/shanehans/FI2010/resolve/main/FI2010_train.csv"
    ], check=True)
    df = pd.read_csv("fi2010_train.csv", header=None)
    arr = df.to_numpy(dtype=np.float32)
    assert arr.shape[1] >= 44, f"expected >=44 cols, got {arr.shape[1]}"
    np.save(npy_path, arr)
    print("saved", npy_path, arr.shape, arr.dtype)
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
