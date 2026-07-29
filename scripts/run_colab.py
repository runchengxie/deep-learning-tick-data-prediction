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
from pathlib import Path

# 仓库根目录（scripts/ 的上一级），Colab 上 cwd 会变，用绝对路径最稳
REPO_ROOT = Path(__file__).resolve().parent.parent

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
folds_path = os.path.join(DATA_DIR, "FI2010_folds.npy")
if not os.path.exists(npy_path):
    raise FileNotFoundError(
        f"Missing {npy_path}. Prepare it locally:\n"
        "  python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets \\\n"
        "      --auction NoAuction --norm z-score --folds 1 2 3 4 5 6 7 8 9 \\\n"
        "      --out FI2010_normalised.npy\n"
        "then upload BOTH FI2010_normalised.npy AND FI2010_folds.npy to\n"
        "MyDrive/DeepLOB/data/ on Drive.\n"
        "Do NOT use the shanehans/FI2010 CSV mirror: it has 130 features + 15 junk "
        "columns + dirty labels (150 cols), not the official 144+5 layout."
    )
else:
    print("reusing", npy_path)

# 4. Install deps
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# 4b. Decide device. If the Colab runtime has no GPU enabled, fall back to CPU
#     (slow, but it runs) instead of crashing on .to('cuda').
import torch  # available after the pip install above
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("=" * 60)
    print("WARNING: CUDA not available -> training on CPU (very slow).")
    print("To use the T4 GPU: Runtime -> Change runtime type -> Hardware")
    print("accelerator = GPU -> Save, then re-run this cell.")
    print("=" * 60)

# 5. Train. 用 configs/colab.yaml 作为基础配置（epochs=100、device=cuda 等），
#    再覆盖运行时才确定的动态值（data_path、device、checkpoint_dir）。
#    若 fold-id 数组存在则跑 9 折交叉验证（论文 Setup 2），否则退回到
#    简单的 70/15/15 切分（数字不能与论文 Table II 比较）。
train_cmd = [
    "python", str(REPO_ROOT / "src" / "train.py"),
    "--config", str(REPO_ROOT / "configs" / "colab.yaml"),
    "--dataset", "fi2010",
    "--data_path", npy_path,
    "--device", device,
    "--checkpoint_dir", CKPT_DIR,
]
if os.path.exists(folds_path):
    print("fold-id array found -> running 9-fold cross-validation")
    train_cmd += ["--cv", "--folds_path", folds_path, "--num_folds", "9"]
else:
    print("WARNING: FI2010_folds.npy not found -> using 70/15/15 split "
          "(numbers NOT comparable to the paper's Table II).")

subprocess.run(train_cmd, check=True)

print("DONE. Checkpoints at:", CKPT_DIR)
