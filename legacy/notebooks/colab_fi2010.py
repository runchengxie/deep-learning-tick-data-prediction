# Generated from the retired notebook colab.ipynb.
# This file is a historical Python snapshot and is not a production entry point.

# %% [markdown] cell 1
# # 在 Colab 上运行 DeepLOB
#
# ## 目标
#
# 本 notebook 按顺序完成环境准备、1 epoch 真实数据验收、论文 Table II 的 Setup 2 正式训练、结果汇总和曲线绘制。检查点保存在 Google Drive，Colab 中断后重新运行对应训练单元格即可恢复。
#
# 运行前请确认 `MyDrive/DeepLOB/data/` 包含：
#
# - `FI2010_normalised.npy`
# - `FI2010_normalised_meta.json` 或 `FI2010_meta.json`

# %% [markdown] cell 2
# ## 1. 准备环境并检查数据
#
# 已有仓库时会快进到最新 `main`。这个单元格也会确认 Drive 文件和 CUDA GPU 可用。训练入口随后会把数据复制到 `/content/DeepLOB/data/`，并在同一运行时内复用本地副本。检查点仍保存在 Drive。

# %% code cell 3
from pathlib import Path

import torch
from google.colab import drive

drive.mount("/content/drive")
repository = Path("/content/deep-learning-tick-data-prediction")
if not repository.exists():
    pass  # IPython command: !git clone https://github.com/runchengxie/deep-learning-tick-data-prediction.git {repository}
else:
    pass  # IPython command: !git -C {repository} pull --ff-only

# IPython command: %cd /content/deep-learning-tick-data-prediction

data_dir = Path("/content/drive/MyDrive/DeepLOB/data")
checkpoint_dir = Path("/content/drive/MyDrive/DeepLOB/checkpoints")
data_path = data_dir / "FI2010_normalised.npy"
meta_candidates = [
    data_dir / "FI2010_normalised_meta.json",
    data_dir / "FI2010_meta.json",
]
meta_path = next((path for path in meta_candidates if path.is_file()), None)

if not data_path.is_file() or meta_path is None:
    raise FileNotFoundError(f"FI-2010 数据不完整：{data_dir}")
if not torch.cuda.is_available():
    raise RuntimeError("未检测到 CUDA。请在运行时设置中选择 GPU。")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"数据：{data_path}")
print(f"元数据：{meta_path}")
print(f"GPU：{torch.cuda.get_device_name(0)}")

# %% [markdown] cell 4
# ## 2. 运行 1 epoch 真实数据验收
#
# 这个单元格只验证完整数据链路，不产生可与论文比较的最终指标。它会生成可恢复的 `setup2.k10` 检查点。

# %% code cell 5
# IPython command: %%time
# IPython command: !python scripts/run_colab.py --protocol setup2 --k 10 --epochs 1

# %% [markdown] cell 6
# ## 3. 运行 Table II：Setup 2
#
# 依次运行下面五个单元格。`k=10` 会从验收检查点的第 2 epoch 继续；其他预测跨度会新建各自检查点。每次训练最多 100 epochs，验证准确率连续 20 epochs 未提升时提前停止。Colab 中断后重新运行同一个单元格即可恢复。

# %% code cell 7
# IPython command: %%time
# IPython command: !python scripts/run_colab.py --protocol setup2 --k 10

# %% code cell 8
# IPython command: %%time
# IPython command: !python scripts/run_colab.py --protocol setup2 --k 20

# %% code cell 9
# IPython command: %%time
# IPython command: !python scripts/run_colab.py --protocol setup2 --k 30

# %% code cell 10
# IPython command: %%time
# IPython command: !python scripts/run_colab.py --protocol setup2 --k 50

# %% code cell 11
# IPython command: %%time
# IPython command: !python scripts/run_colab.py --protocol setup2 --k 100

# %% [markdown] cell 12
# ## 4. 汇总 Setup 2 结果
#
# 已完成的预测跨度会显示 Accuracy、macro F1 和 weighted F1。缺少结果的跨度会被跳过。

# %% code cell 13
import json

import pandas as pd
from IPython.display import display

rows = []
for horizon in (10, 20, 30, 50, 100):
    result_path = checkpoint_dir / f"result.setup2.k{horizon}.json"
    if not result_path.is_file():
        continue
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["test"]
    rows.append(
        {
            "k": horizon,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "duration_seconds": result["duration_seconds"],
        }
    )

if rows:
    display(pd.DataFrame(rows).sort_values("k").reset_index(drop=True))
else:
    print("尚未找到 Setup 2 结果文件。")

# %% [markdown] cell 14
# ## 5. 绘制训练曲线

# %% code cell 15
from IPython.display import Image

curve_path = checkpoint_dir / "train_curves.png"
# IPython command: !python scripts/plot_curves.py --checkpoint-dir {checkpoint_dir} --out {curve_path}
display(Image(filename=str(curve_path)))

# %% [markdown] cell 16
# ## 6. 可选：运行 Table I 的 Setup 1
#
# Setup 1 对五个预测跨度分别运行九个 CF，共 45 次训练。确认 Colab 时长和 Drive 空间后，把 `RUN_SETUP1` 改为 `True`。检查点同样支持中断恢复。

# %% code cell 17
RUN_SETUP1 = False

if RUN_SETUP1:
    for horizon in (10, 20, 30, 50, 100):
        print(f"\n===== Setup 1：k={horizon} =====", flush=True)
        pass  # IPython command: !python scripts/run_colab.py --protocol setup1 --k {horizon}
else:
    print("Setup 1 未启动。确认后把 RUN_SETUP1 改为 True。")
