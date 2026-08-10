"""生成 2024 validation-only 多周期 Colab notebook。"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "notebooks" / "nextday_multi_horizon_validation_colab.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT_PATH.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Raw-200 多周期验证（2024 Validation Only）

## Goal

加载 `capacity_1m` 的 seed 0、1、2 三个既有 **best checkpoint**，只在 2024 validation
计算 IC@1D/3D/5D、月度稳定性、Newey–West 标准误和非重叠抽样。

这个 notebook **不训练、不续训、不创建 optimizer，也不对 2025 test 做模型推理或指标计算**。
结果自动写回 Google Drive；2025 test 继续锁定。"""
        ),
        nbf.v4.new_markdown_cell(
            """## Setup

### 固定实验合同

- 特征：信号日 14:55 前最后 200 个十档 snapshot。
- 模型：1,033,383 参数 `capacity_1m`，使用已完成训练的三个 best checkpoint。
- 收益：`T+1 open → T+H close` 相对中证全指同期收益，`H ∈ {1, 3, 5}`。
- 样本：仅 `2024-01-01` 至 `2024-12-31` validation；跨边界收益会被 purge。
- 决策：先观察固定一日模型的跨周期排序能力，再决定是否训练独立 H=5 模型。"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path

SEEDS = (0, 1, 2)
HORIZONS = (1, 3, 5)
INFERENCE_BATCH_SIZE = 128
NUM_WORKERS = 2
VERIFY_DATA_CHECKSUMS = True

PROJECT_REPOSITORY = "https://github.com/runchengxie/deep-learning-tick-data-prediction.git"
PROJECT_REVISION = "main"
DRIVE_ROOT = Path("/content/drive/MyDrive/deep-learning-tick-data-prediction")
DRIVE_DATA_DIR = DRIVE_ROOT / "ticknet-data/nextday-raw-200"
DRIVE_TARGET_DIR = DRIVE_ROOT / "ticknet-data/nextday-raw-200-targets-v1"
DRIVE_RUN_DIR = DRIVE_ROOT / "ticknet-runs/raw-200-capacity_1m"
DRIVE_OUTPUT_DIR = DRIVE_RUN_DIR / "multi-horizon-validation-2024"

LOCAL_PROJECT_DIR = Path("/content/deep-learning-tick-data-prediction")
LOCAL_DATA_DIR = Path("/content/nextday-raw-200")
LOCAL_TARGET_DIR = Path("/content/nextday-raw-200-targets-v1")
CHECKPOINT_NAME = "raw-200-dual-head-capacity_1m"
"""
        ),
        nbf.v4.new_markdown_cell("### 1. Mount Drive and fetch the current `main`"),
        nbf.v4.new_code_cell(
            """import importlib
import shutil
import subprocess
import sys

from google.colab import drive

drive.mount("/content/drive")

if LOCAL_PROJECT_DIR.exists():
    shutil.rmtree(LOCAL_PROJECT_DIR)
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        PROJECT_REVISION,
        PROJECT_REPOSITORY,
        str(LOCAL_PROJECT_DIR),
    ],
    check=True,
)
source_revision = subprocess.run(
    ["git", "-C", str(LOCAL_PROJECT_DIR), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(LOCAL_PROJECT_DIR)],
    check=True,
)
PROJECT_SRC = LOCAL_PROJECT_DIR / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
importlib.invalidate_caches()
ticknet = importlib.import_module("ticknet")

print({"ticknet": ticknet.__file__, "source_revision": source_revision})"""
        ),
        nbf.v4.new_markdown_cell("### 2. Verify GPU, storage, sidecar, and all best checkpoints"),
        nbf.v4.new_code_cell(
            """import json

import torch

assert torch.cuda.is_available(), "当前会话没有 CUDA GPU，请在 Runtime 设置中选择 GPU"
feature_manifest_path = DRIVE_DATA_DIR / "manifest.json"
target_manifest_path = DRIVE_TARGET_DIR / "horizon-labels.json"
assert feature_manifest_path.is_file(), f"找不到特征 manifest：{feature_manifest_path}"
assert target_manifest_path.is_file(), f"找不到多周期标签侧车：{target_manifest_path}"

feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
assert target_manifest["horizons"] == [1, 3, 5]
assert target_manifest["source_dataset_fingerprint"] == feature_manifest["dataset_fingerprint"]

checkpoint_paths = [DRIVE_RUN_DIR / f"{CHECKPOINT_NAME}.seed{seed}.best.pt" for seed in SEEDS]
missing_checkpoints = [str(path) for path in checkpoint_paths if not path.is_file()]
assert not missing_checkpoints, f"缺少 best checkpoint：{missing_checkpoints}"

feature_bytes = sum(
    (DRIVE_DATA_DIR / shard["path"]).stat().st_size for shard in feature_manifest["shards"]
)
free_bytes = shutil.disk_usage("/content").free
assert free_bytes > feature_bytes * 1.2, "Colab 临时盘不足以复制 raw-200 工作集"
print(
    {
        "gpu": torch.cuda.get_device_name(0),
        "feature_samples": len(feature_manifest["samples"]),
        "feature_fingerprint": feature_manifest["dataset_fingerprint"],
        "target_fingerprint": target_manifest["sidecar_fingerprint"],
        "feature_gib": feature_bytes / 2**30,
        "free_gib": free_bytes / 2**30,
        "checkpoints": [path.name for path in checkpoint_paths],
        "test_status": "locked_not_accessed",
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Steps

### 3. Stage features and the small label sidecar on local ephemeral disk

NPY 随机读取不走 Drive 挂载点。raw-200 约 7.2 GiB，只复制一次；约 22 MiB 的 sidecar
单独复制。checkpoint 和聚合结果仍保存在 Drive。"""
        ),
        nbf.v4.new_code_cell(
            """for local_path in (LOCAL_DATA_DIR, LOCAL_TARGET_DIR):
    if local_path.exists():
        shutil.rmtree(local_path)
shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
shutil.copytree(DRIVE_TARGET_DIR, LOCAL_TARGET_DIR)

local_manifest_path = LOCAL_DATA_DIR / "manifest.json"
local_sidecar_path = LOCAL_TARGET_DIR / "horizon-labels.json"
staged_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
staged_sidecar = json.loads(local_sidecar_path.read_text(encoding="utf-8"))
assert staged_manifest["dataset_fingerprint"] == feature_manifest["dataset_fingerprint"]
assert staged_sidecar["sidecar_fingerprint"] == target_manifest["sidecar_fingerprint"]
print(
    {
        "local_features": str(LOCAL_DATA_DIR),
        "local_sidecar": str(LOCAL_TARGET_DIR),
        "shards": len(staged_manifest["shards"]),
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Reconstruct the exact original training signature

这些字段必须与三个 checkpoint 的训练配置一致，才能通过签名校验。`batch_size=32` 是原训练
配置；实际只读推理使用独立的 `INFERENCE_BATCH_SIZE`。"""
        ),
        nbf.v4.new_code_cell(
            """from ticknet.nextday.train import NextDayConfig

training_config = NextDayConfig(
    manifest_path=str(local_manifest_path),
    train_start="2021-01-01",
    train_end="2023-12-31",
    val_start="2024-01-01",
    val_end="2024-12-31",
    test_start="2025-01-01",
    test_end="2025-12-31",
    epochs=30,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    patience=8,
    seed=0,
    num_workers=NUM_WORKERS,
    device="cuda",
    resume=True,
    evaluate_test=False,
    verify_data_checksums=VERIFY_DATA_CHECKSUMS,
    checkpoint_dir=str(DRIVE_RUN_DIR),
    checkpoint_name=CHECKPOINT_NAME,
    conv_channels=32,
    inception_channels=64,
    intraday_embedding_size=320,
    day_hidden_size=192,
    day_layers=1,
    dropout=0.1,
    class_weighting="balanced",
    selection_metric="daily_rank_ic_mean",
    min_symbols_per_day=100,
    portfolio_quantile=0.1,
    classification_loss_weight=1.0,
    regression_loss_weight=0.5,
    gradient_accumulation_steps=1,
    amp=True,
)
training_config.validate()
assert training_config.evaluate_test is False
training_config"""
        ),
        nbf.v4.new_markdown_cell(
            """### 5. Run checkpoint-only inference and multi-horizon evaluation

这一单元只加载 `.best.pt` 并做前向推理。模块会先验证全部 checkpoint 签名，再开始计算；
任一 seed 不匹配会立即停止，不会写出半套结论。"""
        ),
        nbf.v4.new_code_cell(
            """from ticknet.nextday.horizon_evaluation import evaluate_validation_horizons

DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result = evaluate_validation_horizons(
    training_config,
    local_sidecar_path,
    seeds=SEEDS,
    horizons=HORIZONS,
    output_dir=DRIVE_OUTPUT_DIR,
    inference_batch_size=INFERENCE_BATCH_SIZE,
    source_revision=source_revision,
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Checks

### 6. Review IC decay, uncertainty, and monthly stability

`daily_rank_ic_mean` 是逐信号日横截面 Rank IC 的均值；Newey–West lag 固定为 `H-1`。
`positive_month_ratio` 表示月均 IC 为正的月份占比。"""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd
from IPython.display import display

summary_rows = []
model_order = [*(f"seed_{seed}" for seed in SEEDS), "ensemble"]
for horizon in HORIZONS:
    horizon_result = result["results"][str(horizon)]
    for model_name in model_order:
        metrics = horizon_result["models"][model_name]
        summary_rows.append(
            {
                "horizon": horizon,
                "model": model_name,
                "days": metrics["dates"],
                "mean_ic": metrics["daily_rank_ic_mean"],
                "ic_ir": metrics["daily_rank_ic_ir"],
                "nw_lag": metrics["newey_west"]["lag"],
                "nw_se": metrics["newey_west"]["standard_error"],
                "nw_t": metrics["newey_west"]["t_stat"],
                "positive_month_ratio": metrics["monthly_stability"]["positive_month_ratio"],
                "worst_non_overlap_phase": metrics["non_overlapping"]["min_phase_mean"],
            }
        )
summary_table = pd.DataFrame(summary_rows)
display(
    summary_table.style.format(
        {
            "mean_ic": "{:.3%}",
            "ic_ir": "{:.3f}",
            "nw_se": "{:.4f}",
            "nw_t": "{:.2f}",
            "positive_month_ratio": "{:.1%}",
            "worst_non_overlap_phase": "{:.3%}",
        }
    )
)"""
        ),
        nbf.v4.new_code_cell(
            """import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots(figsize=(8, 4.5))
for model_name in model_order:
    values = [
        result["results"][str(horizon)]["models"][model_name]["daily_rank_ic_mean"]
        for horizon in HORIZONS
    ]
ax.plot(HORIZONS, values, marker="o", label=model_name)
ax.axhline(0, color="black", linewidth=0.8)
ax.set(
    title="2024 validation IC decay",
    xlabel="Holding horizon (trading days)",
    ylabel="Mean daily Rank IC",
)
ax.set_xticks(HORIZONS)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(ncol=2)
ax.grid(alpha=0.25)
plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """monthly_rows = []
for horizon in HORIZONS:
    monthly_rows.extend(
        {"horizon": horizon, **row}
        for row in result["results"][str(horizon)]["models"]["ensemble"]["monthly_stability"][
            "monthly"
        ]
    )
monthly_table = pd.DataFrame(monthly_rows)
monthly_pivot = monthly_table.pivot(index="horizon", columns="month", values="mean")

fig, ax = plt.subplots(figsize=(12, 3.2))
limit = max(abs(monthly_pivot.min().min()), abs(monthly_pivot.max().max()))
image = ax.imshow(monthly_pivot.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(len(monthly_pivot.columns)), monthly_pivot.columns, rotation=45, ha="right")
ax.set_yticks(range(len(monthly_pivot.index)), [f"H={value}" for value in monthly_pivot.index])
ax.set_title("Ensemble monthly mean Rank IC")
fig.colorbar(image, ax=ax, format=PercentFormatter(1.0), label="Mean Rank IC")
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell("### 7. Inspect the non-overlapping five-day robustness gate"),
        nbf.v4.new_code_cell(
            """phase_rows = []
for model_name in model_order:
    phases = result["results"]["5"]["models"][model_name]["non_overlapping"]["phases"]
    phase_rows.extend({"model": model_name, **row} for row in phases)
phase_table = pd.DataFrame(phase_rows)
display(phase_table.style.format({"mean": "{:.3%}", "std": "{:.3%}"}))

h5_gate = result["results"]["5"]["roadmap_gate"]
print(json.dumps(h5_gate, ensure_ascii=False, indent=2))"""
        ),
        nbf.v4.new_code_cell(
            """assert result["mode"] == "fixed_best_checkpoint_multi_horizon_validation"
assert result["test_status"] == "locked_not_accessed"
assert result["training_status"] == "not_run"
assert result["seeds"] == list(SEEDS)
assert result["horizons"] == list(HORIZONS)
assert all(Path(path).is_file() for path in result["artifacts"].values())
assert all(len(row["sha256"]) == 64 for row in result["checkpoints"])

print(
    {
        "evaluation_complete": True,
        "test_status": result["test_status"],
        "h5_meets_roadmap_gate": h5_gate["meets_roadmap_gate"],
        "summary": result["artifacts"]["summary"],
        "daily_ic": result["artifacts"]["daily_rank_ic"],
        "scores": result["artifacts"]["scores"],
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Next Steps

1. 保存本次已执行 notebook 的 `.ipynb` 或 HTML 快照到
   `ticknet-runs/raw-200-capacity_1m/multi-horizon-validation-2024/`。
2. 若 H=5 的三个 seed 均值为正、多数月份为正且五个非重叠 phase 不反转，再开独立 PR
   训练 raw-200、1M 参数的 H=5 单目标模型。
3. 若门槛未通过，保留现有一日模型并停止多周期训练；先分析失败月份与 phase。
4. 无论结果如何，本 notebook 都不解锁 2025 test；先冻结 horizon、收益处理、模型结构和 seed。
5. 只有独立 H=5 模型证明有稳定增量后，才进入 raw-500 / raw-1000 数据扩容。"""
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"
        cell["source"] = cell["source"].strip()
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    nbf.write(notebook, OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
