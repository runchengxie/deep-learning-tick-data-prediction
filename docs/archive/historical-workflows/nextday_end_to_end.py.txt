# Generated from the retired notebook nextday_end_to_end_colab.ipynb.
# This file is a historical Python snapshot and is not a production entry point.

# %% [markdown] cell 1
# # 原始 tick → 次日信号：Colab 端到端训练
#
# ## Goal
#
# 这个 notebook 训练共享 DeepLOB + GRU 双头模型，并支持 87k 基线与 1.03M 容量扩展实验。输入是每个股票日 14:55 前最后 200 个十档盘口 snapshot tick；输出是次日开盘到收盘超额收益分数，以及下跌/中性/上涨概率。
#
# 训练数据应先在本地从移动硬盘生成。Drive 只保存筛选后的 float16 工作集、配置、checkpoint 和结果；原始 Parquet 不上传。

# %% [markdown] cell 2
# ## Setup
#
# ### Key assumptions
#
# - `RUN_MODE = "smoke"` 使用 9 个交易日的真实小样本验证 Drive → Colab → checkpoint 闭环。
# - `RUN_MODE = "pilot"` 使用 2024 全年动态前 100 股票。`EVALUATE_LOCKED_TEST = False` 时不会计算 2024Q4 测试指标。
# - `RUN_MODE = "full"` 使用五年 raw-200 工作集。`EVALUATE_LOCKED_TEST = False` 时不会计算 2025 测试指标。
# - `MODEL_PROFILE = "baseline"` 使用 86,775 参数基线；`MODEL_PROFILE = "capacity_1m"` 使用 1,033,383 参数容量扩展。两者写入不同 checkpoint 目录。
# - `EVALUATE_LOCKED_TEST = True` 只调用纯评估入口，一次读取 `LOCKED_TEST_SEEDS` 对应的固定 best checkpoint，不恢复训练。
# - 项目源码从 GitHub `main` 拉取并打印实际 commit；`DRIVE_DATA_DIR` 包含 `manifest.json` 和 `shards/*.npy`。
# - Colab GPU 和运行时长不保证，因此所有正式训练都把可恢复 checkpoint 写回 Drive。

# %% code cell 3
from pathlib import Path

RUN_MODE = "pilot"  # "smoke"、"pilot" 或 "full"
MODEL_PROFILE = "baseline"  # "baseline" 或 "capacity_1m"
SEED = 4
LOCKED_TEST_SEEDS = (0, 1, 2, 3, 4)
EVALUATE_LOCKED_TEST = False  # 默认永久锁定测试集；解锁需显式确认字符串
UNLOCK_CONFIRMATION = ""  # 解锁测试集须填：EVALUATE-LOCKED-TEST-ONCE
VERIFY_DATA_CHECKSUMS = True
PROJECT_REPOSITORY = "https://github.com/runchengxie/deep-learning-tick-data-prediction.git"
PROJECT_REVISION = "main"
DRIVE_ROOT = Path("/content/drive/MyDrive/deep-learning-tick-data-prediction")
LOCAL_PROJECT_DIR = Path("/content/deep-learning-tick-data-prediction")
MODEL_PROFILES = {
    "baseline": {
        "conv_channels": 16,
        "inception_channels": 32,
        "intraday_embedding_size": 64,
        "day_hidden_size": 64,
    },
    "capacity_1m": {
        "conv_channels": 32,
        "inception_channels": 64,
        "intraday_embedding_size": 320,
        "day_hidden_size": 192,
    },
}

if RUN_MODE == "smoke":
    DRIVE_DATA_DIR = DRIVE_ROOT / "ticknet-data/nextday-raw-smoke-v2"
    DRIVE_RUN_DIR = DRIVE_ROOT / "ticknet-runs/raw-200-smoke-colab-v2"
    LOCAL_DATA_DIR = Path("/content/nextday-raw-smoke-v2")
    DATE_SPLIT = {
        "train_start": "2024-01-02",
        "train_end": "2024-01-05",
        "val_start": "2024-01-08",
        "val_end": "2024-01-10",
        "test_start": "2024-01-11",
        "test_end": "2024-01-15",
    }
    EPOCHS = 2
    BATCH_SIZE = 8
    BENCHMARK_BATCHES = 10
    MIN_SYMBOLS_PER_DAY = 10
    PORTFOLIO_QUANTILE = 0.2
    PATIENCE = 2
    CHECKPOINT_NAME = "raw-200-smoke-colab-v2"
    EVALUATE_TEST = True
elif RUN_MODE == "pilot":
    DRIVE_DATA_DIR = DRIVE_ROOT / "ticknet-data/nextday-raw-pilot-2024-top100"
    DRIVE_RUN_DIR = DRIVE_ROOT / "ticknet-runs/nextday-pilot-2024-top100"
    LOCAL_DATA_DIR = Path("/content/nextday-raw-pilot-2024-top100")
    DATE_SPLIT = {
        "train_start": "2024-01-01",
        "train_end": "2024-06-30",
        "val_start": "2024-07-01",
        "val_end": "2024-09-30",
        "test_start": "2024-10-01",
        "test_end": "2024-12-31",
    }
    EPOCHS = 10
    BATCH_SIZE = 32
    BENCHMARK_BATCHES = 100
    MIN_SYMBOLS_PER_DAY = 80
    PORTFOLIO_QUANTILE = 0.1
    PATIENCE = 4
    CHECKPOINT_NAME = "raw-200-pilot-2024-top100"
    EVALUATE_TEST = False
elif RUN_MODE == "full":
    DRIVE_DATA_DIR = DRIVE_ROOT / "ticknet-data/nextday-raw-200"
    DRIVE_RUN_DIR = DRIVE_ROOT / "ticknet-runs/raw-200"
    LOCAL_DATA_DIR = Path("/content/nextday-raw-200")
    DATE_SPLIT = {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "val_start": "2024-01-01",
        "val_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    }
    EPOCHS = 30
    BATCH_SIZE = 32
    BENCHMARK_BATCHES = 100
    MIN_SYMBOLS_PER_DAY = 100
    PORTFOLIO_QUANTILE = 0.1
    PATIENCE = 8
    CHECKPOINT_NAME = "raw-200-dual-head"
    EVALUATE_TEST = False
else:
    raise ValueError(f"未知 RUN_MODE: {RUN_MODE!r}")

if MODEL_PROFILE not in MODEL_PROFILES:
    raise ValueError(f"未知 MODEL_PROFILE: {MODEL_PROFILE!r}")
MODEL_CONFIG = MODEL_PROFILES[MODEL_PROFILE]
if MODEL_PROFILE != "baseline":
    DRIVE_RUN_DIR = DRIVE_RUN_DIR.with_name(f"{DRIVE_RUN_DIR.name}-{MODEL_PROFILE}")
    CHECKPOINT_NAME = f"{CHECKPOINT_NAME}-{MODEL_PROFILE}"

# %% [markdown] cell 4
# ### 1. Mount Drive and fetch the project

# %% code cell 5
import importlib
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
assert (PROJECT_SRC / "ticknet").is_dir(), f"找不到项目源码：{PROJECT_SRC}"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
importlib.invalidate_caches()
ticknet = importlib.import_module("ticknet")

print(f"ticknet loaded from: {ticknet.__file__}")
print(f"source revision: {source_revision}")

# %% [markdown] cell 6
# ### 2. Verify GPU, disk, and source artifacts

# %% code cell 7
import json

import torch

assert torch.cuda.is_available(), "当前会话没有 CUDA GPU，请在 Runtime 设置中选择 GPU"
assert (DRIVE_DATA_DIR / "manifest.json").is_file(), "Drive 数据清单不存在"
manifest = json.loads((DRIVE_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
feature_bytes = sum((DRIVE_DATA_DIR / shard["path"]).stat().st_size for shard in manifest["shards"])
free_bytes = shutil.disk_usage("/content").free
assert free_bytes > feature_bytes * 1.2, "Colab 临时盘不足以复制当前工作集"
print(
    {
        "run_mode": RUN_MODE,
        "gpu": torch.cuda.get_device_name(0),
        "samples": len(manifest["samples"]),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "data_gib": feature_bytes / 2**30,
        "free_gib": free_bytes / 2**30,
    }
)

# %% [markdown] cell 8
# ## Steps
#
# ### 3. Copy the current workset to local ephemeral disk
#
# 训练期间不要通过 Drive 挂载点随机读取 NPY；先复制到 `/content`，checkpoint 再写回 Drive。

# %% code cell 9
if LOCAL_DATA_DIR.exists():
    shutil.rmtree(LOCAL_DATA_DIR)
shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
local_manifest_path = LOCAL_DATA_DIR / "manifest.json"
staged_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
assert staged_manifest.get("dataset_fingerprint"), "工作集缺少 dataset_fingerprint"
missing_checksums = [
    shard["path"] for shard in staged_manifest["shards"] if not shard.get("sha256")
]
assert not missing_checksums, f"工作集分片缺少 sha256：{missing_checksums[0]}"
assert staged_manifest["dataset_fingerprint"] == manifest["dataset_fingerprint"], (
    "本地工作集与 Drive manifest 指纹不一致"
)
print(
    {
        "local_workset": str(LOCAL_DATA_DIR),
        "dataset_fingerprint": staged_manifest["dataset_fingerprint"],
        "shards": len(staged_manifest["shards"]),
    }
)

# %% [markdown] cell 10
# ### 4. Build a dated split and benchmark throughput

# %% code cell 11
import time

from torch.utils.data import DataLoader

from ticknet.nextday.dataset import NextDayShardDataset
from ticknet.nextday.model import build_nextday_model
from ticknet.nextday.splits import WalkForwardSplit

date_split = WalkForwardSplit.from_strings(**DATE_SPLIT)
train_data = NextDayShardDataset(
    LOCAL_DATA_DIR / "manifest.json", date_split=date_split, split="train"
)
loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)
model = build_nextday_model(
    chunks_per_sample=train_data.chunks_per_sample,
    chunk_size=train_data.chunk_size,
    **MODEL_CONFIG,
).cuda()
model.train()
parameter_count = sum(parameter.numel() for parameter in model.parameters())
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
seen = 0
for batch_index, (features, labels, targets) in enumerate(loader):
    if batch_index >= BENCHMARK_BATCHES:
        break
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(features.cuda(non_blocking=True))
        loss = torch.nn.functional.cross_entropy(
            output.logits, labels.cuda(non_blocking=True)
        ) + 0.5 * torch.nn.functional.smooth_l1_loss(output.score, targets.cuda(non_blocking=True))
    loss.backward()
    model.zero_grad(set_to_none=True)
    seen += features.shape[0]
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
samples_per_second = seen / elapsed
print(
    {
        "run_mode": RUN_MODE,
        "model_profile": MODEL_PROFILE,
        "parameter_count": parameter_count,
        "batches": min(len(loader), BENCHMARK_BATCHES),
        "samples_per_second": samples_per_second,
        "estimated_epoch_minutes": len(train_data) / samples_per_second / 60,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
)
del model, loader, train_data
torch.cuda.empty_cache()

# %% [markdown] cell 12
# ### 5. Train or resume the dual-head model

# %% code cell 13
from ticknet.nextday.train import NextDayConfig, train

DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
training_config = NextDayConfig(
    manifest_path=str(LOCAL_DATA_DIR / "manifest.json"),
    train_start=DATE_SPLIT["train_start"],
    train_end=DATE_SPLIT["train_end"],
    val_start=DATE_SPLIT["val_start"],
    val_end=DATE_SPLIT["val_end"],
    test_start=DATE_SPLIT["test_start"],
    test_end=DATE_SPLIT["test_end"],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=1e-3,
    weight_decay=1e-4,
    patience=PATIENCE,
    seed=SEED,
    num_workers=2,
    device="cuda",
    resume=True,
    evaluate_test=EVALUATE_TEST,
    verify_data_checksums=VERIFY_DATA_CHECKSUMS,
    checkpoint_dir=str(DRIVE_RUN_DIR),
    checkpoint_name=CHECKPOINT_NAME,
    conv_channels=MODEL_CONFIG["conv_channels"],
    inception_channels=MODEL_CONFIG["inception_channels"],
    intraday_embedding_size=MODEL_CONFIG["intraday_embedding_size"],
    day_hidden_size=MODEL_CONFIG["day_hidden_size"],
    dropout=0.1,
    classification_loss_weight=1.0,
    regression_loss_weight=0.5,
    gradient_accumulation_steps=1,
    amp=True,
    min_symbols_per_day=MIN_SYMBOLS_PER_DAY,
    portfolio_quantile=PORTFOLIO_QUANTILE,
)
result = None
if EVALUATE_LOCKED_TEST and RUN_MODE != "smoke":
    print("locked-test 模式只构建配置，本单元不会恢复或继续训练。")
else:
    result = train(training_config)

# %% [markdown] cell 14
# ### 6. Evaluate fixed best checkpoints without training
#
# 这个单元会先确认 `LOCKED_TEST_SEEDS` 的所有 best checkpoint 和实验签名都存在且一致，再计算测试指标。它不创建优化器，不读取 last checkpoint，也不改写训练 checkpoint。把 `EVALUATE_LOCKED_TEST` 改为 `True` 代表正式解锁测试集。

# %% code cell 15
from ticknet.nextday.train import evaluate_best_checkpoints

locked_test_result = None
if RUN_MODE == "smoke":
    print("smoke 的测试闭环已由训练单元执行。")
elif not EVALUATE_LOCKED_TEST:
    print("locked test 仍保持锁定。")
else:
    if UNLOCK_CONFIRMATION != "EVALUATE-LOCKED-TEST-ONCE":
        raise RuntimeError("解锁测试集需要 UNLOCK_CONFIRMATION == 'EVALUATE-LOCKED-TEST-ONCE'")
    locked_test_result = evaluate_best_checkpoints(
        training_config,
        LOCKED_TEST_SEEDS,
    )

# %% [markdown] cell 16
# ## Checks
#
# ### 7. Inspect the evaluation artifact
#
# `smoke` 展示单个训练结果。`pilot` 和 `full` 的 locked test 展示固定种子间的均值和样本标准差。正式汇报还需要类别分布、按月稳定性和含成本回测。

# %% code cell 17
if locked_test_result is not None:
    aggregate = locked_test_result["aggregate"]
    summary = {
        "seeds": locked_test_result["seeds"],
        "samples": locked_test_result["samples"],
        "dataset_fingerprint": locked_test_result["dataset_fingerprint"],
        "test_status": "evaluated_from_fixed_best_checkpoints",
        "test_rank_ic": aggregate["daily_rank_ic_mean"],
        "test_macro_f1": aggregate["macro_f1"],
        "evaluated_test_dates": locked_test_result["per_seed"][0]["test"]["evaluated_dates"],
        "result_file": locked_test_result["result_file"],
    }
elif result is not None:
    result_path = Path(result["result_file"])
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    test_metrics = saved_result["test"]
    summary = {
        "samples": saved_result["samples"],
        "dataset_fingerprint": saved_result["dataset_fingerprint"],
        "best_validation_metric": saved_result["best_selection_value"],
        "test_status": "evaluated" if test_metrics is not None else "locked",
    }
    if test_metrics is not None:
        summary.update(
            test_rank_ic=test_metrics["daily_rank_ic_mean"],
            test_macro_f1=test_metrics["macro_f1"],
            evaluated_test_dates=test_metrics["evaluated_dates"],
        )
else:
    summary = {"test_status": "locked"}
summary

# %% [markdown] cell 18
# ## Next Steps
#
# 1. `smoke` 用于验证 Drive、GPU、训练、checkpoint 和结果读取。
# 2. `pilot` 先保持 `EVALUATE_LOCKED_TEST = False`，完成固定种子的验证期比较并冻结配置。
# 3. 解锁时把 `EVALUATE_LOCKED_TEST` 改为 `True`，从参数单元开始顺序运行。训练单元会跳过训练，纯评估单元一次读取五个 best checkpoint。
# 4. 保存 `locked_test.*.json`，报告跨 seed 的测试 Rank IC 和 Macro F1 均值与标准差，不按测试结果选择 seed。
# 5. 2024Q4 解锁后不再用它调整 pilot。后续 500 tick 或正式五年实验使用新的时间外测试区间。
# 6. 增加涨跌停、ST、停牌和实际交易成本约束。
# 7. 结果展示只使用可复现的 checkpoint、输入契约和样本外指标，不把工程 smoke test 描述成盈利证明。
# 8. `capacity_1m` 先作为开发期容量实验；2024Q4 已经查看过，不重新作为 locked test 选模。
