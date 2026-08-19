# 原始盘口端到端主线

本文说明原始十档快照从本地加工到锁定测试的操作流程。资源门槛见[资源策略与试验门槛](../research/resource-strategy-and-pilot-gates.md)。本文保留 raw-200 pilot 的执行记录，当前状态以[项目现状](../project-status.md)为准。

## 1. 目标与现状

### 目标

输入是每个股票日 14:55 前最后 200 个十档盘口 snapshot tick，输出次日开盘到收盘的超额收益分数，以及下跌、中性、上涨三分类概率。模型用共享 DeepLOB 编码 2 个 100-event 块，GRU 汇总后接双头输出。默认配置有 86,775 个参数，容量扩展配置有 1,033,383 个参数，两者都可以在单卡 Colab 会话内训练。

### 现状

- `raw_snapshot.py` 数据准备链路、`train.py` 训练器和 raw 配置都已就位。raw-200、raw-1000、约 1M 参数和 100M 参数均有真实训练结果。
- 五年 Top-400 raw-200 工作集和五年 Top-100 raw-1000 工作集已经生成。四格三 seed 受控矩阵选择 `1M/raw-200` 作为唯一候选，继续扩容和加长窗口已经停止。完整数字见 [multi-horizon-data-expansion-roadmap.md](multi-horizon-data-expansion-roadmap.md)。
- 数据加工在本地主机完成，Colab 只做训练。现行调度入口为 `scripts/run_colab_nextday.py`。
- 本地主机 CPU 实测约 8.5 股票日每秒，三年分块 DeepLOB 一个 epoch 约 9.8 小时，正式训练必须上 Colab 或云 GPU。

## 2. 数据加工（本地主机，不占 Colab 额度）

训练数据在本地从移动硬盘生成。Drive 只保存筛选后的 float16 工作集、配置、checkpoint 和结果，原始 Parquet 不上传。

### 入口与配置

```bash
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw-pilot.yaml
```

关键参数来自 `configs/nextday-raw-pilot.yaml`：

- `start_date / end_date`：2024 全年。
- `scan_start_time_ms`：只读 14:30 起的 tick。
- `signal_time_ms`：14:55，作为信号时点，之后的数据绝不进入样本。
- `chunks_per_sample: 2`、`chunk_size: 100`：200 tick 切成 2 个 100-event 块。
- `min_valid_events: 200`：不足 200 个有效 tick 的股票日丢弃。
- `top_n: 100`：每天动态选前 100 只股票，股票池只使用信号时点前已知信息。
- `storage_dtype: float16`：Colab 工作集。
- `samples_per_shard: 2048`：分片粒度。

### 产物

`data/nextday-raw-pilot-2024-top100/`：

- `manifest.json`：数据清单，含分片路径、样本行号、sha256 和 dataset_fingerprint。
- `shards/part-*.npy`：float16 分片，布局 `samples × chunks × time × 40`。
- `data-audit.json`：股票池覆盖、抽取统计、标签分布。

### 数据加工验收

- 输入时间严格早于或等于 14:55 信号时点。
- 输入日和标签日是相邻交易日，日期切分无交集。
- 每个股票日只有一个样本。
- 分片 sha256 齐全且 manifest 指纹一致，Colab 端可以校验。

## 3. 门槛式推进（阶段记录）

```text
本地数据加工（第 2 节，占 CPU 但不占 Colab 额度）
   ↓ 验收数据审计
Logistic 基线（本地 CPU，证明分片有信息）
   ↓ 验证期 IC 为正、无泄漏
Colab 吞吐测试 + 100 batch 试跑（确认预算与断点续训）
   ↓ 每 epoch 时间可接受、resume 正常
pilot 训练（2024H1 训练 / Q3 验证 / Q4 测试锁定）
   ↓ 固定种子验证期比较，冻结配置
解锁锁定测试（只评估一次，不再选模型）
```

### 3.1 Logistic 基线（先于深度模型）

roadmap 要求先过 Logistic，证明分片里真有信息，同时校验管线无泄漏。

### 3.2 Colab 吞吐与 100 batch 试跑

Colab 自动化由 `scripts/run_colab_nextday.py` 负责。正式运行前先加 `--dry-run` 检查会话、配置、数据目录和输出目录。下面的命令展示 `1M/raw-200` 单 seed 训练入口，测试区继续锁定：

```bash
python scripts/run_colab_nextday.py \
  --workflow capacity-matrix-train \
  --matrix-cell 1m-raw200 \
  --session ticknet-capacity-matrix-1m-raw200 \
  --gpu A100 \
  --seeds 0 \
  --no-evaluate-test \
  --local-output-dir artifacts/capacity-matrix/1m-raw200/seed0
```

运行时确认：

- GPU 可达，`torch.cuda.is_available()` 为真。
- 工作集能从 Drive 复制到 `/content` 临时盘，且校验指纹一致。
- 训练能断点续训，checkpoint 写回 Drive。

### 3.3 pilot 训练与锁定测试

- 2024H1 训练、Q3 验证、Q4 测试锁定。
- 固定多个随机种子做验证期比较并冻结配置。
- `EVALUATE_LOCKED_TEST` 只有显式填确认字符串才解锁，解锁后只评估一次。
- 报告跨 seed 的测试 Rank IC 和 Macro F1 均值与标准差，不按测试结果选 seed。

### 3.4 百万参数容量实验

`configs/nextday-raw-1m-pilot.yaml` 只改变模型容量，复用 raw-200 pilot 的输入、标签、日期切分和训练超参：

| 结构参数 | 86k 基线 | 1.03M 容量实验 |
|---|---:|---:|
| `conv_channels` | 16 | 32 |
| `inception_channels` | 32 | 64 |
| `intraday_embedding_size` | 64 | 320 |
| `day_hidden_size` | 64 | 192 |
| 总参数量 | 86,775 | 1,033,383 |

容量实验通过独立 YAML 固定模型结构和数据合同，再用 Colab CLI 分别运行 seed 0、1、2。2024Q4 已用于既有 pilot 结果，只作开发诊断。本轮用验证期 Rank IC、Macro F1、训练耗时和跨 seed 波动比较容量增量，该区间不再作为 locked test。只有训练指标提高而验证指标不提高时，停止继续扩容。

## 4. 充分利用 Colab 的约定

1. 训练期间不通过 Drive 挂载点随机读 NPY，先复制到 `/content`。
2. checkpoint 走 Drive，会话断了可以接着跑。
3. 锁定测试由纯评估入口完成，不创建优化器，不读 last checkpoint。
4. 只用 sha256 校验过的分片，指纹不一致就停止。

## 5. 停止信号

出现任一停止信号就停止扩大模型，停止本身是有效的研究结论。完整的停止信号清单见 [硬件约束与分阶段实验路线](hardware-constraints-and-experiment-roadmap.md) 的评估和停止规则一节。

## 6. 当前动作

1. 固定 `1M/raw-200` 的三个 checkpoint、seed 聚合方式和验收指标。
2. 预先写明 2025 锁定测试的触发条件和通过条件。
3. 满足门槛后一次性评估 2025，测试结果不得用于重新选择模型。

项目整体工作顺序见[项目现状](../project-status.md)。

参考：`configs/nextday-raw-pilot.yaml`、`configs/nextday-pilot.yaml`、`configs/nextday-raw-1m-pilot.yaml`、[Colab CLI 自动化](../dev/colab-cli-automation.md)和[原始数据扩容路线](raw-data-expansion-roadmap.md)。历史交互入口归档在 `legacy/notebooks/nextday_end_to_end_colab.ipynb`。
