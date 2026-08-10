# 原始盘口 200 tick 端到端主线：本地加工到 Colab 训练

> 适用范围：验证深度模型直接接受原始十档 snapshot tick 并输出次日方向的端到端能力。
> 本文与 `resource-strategy-and-pilot-gates.md` 互补：那份讲资源怎么省着用，
> 这份讲 raw-200 这条主线怎么从本地数据一路走到锁定测试，并且每一步用什么门槛
> 拦住。

## 1. 目标与现状

### 目标

输入是每个股票日 14:55 前最后 200 个十档盘口 snapshot tick，输出次日开盘到收盘的
超额收益分数，以及下跌、中性、上涨三分类概率。模型是共享 DeepLOB 编码 2 个
100-event 块，GRU 汇总后接双头输出。默认配置有 86,775 个参数；容量扩展配置有
1,033,383 个参数。两者均可在单卡 Colab 会话内训练。

### 现状

- `raw_snapshot.py` 数据准备链路、`train.py` 训练器和 raw-200 pilot 配置都已就位；默认
  86k 模型和百万参数版本均已有真实 2024 pilot 多 seed 结果。百万参数版本没有提高跨 seed
  平均验证 Rank IC，下一阶段转向五年 Top-400 数据扩充。
- 数据加工在本地主机完成，Colab 只做训练，这正是参考
  `notebooks/nextday_end_to_end_colab.ipynb` 的分工。
- 本地主机 CPU 实测约 8.5 股票日/秒，三年分块 DeepLOB 一个 epoch 约 9.8 小时，
  所以正式训练必须上 Colab 或云 GPU。

## 2. 数据加工（本地主机，不占 Colab 额度）

参考 notebook 的约定：训练数据在本地从移动硬盘生成，Drive 只保存筛选后的 float16
工作集、配置、checkpoint 和结果，原始 Parquet 不上传。

### 入口与配置

```bash
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw-pilot.yaml
```

关键参数（`configs/nextday-raw-pilot.yaml`）：

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
- 分片 sha256 齐全且 manifest 指纹一致，Colab 端可校验。

## 3. 门槛式推进（每步通过才进下一步）

```text
本地数据加工（第 2 节，费 CPU 免费）
   ↓ 验收数据审计
Logistic 基线（本地 CPU，证明分片有信息）
   ↓ 验证期 IC 为正、无泄漏
Colab 吞吐测试 + 100 batch 试跑（确认预算与断点续训）
   ↓ 每 epoch 时间可接受、resume 正常
pilot 训练（2024H1 train / Q3 val / Q4 test 锁定）
   ↓ 固定种子验证期比较，冻结配置
解锁锁定测试（只评估一次，不再选模型）
```

### 3.1 Logistic 基线（先于深度模型）

roadmap 要求先过 Logistic，证明分片里真有信息，同时校验管线无泄漏。

### 3.2 Colab 吞吐与 100 batch 试跑

在 `notebooks/nextday_end_to_end_colab.ipynb` 的 smoke 模式跑通
Drive → Colab → checkpoint 闭环。确认：

- GPU 可达，`torch.cuda.is_available()` 为真。
- 工作集能从 Drive 复制到 `/content` 临时盘，且校验指纹一致。
- 训练能断点续训，checkpoint 写回 Drive。

### 3.3 pilot 训练与锁定测试

- 2024H1 训练、Q3 验证、Q4 测试锁定。
- 固定多个随机种子做验证期比较并冻结配置。
- `EVALUATE_LOCKED_TEST` 只有显式填确认字符串才解锁，解锁后只评估一次。
- 报告跨 seed 的测试 Rank IC 和 Macro F1 均值与标准差，不按测试结果选 seed。

### 3.4 百万参数容量实验

`configs/nextday-raw-1m-pilot.yaml` 只改变模型容量，复用 raw-200 pilot 的输入、标签、日期切分
和训练超参：

| 结构参数 | 86k 基线 | 1.03M 容量实验 |
|---|---:|---:|
| `conv_channels` | 16 | 32 |
| `inception_channels` | 32 | 64 |
| `intraday_embedding_size` | 64 | 320 |
| `day_hidden_size` | 64 | 192 |
| 总参数量 | 86,775 | 1,033,383 |

先在 Colab notebook 选择 `MODEL_PROFILE = "capacity_1m"` 运行 100 batch，记录吞吐与峰值显存，
再分别以 seed 0、1、2 训练。2024Q4 已用于既有 pilot 结果，只能作为开发诊断；本轮用验证期
Rank IC、Macro F1、训练耗时和跨 seed 波动比较容量增量，不重新把该区间称为 locked test。
只有训练指标提高而验证指标不提高时，停止继续扩容。

## 4. 用满 Colab 的约定（与 notebook 一致）

1. 训练期间不通过 Drive 挂载点随机读 NPY，先复制到 `/content`。
2. checkpoint 走 Drive，会话断了能接。
3. 锁定测试由纯评估入口完成，不创建优化器、不读 last checkpoint。
4. 只用 sha256 校验过的分片，指纹不一致就停止。

## 5. 判断口径

出现以下任一信号，就应停止扩大模型。停止也是有效结论：

- 只有训练集指标变好，验证集 Rank IC 没变。
- 改一个月份或随机种子结论就反转。
- 原始盘口没有稳定超过分钟模型。
- 收益全部来自涨跌停、停牌恢复或极低流动性股。
- 扣除合理成本后分组收益消失。

## 6. 下一步动作

1. 使用 `configs/nextday-raw-200-preflight.yaml` 完成单月 Top-400 数据预检。
2. 使用 `configs/nextday-raw.yaml` 生成 2021 至 2025 五年 Top-400 raw-200 工作集。
3. 完成分片校验、日期覆盖和 train/val/test purge 审计后再上传 Drive。
4. 在正式工作集上重新比较 86k 与 1.03M，容量增量跨 seed 稳定后才增加参数或窗口。

参考：`notebooks/nextday_end_to_end_colab.ipynb`、`configs/nextday-raw-pilot.yaml`、
`configs/nextday-pilot.yaml`、`configs/nextday-raw-1m-pilot.yaml`、
`docs/raw-data-expansion-roadmap.md`。
