# 原始盘口数据扩充路线

## 决策

2024 Top-100 raw-200 pilot 的 86,775 参数基线和 1,033,383 参数容量实验已经完成。
五年 Top-400 raw-200 已完成 470,815 个股票日样本。容量实验在 seed 0、1、2 上的最佳
2024 验证期日均 Rank IC 分别为 `0.02145`、`0.02054` 和 `0.01893`，平均 `0.02031`，
seed 间样本标准差 `0.00127`。下一阶段先加入 1/3/5 日标签和固定 checkpoint 评估，不继续
增加模型参数。后续执行门槛见 `docs/multi-horizon-data-expansion-roadmap.md`。

正式工作集固定为 2021 至 2025 年、动态 Top-400、每个股票日最后 200 个有效 snapshot
事件。训练、验证和测试沿用 `configs/nextday.yaml`：2021 至 2023 年训练，2024 年验证，
2025 年保持锁定。数据生成不得读取或计算 2025 年模型指标。

## 已确认资源

2026-08-09 只读盘点结果：

| 项目 | 状态 |
|---|---:|
| 2021 snapshot 月文件 | 12 个，157.48 GiB |
| 2022 snapshot 月文件 | 12 个，174.18 GiB |
| 2023 snapshot 月文件 | 12 个，168.74 GiB |
| 2024 snapshot 月文件 | 12 个，182.16 GiB |
| 2025 snapshot 月文件 | 12 个，201.58 GiB |
| 60 个规范月文件 | 全部存在，共 884.14 GiB |
| 远程 NVMe 可用空间 | 610 GiB |
| 原始数据盘可用空间 | 761 GiB |
| Google Drive 套餐 | 200GB，2026-08-10 已升级 |

预计最多约 48 万个股票日。raw-200 的 float16 紧凑工作集约 7 至 8 GiB，适合先写入远程
NVMe，通过审计后再上传 Drive。原始 Parquet 保留在数据盘，不上传 Drive。

## 阶段 0：冻结数据契约

正式配置使用 `configs/nextday-raw.yaml`，以下字段在本轮生成期间保持不变：

- `start_date: 2021-01-01`、`end_date: 2025-12-31`
- `top_n: 400`
- `scan_start_time_ms: 18000000`、`signal_time_ms: 19500000`
- `chunks_per_sample: 2`、`chunk_size: 100`
- `min_valid_events: 200`
- `storage_dtype: float16`
- 动态股票池只使用历史 20 日成交额，至少 15 个有效观测

任何字段变化都生成新的输出目录和数据指纹，不覆盖正式工作集。

## 阶段 1：单月 Top-400 预检

先运行隔离配置：

```bash
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw-200-preflight.yaml
```

产物写入 `data/nextday-raw-200-preflight-202101-top400/`。通过条件：

- `manifest.json`、`data-audit.json` 和全部 shard 均存在
- universe 的最小值、中位数和最大值均有审计统计，中位股票数为 400
- `written_samples > 0`，不存在重复股票日
- 所有 shard 的 SHA-256 与 manifest 一致
- `last_event_timestamp <= signal_timestamp`
- 读取 `NextDayShardDataset` 后样本形状为 `2 × 100 × 40`

预检只验证工程链路，不用于模型结论。

## 阶段 2：生成五年 raw-200

预检通过后运行：

```bash
mkdir -p logs
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw.yaml \
  > logs/prepare-nextday-raw-200.log 2>&1
```

目标目录是 `data/nextday-raw-200/`。2024 Top-100 pilot 用时约 35 分钟，五年 Top-400 需要
扫描更大的股票集合，预算按 8 至 14 小时安排。生成期间不要启动第二个同目录任务。

当前 writer 对单个 shard 使用原子替换，完整 manifest 在全部样本写完后生成，尚不支持真正的
月级断点续跑。进程中断后保留的 shard 不能视为完整数据集，重跑前应确认没有仍在运行的任务。
后续若多次生成 raw-500 或 raw-1000，再实现按月物化和合并 manifest，避免重复扫描。

进度检查：

```bash
pgrep -af "ticknet-nextday-prepare-snapshot"
du -sh data/nextday-raw-200
find data/nextday-raw-200/shards -maxdepth 1 -name "part-*.npy" | wc -l
tail -n 50 logs/prepare-nextday-raw-200.log
```

## 阶段 3：完整性和覆盖审计

生成完成后执行：

1. 校验 manifest 指纹和全部 shard SHA-256。
2. 汇总每年、每月、每交易日样本数和股票覆盖。
3. 汇总 `missing_snapshot`、`insufficient_events`、`invalid_lob_rows` 和逐日备份回退月份。
4. 用 `configs/nextday.yaml` 构造 train、val、test 数据集，确认日期与标签日期均在各自区间。
5. 抽样检查归一化数值范围、类别比例和连续目标分布。
6. 保存只含聚合统计的审计摘要，不把真实数据或完整 manifest 提交到 Git。

最低验收门槛：

- 2021 至 2025 每个有目标的月份均有样本
- 数据指纹稳定，重复校验得到同一结果
- 训练、验证、测试日期无交集，跨边界标签已 purge
- 每日有效股票数足以计算 Top-400 口径的横截面指标
- fallback 使用的月份和文件数有明确记录

## 阶段 4：上传 Drive

本地审计通过后，上传紧凑工作集：

```bash
rclone copy data/nextday-raw-200 \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-200 \
  --checksum --transfers 4 --checkers 8 --progress
```

上传后用 `rclone check` 复核。Drive 长期只保留当前 raw-200 工作集、配置、checkpoint 和结果。
raw-500、raw-1000 进入下一阶段时采用轮换保存，避免三个完整版本与临时副本同时占用空间。

## 阶段 5：数据优先的模型比较

先固定相同数据、日期和训练预算，比较：

1. Logistic 或 HGB 聚合基线
2. 86,775 参数 raw-200 模型
3. 1,033,383 参数 raw-200 模型

验证期至少使用 seed 0、1、2，汇报均值、样本标准差、月度 IC 和正 IC 月份比例。2025 测试集
在模型、超参数和 seed 列表冻结前保持锁定，不按测试结果选择模型。

继续扩大容量的条件：1M 模型在多个 seed 和月份上稳定优于 86k 模型，并且改善来自多数月份，
而不是少数极端交易日。条件未满足时保留小模型。

## 阶段 6：窗口扩展

raw-200 正式比较完成后才生成更长窗口：

| 工作集 | 五年预计空间 | 启动条件 |
|---|---:|---|
| raw-200 Top-400 | 约 8 GiB | 当前执行 |
| raw-500 Top-100 | 约 5 GiB | 先检查长窗口是否有增量 |
| raw-500 Top-400 | 约 20 GiB | Top-100 增量稳定 |
| raw-1000 Top-400 | 约 40 GiB | raw-500 明确优于 raw-200 |

每一级都与相同日期、股票池和标签下的分钟模型比较。验证期 Rank IC 没有稳定增量、结果随
随机种子反转或成本后收益消失时停止扩展。
