# H5 Rank IC 与事件流滚动路线

## 决策摘要

本研究系列使用处理后的五日收益作为训练目标，用 validation 每日横截面 Rank IC@5D 选择 checkpoint。IC@3D 用来观察信号是否提前生效，IC@1D 只作诊断。每个完整滚动折包含三个月训练、一个月 validation 和一个月 OOS。

2021 至 2024 是历史研究区。2025 已经用于成本审计、模型比较和假设形成，因此属于可见的开发与滚动验证区。2026 是锁定区，在独立审批和完整样本达到协议门槛前不得读取。

最近折使用以下窗口：

```text
train       2025-08-01 至 2025-10-31
validation  2025-11-01 至 2025-11-30
OOS         2025-12-01 至 2025-12-31
locked      2026-01-01 起
```

2021 折保留为基础设施和跨年份吞吐基线。2025 的事件密度更高，正式训练耗时使用 2025 pack 的实测吞吐估算。

## 固定实验合同

```text
信号时点：T 日 14:55 前
进入：T+1 开盘
退出：T+H 收盘，H ∈ {1, 3, 5}
连续目标：个股收益减去中证全指同期收益
主训练目标：H=5 连续收益，SmoothL1 回归
辅助目标：下一事件与订单类型交叉熵
主选择指标：validation 每日横截面 Rank IC@5D
监控指标：Rank IC@3D、月度正 IC 比例、Newey-West t 值、五组非重叠抽样
诊断指标：Rank IC@1D
```

Rank IC 先在同一信号日内跨股票计算，再跨日平均。第一轮不把 Rank IC 直接写入 loss，因为随机事件窗口 batch 不包含完整的日度横截面。H5 回归基线通过后，再评估按日 batch 的 pairwise 或 listwise ranking loss。

H5 purge 要求 `trading_date`、`entry_date` 和 `return_end_date` 全部位于同一个 split。validation 只用于 checkpoint 选择，OOS 只用于该折的一次性报告。2021 年 1 月至 2025 年 12 月可以生成 56 个完整折，最后一个是 `fold-55-oos-202512`。rolling plan 不包含 2026。

## 2025 数据审计

2025 年 8 月至 12 月的 order、trades 和 snapshot 三路输入完整，共 103 个交易日。原始数据留在 Linux 数据盘，不上传 Drive。

| 月份 | 交易日 | 原始字节数 |
|---|---:|---:|
| 2025-08 | 21 | 102,662,250,487 |
| 2025-09 | 22 | 107,672,938,318 |
| 2025-10 | 17 | 77,376,235,803 |
| 2025-11 | 20 | 89,256,118,925 |
| 2025-12 | 23 | 100,099,208,577 |
| 合计 | 103 | 477,066,752,110 |

preflight、逐日股票池、rolling plan 和标签位于 `artifacts/eventstream-h5-recent-fold/`。H3 在 train、validation 和 OOS 分别保留 22,058、6,578 和 7,732 条标签。H5 分别保留 21,266、5,807 和 6,966 条标签。跨 split 的收益标签已经清除。

五个月 Top-400 pack 已经完整生成，共 412 个文件、313.11 GiB，103 个日索引覆盖 2025-08-01 至 2025-12-31，没有遗留的 partial 文件。2025 年 8 月 pack 约为 68.58 GB，已完成审计和上传。截至 2026-08-16，Google Drive 总额为 200 GiB，当前约使用 98.1 GiB，剩余约 100.5 GiB。正式 3/1/1 训练需要选择 400GB Drive、GCS 或可恢复的月度流式暂存，不能通过删除字段降低精度。

本机没有可用 CUDA GPU。现有 `run_colab_nextday.py` 只覆盖事件流 benchmark、batch sweep 和 input profile，尚无正式训练 workflow。压缩抽样约为原始体积的 24% 至 27%，目前还没有形成正式暂存方案。

## 分阶段进度

| 阶段 | 状态 | 动作与结果 |
|---|---|---|
| R0 | 完成 | 生成 2021-01 至 2025-12 的 56 折计划，确认不含 2026 |
| D0 | 完成 | 五个月原始数据、preflight 和股票池完整，源 manifest 指纹固定 |
| F0 | 完成 | 前收盘价逐股票回退到最近有效正值，测试覆盖 null、NaN 和缺列 |
| D1 | 完成 | 2025-08-01 四类文件齐全，随机读取和 H5 标签覆盖通过 |
| D2 | 完成 | 2025 年 8 月 21 日 pack 完成审计并上传 |
| B0 | 完成 | A100 输入分析完成，优化后为 149.40 samples/s，20 epoch 外推 4.46 小时每 seed |
| D3 | 完成 | 2025 年 9 月至 12 月全部打包，103 个交易日均有完整日索引 |
| S0a | 完成 | 增加按月逻辑存储清单、直存远端文件核对、完整复制容量检查和落盘内容核对，固定 103 个交易日并阻断 2026 |
| S0b | 待执行 | 选择可恢复的月度流式暂存或足够容量的远端存储，定义传输清单和恢复边界，补齐正式训练与产物回传 workflow |
| T0 | 待前置 | S0b 完成后运行最近折正式 seed 0，H5 选择 checkpoint，H3 只作监控，门槛见[事件流主线](eventstream.md#正式训练与扩容门槛) |
| T1 | 待门槛 | seed 0 通过后补 seed 1 和 2，报告三 seed 均值、方向一致性与成本敏感性 |
| E0 | 待门槛 | 100M checkpoint 通过后生成冻结 embedding，接入 AgentX M4 的相同下游合同 |
| C0 | 待门槛 | 100M 信号或迁移门槛通过后，运行 `probe150m` benchmark 与 seed 0 容量消融 |
| L0 | 封存 | 2026 达到协议门槛并获批后一次性评估，不据此回调本轮模型 |

## 可复现命令

生成不触及 2026 的 rolling plan：

```bash
python -m ticknet.nextday.rolling \
  --start-month 2021-01 \
  --end-month 2025-12 \
  --target-horizon 5 \
  --output artifacts/eventstream-h5-recent-fold/rolling-plan.json
```

生成最近折的 H3 和 H5 标签：

```bash
ticknet-eventstream-prepare-horizon-labels \
  --sidecar artifacts/eventstream-h5-fold0/horizon-sidecar/horizon-labels.json \
  --feature-manifest data/nextday-raw-200/manifest.json \
  --output-dir artifacts/eventstream-h5-recent-fold/fold-labels \
  --horizons 3 5 \
  --train-start 2025-08-01 --train-end 2025-10-31 \
  --val-start 2025-11-01 --val-end 2025-11-30 \
  --test-start 2025-12-01 --test-end 2025-12-31
```

打包指定交易日：

```bash
python -m ticknet.eventstream.pack \
  --days 20250801 \
  --universe artifacts/eventstream-h5-recent-fold/202508/universe.json \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1
```

首日实测墙钟为 311 秒，峰值内存为 24.8 GiB，swap 峰值为 224 KiB。396 只股票产生 60,012,903 条委托、33,284,058 条成交和 1,876,631 条快照，pack 共 2,742,987,628 字节。数据集可以构造 2,000 个训练样本，日标签覆盖 396 只股票。三处随机读取均得到 `512 × 80` 的有限特征。

2025 年 8 月 pack 上传后，可以运行 A100 基准：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-capacity-benchmark \
  --session ticknet-eventstream-h5-recent-100m-a100 \
  --gpu A100 \
  --benchmark-batches 100 --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/benchmarks/a100
```

## 完成标准

- rolling plan、股票池、标签和 pack 绑定源数据指纹或确定性文件集合
- 2025 可以用于开发结论，训练和调参路径不得读取 2026
- 训练只用 validation 选择 checkpoint，OOS 不反向影响同一研究系列
- IC@5D 同时报告逐日、月度、Newey-West 和非重叠五日结果，IC@3D 单独监控
- 真实数据、股票列表、逐日 pack、checkpoint 和预测不提交 Git，仓库只提交代码、配置与聚合审计
