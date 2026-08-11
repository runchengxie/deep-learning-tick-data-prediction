# H5 Rank IC、3/1/1 滚动验证与全天事件数据路线图

## 决策摘要

本研究系列以处理后的五日收益为训练目标，以 validation 每日横截面 Rank IC@5D 选择
checkpoint；IC@3D 是信号提前生效的监控指标，IC@1D 只作诊断。每个 fold 固定使用三个完整
自然月训练、一个月 validation、一个月 OOS，并逐月滚动。

数据边界统一为：2021–2024 是历史研究区，2025 已被用于成本审计、模型比较和假设形成，因此是
可见的开发与滚动验证区；2026 才是 locked 区。任何 2025 结果都不得再描述成“未见最终测试”。
2026 在独立审批且完整样本达到研究协议门槛前不读取、不调参、不出正式结论。

正式 seed 0 使用当前最新完整的五个月事件流窗口：

```text
train       2025-08-01 .. 2025-10-31
validation  2025-11-01 .. 2025-11-30
OOS         2025-12-01 .. 2025-12-31
locked      2026-01-01 起
```

2021 fold 继续保留为基础设施和跨年份吞吐基线，但不能代替 2025 pack 的短 benchmark。2025 的事件
密度显著更高，正式训练耗时只能由 2025 实测吞吐外推。

## 不可变实验合同

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

Rank IC 先在同一信号日内跨股票计算，再跨日平均。第一轮不把 Rank IC 直接写进 loss，因为随机
事件窗口 batch 不包含完整交易日横截面。只有 H5 回归基线通过后，才评估按日 batch 的
pairwise/listwise ranking loss。

H5 purge 由数据合同强制：`trading_date`、`entry_date` 和 `return_end_date` 必须全部落在同一
split。validation 只用于 checkpoint 选择；OOS 只用于该 fold 的一次性报告。2021-01 至
2025-12 可生成 56 个完整 fold，最后一个是 `fold-55-oos-202512`；rolling plan 不包含 2026。

## 2025 数据审计与标签

2025-08 至 2025-12 的 order、trades、snapshot 三路输入均完整，共 103 个交易日。raw 永久留在
Linux，不上传 Drive。

| 月份 | 交易日 | raw bytes |
|---|---:|---:|
| 2025-08 | 21 | 102,662,250,487 |
| 2025-09 | 22 | 107,672,938,318 |
| 2025-10 | 17 | 77,376,235,803 |
| 2025-11 | 20 | 89,256,118,925 |
| 2025-12 | 23 | 100,099,208,577 |
| 合计 | 103 | 477,066,752,110 |

preflight、逐日 universe、rolling plan 和标签位于
`/home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-recent-fold/`。
H3 共保留 train/validation/OOS 22,058/6,578/7,732 条，H5 共保留
21,266/5,807/6,966 条；其余跨 split 的收益标签已 purge。

200GB Drive 继续按 120GB 稳定占用、150GB 峰值软门槛管理。根据 2021 实测压缩比，完整五个月
Top-400 pack 很可能超过 200GB：本地先完整生成，Colab B0 只上传 2025-08 benchmark pack；正式
3/1/1 训练在得到实测 pack 体积后选择 400GB Drive、GCS 或按月流式 staging，不能靠删字段降低
精度。

## 分阶段推进

| 阶段 | 状态 | 动作与通过门槛 |
|---|---|---|
| R0 | 完成 | 生成 2021-01 至 2025-12 的 56-fold plan，确认 2026 不出现 |
| D0 | 完成 | 五个月 raw/preflight/universe 完整，源 manifest 指纹固定 |
| F0 | 完成 | 前收盘价逐股票回退到最近有效正值；测试覆盖 null/NaN/缺列 |
| D1 | 完成 | 2025-08-01 四文件齐全，随机读取和 H5 标签覆盖通过，无 OOM |
| D2 | 待执行 | 打包 2025-08，审计实际体积、事件数、峰值 RSS 与耗时 |
| B0 | 待执行 | 上传 2025-08 pack，在 T4/A100 跑 100 batch recent benchmark |
| D3 | 待执行 | 本地断点续跑 2025-09 至 2025-12，确认无 partial day |
| T0 | 待执行 | recent fold 正式 seed 0；H5 选 checkpoint，H3 只监控 |
| T1 | 待执行 | seed 1/2 与更多 rolling fold，报告最差 OOS 月和成本敏感性 |
| L0 | 封存 | 2026 数据达到协议门槛且获批后一次性评估；不得据此回调本轮模型 |

## 可复现命令

生成不触及 2026 的 rolling plan：

```bash
python -m ticknet.nextday.rolling \
  --start-month 2021-01 \
  --end-month 2025-12 \
  --target-horizon 5 \
  --output /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-recent-fold/rolling-plan.json
```

生成 recent fold 的 H3/H5 标签：

```bash
ticknet-eventstream-prepare-horizon-labels \
  --sidecar /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-fold0/horizon-sidecar/horizon-labels.json \
  --feature-manifest data/nextday-raw-200/manifest.json \
  --output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-recent-fold/fold-labels \
  --horizons 3 5 \
  --train-start 2025-08-01 --train-end 2025-10-31 \
  --val-start 2025-11-01 --val-end 2025-11-30 \
  --test-start 2025-12-01 --test-end 2025-12-31
```

D1 只打包首日；成功后才启动整月：

```bash
python -m ticknet.eventstream.pack \
  --days 20250801 \
  --universe /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-recent-fold/202508/universe.json \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1
```

D1 实测墙钟 311 秒、峰值内存 24.8 GiB、swap 峰值 224 KiB。396 只股票产生
60,012,903 条委托、33,284,058 条成交和 1,876,631 条快照，pack 共 2,742,987,628 bytes。
数据集可构造 2,000 个训练样本，日标签覆盖 396/396；三处随机读取均为 `512 × 80` 有限特征。

2025-08 pack 上传后，运行 recent A100 benchmark：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-capacity-benchmark \
  --session ticknet-eventstream-h5-recent-100m-a100 \
  --gpu A100 \
  --benchmark-batches 100 --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-recent-fold/benchmarks/a100
```

## 完成定义

- rolling plan、股票池、标签和 pack 均绑定源数据指纹或确定性文件集合。
- 2025 可用于开发结论但不称 locked；任何训练或调参路径都不读取 2026。
- 训练只用 validation 选 checkpoint，OOS 不反向影响同一研究系列。
- IC@5D 同时报告逐日、月度、Newey-West 和非重叠五日结果，IC@3D 独立监控。
- 真实数据、股票列表、逐日 pack、checkpoint 和预测不提交 Git；仓库只提交代码、配置与聚合审计。
