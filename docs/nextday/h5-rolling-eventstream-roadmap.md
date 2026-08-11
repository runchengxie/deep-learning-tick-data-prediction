# H5 Rank IC、3/1/1 滚动验证与全天事件数据路线图

## 决策摘要

本轮把处理后的五日收益设为主要训练标签，把每日横截面 Rank IC@5D 设为 checkpoint 选择
指标。IC@3D 是信号提前生效的监控指标，IC@1D 只作诊断。滚动实验固定使用三个月训练、一个月
验证、一个月 OOS，并按月向后移动。2025 保持最终 locked test，不参与滚动调参。

现有 H=1/3/5 标签侧车、月度 IC、Newey-West、非重叠五日抽样和独立 H5 训练已经实现。本路线
不重复这些能力，新增可指纹化的 rolling plan、全天 order/trade/snapshot 月度容量审计和逐日
Top-400 eventstream pack pilot。

截至 2026-08-11，Linux 上 2021-01 原始 order、trades 和 snapshot 分别约 8.8GB、19GB 和
13GB，合计约 40.8GB。整个原始 L2 数据湖约 4.3TB，数据盘剩余约 764GB，机器内存 31GB。
eventstream 正式 pack 目录尚无完成交易日。月度打包体积和峰值内存必须由 pilot 实测，不能用
朋友的 100GB/月口径替代本机审计结果。

## 不可变实验合同

第一轮沿用已经验证过的多周期标签合同：

```text
信号时点：T 日 14:55 前
进入：T+1 开盘
退出：T+H 收盘，H ∈ {1, 3, 5}
连续目标：个股收益减去中证全指同期收益
主训练目标：H=5 连续收益，SmoothL1 回归
辅助目标：三分类交叉熵
主选择指标：validation 每日横截面 Rank IC@5D
监控指标：Rank IC@3D、月度正 IC 比例、Newey-West t 值、五组非重叠抽样
诊断指标：Rank IC@1D
```

Rank IC 是同一信号日内跨股票计算后再跨日平均，不把不同日期混成一个横截面。第一轮不直接把
Rank IC 写成 loss，因为随机窗口 batch 不包含完整交易日横截面。只有 H5 回归基线通过后，才
评估按日 batch 的 pairwise/listwise ranking loss。

滚动切分固定为：

```text
3 个完整自然月 train
1 个完整自然月 validation
1 个完整自然月 OOS
每次向后滚动 1 个月
```

H5 purge 由数据合同强制：`trading_date`、`entry_date` 和 `return_end_date` 必须全部落在同一个
split。训练月末与验证月末会自动清除跨边界标签。OOS 只用于该 fold 的一次性报告，不参与
checkpoint 选择。2021-01 至 2024-12 共生成 44 个完整 fold，2025 留作最终 locked test。

## 数据与云端边界

原始 Parquet 永久留在 Linux，不上传 Drive。Linux 先按日筛选历史 Top-400，再打包无损整数镜像。
Colab 只接收当前 3/1/1 窗口所需的模型就绪 pack、标签和配置。每个 fold 完成后保留 checkpoint、
预测和聚合指标，旧的临时 pack 可以轮换。

200GB Drive 的稳定占用继续控制在 120GB 以内，计划峰值控制在 150GB 以内。如果一个月 Top-400
pack 乘以五超过 150GB，停止上传 Drive，改用 GCS、持久云盘或本地 GPU。不得通过删字段、降低
时间精度或重复覆盖正式产物来勉强通过容量门槛。

## 分阶段推进

| 阶段 | 动作 | 产物 | 通过门槛 | 失败动作 |
|---|---|---|---|---|
| R0 | 生成 2021-01 至 2024-12 的 3/1/1 rolling plan | `rolling-plan.json` 与 SHA-256 指纹 | 44 folds；2025 不出现；H5 purge 合同固定 | 修复日期协议 |
| D0 | 审计一个月 raw 三流与 Top-400 日股票池 | `pilot-audit.json`、`universe.json` | 输入日完整；股票池绑定 raw-200 manifest 指纹 | 修复源文件或股票池 |
| D1 | 只打包 2021-01-04 一个交易日 | 四个按日 pack 文件和资源日志 | 无 OOM；四文件齐全；随机读取通过；峰值 RSS 可接受 | 优化 lazy scan 或缩小 universe |
| D2 | 打包完整 2021-01 Top-400 | 月度 pack 审计 | 无 partial day；五个月投影不超过选定存储后端 | 切 GCS/本地训练 |
| B0 | 数据加载与 100 batch benchmark | 吞吐、显存、I/O 等待比例 | A100 单 seed 预计可在可接受会话内完成 | 调 seq_len、batch 或缓存 |
| T0 | 第一个 H5 rolling fold，seed 0 | checkpoint、IC@3D/5D、月度与非重叠摘要 | H5 IC 为正且不由单月/单 phase 驱动 | 停止扩 seed |
| T1 | 多 fold、三 seed | walk-forward 聚合与最差窗口 | 多数 OOS 月为正；最差窗口和成本可接受 | 回退模型/数据 |
| L0 | 冻结后一次性评估 2025 | locked-test artifact | 只报告，不按结果改本轮模型 | 新研究系列重新开始 |

## 当前执行命令

生成可指纹化的滚动计划：

```bash
python -m ticknet.nextday.rolling \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --target-horizon 5 \
  --output /home/richard/code/.artifacts/deep-learning-tick-data-prediction/h5-rolling/rolling-plan.json
```

生成一个月 raw 容量审计和逐日 Top-400 universe：

```bash
python -m ticknet.eventstream.pilot \
  --month 2021-01 \
  --feature-manifest data/nextday-raw-200/manifest.json \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1 \
  --universe-output /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-202101-top400/universe.json \
  --output /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-202101-top400/pilot-audit.json
```

D0 通过后只打包首日：

```bash
python -m ticknet.eventstream.pack \
  --days 20210104 \
  --universe /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-202101-top400/universe.json \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1
```

首日完成后重新运行 pilot 审计。D1 的报告先检查实际 pack bytes、ticker 数、三流事件数、处理时间
和峰值 RSS，再决定是否启动完整月度 systemd job。完整月度任务必须支持断点续跑，不覆盖其他
pack 版本。多日 pack 默认让每个交易日运行在独立子进程中，防止 Polars 与 Python allocator
保留的常驻内存跨日累积；已完成日期由四文件完整性检查自动跳过。

## 完成定义

- rolling plan、股票池和 pack 都绑定源数据指纹或确定性文件集合。
- 训练只用 validation 选 checkpoint，月度 OOS 不反向影响同一研究系列。
- IC@5D 同时报告逐日、月度、Newey-West 和非重叠五日结果。
- 云端不保存原始三流，200GB 容量判断使用实测 pack，不使用 raw 体积猜测。
- 真实数据、股票列表、逐日 pack、checkpoint 和预测不提交 Git；仓库只提交代码、配置和聚合审计。

## 2026-08-11 执行进度

R0、D0、D1 和 D2 已完成。2021-01 共 20 个交易日，逐日 Top-400 实际范围为 364 至 392 只股票，pack 为 31,129,493,499 bytes，包含约 9.72 亿条事件。没有缺失或 partial day。五个月 packed 投影为 155,647,467,495 bytes，超过 150GB 软门槛，因此原始 Parquet 继续留在 Linux，完整 fold pack 优先使用本地或 GCS。聚合审计见 `docs/reports/eventstream-top400-pilot-202101.json`。

首个 fold 的 H5 长表已由来源 manifest 指纹绑定生成。B0 前使用下面的命令生成带边界 purge 的 H3/H5 宽表：

```bash
ticknet-eventstream-prepare-horizon-labels \
  --sidecar /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-fold0/horizon-sidecar/horizon-labels.json \
  --feature-manifest data/nextday-raw-200/manifest.json \
  --output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-fold0/fold-labels \
  --horizons 3 5 \
  --train-start 2021-01-01 --train-end 2021-03-31 \
  --val-start 2021-04-01 --val-end 2021-04-30 \
  --test-start 2021-05-01 --test-end 2021-05-31
```

真实标签转换已通过。H5 在 train、validation 和 OOS 分别保留 20,462、6,213 和 5,013 条，边界 purge 分别清除 1,947、1,939 和 1,941 条。H3 分别保留 21,239、6,996 和 5,787 条。训练按 H5 Rank IC 选择 checkpoint，H3 只写入每轮监控与最终结果，不参与选择。

2月至5月 pack 与标签审计通过后，先运行100个 batch：

```bash
ticknet-eventstream-benchmark \
  --config configs/eventstream-h5-fold0-capacity100m.yaml \
  --output /home/richard/code/.artifacts/deep-learning-tick-data-prediction/eventstream-h5-fold0/benchmarks/a100.json \
  --batches 100 --warmup-batches 5 \
  --requested-gpu A100 \
  --expected-parameter-count 100604180
```
