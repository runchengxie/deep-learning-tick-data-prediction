# 多周期标签与数据扩容路线图

## 当前决策

截至 2026-08-10，五年 Top-400 raw-200 工作集已经生成，共 470,815 个股票日样本，目录约 7.2 GiB。1,033,383 参数模型的 seed 0、1、2 已完成 2024 验证期训练，最佳日均 Rank IC 分别为 0.02145、0.02054 和 0.01893，平均 0.02031，seed 间样本标准差 0.00127。2025 测试期继续锁定。

Google Drive 已升级为 200GB，当前容量足以保存 raw-200、多周期标签、raw-1000 和实验产物。Stage B 已完成。三个一日模型在 2024 validation 上的 H=5 平均 IC 为 0.07750，ensemble IC 为 0.08264，Newey-West t 值为 3.11，正 IC 月份占 83.3%，五组非重叠抽样最低 IC 为 0.07324，门槛通过。独立 H=5 seed 0 没有显示相对原一日模型的稳定增量，因此主目标继续使用 H=1，H=3 和 5 作为监控指标。客户要求的 100M 模型进入受控容量试验，先生成 raw-1000 Top-100 单月 preflight，再用精确 100,817,575 参数配置分别跑 T4 和 A100 的 100-batch benchmark。这一步不训练完整模型，也不访问 2025 test。

2026-08-11 benchmark 已完成。T4 为 21.13 samples/s，A100 为 80.23 samples/s，A100 加速 3.80 倍，峰值 reserved 显存分别为 2.40 和 2.35 GiB。按 75,000 train 样本、30 epochs 暂估，T4 为 29.58 小时每 seed，A100 为 7.79 小时每 seed。容量门槛通过，正式 100M 训练选择 A100，五年 raw-1000 Top-100 pilot 已进入生成阶段。

## 不可变研究合同

多周期收益固定为：

```text
信号：T 日 14:55 前最后 N 个 snapshot
进入：T+1 交易日开盘
退出：T+H 交易日收盘，H ∈ {1, 3, 5}
目标：个股收益减去中证全指同进入、退出时点收益
分类：按 T 日可用样本横截面的 20% / 80% 分位点
```

标签侧车保存 `entry_date` 和 `return_end_date`。样本只有在 `trading_date`、`entry_date` 和 `return_end_date` 全部位于同一 train、val 或 test 区间时才能进入该切分。H=1 直接复用现有 manifest 的标签和收益，作为向后兼容控制组。H=3 和 5 从同一日线与基准价格生成。

每份侧车绑定原特征 `dataset_fingerprint`。修改收益合同、分位点、股票样本集合或源数据时必须生成新目录，不能覆盖旧标签。特征 NPY 分片保持不变，因此这一阶段不会复制 7.2 GiB 工作集。

## 执行阶段与门槛

| 阶段 | 执行动作 | 通过条件 | 失败后的动作 |
|---|---|---|---|
| A. 标签侧车 | 生成 H=1/3/5 Parquet 和版本化 JSON 合同 | H=1 与旧 manifest 精确一致，H=3/5 日期与收益有限，边界 purge 测试通过 | 修复标签，不训练 |
| B. 固定模型评估 | 用三个现有 best checkpoint 只评估 2024 val 的 IC@1D/3D/5D | 三 seed 均值为正，多数月份方向一致，非重叠 5 日抽样不反转 | 停止多周期模型 |
| C. 独立 H=5 模型 | 保持 raw-200 和 1M 结构，只把目标改为 H=5 | 相对一日模型直接评估 H=5 有稳定增量，成本后 Top-K 改善 | 保留一日模型 |
| D. raw-500 | 先做 Top-100，再做 Top-400 | 同标签、同模型下多 seed 验证 IC 稳定高于 raw-200 | 停止扩大 tick 窗口 |
| E. raw-1000 | 生成约 35 至 40 GiB 的五年 Top-400 工作集 | 增量覆盖多数月份，且训练时间和成本可接受 | 回退 raw-200 或 raw-500 |
| F. 全天 Tick 试制 | 只生成一个月，测体积、吞吐和随机读取 | 预计五个月峰值不超过存储门槛，100 batch 基准可完成单 seed | 改用分段或 embedding，不直接全量 |

客户容量试验使用 2×2 归因矩阵。`1M/raw-200` 是现有控制组，`100M/raw-200` 只改变容量，`1M/raw-1000` 只改变窗口，`100M/raw-1000` 才是候选交付模型。本轮只完成 raw-1000 Top-100 数据 pilot 和 100M 资源 benchmark，不把两项变化直接解释为预测增益。

阶段 B 只访问 2024 验证集。horizon、收益处理、模型结构和 seed 列表冻结后，才允许一次性评估 2025 测试集。5 日标签高度重叠，报告同时给出逐日结果、每 5 个交易日非重叠抽样、月度结果和 Newey-West 或分块 bootstrap 不确定性。

## 近期执行命令

代码合并并同步到远程主机后，在远程项目目录运行：

```bash
.venv/bin/ticknet-nextday-prepare-horizon-labels \
  --manifest data/nextday-raw-200/manifest.json \
  --basic-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2/basic \
  --benchmark-path /mnt/data/hdd6t/quant-data-lake/reference/cn_market_reference/csi_all_a_000985.CSI.parquet \
  --output-dir data/nextday-raw-200-targets-v1 \
  --horizons 1 3 5 \
  --min-cross-section 100
```

产物预计远小于 1 GiB。审计通过后上传 Drive，但不重新上传 NPY 特征：

```bash
rclone copy data/nextday-raw-200-targets-v1 \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-200-targets-v1 \
  --checksum --transfers 4 --checkers 8 --progress
```

训练配置选择侧车时增加：

```yaml
target_sidecar_path: ./data/nextday-raw-200-targets-v1/horizon-labels.json
target_horizon: 5
```

阶段 B 不会用 H=3/5 重新训练。它加载现有一日 checkpoint 的分数，只替换验证目标来画 IC 衰减曲线。阶段 C 才使用上面的训练配置。

阶段 B 已通过 CLI 完成，notebook 只作为历史快照保留。Stage C 的 seed 0 入口为：

```bash
python scripts/run_colab_nextday.py \
  --workflow h5-train \
  --seeds 0 \
  --keep-on-failure \
  --session ticknet-h5-seed0 \
  --gpu T4 \
  --local-output-dir artifacts/raw-200-capacity_1m-h5/seed0
```

使用 `configs/nextday-raw-200-capacity-1m-h5.yaml`，只改变 target horizon 和独立产物目录。seed 0 只有在 2024 validation 的 H=5 IC、月度稳定性和成本后 Top-K 相对一日模型有增量时，才补 seed 1 和 2。2025 test 保持锁定。

## 200GB 到 400GB 的容量门槛

Drive 峰值按以下口径估算：

```text
现有正式数据 + 新工作集 + 上传/校验临时副本 + checkpoint/结果 + 25% 安全余量
```

在 200GB 套餐下，稳定占用控制在 120GB 以内，计划峰值控制在 150GB 以内。满足任一条件时，在生成或上传完整数据之前升级到 400GB：

- 计划峰值超过 150GB
- 一个月全天 Tick 试制超过 20GB，推算五个月超过 100GB
- 需要同时保留两份 60GB 以上的正式工作集
- Drive 其他个人数据使项目可用空间低于 50GB

raw-1000 预计 35 至 40 GiB，加上当前 raw-200、标签和 checkpoint，200GB 仍有充分余量。400GB 的实际决策点位于一个月全天 Tick 试制之后、五个月全量生成之前。

## Colab Pro+ 门槛

资源判断使用单个 seed 的连续运行时间，不使用所有 seed 的合计时间：

| 单 seed 100 batch 外推 | 资源决策 |
|---:|---|
| 少于 8 小时 | 当前 Colab 方案，逐 seed 运行并保存 checkpoint |
| 8 至 12 小时 | 优先 A100 或按量 compute units，Pro+ 可选 |
| 12 至 20 小时 | Pro+ 有价值，但仍要求断点续训和独立产物 |
| 超过 20 小时 | 不依赖单个 Colab 会话，改用专用云 GPU 或本地 5090 |

当前 1M raw-200 实测约 51 至 68 分钟每 seed，不需要 Pro+。raw-1000 先跑 100 batch 基准，只有外推单 seed 超过 8 小时才重新评估订阅。Pro+ 不作为获得 A100 的保证条件。

## raw-1000 Top-100 与 100M benchmark

先生成不覆盖任何旧数据的单月 preflight：

```bash
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw-1000-preflight.yaml
```

源 snapshot 约每 3 秒一条。raw-200 的 14:30 起点在 14:55 前通常只有 500 条，因此 raw-1000 固定从 13:30 开始扫描，再严格保留信号时点前最后 1000 个有效事件。扫描起点只是数据提取窗口，更早的多余事件不会进入样本。

审计 `manifest.json`、`data-audit.json`、分片大小和校验和后，把它同步到：

```text
gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-1000-preflight-202101-top100
```

然后用 `scripts/run_colab_nextday.py --workflow capacity-benchmark` 分别申请 T4 和 A100。两次运行固定 5 个 warmup batch 和 100 个 measured batch，默认 ephemeral，完成后自动关闭 runtime。比较 `capacity-benchmark.json` 的真实 GPU 名称、samples/s、peak reserved GiB 和按 75,000 个训练样本外推的单 seed 小时数。若单 seed 小于 8 小时且显存留有至少 20% 余量，再生成五年 `configs/nextday-raw-1000-top100.yaml`，否则先缩小有效 batch 或模型宽度。

## 交付清单

- `horizon-labels.json`：合同、源特征指纹、horizon、行数和 Parquet SHA-256
- `labels.parquet`：股票、信号日、进入日、退出日、H、收益和标签
- 三 seed 的 IC@1D/3D/5D 总表、月表和非重叠抽样结果
- raw-500/raw-1000 每级的数据审计、100 batch 基准和继续或停止决定
- 一个月全天 Tick 的实际体积与五个月峰值估算

所有真实行情、标签 Parquet、checkpoint 和 notebook HTML 保留在实验产物目录或 Drive，不提交到 Git。仓库只提交代码、聚合报告、配置和不含真实逐股票记录的审计摘要。
