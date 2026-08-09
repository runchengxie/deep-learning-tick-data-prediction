# M3 Top-K、buffer 与成本矩阵诊断

## 当前结论

M3 的工程链路已经可以把一份预测明细展开为完整的 K、buffer、成本笛卡尔积，并生成一个
确定性的甜点区判断。诊断同时检查成本后相对股票池等权收益、净 Sharpe、正超额收益月份、
超额收益极端日集中度和 buffer 的净收益效率。所有组合必须使用相同的评估日期，否则整个矩阵
失败。

2025 HGB Top-100 历史预测已完成 64 组 smoke。10bp 单边成本和 5bp 卖出印花税下，没有组合
在扣除交易成本后跑赢当日预测股票池等权收益，结果为 `NO_TRADEABLE_REGION`。buffer 明显降低
换手并改善净收益，但当前样本中最好的相对等权收益盈亏平衡单边成本只有约 6.55bp，仍低于
10bp 决策成本。

这不是正式交易结论。源预测使用 next-open-to-same-close 历史标签，没有 `can_buy`、`can_sell`
字段，也没有绑定当前 Registry 的数据指纹。125 个日期中只有 91 个日期达到 100 只股票，无法
替代动态 Top-400、open-to-following-open 和完整交易状态的正式实验。

## 诊断契约

`topk_cost_sweep` 支持两种互斥来源：

- `predictions_path`：只用于开发 smoke。输出记录源文件 SHA-256，但没有 Registry 数据指纹。
- `source_experiment_id`：从 Registry 选择唯一 seed 和 artifact，读取前校验状态、路径、SHA-256
  和锁定日期边界，再把文件物化到本次独立 seed 目录。源实验的数据指纹会传递到 M3 实验。

`evaluation_mode: formal` 还会强制以下条件：

- 使用 `source_experiment_id`，不能使用任意文件路径。
- `require_tradability: true`。
- `require_universe_membership: true`，并按 `expected_universe_size` 校验每日候选数。
- `missing_holding_policy: error`。
- `target_return_contract: next_open_to_following_open`。
- buffer 网格包含 0，作为换手和收益变化的固定对照。

正式模式不再只相信上游 spec 的文字声明。`import_predictions` 会先校验 Parquet 内容与 metadata，
通过后才物化为 Registry prediction artifact；`topk_cost_sweep` 消费时会用 Registry
`dataset_fingerprint` 再校验一次。metadata 必须包含：

- `ticknet.dataset_fingerprint`
- `ticknet.target_return_contract=next_open_to_following_open`
- `ticknet.universe_contract=lagged_turnover_top_n`
- `ticknet.tradability_contract=next_open_suspension_one_price_limit`
- `ticknet.suspended_mark_policy=previous_close`

内容必须包含 `symbol`、`trading_date`、`label_date`、`return_end_date`、`target_return`、
`score`、`can_buy`、`can_sell` 和 `in_universe`。`return_end_date` 必须晚于 `label_date`，且
每个 `label_date` 只能对应一个收益结束日。每个 `label_date` 恰有 `expected_universe_size` 个
`in_universe=true` 候选；调出股票可用 `in_universe=false` 状态行表达次日可卖状态。Audit、排名和
股票池基准忽略状态行，组合状态机仍用它处理旧持仓退出或停牌强制持有。

正式预测首次登记使用：

```yaml
experiment_type: prediction_export
executor: import_predictions
inputs:
  predictions_path: results/predictions-hgb-top400-open2open.parquet
  evaluation_mode: formal
  target_return_contract: next_open_to_following_open
  expected_universe_size: 400
```

## 正式输入生成状态

`configs/nextday-minute-formal-2025.yaml` 已把正式 HGB 输入固定为：2021–2024 训练、2025
上半年验证、2025 下半年输出；每日股票池只使用信号日以前 20 个交易日成交额，严格保留 400
只候选。模型监督目标是个股 T+1 open 到 T+2 open 减同期基准收益，prediction 中供组合核算的
`target_return` 则保存未减基准的个股持有收益。停牌以此前最近有效收盘价估值，一字涨停不可买、
一字跌停不可卖。缺少分钟窗口的候选不从股票池删除，而是保留全 NaN 特征交给 HGB 的缺失值
分支，并写出 `feature_available=false`。

先把约 110 GB 的源缓存按月物化为可恢复的聚合特征：

```bash
uv run python scripts/materialize_minute_features.py \
  --config configs/nextday-minute-formal-2025.yaml \
  --output results/m3-formal-minute-features-v1
```

每个月只有在 Parquet 原子落盘并完成 SHA-256 后才会进入 manifest。重复执行相同命令会校验并
跳过已有月份。需要单月诊断时可重复传入 `--period YYYY-MM`。manifest 绑定全部 484,000 个
目标股票日、30 个原始分钟特征的列顺序、窗口参数以及 15 个年度三模态源文件的大小和修改时间。
任何身份变化都会停止续跑。正式数据指纹还会对加载后的 120 维聚合特征、标签和交易状态逐值
哈希。

全部 60 个月完成后运行 HGB：

```bash
uv run python scripts/run_minute_baseline.py \
  --config configs/nextday-minute-formal-2025.yaml \
  --materialized-features results/m3-formal-minute-features-v1 \
  --evaluate-test \
  --save-predictions results/predictions-hgb-top400-open2open-2025.parquet \
  --output results/nextday-minute-formal-2025.json
```

本地真实数据证据：

- 2021-01-04 至 2025-12-29 共 1,210 个完整信号日，每日均为 400 个候选，合计 484,000
  个候选标签；没有不完整股票池或缺失市场状态日期。
- 另生成 13,329 条调出股票状态行；候选及状态中记录到 3,282 条停牌、264 条一字涨停和
  209 条一字跌停状态。
- 2025-07-01 的真实 L2 单日抽取请求 400 个候选，399 个有完整三模态分钟行，1 个走全 NaN
  缺失特征路径；读取 6 个相关 row group，按日期元数据跳过 835 个无关 row group。
- 按月物化已真实完成 2025-07 的 9,200 个候选，9,193 个有特征，7 个保留为全 NaN；读取
  83 个相关 row group、跳过 758 个，耗时 100.3 秒，峰值内存约 2.26 GB，压缩分片约 4.2 MB。
  同月重跑通过 manifest 与分片 SHA-256 校验后直接跳过。
- 当前 manifest 状态为 `in_progress`，完成 1/60 个月。正式 HGB 已实测拒绝该残缺 manifest，
  并列出缺失的 59 个月，不会用部分数据训练。
- 所有日线面板显式截断到 2025-12-31；prediction 契约新增 `return_end_date`，防止 2025
  样本借用 2026 locked 收益。

这些结果证明数据口径、资源边界、恢复和缺失路径可执行，不构成模型效果或可交易性结论。正式
结论仍需完成其余 59 个月、prediction 登记和成本矩阵。

## 甜点区判定

默认在单边 10bp 成本下逐个评估 K 和 buffer。候选必须同时满足：

1. 至少 60 个完全可比评估日。
2. 净 Sharpe 大于 0。
3. 扣除本策略交易成本后，日均收益高于未扣成本的当日股票池等权收益。
4. 至少一半月份的成本后日均超额收益为正。
5. 毛超额收益绝对值最大的 5 天贡献不超过全部交易日绝对毛超额收益的 50%。
6. 非零 buffer 相对 buffer=0 降低换手，且成本节省足以覆盖毛收益变化。

第 3 条使用未扣成本的股票池等权收益，属于偏严格门槛。未来若实现可实际交易的无信号组合，
应把它以同一交易状态和成本模型作为独立基线，同时保留当前门槛，避免弱 alpha 被基准换手掩盖。

每个 K、buffer 还输出两种解析成本门槛：

- `absolute_return_breakeven_per_side_bps`：组合绝对净收益归零时的单边成本。
- `active_return_breakeven_per_side_bps`：相对当日股票池等权收益归零时的单边成本。

第二个指标用于判断预测信号是否覆盖交易成本。第一个指标会受到市场整体涨跌影响，不能单独
解释为模型 alpha。

## Artifacts

每次运行生成：

- `source-predictions.parquet`：经过 SHA-256 校验后物化的本次输入。
- `topk-sweep.json`：全部组合的 M1 summary 和 M3 诊断。
- `m3-diagnostic.json`：精简的矩阵、阈值、候选排序、成本门槛、来源身份和最终状态。
- `topk/k*.buffer*.cost*/`：每个组合的 summary、daily、holdings 和 trades。

正式 spec 的 `inputs` 片段：

```yaml
experiment_type: cost_analysis
executor: topk_cost_sweep
inputs:
  source_experiment_id: PRED-HGB-400-OPEN2OPEN-001
  source_seed: 0
  artifact_name: predictions
  evaluation_mode: formal
  target_return_contract: next_open_to_following_open
  top_k: [25, 50, 75, 100]
  exit_buffer: [0, 10, 25, 50]
  cost_bps: [5, 10, 15, 20]
  decision_cost_bps: 10
  sell_stamp_tax_bps: 5
  min_symbols_per_day: 400
  require_tradability: true
  require_universe_membership: true
  missing_holding_policy: error
  expected_universe_size: 400
```

完整运行仍通过统一入口：

```bash
ticknet-research \
  --registry results/registry.sqlite \
  --artifacts research/experiments \
  run --spec path/to/m3-formal.yaml --id TRD-TOPK-400-001
```

## 2025 工程 smoke 证据

输入文件：

```text
results/predictions-rolling-2025.parquet
SHA-256 ee2f8c2c4dd1be56c45138f8e6ca5de48539a8a899c5f9ea3dbcc4c15867eeba
```

输出位置：

```text
results/m3-topk-smoke-2025/TRD-TOPK-SMOKE-001/
```

关键结果：

| 项目 | 结果 |
|---|---:|
| 完整组合数 | 64 |
| 可比日期 | 91 |
| 日期范围 | 2025-07-02 至 2025-12-31 |
| 10bp 甜点区数量 | 0 |
| K25、buffer50 日均单边换手 | 25.42% |
| K25、buffer50 成本后相对等权日均收益 | -1.75bp |
| K25、buffer50 相对收益盈亏平衡单边成本 | 6.55bp |
| K25、buffer0 日均单边换手 | 69.22% |
| K25、buffer50 相对 buffer0 的日均净收益改善 | 10.57bp |

工程 gate `diagnostic.grid.validated_combinations >= 64` 通过，Runner 决策为 `EXTEND`。这里的
`EXTEND` 只表示诊断链路完整，不表示交易策略通过。交易判断由
`diagnostic.decision.status=NO_TRADEABLE_REGION` 给出。

## 下一步

M3 正式结论仍需：

1. 从当前 manifest 继续物化其余 59 个月，再运行正式 HGB 生成动态 Top-400、
   open-to-following-open、带完整交易状态和 metadata 的 predictions，通过
   `import_predictions` 登记。
2. 使用同一数据指纹运行 `TRD-TOPK-400-001` 完整矩阵。
3. 对有希望的 buffer 区域运行滚动年份或月份稳健性检查，形成
   `TRD-BUFFER-400-001`。
4. 若正式矩阵仍没有 10bp 甜点区，进入 M4 的 HGB 与 LambdaMART 同口径比较，不进入高成本
   盘口预训练或神经排序损失。

当前本地 `results/registry.sqlite` 是旧 schema，包含重复 run、metric 和 review，不能由 v2
Registry 静默迁移。本次 smoke 使用独立 Registry。正式实验开始前应显式导入或重建干净的 v2
Registry，保留旧文件作为只读历史证据。

现有 `results/predictions-rolling-2025.parquet` 已用正式 validator 实测拒绝：缺少 `can_buy`、
`can_sell` 和 `in_universe`。这是一条确定性失败边界，不会用默认可交易状态伪装成正式输入。
