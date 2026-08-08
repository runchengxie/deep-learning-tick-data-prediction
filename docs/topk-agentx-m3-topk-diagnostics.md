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
- `missing_holding_policy: error`。
- `target_return_contract: next_open_to_following_open`。
- buffer 网格包含 0，作为换手和收益变化的固定对照。

正式模式仍依赖上游实验如实登记标签契约。后续正式 prediction export 应把标签、股票池和交易
状态版本写入上游 spec 和数据指纹，不能仅依靠文件名推断。

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
  missing_holding_policy: error
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

1. 生成并登记动态 Top-400、open-to-following-open、带完整交易状态的 HGB predictions。
2. 使用同一数据指纹运行 `TRD-TOPK-400-001` 完整矩阵。
3. 对有希望的 buffer 区域运行滚动年份或月份稳健性检查，形成
   `TRD-BUFFER-400-001`。
4. 若正式矩阵仍没有 10bp 甜点区，进入 M4 的 HGB 与 LambdaMART 同口径比较，不进入高成本
   盘口预训练或神经排序损失。

当前本地 `results/registry.sqlite` 是旧 schema，包含重复 run、metric 和 review，不能由 v2
Registry 静默迁移。本次 smoke 使用独立 Registry。正式实验开始前应显式导入或重建干净的 v2
Registry，保留旧文件作为只读历史证据。
