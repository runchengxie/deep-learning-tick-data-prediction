# M1 Top-K long-only 评估内核

## 结论

`ticknet.research.portfolio` 现在是模型无关的组合评估入口。HGB、LambdaMART、TCN、
DeepLOB 和后续 AgentX executor 只要产出同一预测契约，就能共享 fixed-K 选股、排名缓冲、
换仓门槛、不可交易约束、成本与稳定性指标。

历史 `portfolio_quantile` 多空回测仍可运行，但返回的 `mode` 明确标记为
`legacy_quantile_long_short_diagnostic`。它不等同于新研究系列的 Top-K long-only 正式策略。

## 输入契约

预测 Parquet 必须包含：

| 字段 | 含义 |
|---|---|
| `symbol` | 股票代码；同一 `label_date` 内必须唯一 |
| `trading_date` | T 日信号日期 |
| `label_date` | T+1 开盘调仓日期 |
| `score` | 只使用 T 日及以前信息得到的横截面分数 |
| `target_return` | 正式口径为 T+1 open 到 T+2 open |

可选的 `can_buy` 与 `can_sell` 必须成对提供。缺省时 smoke 会假设可交易；正式结果必须使用
`--require-tradability --missing-holding-policy error`，否则与 M0 的交易状态契约不一致。

评估不会根据未来收益过滤候选。如果选中持仓的 `target_return` 缺失，会直接失败，避免用
收益是否存在进行隐含的前视筛选。

## 组合状态机

每日按如下顺序确定持仓：

1. 不可卖的旧持仓强制保留。
2. 排名未跌出 `top_k + exit_buffer` 的旧持仓保留。
3. 其余名额在旧持仓和可买的新股票之间比较；新分数至少高出 `min_score_gap` 才换仓。
4. 尽量恢复等权，但不可买/不可卖约束形成权重上下界，不会通过调权隐式成交。
5. 持有期结束后按个股实际收益漂移权重；下一日恢复等权产生的交易也计入换手与成本。

动态股票池里消失的旧持仓有两种策略：`liquidate` 明确记录 `universe_exit` 卖出，适合 smoke；
`error` 直接停止，适合要求完整交易状态的正式评估。

## 成本与指标

交易明细按目标权重差计算：

```text
buy_cost  = buy_notional  * per_side_bps
sell_cost = sell_notional * (per_side_bps + sell_stamp_tax_bps)
net_return = gross_return - buy_cost - sell_cost
```

初始建仓的买入名义本金为 1，明确计入成本。日度输出分别保存买入、卖出和单边平均换手，
所以 buffer 引起的每一次换手变化都可以回溯到股票级交易。

汇总包括：

- 毛/净日收益、年化收益、波动率、Sharpe、累计收益和最大回撤
- Top-K 实现收益重合度、相对全股票池收益、选中股票内部 Rank IC
- 月度累计收益与正收益日比例
- 极端 1/5/10 日的绝对收益贡献
- 持仓数、净/毛暴露、最大权重和 HHI 集中度

## CLI 与 artifacts

示例：

```bash
python scripts/evaluate_cost_adjusted.py \
  --predictions results/predictions.parquet \
  --top-k 50 \
  --exit-buffer 20 \
  --min-score-gap 0.05 \
  --cost-bps 10 \
  --stamp-tax-bps 5 \
  --require-tradability \
  --missing-holding-policy error \
  --output-dir results/topk-k50-buffer20-cost10
```

`--output-dir` 会生成：

- `summary.json`：组合、成本、排序、稳定性和风险汇总
- `daily.parquet`：每日换手、收益、成本、暴露和 Top-K 指标
- `holdings.parquet`：每日股票、排名、分数、权重、收益贡献和保留原因
- `trades.parquet`：每笔权重变化、买卖方向、原因、名义金额和成本

不传 `--top-k` 时保持历史分位数多空 CLI 兼容。新实验和 AgentX 不应使用该 legacy 模式作为
Top-K 成功门槛。

## 工程 smoke

使用冻结的 2025 HGB Top-100 预测明细，以 K=50、buffer=20、单边 10 bp 和卖出印花税
5 bp 跑通了 125 日链路，生成 125 行日度、6,250 行持仓和 7,958 行交易明细，日均单边换手
约 28.6%。该文件的 `target_return` 是历史 next-open-to-same-close 标签且没有交易状态列，
所以这里只验证实现和 artifact，不作为 M0 新 open-to-open 交易契约的收益结论。
