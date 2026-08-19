# Top-K 可交易预测与 AgentX 研究路线图

## 目标

本路线把模型研究和 AgentX 研究系统合并为一条可依赖、可验收的执行链。最终要回答：

1. 盘中微观结构信息能否稳定改善 Top 50 至 100 股票的成本后收益。
2. 短周期盘口表征能否作为隐藏特征，改善次日及多日横截面预测。
3. AgentX 闭环能否自动提出、执行、审计并积累可信的量化研究结论。

路线按最低成本的可证伪实验推进。一个阶段未通过门槛时，不进入依赖它的高成本阶段。历史实验文档继续保存已经发生的事实，本文件作为后续工作的执行清单和状态入口。

## 当前证据

截至 2026-08-19，仓库已有以下真实结果：

- 分钟聚合特征 HGB 在 2022 至 2025 四个历史滚动样本外年份的 Rank IC 约为 0.022 至 0.035，信号弱但跨年为正。2021 年上半年委托源缺少沪市记录，这组结果带有早期市场覆盖偏差。
- 2025 H2 的日频 top/bottom 10% 组合换手约 83%。单边 10 bp 成本下净收益为负，盈亏平衡成本约为 5 至 6 bp。
- 组合毛利集中在少数极端交易日，前 5 日贡献超过全部收益。
- 分钟 TCN 的验证集优势没有稳定泛化到测试集，聚合特征 HGB 更稳健。
- 固定 Top-100 样本的原始盘口四格三 seed 矩阵已经完成。`1M/raw-200` 的验证 Rank IC 为 `0.03748 ± 0.00096`，表现最好且波动最小。扩大到 100M 参数或 raw-1000 都没有形成稳定增益。
- L2 事件流 100M 最近折三 seed 已完成。H5 validation Rank IC 为 0.04345、0.09403、0.08029，均值为 0.07259。2025 年 12 月 OOS 为 0.05879、0.03730、0.03291，均值为 0.04300。三个 seed 在两段的方向全部为正，100M 信号门槛已经通过。
- 相邻滚动折 `fold-54-oos-202511` 的 seed 0 已完成。H5 validation 与 OOS Rank IC 分别为 0.08735 和 0.03305，H3 监控分别为 0.04840 和 0.04231。H5 OOS 极端组收益差为 -0.00512，跨窗口排序方向为正，头部组合仍需调整。
- 冻结表征最近折已经完成。HGB E2 三 seed 的 OOS Rank IC 为 0.04333、0.05644、0.05912，均高于 E0 的 0.04010。三 seed 预测均值为 0.05701，逐日配对增量为 0.01691，bootstrap 95% 区间为 0.00596 至 0.02851。Top-100 日均成本后主动收益从 -11.51bp 改善到 -4.80bp，仍未转正。LambdaMART 的增量缺少跨 seed 和跨月稳定性。
- 联合端到端三 seed 已完成。validation Rank IC 为 `0.05917 ± 0.01400`，2025 年 12 月 OOS Rank IC 为 `0.06398 ± 0.00785`。OOS `NDCG@100` 为 `0.54507 ± 0.00450`，`Precision@100` 为 `0.23921 ± 0.01611`，Top-100 日均成本后主动收益为 `-9.67 ± 3.71bp`，日均单边换手为 `49.74% ± 5.79%`。
- 事件流半衰期与交易转换诊断已完成。两个窗口到 H10 都没有出现清晰 IC 衰减。EMA 加 5bp 换仓收益差把 11 月和 12 月的 H1 单边换手分别降到 13.90% 和 8.90%，成本后主动收益仍为 -13.02bp 和 -10.99bp。H5 五组错峰在两个月的成本后主动收益为 -15.48bp 和 17.12bp，方向没有跨窗口重复。
- AgentX 的 M0 至 M2 已完成，已经具备受控执行器、预测导入导出、多 seed 对比、walk-forward 汇总、Registry 上下文和一次性锁定测试批准。
- M3 v1 在完成 14/60 个月后停止。M3 v2 已完成 54 个月、436,800 个候选的物化，完整特征覆盖率为 99.88%。HGB 的 2025 下半年 Rank IC 为 0.06994，六个月月度 IC 均为正。正式 64 组 Top-K 矩阵结果为 `NO_TRADEABLE_REGION`，所有组合在单边 10bp 下的成本后主动收益均为负。

M3 已形成正式停止结论。M4 与 M5 的最近折同口径增量实验已经完成，HGB 组合输入和联合训练都得到正 Rank IC。相邻滚动折 seed 0 继续得到正 Rank IC。半衰期、错峰持有、交易门槛和已知风格暴露检查已经完成，跨窗口成本后收益门槛没有通过。下一阶段进入任务梯度、标签尺度和监督位置消融。原始盘口扩容、相邻折 seed 1、2 和 150M 事件流容量消融继续暂停。

## 外部对比带来的验证队列

[外部 L2 研究项目对比](external-l2-research-comparison.md) 已把外部稿件中的判断分成仓库事实、外部项目自述、机制假设和研究决策。当前采纳的顺序如下：

1. [x] `EVT-HALFLIFE-001` 计算最近折和相邻滚动折的 `IC(1)` 至 `IC(10)` 与按建仓日分组的组合收益
2. [x] `TRD-STAGGERED-H5-001` 比较每日全量换仓和 H5 五组错峰持有
3. [x] `TRD-RANK-EMA-001` 比较排名 EMA、绝对开仓门槛、换仓收益差门槛和现金仓位
4. [x] `RISK-ATTR-001` 完成规模、流动性和波动率归因。行业归因等待本地分类数据
5. [ ] `EVT-GRAD-AUDIT-001` 记录各任务对共享主干的梯度范数和夹角
6. [ ] `EVT-LABEL-SCALE-001` 比较原始连续收益与每日截面去极值 z 标签
7. [ ] `EVT-SUPERVISION-POSITION-001` 比较全部位置、最后位置和尾部加权监督

前四项已复用已有预测完成，正式结论见[事件流信号半衰期与交易转换诊断](eventstream-signal-trading-diagnostics.md)。后三项每次只改变一个训练机制。单 seed 需要同时改善 validation 和相邻 OOS，才补更多 seed 与滚动窗口。容量扩张排在这组验证之后。

## 统一研究原则

### 交易口径先于模型指标

主要目标是动态流动性股票池中的 long-only Top-K 组合。全横截面 Rank IC、Macro F1 和 long-short spread 保留为诊断指标，不能替代成本后组合结果。

正式报告至少包含：

- `NDCG@50`、`NDCG@100`
- `Precision@50`、`Precision@100`
- Top-K 平均超额收益和命中率
- 持仓换手率、交易成本、净收益和净 Sharpe
- 月度稳定性、极端日贡献和 winsorize 敏感性
- 行业、规模、波动率和流动性暴露

### 研究集、验证集和最终确认集分离

2025 数据已经被用于成本审计、模型比较和新假设形成。对本路线产生的新方法而言，2025 只能作为开发与滚动验证证据，不能继续充当未见过的最终 locked test。

M0 需要先审计 2026 数据可用性，再选择一个未被查看的 2026 区间作为新研究系列的最终确认集。若当前没有完整 2026 数据，则从未来第一个完整可交易区间开始前瞻模拟，期间不按结果修改模型。锁定区间在审计完成前保持 `TBD`，不得为了方便写死日期。

### 一次实验只改变一个主要机制

结构、标签、损失、数据窗口和组合规则应分开消融。每个实验必须有固定对照、主要指标、可证伪条件和停止条件。负面结果同样进入 Registry。

### 先冻结特征，再增加模型复杂度

新的盘口 embedding 先以冻结特征接入 HGB 或 LambdaMART。只有冻结 embedding 稳定提供增量后，才做联合微调、神经排序损失和多日深度模型。

## 路线总览

| 里程碑 | 方向 | 主要产物 | 状态 |
|---|---|---|---|
| M0 | 重置研究契约 | 新 locked 协议、Top-K 与成本口径、基线快照 | 已完成 |
| M1 | Top-K 评估内核 | fixed-K long-only、缓冲区、成本与稳定性指标 | 已完成 |
| M2 | AgentX 确定性闭环修复 | typed executor、artifact contract、完整 Registry | 已完成 |
| M3 | 无重训组合诊断 | Top-K、缓冲区和成本敏感性结论 | 已完成 |
| M4 | 横截面排序基线 | HGB 与 LambdaMART 同口径比较 | 进行中 |
| M5 | 事件流表征与冻结 embedding | 100M 多 seed、缓存 embedding、下游增量实验 | 进行中 |
| M6 | 神经 Top-K 损失 | DayBatchSampler、pairwise 或 LambdaRank 损失 | 待开始 |
| M7 | 多日与多期限模型 | 5/10/20 日输入、1 日与 5 日预测头 | 待开始 |
| M8 | AgentX 研究智能 | Context Builder、候选排序、Evaluation 回流 | 待开始 |
| M9 | Developer Agent 与 Harness Evolution | 受控代码修改、轨迹 replay、harness 准入 | 暂缓 |

状态只使用 `待开始`、`进行中`、`已完成`、`已停止`。进入一个里程碑时，应把该行改为 `进行中`。验收证据、实验 ID 和结论写入对应章节后，才能改为 `已完成`。

## M0：重置研究契约

### 目标

在新增评估代码和模型实验前，固定本研究系列的可交易问题、数据权限和对照基线。

### 工作项

- [x] 审计本地及远端是否具有完整 2026 分钟、snapshot、日线和交易状态数据。
- [x] 为新研究系列定义 `research_end`、`validation_end` 和 `locked_start`。
- [x] 将 2025 标记为已见开发证据，不再称为新方法的 locked test。
- [x] 固定信号时点、执行价格、收益区间和不可交易样本处理。
- [x] 固定股票池为历史动态流动性 Top 400，并保留 Top 100 作为快速冒烟口径。
- [x] 固定 long-only `K = 25 / 50 / 75 / 100` 和成本档位 `5 / 10 / 15 / 20 bp`。
- [x] 定义初始建仓、卖出印花税、冲击成本、停牌和涨跌停的处理规则。
- [x] 保存当前 HGB、TCN、raw-200 结果的基线快照与文件指纹。
- [x] 版本化 `ResearchProtocol`，让研究系列显式引用协议版本，不再依赖硬编码日期。

### 验收门槛

- 协议能明确回答某个日期和数据文件是否允许用于研究、验证或最终确认。
- 同一预测文件在相同协议下得到确定性一致的股票池、收益和成本结果。
- 新实验不能把 2025 结果描述为本路线的全新锁定测试结果。

### 产物

- 协议配置或决策记录
- 基线 artifact 清单及 SHA-256
- M1 使用的 Top-K 策略参数定义

### 完成记录（2026-08-08）

- 研究契约与数据审计：[topk-agentx-m0-research-contract.md](topk-agentx-m0-research-contract.md)
- 权限协议：`configs/research-protocol-topk-v1.yaml`
- 交易契约：`configs/topk-portfolio-v1.yaml`
- 基线清单：`../baselines/topk-agentx-v1.json`
- 2026 数据已经封存，但当前完整对齐仅有 73 个交易日，未达到正式确认所需的 120 日。
- 正式日频收益改为 next-open-to-following-open，使持仓换手与收益周期一致。

## M1：实现 Top-K long-only 评估内核

### 目标

建立一个不依赖具体模型的确定性评估内核，让 HGB、LambdaMART、TCN 和 DeepLOB 使用完全相同的组合与成本口径。

### 最小实现

- [x] 从预测 Parquet 读取 `symbol`、`trading_date`、`label_date`、`score` 和 `target_return`。
- [x] 支持固定 `top_k`，不再只支持分位数。
- [x] 支持 long-only 等权组合。
- [x] 支持排名缓冲区，例如目标 K 为 50，老持仓跌出 70 才卖出。
- [x] 支持最低换仓收益门槛或最小分数差。
- [x] 正确处理动态股票池成分变化和持仓股票次日缺失。
- [x] 输出逐日持仓、交易、换手、毛收益、成本和净收益明细。
- [x] 输出 Top-K 排序指标、月度稳定性、极端日贡献和风险暴露。
- [x] 把 `scripts/evaluate_cost_adjusted.py` 保持为薄 CLI，核心逻辑进入可复用模块。
- [x] 增加合成数据测试，覆盖初始建仓、缓冲区、动态成分、缺失收益和成本计算。

### 建议模块边界

```text
src/ticknet/research/portfolio.py
  PortfolioPolicy
  CostModel
  evaluate_topk_portfolio()

scripts/evaluate_cost_adjusted.py
  参数解析
  调用核心模块
  写出 JSON 与日度 Parquet
```

`ticknet.research` 不直接导入 `ticknet.nextday` 的训练实现，评估只消费稳定的预测文件契约。

### 验收门槛

- 同一个 Top-K 结果可以被 CLI、测试和 AgentX executor 复用。
- long-only 与现有 long-short 结果明确区分。
- 成本为零时净收益严格等于毛收益。
- 增加缓冲区后，换手变化能由逐日交易明细解释。

### 完成记录（2026-08-08）

- 核心模块：`src/ticknet/research/portfolio.py`
- 使用与 artifact 契约：[topk-agentx-m1-portfolio-evaluator.md](topk-agentx-m1-portfolio-evaluator.md)
- 历史分位数多空路径保留为 `legacy_quantile_long_short_diagnostic`。
- Top-K 路径支持 `can_buy`、`can_sell`、严格缺失持仓策略和交易后权重漂移。
- M2 可以把同一个 `evaluate_topk_portfolio()` 接入 typed `topk_cost_sweep` executor。

## M2：修复 AgentX 确定性研究闭环

### 目标

让每个 ExperimentSpec 触发语义正确的执行器，并产生不可混淆、可比较、可回放的证据。

### ExperimentSpec v2

已经增加以下结构化字段：

```text
objective
executor
inputs
config_overrides
primary_metrics
success_gates
falsification_condition
artifact_contract
budget
parent_id
novelty_signature
stage
```

`entry_point` 不再接受任意字符串，只能选择程序白名单中的 executor。

### Typed executors

- [x] `train_nextday`
- [x] `train_minute_tcn`
- [ ] `train_ranker`（推迟到 M4 选定排序库和固定命令）
- [x] `export_predictions`
- [x] `audit_predictions`
- [x] `topk_cost_sweep`
- [x] `walk_forward_robustness`
- [x] `compare_experiments`

`data_audit` 和 `cost_analysis` 必须调用各自的确定性实现，不能回退为普通模型训练。

### Artifact contract

每个实验和 seed 使用独立目录，至少保存：

- resolved spec 与 config
- git SHA 和工作区状态
- dataset fingerprint
- stdout、stderr、退出码和运行时长
- checkpoint、结果 JSON、预测 Parquet 及其 SHA-256
- Evaluation 结论和异常列表

不得继续复用不同实验之间相同的 checkpoint 名称和结果目录。

### Registry 修复

- [x] 递归登记 `validation.*`、`test.*`、`topk.*` 和 `cost.*` 数值指标。
- [x] 为 experiment、run、metric、review 和 artifact 增加唯一性或外键约束。
- [x] 在执行前登记 experiment，失败、超时和被拒实验也保留状态与原因。
- [x] 保存 result path 和 artifact checksum。
- [x] `compare` 能直接比较主要指标、对照差值和多 seed 波动。
- [x] 防止同一 experiment ID 重复执行产生重复 run 和 metric 行。

### Evaluation 与权限

- [x] 由确定性规则输出 `KEEP`、`EXTEND` 或 `DISCARD`。
- [x] 使用结构化 `success_gates` 表达可计算证伪门槛，不只保存自然语言。
- [x] Audit 异常自动写回 Registry。
- [x] Registry 中的 Audit 异常自动进入下一轮 ResearchContext。
- [x] locked approval 绑定 experiment spec、checkpoint、predictions、数据指纹和一次性批准记录。
- [x] CLI 不再提供默认等于有效批准值的 token。

### 验收门槛

构造三类端到端测试：

1. `cost_analysis` 确实产生成本报告，不启动训练。
2. 训练实验自动导出预测、执行 Audit、登记嵌套指标并给出 Evaluation 决策。
3. 任意入口、locked 数据、重复 ID 和 artifact 冲突均被确定性拒绝。

### 阶段记录：M2a（2026-08-08）

- 实现说明与剩余边界：[topk-agentx-m2a-deterministic-loop.md](topk-agentx-m2a-deterministic-loop.md)
- `ExperimentSpec.from_dict()` 严格拒绝未知字段和旧 `entry_point`，Runner 只分派白名单 executor。尚未实现的白名单入口会显式失败，不会回退成训练。
- 训练产出的预测明细也会再次检查协议日期，并强制执行 Audit，不能通过 artifact 绕过 locked 数据边界。
- Registry v2 保存递归指标、失败状态、review、artifact 路径、大小与流式 SHA-256。
- 本阶段仍不把 M2 标为完成。完整 compare 统计、ranker、export、walk-forward executor 和 Registry 到 ResearchContext 回流留给后续阶段。

### 阶段记录：M2b（2026-08-08）

- 使用与安全边界：[topk-agentx-m2b-locked-approval.md](topk-agentx-m2b-locked-approval.md)
- 新增 `approve-locked-test` 与 `locked-test` 两步流程。静态 `APPROVED` 字符串不再具有权限。
- 签发要求实验为 `stage=release`、状态 `completed`、Evaluation 为 `KEEP`，并具有数据指纹和已登记的 `best_checkpoint` artifact。
- 随机 bearer token 只显示一次，Registry 只保存 token SHA-256。批准绑定 spec、全部 seed 的 checkpoint bundle、locked predictions 和 dataset fingerprint。
- token 在审计前原子消费，成功、失败或重放都有确定性状态。预测或 checkpoint 改变会在消费前使批准失效。

### 阶段记录：M2c（2026-08-09）

- 使用与指标语义：[topk-agentx-m2c-executors-comparison.md](topk-agentx-m2c-executors-comparison.md)
- `export_predictions` 只物化 Registry 中 checksum 匹配的 prediction artifact。Runner 对导出文件重新执行日期协议与 Audit，不能借此读取 locked 数据。
- `compare_experiments` 输出 seed 分布、相对基线均值差和同 seed 配对差，并提供按 `higher` 和 `lower` 方向归一、正值恒为改善的字段。
- `walk_forward_robustness` 默认要求不同数据指纹的至少三个窗口，按指标方向报告跨窗口波动和最差窗口。
- M2c 完成时，Registry 到 ResearchContext 的回流仍待实现。该项随后在 M2d 完成。

### 阶段记录：M2d（2026-08-09）

- 使用与安全边界：[topk-agentx-m2d-registry-context.md](topk-agentx-m2d-registry-context.md)
- `ResearchContextBuilder` 从 Registry 选择 KEEP、EXTEND 基线，并汇总近期实验、parent、失败原因、Evaluation 决策、Audit 异常、指标、数据指纹和全历史 novelty signature。
- 同一 Registry 状态生成稳定 SHA-256。Orchestrator 将完整快照写入 `research_context` review，Brainstorm 与 Critic 消费同一份上下文。
- Brainstorm 和 Critic 双重阻止历史 novelty 重复。Critic 还检查允许动作、已实现 executor 和本轮算力预算。重复候选在 reserve 前失败，不污染 Registry。
- ResearchContext 明确不具有 locked-test 权限，也不从 dataset fingerprint 推断日期。
- M2 的确定性闭环验收完成。`train_ranker` 仍未实现，状态见 [topk-agentx-m2a-deterministic-loop.md](topk-agentx-m2a-deterministic-loop.md)。

## M3：无重训 Top-K 与成本诊断

### 目标

在不改变模型的情况下，判断组合宽度、持仓缓冲和成本门槛是否存在可交易甜点区。

### 实验矩阵

第一步用现有 Top 100 预测做冒烟，只验证代码和方向。正式结论使用动态 Top 400 预测。参数网格以 [m0 交易契约](topk-agentx-m0-research-contract.md) 为准，执行值见 [m3](topk-agentx-m3-topk-diagnostics.md) 的正式 spec。

### 当前进展

- [x] `topk_cost_sweep` 支持完整笛卡尔积、相同日期样本校验和确定性甜点区判断。
- [x] 支持 Registry prediction artifact 的状态、唯一性、SHA-256、锁定日期和数据指纹绑定。
- [x] 正式模式强制交易状态、严格缺失持仓策略和 open-to-following-open 标签声明。
- [x] 正式 Parquet metadata、每日候选数与数据指纹由 `import_predictions` 校验并登记。
- [x] `in_universe=false` 状态行支持动态股票池退出和不可卖旧持仓，不污染排名与 Audit。
- [x] `TRD-TOPK-SMOKE-001` 已运行 64 组冒烟，工程 gate 通过。
- [x] 正式 HGB 生成器已实现动态 Top-400、T+1 至 T+2 open 收益、交易状态、状态行、缺失特征保留、数据指纹和 `return_end_date` 防泄漏校验。真实日线全区间审计与单日 L2 冒烟已通过。
- [x] M3 v2 完成 54 个按月原子 Parquet、源身份和分片 SHA-256 校验、断点续跑及完整 manifest 强制门槛。436,800 个候选中有 436,256 个具备完整特征。
- [x] 动态 Top-400 正式 prediction 已生成并登记为 `PRED-HGB-400-OPEN2OPEN-001`。
- [x] `TRD-TOPK-400-001` 已完成 64 组正式矩阵并形成 M3 结论。没有候选通过 10bp 甜点区门槛，因此未启动依赖候选区域的 `TRD-BUFFER-400-001` 稳健性扩展。

正式矩阵在 10bp 下得到 `NO_TRADEABLE_REGION`。绝对收益最好的 `K=100、buffer=50` 仍有正净收益，成本后相对 Top-400 等权基准的日均主动收益为 -4.75bp。全部策略的主动收益盈亏平衡单边成本最高约为 4.33bp。完整契约、结果和限制见 [topk-agentx-m3-topk-diagnostics.md](topk-agentx-m3-topk-diagnostics.md)。

### 对照

- 当前 HGB 分数
- 全市场或股票池等权基准
- 纯价格动量、反转和流动性简单基线

### 验收与停止条件

进入 M4 的最低条件：

- 至少存在一个 Top-K 口径，在合理成本下明显优于无信号基准。
- 改善不能只来自一个月份或少数极端交易日。
- 缓冲区降低换手后，毛利下降速度没有持续快于成本下降速度。

若所有 K、buffer 和成本组合都没有改善区域，仍可进入 M4 检查排序目标是否错配，但不进入高成本盘口预训练。

### 计划实验 ID

- `TRD-TOPK-SMOKE-001`
- `TRD-TOPK-400-001`
- `TRD-BUFFER-400-001`

## M4：HGB 与 LambdaMART 横截面排序基线

### 目标

回答当前瓶颈是否来自点预测目标与 Top-K 交易目标错配。

### 最小实验

- [x] 保持相同分钟聚合特征、日期、股票交集和标签。
- [x] HGB 继续作为 pointwise 对照。
- [x] LambdaMART 按交易日提供 group。
- [x] 将未来收益转换为非负的横截面 relevance 等级，用于 NDCG。
- [x] 训练目标与评估目标均明确包含 `NDCG@50/100`。
- [x] 输出同一 schema 的预测 Parquet，并复用 M1 的 Top-K 成本评估。
- [ ] 使用滚动年份验证，记录每年、每月和不同市场状态下的差异。

最近折实现使用 LightGBM 的 `LGBMRanker`，依赖已经进入 `pyproject.toml` 和 `uv.lock`。配置位于 `configs/embedding-frozen-recent-2025.yaml`。第一轮固定为 2025 年 8 月至 10 月训练、11 月验证、12 月 OOS。它只能形成最近折证据，滚动年份需要补充更长区间的事件流缓存后再执行。

最近折结果显示，LambdaMART E0 的 validation Rank IC 为 -0.04334，OOS 为 0.00766。E1 三 seed 预测均值对应 -0.01117 和 0.03414，E2 对应 -0.05081 和 0.01389。E1 的 OOS Top-100 日均成本后主动收益为 15.53bp，validation 为 -15.34bp。排序目标在当前两个月没有形成稳定优势，M4 保持进行中，滚动年份验证仍是完成条件。

排序库在本里程碑开始时根据本地环境和可复现性选择。新增依赖必须进入 `pyproject.toml` 和 `uv.lock`，不能依赖未固定的系统安装。

### 验收门槛

- LambdaMART 在多个滚动窗口提高 Top-K 排序或成本后收益。
- 改善不能由预测范围、标签或股票池口径变化造成。
- 训练时间和内存适合重复多次运行。

若 LambdaMART 不优于 HGB，优先判断特征信息不足，不立即实现复杂神经排序损失。

### 计划实验 ID

- `MDL-RANK-SMOKE-001`
- `MDL-RANK-ROLLING-001`

## M5：事件流表征与冻结 embedding

### 目标

从 A 股逐笔事件中提取隐藏表征，并验证其相对分钟聚合特征能否改善 Top-K 交易结果。当前先完成 `capacity100m` 最近折，再把通过门槛的 checkpoint 作为冻结编码器接入 M4。

### 当前入口与执行顺序

最近折使用 `configs/eventstream-h5-recent-capacity100m.yaml`，训练期为 2025 年 8 月至 10 月，validation 为 2025 年 11 月，OOS 为 2025 年 12 月，2026 继续锁定。`capacity100m` 有 100,604,180 个参数，输入打包与 A100 吞吐基准已经完成。

截至 2026-08-19，本机没有可用 CUDA GPU。通过 `rclone about gdrive:` 核对，Google Drive 总额为 200 GiB，已用 145.292 GiB，剩余 53.305 GiB。完整五个月 pack 约为 313.11 GiB，正式训练改用每个 seed 约 25 GiB 的固定窗口缓存。最近折三个 seed 和相邻滚动折 seed 0 的缓存、checkpoint 与结果均已完成并通过指纹核对。

1. [x] 完成 T0 基础设施门槛、真实缓存物化、远端核对和 checkpoint 恢复测试。
2. [x] 完成 seed 0，H5 validation 每日 Rank IC 用于选择 checkpoint，H3 只作监控。
3. [x] seed 0 通过门槛后完成 seed 1 和 2。
4. [x] 三 seed 的 validation 与 OOS H5 平均 Rank IC 均为正，100M 信号门槛通过。
5. [x] 完成 seed 共用尾盘缓存的本地生成、全量核对、远端上传和远端逐文件核对。
6. [x] 使用固定 checkpoint 生成每日冻结 embedding，分别接入 M4 的 HGB 和 LambdaMART。
7. [x] 完成当前折三 seed 配对增量与预测组合检查，HGB E2 进入联合端到端小实验。
8. [x] 实现轻量联合特征缓存、100M 主干微调、同口径评估和 Colab 恢复工作流。
9. [x] 运行 `FEAT-EVENTSTREAM-JOINT-001` seed 0、1、2，并与 frozen E2 使用同一股票日和评估口径。
10. [x] 完成 `fold-54-oos-202511` seed 0，H5 与 H3 的 validation 和 OOS Rank IC 均为正。
11. [x] 完成半衰期、H5 错峰持有、排名平滑、开仓门槛和已知风险暴露检查。行业归因等待本地分类数据。
12. [ ] 完成任务梯度、标签尺度和监督位置的单变量消融。
13. [ ] 根据跨窗口和交易指标结果决定是否补 seed 1、2 以及启动 `probe150m`。

冻结表征输入使用一份 seed 共用的尾盘窗口缓存。每个股票日取收盘前最后 512 个事件，编码后保存最后一个有效事件的 960 维隐藏状态。缓存和 embedding manifest 记录源数据指纹、checkpoint SHA-256、训练缓存指纹、源码 revision、日期、股票、锚点和 schema。三个 checkpoint 分别训练下游模型，最后汇总指标或平均预测分数。向量本身不跨 seed 逐维平均。

事件流 pack 与 M3 动态 Top-400 分钟候选每天约有 368 至 397 个共同股票，整体覆盖约 96%。E0、E1、E2 固定使用同一股票日交集，保证增量对照公平。第一轮结论只适用于这个高覆盖子集，不能直接替代完整 Top-400 结论。

冻结 embedding 迁移门槛沿用本节的增量验收条件。`probe150m` 当前只是代码中的模型预设，正式实验开始前需要补齐配置、精确参数量测试、预算和停止条件。

`FEAT-EMB-FROZEN-001` 已完成 22,409 个训练样本、6,963 个 validation 样本和 8,125 个 OOS 样本的同口径比较，分钟候选覆盖率为 96.69%。HGB E2 三 seed 在 validation 的 Rank IC 为 0.02462、0.02988、0.02323，在 OOS 为 0.04333、0.05644、0.05912。三 seed 预测均值在 validation 为 0.02833，在 OOS 为 0.05701，对应 E0 为 0.01808 和 0.04010。

当前折通过表征增量的小实验门槛，并完成固定 100M 容量的联合端到端三 seed 对照。M5 的完整验收门槛仍需额外时间窗口、风险暴露和成本后主动收益共同支持。联合实验使用与 frozen E2 相同的股票日交集、日期、标签和评估方式。`probe150m` 保持等待。

联合实验缓存只保存分钟特征、目标、尾盘缓存相对分片和行号，避免复制事件数组。训练加载同 seed 的固定 checkpoint，先训练新增特征塔和分类头，再以较小学习率解冻事件流主干。正式结果输出 validation 与 OOS 的 Rank IC、NDCG、Precision、Top-K 收益和换手率，并记录缓存、checkpoint 和源码指纹。

`FEAT-EVENTSTREAM-JOINT-001` 三 seed 已完成。轻量缓存包含 37,497 个模型样本和 40,274 个组合目标，共 17,948,094 字节，数据指纹为 `e4f54a62e4be3f36ac0693db59ebcdb120cd753d2dc36415b8686adaa13c1bb6`。seed 0 使用源码 revision `da01954b22a1a1506c9e91f8558fcd80bf8184e8`，seed 1、2 使用 `92426f67060e7ebb24cb3400ada6aa8af38ae804`。三个 seed 的股票、日期、标签和行数完全一致，每个 seed 的本地与 Drive 7 个最终文件核对一致。2026 locked 数据没有进入训练或评估。

seed 0、1、2 的最佳 checkpoint 分别出现在第 2、1、1 个 epoch。validation Rank IC 为 `0.05917 ± 0.01400`，OOS Rank IC 为 `0.06398 ± 0.00785`。联合模型的 OOS `NDCG@100` 为 `0.54507 ± 0.00450`，`Precision@100` 为 `0.23921 ± 0.01611`，Top-100 日均单边换手为 `49.74% ± 5.79%`，日均成本后主动收益为 `-9.67 ± 3.71bp`。Rank IC 在三个 seed 中均为正，成本后主动收益在三个 seed 中均为负。

半衰期与交易转换阶段的决定为 `HOLD`。EMA 和换仓收益差门槛降低了换手，H5 五组错峰在 12 月为正，在 11 月为负。两个窗口的成本后主动收益方向没有重复，收益仍集中在少数日期。相邻折 seed 1、2 与 `probe150m` 继续暂停。下一步运行任务梯度、标签尺度与监督位置消融。完整结果见[事件流信号半衰期与交易转换诊断](eventstream-signal-trading-diagnostics.md)。

### 数据与标签

短周期预训练使用 A 股自身数据，不迁移 FI-2010 权重。候选辅助目标包括：

- 未来 10、50、100、500 个盘口事件的中间价方向或收益
- 未来 1、5、30 分钟收益
- 价差变化、盘口失衡变化和短期冲击恢复

所有辅助标签只能使用输入窗口之后的数据，输入不得跨越各自信号时点。标准化、股票池和日期切分继续遵守训练期可得原则。

### 日内采样

冻结特征实验至少比较：

- 单一尾盘 embedding
- 10:00、11:30、14:00、14:55 多锚点 embedding
- 分钟聚合特征与盘口 embedding 拼接

每日 embedding 使用固定版本编码器一次性生成并缓存。缓存记录编码器 checkpoint 指纹、输入协议、日期、股票和特征 schema。

### 增量实验

| 实验 | 输入 | 下游模型 |
|---|---|---|
| E0 | 分钟聚合特征 | HGB 或 M4 最佳 ranker |
| E1 | 冻结盘口 embedding | 同一 ranker |
| E2 | 聚合特征 + 冻结 embedding | 同一 ranker |
| E3 | 多锚点 embedding | 同一 ranker |

### 验收门槛

- E2 或 E3 在多个滚动窗口稳定优于 E0。
- Top-K 净收益、NDCG 和月度稳定性至少有一组一致改善，训练集指标不单独作为通过依据。
- embedding 增量不能由股票代码、日期或未来数据泄漏解释。
- 冻结 embedding 通过门槛后，才允许联合微调。

未通过时保留聚合特征路线，并停止联合微调和新的容量扩张。原始盘口四格矩阵已经形成停止结论，本里程碑不重新启动 raw-200、raw-1000 的容量或窗口扩张。

### 计划实验 ID

- `FEAT-EVENTSTREAM-100M-SEED0-001`
- `FEAT-EVENTSTREAM-100M-ROBUSTNESS-001`
- `FEAT-EMB-FROZEN-001`
- `FEAT-EVENTSTREAM-JOINT-001`
- `FEAT-EMB-MULTI-ANCHOR-001`
- `FEAT-EVENTSTREAM-150M-ABLATION-001`

## M6：神经 Top-K 排序损失

### 进入条件

M4 证明排序目标有价值，或 M5 证明 embedding 有稳定增量。否则本阶段保持待开始。

### 最小实现

- [ ] 新增按 `label_date` 分组的 `DayBatchSampler`。
- [ ] 一个训练 step 能识别同一天股票之间的横截面关系。
- [ ] 实现 pairwise logistic ranking loss。
- [ ] 对 Top-K 边界附近的 pair 加权。
- [ ] 保留 Smooth L1 作为收益尺度锚点，分类头降为辅助目标。
- [ ] 必要时再实现 LambdaRank 或可微 NDCG，不在第一版同时引入多个复杂损失。

候选目标：

```text
L = lambda_point * SmoothL1
  + lambda_class * CrossEntropy
  + lambda_rank * TopKPairwiseLoss
```

### 对照矩阵

- pointwise 原损失
- 仅 pairwise
- pointwise + pairwise
- 普通 pairwise 与 Top-K 边界加权 pairwise

### 验收门槛

- 多 seed、多个滚动窗口下的改善方向一致。
- 提升要同时出现在 Top-K 净收益与排序指标上，只出现在训练 loss 上不算。
- 显存和训练时长没有因完整日 batch 失控。

## M7：多日 embedding 与多期限目标

### 进入条件

M5 的每日 embedding 已证明有增量，且缓存协议固定。

### 模型路径

```text
股票每日 embedding
  -> 最近 5 / 10 / 20 日序列
  -> 小型 TCN 或 GRU
  -> 次日头 + 未来 5 日头
```

输入同时加入常规日频价格、波动率、流动性、行业和规模因子，用于判断盘口 embedding 是独立信息还是已有风险暴露的代理。

### 标签与切分

- 次日目标继续使用明确定义的可执行收益。
- 5 日目标需要定义起止价格和基准收益，不能把日频分数简单持有五天当作周频训练。
- 5 日标签跨越切分边界时必须 purge，并增加与标签长度一致的 embargo。
- 重叠 5 日标签的统计显著性需要使用区块 bootstrap 或相应稳健误差。

### 对照

- 当日 embedding 预测次日
- 5/10/20 日 embedding 预测次日
- 5/10/20 日 embedding 预测未来 5 日
- 仅常规日频因子
- 常规因子 + embedding

### 停止条件

- 5 日目标在多个窗口没有稳定超过常规日频基线。
- 换手下降但毛利下降得更快。
- 结果对序列长度或一个年份高度敏感。

## M8：AgentX 研究智能

### 进入条件

M1 与 M2 已提供稳定的 evaluator、executor、artifact 和 Registry 数据。

### Context Builder

ResearchContext 不再由 CLI 填写一个问题和异常字符串，而是从确定性来源构建：

- 当前生产或研究基线
- 最近实验及其 KEEP、EXTEND、DISCARD 结论
- 相关 parent DAG 和失败原因
- Audit 异常与未解释观测
- 可用 executor、数据和预算
- 已见日期与 locked 权限

### Brainstorm

- [ ] 每轮生成 3 至 5 个候选，不再只生成单个提案。
- [ ] 标记 `ready`、`probe_first` 或 `backlog`。
- [ ] 按目标对齐、证据、可行性、成本、风险和新颖性排序。
- [ ] 使用 novelty signature 去重，避免重复已否定实验。
- [ ] 每个候选说明机制、可观察量、对照和可证伪条件。

### Critic 与 Evaluation

- [ ] Critic 检查历史重复、数据泄漏、预算、入口可用性和指标是否能计算。
- [ ] LLM 评审只提供建议，Policy 和统计 gate 保留最终决定权。
- [ ] Evaluation 对声明的因果链逐项输出 `verified`、`broken` 或 `unclear`。
- [ ] 指标改善但机制归因不清晰时，不自动记为成功。
- [ ] KEEP、EXTEND、DISCARD 和负面教训自动回写下一轮上下文。

### 真实 LLM 接入顺序

先用固定 replay cases 验证模板和结构化输出，再接真实模型。真实 LLM 不能扩大 executor、数据或 locked 权限。模型版本、prompt 版本、输入上下文摘要和原始输出都进入 artifact。

### 验收门槛

- Agent 能从真实 Registry 识别已有失败，不重复提出相同实验。
- 选择出的候选能被现有 executor 无人工补配置地执行。
- 一轮执行后，Audit 和 Evaluation 结果能自动影响下一轮候选。
- LLM 输出错误时，系统安全失败且不污染 Registry 或 locked 数据。

## M9：Developer Agent 与 Harness Evolution

### Developer Agent

只有 M8 稳定后才开放代码修改。Developer Agent 使用独立 worktree，并受到以下约束：

- 明确允许修改的文件和模块
- 预期机制与必须出现的可观察量
- 必须新增或更新的测试
- diff 大小、执行时间和重试次数预算
- 禁止修改数据切分、locked 协议和历史 artifact
- Ruff、ty、pytest 和冒烟全部通过后才进入人工 diff review

Developer Agent 不直接合并代码，也不能自行批准 locked test。

### Harness Evolution

当前轨迹数量不足，不启动 SGPO。至少积累一批结构完整、能 replay 的成功、失败、被拒和平台故障轨迹后，再建立：

- 固定 replay 数据集
- 旧 harness 与候选 harness 的 paired replay
- 语义正确性、任务覆盖、文件覆盖和安全评分
- 只修改一个 subagent harness 的准入规则
- 回归时 no-op 并保存失败原因

是否启动不以轨迹条数单独决定，还要确认历史 artifact、指标和决策可以被确定性重放。

## 每个里程碑的执行模板

进入新里程碑时，在本节下方复制一份执行记录，或建立对应实验说明：

```yaml
milestone: M1
status: in_progress
owner: human-and-codex
hypothesis: ""
baseline: ""
change: ""
primary_metrics: []
success_gates: []
falsification_condition: ""
data_protocol_version: ""
experiment_ids: []
artifacts: []
decision: null
next_action: ""
```

完成时补充实际命令、结果路径、指标、异常和 KEEP、EXTEND、DISCARD 决策。不能只写模型训练成功或代码测试通过。

## 当前下一步

M0 至 M3 已完成。M4 与 M5 的最近折冻结表征和联合训练三 seed 对照也已完成，当前优先级如下：

1. `EVT-GRAD-AUDIT-001` 已完成。两折 best checkpoint 的日级梯度比值中位数为 0.01969 和 0.03927，正式决定进入 `EVT-LABEL-SCALE-001`。
2. 在最近折和相邻折运行每日截面去极值 z 标签的 seed 0，对比原始 H5 标签基线的 validation、OOS 和极端组收益差。
3. seed 0 在两折同时改善后，再补 seed 1、2。未形成跨折增量时进入 `EVT-SUPERVISION-POSITION-001`。
4. 获得带日期的行业分类数据后补齐行业暴露归因。规模、流动性和波动率暴露已经完成第一轮核对。
5. 跨窗口和成本后主动收益形成稳定增量后，再评估 `probe150m` 的预算和 seed 0 门槛。

神经排序损失和多日模型继续遵守 M6、M7 的进入条件。原始盘口容量与窗口扩张保持停止。
