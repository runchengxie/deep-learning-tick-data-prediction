# 实验记录

按日期记录带结论的实验，保留关键数字和交付物，供核对。最新结论汇总见 [topk-agentx-research-roadmap.md](topk-agentx-research-roadmap.md) 的当前证据一节，资源使用原则见 [resource-strategy-and-pilot-gates.md](resource-strategy-and-pilot-gates.md)。

## 2026-08-07：Google Drive 目录整理

Drive 上散落了本项目多代命名（旧名 `deeplob`）与个人文件。本轮只做移动改名，全部为 Drive 内元数据操作，可回退，不删除、不上传下载。项目相关全部归入新建的顶层目录 `deep-learning-tick-data-prediction/`：

| 新路径 | 来源 | 说明 |
|---|---|---|
| `ticknet-data/` | `deeplob-data` | 训练工作集（raw-200 pilot、smoke、smoke-v2） |
| `ticknet-runs/` | `deeplob-runs` | 训练结果（5 seed + 锁定测试 + 基线、smoke 系列、本地证据） |
| `fi2010-reproduction/` | `DeepLOB` | FI-2010 复现（数据 + checkpoint） |
| `code-legacy-v1/` | `deeplob-reproduction` | 旧代码快照（`deeplob` 包名） |
| `backup-before-locked-test-2026-08-04/` | 同名 | 锁定测试前备份 |
| `Colab Notebooks/`、`Google AI Studio/` | 同名 | Google 工具默认目录，一并纳入 |

目录名统一为 `ticknet` 前缀或语义化新名，checkpoint 文件名 `deeplob.setup2.*.pt` 改为 `ticknet.setup2.*.pt`。旧代码快照内部的 `deeplob` 包名和文档名保持原样，改包名会使旧快照无法运行，它只是历史存档。

本次整理确认 A 股 raw-200 pilot 已完整跑完：数据 2024 全年 23,515 样本；Logistic 基线验证集 Rank IC 约 0.015、MCC 约 0.059；深度模型 5 seed 锁定测试 Rank IC 约 0.0075 ± 0.015、MCC 约 0.070 ± 0.010、Macro F1 约 0.335 ± 0.028，信号微弱、跨 seed 不稳定，未明显超过基线。另设 1,033,383 参数容量实验，与 86,775 参数基线保持相同数据和训练口径。

Drive 操作全部经远程主机 rclone。本机 `gdrive:` 的 OAuth token 会过期，若要更持久，配置自定义 Google Cloud OAuth 凭据（见私有记录，不写入本仓库）。

## 2026-08-07：分钟序列 TCN 与聚合 HGB 对比

在完全同口径下对比未聚合的分钟序列 TCN 与聚合特征 HGB，回答时序建模是否比聚合特征多提供次日横截面排序信息。

数据：全量 L2 分钟分片 2024 全年 24,188 样本（与 HGB 基线 written_samples 一致），12 个分片共 174 MB，布局 `samples x 60 x 30` float32。HGB 使用同一股票池、同标签、同切分、同 60 分钟窗口，输入聚合为 120 维特征。L2 分钟特征存在逐列缺失（25/30 列，11,251 个样本含 NaN），用训练区间逐列中位数填充，短窗口（21 个）尾部补 NaN 后统一填充。切分为 2024-01-01 至 06-30 训练（11,595）、07-01 至 09-30 验证（6,293）、10-01 至 12-31 测试（6,000）。

TCN 为 4 层膨胀因果卷积（核 3）、64 通道、weight-norm、双头输出，与 HGB 相同的类别加权、选择指标和早停。训练 12 epoch 内早停，best 在 epoch 7（val Rank IC 0.058）。HGB 为 HistGradientBoosting，max_iter 500、leaf 31。

单 seed 对比：

| 指标 | HGB val | TCN val（best） | HGB test | TCN test（seed 0） |
|---|---|---|---|---|
| daily_rank_ic_mean | 0.0325 | 0.0579 | 0.0111 | -0.0081 |
| macro_f1 | 0.313 | 0.224 | 0.361 | 0.214 |
| mcc | 0.107 | 0.041 | 0.131 | 0.032 |

TCN 3 seed 锁定测试聚合：

| 指标 | HGB test | TCN test（3 seed mean ± std） |
|---|---|---|
| daily_rank_ic_mean | 0.0111 | 0.0093 ± 0.016 |
| macro_f1 | 0.361 | 0.253 ± 0.033 |
| mcc | 0.131 | 0.060 ± 0.024 |
| balanced_accuracy | 0.379 | 0.358 ± 0.014 |

结论：验证集上 TCN 的排序能力（Rank IC 0.047 至 0.058）超过 HGB（0.032），但分类指标弱于 HGB。3 seed 测试集上 HGB 全面占优，TCN 的验证集选模信号未能泛化到测试集，存在系统性过拟合。聚合特征加 HGB 是更稳健的分钟级基线。本次对比交付了受控实验设计、L2 分钟分片管线（含 NaN 中位数填充与短窗口补齐）和 TCN 训练入口（`ticknet-minute-tcn-train` 与 `-evaluate`）。分钟分片 174 MB 显著小于原始 L2 parquet（年度 23 GB），TCN CPU 训练约 6.6 分钟每 seed。

## 2026-08-08：分钟 HGB 多年份滚动稳健性验证

用同一套分钟 HGB 管线（聚合 120 维特征、max_iter 500）逐年滚动，得到 4 个独立样本外 test 年，回答信号在严格样本外是否仍为正。

数据为 L2 分钟缓存 2021 至 2025、动态前 100 股票、窗口 60 分钟。对每个 test 年，train 为之前全部年份，val 为 test 年前半年，test 为目标年 H2。配置见 `configs/nextday-minute-rolling-{2022,2023,2024,2025}.yaml`，结果见 `results/nextday-minute-rolling-{2022,2023,2024,2025}.json`。

| test 年 | 样本外 Rank IC | 交易日数 | test mcc | train 样本 |
|---|---|---|---|---|
| 2022 | 0.0218 | 124 | 0.060 | 19,559 |
| 2023 | 0.0326 | 123 | 0.110 | 43,734 |
| 2024 | 0.0353 | 124 | 0.113 | 67,930 |
| 2025 | 0.0304 | 125 | 0.081 | 92,118 |

四个 test 年 Rank IC 全部为正，区间 0.022 至 0.035，分类指标也全部为正。结论：分钟聚合特征 HGB 的信号在 4 个独立样本外年一致为正，信号真实存在但弱，约 Rank IC 0.02 至 0.035，跨年稳定，尚不足以直接构成扣除成本后盈利的策略。工程改动：`run_minute_baseline.py` 增加按年流式读取（`_build_samples_by_year`），峰值内存从 20 GB 以上降至约 9 GB；`minute_baseline.py` 分钟特征由 float64 降为 float32，内存再减半，指标不变。

## 2026-08-08：成本后多空组合收益评估

判断弱信号（Rank IC 0.02 至 0.035）在真实交易成本下是否仍为正。用 2025 独立样本外年的 HGB 预测明细（`results/predictions-rolling-2025.parquet`）做多空组合回测，每日按 score 取 top/bottom 10% 等权持仓。成本模型为单边成本（佣金加冲击）按档位 0、3、5、10、20 bp，卖出另加印花税 0.05%，换手率按相邻调仓日组合成分差异计算。脚本为 `scripts/evaluate_cost_adjusted.py`，支持 `--rebalance-days` 控制调仓频率。

结果（2025 H2，125 个交易日，换手率 83% 每日）：

| 单边成本 | 无成本年化 | 净年化 | 净夏普 | 日均成本 |
|---|---|---|---|---|
| 0 bp | +27.9% | +27.9% | 0.88 | 0.04% |
| 3 bp | +27.9% | +15.7% | 0.49 | 0.09% |
| 5 bp | +27.9% | +7.5% | 0.24 | 0.13% |
| 10 bp | +27.9% | -12.8% | -0.40 | 0.21% |
| 20 bp | +27.9% | -53.6% | -1.68 | 0.38% |

日频换手率 83%，盈亏平衡单边成本约 5 至 6 bp，低于 A 股实际成本（佣金加冲击通常不低于 10 bp）。周频（每 5 日调仓）换手降到 17%、日均成本降到 4.3 bp，但无成本毛利也跌到 -30%（信号是日频短期动量），净收益 -40.8% 更差。

结论：信号真实但属日频短期信号，真实成本下净收益转负，盈亏平衡成本低于可实现成本，降低调仓频率也不解决问题。这是项目停止判断的完整证据链最后一块，信号存在、跨年稳健、但不可交易。以上是当时的分位数多空诊断结论，新 Top-K 研究系列保留它作为历史证据，正式策略改用 `ticknet.research.portfolio` 的 fixed-K long-only、open-to-open 和股票级成交明细口径，见 [topk-agentx-m1-portfolio-evaluator.md](topk-agentx-m1-portfolio-evaluator.md)。工程改动：`run_minute_baseline.py` 增加 `--save-predictions`，新增 `scripts/evaluate_cost_adjusted.py`，`tests/test_evaluate_cost_adjusted.py` 有 3 个测试覆盖。

### 同日：IC 与 spread 背离的审计归因

第 2 节观察到 Rank IC 约 0.01 但无成本 spread 约 27.9 bp 每日的矛盾。用新增的 `ticknet-research audit`（`src/ticknet/research/audit.py`）对 2025 预测明细做诊断：

| 指标 | 值 | 含义 |
|---|---|---|
| daily_rank_ic_mean | 0.030 | 信号真实但弱 |
| daily_ic_ir | 0.211 | 横截面排序不稳定 |
| top 1 日贡献 | 25.7% | 单日贡献超四分之一 |
| top 5 日贡献 | 121% | 前 5 天贡献超过全部收益 |
| top 10 日贡献 | 212% | 前 10 天贡献是全部收益的两倍 |
| spread 中位数 | 0.026% | 典型日 spread 很低 |
| decile 单调性 | 0.41 | 排序信号非线性 |
| 月度 IC | 5 正 1 负 | 2025-10 为 -0.015 |

结论：多空组合的利润基本由极少数大波动日贡献，没有持续稳定的方向性预测能力。spread 均值被极端日拉高，掩盖了典型日的低收益，这解释了为什么成本后收益转负。审计把 IC 接近 0 但 spread 高这个矛盾还原为可解释的结构性事实，也是未来 Evaluation Agent 的观测接口。工程改动：新增 `src/ticknet/research/audit.py`（`PredictionTable` 加 `audit_predictions()`）、`ticknet-research audit` 子命令，`tests/test_research_audit.py` 有 5 个测试覆盖。

## 2026-08-08：AgentX 式自动量化研究闭环落地

参考 [references/agentx-paper-notes.md](../../references/agentx-paper-notes.md)，把项目从人工研究升级为实验系统先机器可调用、再加 Agent。代码结构如下：

```
src/ticknet/research/
  spec.py        ExperimentSpec：假设 + 可证伪条件 + 配置覆盖 + seeds
  policy.py      ResearchPolicy：白/黑名单 + 预算 + stage（程序裁决，非 LLM）
  protocol.py    ResearchProtocol：锁定测试期程序级隔离
  runner.py      ExperimentRunner：唯一执行入口，跑训练、解析结果、登记
  registry.py    SQLite 实验记忆（experiments/runs/metrics/reviews + parent DAG）
  audit.py       PredictionTable + audit_predictions：IC/spread/decile/月频诊断
  locked.py      锁定测试评估，需显式人工批准
  agents/
    client.py    LLMClient 抽象（template / openai / deepseek）
    context.py   ResearchContext：Brainstorm 标准输入
    brainstorm.py  生成 ExperimentSpec（模板或 LLM）
    critic.py    审查可证伪性/重复/泄漏
    orchestrator.py  research_step 闭环
```

闭环为 ResearchContext 到 Brainstorm、Critic、Policy、Runner、Audit、Registry 的单向流程。`ticknet-research` CLI 提供 run、show、compare、audit、approve-locked-test、locked-test 和 agent-step 子命令。关键设计：权限由程序控制，Agent 不能改 `test_end` 等字段；确定性先于 LLM，指标提取和裁决用确定性 Python；每个提案必须声明 falsification_condition；负面结果写入 Registry 形成 parent DAG；测试集物理隔离。

验证结果：全套件 115 个测试通过（新增 research 16 个），端到端 `agent-step` 真实跑通（Brainstorm 从极端日贡献偏高生成 data_audit 提案，TCN 训练完成并登记 EXP-AUTO-TCN2），越权实验被 PolicyViolation 拦截。后续未做：Developer Agent、SGPO 或 Harness Evolution、Brainstorm 接真实 LLM（`--provider deepseek/openai` 已预留接口）。

## 2026-08-08：Agent-driven 调仓频率假设验证

由 LLM 扮演 Brainstorm Agent 驱动的第一轮自动研究，验证系统闭环。审计显示多空收益由极端日驱动（top 5 日贡献 121%）、成本后收益 -12.8%，LLM 提出假设存在中等调仓频率（2 至 3 日）甜点区。

`ticknet-research agent-step` 走完整闭环，Brainstorm 识别成本异常并生成 cost_analysis 提案，TCN 训练完成登记（EXP-LLM-ROUND1，git_sha=86bb130）。成本敏感度（单边 10bp，2025 H2）：

| 调仓频率 | 换手 | 毛利年化 | 净年化 | 净夏普 |
|---|---|---|---|---|
| 1 日 | 0.83 | +38.1% | -12.8% | -0.40 |
| 2 日 | 0.42 | -41.6% | -67.0% | -2.02 |
| 3 日 | 0.29 | -54.6% | -72.2% | -2.50 |
| 5 日 | 0.17 | -30.2% | -40.8% | -1.23 |
| 10 日 | 0.09 | -9.0% | -14.6% | -0.49 |

结论：假设被否定。降低频率虽降换手，但毛利崩溃更快，2 至 3 日反而最差。信号是纯日频动量，alpha 集中在隔夜到次日单日窗口，与成本结构不兼容。系统自主收敛到成本是主要瓶颈的方向，与 LLM 独立分析一致，验证了闭环能产生可复现的研究结论并登记为结构化证据。

## 2026-08-12：eventstream A100 输入流水线优化

在 2025-08 Top-400 train pack 上，用 100,604,180 参数的 `capacity100m` 因果 Transformer 拆分 DataLoader-only、预加载 batch 的 GPU-only 和真实端到端吞吐。实验只读取 2025-08 训练数据，dataset fingerprint 为 `705445378f0fc5842ce80bcfa41a01cdd10236198c5683f108f0ba146f8c3b82`；validation、OOS 和 2026 locked 数据均未访问。

旧 Dataset 的 GPU-only 为 235.28 samples/s。worker sweep 结果为 2、4、8、16 workers 对应端到端 4.62、9.53、13.23、18.19 samples/s，确认瓶颈在输入流水线。按 120,000 个样本和 20 epoch 外推，最佳 16 workers 仍需 36.65 小时每 seed。

旧实现每取一个 512 事件窗口都会对该股票全天 order、trade、snapshot 重新拼接、稳定排序并构造全部 80 维特征。2025-08 有 8,097 个 `(day, ticker)` 和约 24.55 亿事件。`uint32` merge index 全月约 9.15GiB；按正式 shuffle 顺序模拟 20 epoch，16 workers 每个 512MiB、合计 8GiB 的 LRU 命中率只有 5.45%，合计 32GiB 也只有 22.15%，因此不落地 LRU。

优化改为对三条已排序流做时间二分，定位目标合并排名后只稳定归并窗口附近 513 个事件，并从窗口前最后一个有效 snapshot 延续滚动中间价。合成数据穷举同时间戳顺序和全部窗口均与旧实现一致；真实 2025-08 三个交易日、9 个窗口逐元素一致，单样本数据构造加速 15.6 至 18.9 倍。

优化后 A100 runtime 有 12 个 CPU core，GPU-only 为 238.79 samples/s。worker sweep 结果：

| workers | DataLoader-only | 端到端 | 20 epoch 外推 |
|---:|---:|---:|---:|
| 2 | 62.53 | 65.24 | 10.22 小时/seed |
| 4 | 124.11 | 123.71 | 5.39 小时/seed |
| 8 | 140.48 | 149.40 | 4.46 小时/seed |
| 16 | 171.07 | 140.73 | 4.74 小时/seed |

选择 8 workers。相对旧 Dataset 的最佳端到端吞吐提升 8.21 倍，相对最初 batch 8、2 workers 的约 4.4 samples/s 累计提升约 34 倍。三个 seed 串行 20 epoch 上限约 13.39 小时；正式训练还包含 validation、checkpoint I/O 和早停，以真实墙钟为准。当前模型已经是带 RoPE、causal scaled-dot-product attention 和 FFN 的 Transformer，换用 Hugging Face `transformers` 封装不会解决本轮确认的输入瓶颈。

## 2026-08-16：raw-1000 Top-100 100M 三 seed 正式训练

按 [nextday-raw-1000-top100-capacity-100m.yaml](../../configs/nextday-raw-1000-top100-capacity-100m.yaml) 的冻结合同完成 seed 0、1、2。模型有 100,817,575 个参数，目标为下一交易日开盘到收盘的个股超额收益，checkpoint 按 2024 validation 日均 Rank IC 选择。训练期为 2021 至 2023，验证期为 2024。工作集共 118,078 个样本，其中 train 为 70,805 个，validation 为 23,472 个。数据指纹为 `f8a17e63d0716f9e48fd05f9a269bb61cea5bff81e9a7acf90c4a42e47505e5c`。

seed 0 使用源码版本 `56f99d9`，seed 1 和 2 使用合并后的 `95a3a90`。三次训练的模型、数据、目标、优化器、选模指标和早停合同一致。结果如下：

| seed | 最佳 epoch | 实际 epoch | validation Rank IC | Rank ICIR | Macro F1 | MCC | 训练时间 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 17 | 0.03295 | 0.19748 | 0.37910 | 0.09666 | 66.70 分钟 |
| 1 | 20 | 28 | 0.03340 | 0.23013 | 0.38944 | 0.11766 | 109.65 分钟 |
| 2 | 12 | 20 | 0.02822 | 0.18163 | 0.36553 | 0.10205 | 78.27 分钟 |

三 seed 最佳 validation Rank IC 均值为 0.03152，seed 间样本标准差为 0.00287，范围为 0.02822 至 0.03340。串行训练累计 15,276.83 秒，即 4.24 GPU 小时。相对既有 `1M/raw-200` 三 seed 的均值 0.02031，候选组合高 0.01121，约为 55.2%。这个差值只描述两个已运行组合，不是容量的独立因果效应。

当前候选同时改变了模型容量、事件窗口和股票样本集合，学习率与最小日横截面门槛也不同。checkpoint 又是在同一 2024 validation 上按 Rank IC 选择，因此均值带有选模乐观偏差，不能解释为样本外显著性或可交易收益。正式归因需要在同一 Top-100 样本集合和同一训练合同下补齐 `100M/raw-200` 与 `1M/raw-1000`，并与两端组合构成 2×2 对照。

2025 test 仍保持锁定。三个结果文件中的 `test` 均为 `null`，运行摘要为 `locked_not_accessed`。清单中的 23,512 只表示 test 元数据行数，没有执行模型评估。下一步先补齐同口径归因矩阵和 validation 稳定性检查，再冻结一次性 test 评估条件。
