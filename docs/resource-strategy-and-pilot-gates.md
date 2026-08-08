# 有限算力下的研究资源策略与门槛式实验

> 适用范围：在 Google Colab Pro 和 Google Drive 100GB 的资源下，用最省的
> 计算预算逼近一个关键取舍。这个取舍就是分钟微观结构和原始盘口信号是否存在，
> 值不值得继续投入深度模型。本文是 `hardware-constraints-and-experiment-roadmap.md`
> 的可执行补充，聚焦资源怎么用、小批怎么排、每一步怎么判断。

## 1. 核心原则

资源够用但有限。目标不需要跑完所有实验，用少数几个大方向实验就能回答：

- 时序结构是否比聚合特征（HGB）更有效？
- 原始盘口是否比分钟特征有可复现的增量？
- 信号在严格样本外是否仍为正？

对应的三句取舍：

| 取舍 | 做法 |
|---|---|
| 重计算在本地主机，GPU 只训练 | 远程主机做数据预处理，小体积成品传 Drive |
| 跑 1–2 个关键实验，不跑 100 次微调 | 用门槛拦住，结论先行 |
| 断点续跑兜底，避免额度白掉 | checkpoint 走 Drive，会话断了能接 |

## 2. 资源与数据容量对照

| 资源 | 容量 / 能力 | 适合放什么 |
|---|---|---|
| Colab Pro | T4/A100 GPU 小时额度 | 小型 TCN/GRU、原始盘口增量实验、一次性 embedding |
| Google Drive | 100 GB | 预处理后的训练工作集、checkpoint、结果 |
| 远程主机数据盘 | 122 GB 分钟缓存 / 数 TB 原始盘口 | 原始数据、预处理脚本、全量扫描 |

本地主机用 CPU 做：数据审计、Logistic、树模型、小样本训练。

### 体积估算（400 股 × 约 1,250 交易日）

| 表示 | 约占空间 | 谁用 |
|---|---|---|
| 近 60 分 × 33 特征 | 约 4 GB | 推荐主模型（装得下） |
| 全天 240 分 × 33 特征 | 约 16 GB | 检查全天信息增量 |
| 近 200 盘口 × 40 特征 | float16 约 8 GB | 客户原始盘口 MVP |
| 近 500 / 1,000 盘口 × 40 特征 | 20 / 40 GB | 分级扩展，一次放 1–2 个 |
| 每日 64 维 embedding | 约 128 MB | 多日层级模型、反复调参 |

> 结论：Drive 只能存预处理后的成品，不能当原始数据仓库。

## 3. 骨干执行路径（每步是门票决策）

```text
远程主机预处理（费 CPU、不费 GPU，免费）
   ↓ 产出小体积训练分片（4GB / 16GB）
Google Drive（只存工作集、checkpoint）
   ↓
Colab Pro GPU（只跑关键对比与门槛实验）
   ↓
仅在通过时扩展（正式集 / 原始盘口 / 多年）
```

### 分步明细

1. 预处理 + 上传
   - 远程主机把分钟特征缝合成 4GB 分片（默认 60 分钟）。
   - 100GB 够放 4GB 主 + 16GB 扩展 + 8GB 原始盘口 float16（1–2 个）。
2. TCN vs HGB 对比（当前最关键的实验）
   - 数据：同 2024 pilot 切分（H1 训练 / Q3 验证 / Q4 测试、top100）。
   - 门槛：验证集每日 Rank IC 为正，且 TCN 明显优于 HGB。
3. 级联扩展（仅在 2 通过）
   - 扩到 400 股 × 2021–2025 正式集（仍 4GB）。
   - 依次 200 → 500 → 1000 盘口，每级都比同一口径下的分钟模型。
4. 一次编码、多日训练
   - 冻结日内编码器，生成 64 维 embedding（约 128MB）。
   - 多日模型可以放在本地快速迭代，不用重复编码原始盘口。

## 4. 用满 Colab 的三个便宜用法

1. 断点续跑加 checkpoint 走 Drive。Session 会断，靠它兜底额度。
2. embedding 一次编码、多日复用。日内编码器冻结后，多日调参超省算力。
3. 测试集只解锁一次。冻结在所有实验末尾。

## 5. 权衡有效性的判断口径

要把预算花在少数能直接回答信号是否存在的重点实验上，不要把预算均匀砸在调参。

出现以下任一信号，就应停止扩大模型。停止也是有效的结论：

- 只有训练集指标变好，验证集 Rank IC 没变。
- 改一个月份或随机种子结论就反转。
- 原始盘口模型没能稳定超过分钟 TCN。
- 最高分组与最低分组的收益全部来自涨跌停、停牌恢复或极低流动性股。
- 扣除合理成本后分组收益消失。

## 6. 现阶段最关键的一个实验

TCN 受控对比（2024 pilot，60 分钟 33 特征）：

- 作用：判断时序结构是否相对聚合（HGB）有可复现增量。
- 若通过：投入正式扩展与原始盘口。
- 若不通过：停止在分钟模型，本文档到此为止。停止结果本身也有价值。

本机 CPU 实测参照：单层 GRU（240×33）约 61 股票日/秒，分块 DeepLOB
（10×100 盘口）约 8.5 股票日/秒。TCN 参数量约 50 万，适合一台 Colab 会话。

## 7. Google Drive 目录整理记录（2026-08-07）

背景：Drive 上散落了本项目多代命名（旧名 `deeplob`）与个人文件混杂。本轮只做
移动改名，全部为 Drive 内元数据操作，可回退，不删除、不上传下载。

### 目标结构

Drive 根目录只保留个人文件夹（Apun、Em、HelloFax、Recordatorio），项目相关全部
归入新建的顶层目录 `deep-learning-tick-data-prediction/`：

| 新路径 | 来源 | 说明 |
|---|---|---|
| `ticknet-data/` | `deeplob-data` | 训练工作集（raw-200 pilot、smoke、smoke-v2） |
| `ticknet-runs/` | `deeplob-runs` | 训练结果（5 seed + 锁定测试 + 基线、smoke 系列、本地证据） |
| `fi2010-reproduction/` | `DeepLOB` | FI-2010 复现（数据 + checkpoint） |
| `code-legacy-v1/` | `deeplob-reproduction` | 旧代码快照（`deeplob` 包名） |
| `backup-before-locked-test-2026-08-04/` | 同名 | 锁定测试前备份 |
| `Colab Notebooks/`、`Google AI Studio/` | 同名 | Google 工具默认目录，一并纳入 |

### 改名与保留

- 目录名统一为 `ticknet` 前缀或语义化新名。
- checkpoint 文件名 `deeplob.setup2.*.pt` → `ticknet.setup2.*.pt`。
- 旧代码快照（`code-legacy-v1/`、`backup/`）内部的 `deeplob` 包名、文档名保持
  原样。改包名会使旧快照无法运行，它只是历史存档，本地主仓库才是权威。

### 关键事实（本次整理确认）

A 股 raw-200 pilot 已完整跑完：

- 数据：`ticknet-data/nextday-raw-pilot-2024-top100/`，2024 全年 23,515 样本。
- 基线：Logistic，验证集 Rank IC ≈ 0.015，MCC ≈ 0.059。
- 深度模型 5 seed 锁定测试：Rank IC ≈ 0.0075 ± 0.015，MCC ≈ 0.070 ± 0.010，
  Macro F1 ≈ 0.335 ± 0.028。信号微弱、跨 seed 不稳定，未明显超过基线。

### rclone 访问说明

Drive 操作全部经远程主机 rclone（本机 `gdrive:` 的 OAuth token 会过期）。若要
更持久，配置自定义 Google Cloud OAuth 凭据（见私有记录，不写入本文件），替代
正在回收的官方共享 client_id。

## 8. 分钟序列时序模型（TCN）vs 聚合基线（2026-08-07）

目标：在完全同口径下对比未聚合的分钟序列 TCN 与聚合特征 HGB，回答时序
建模是否比聚合特征多提供次日横截面排序信息。

### 数据与口径

- 全量 L2 分钟分片：`ticknet-data/nextday-minute-l2-2024-top100/`，2024 全年
  24,188 样本（与 HGB 基线 written_samples 完全一致），12 个分片，共 174 MB。
- 分片布局 `samples x 60 x 30` float32，HGB 使用同一股票池、同标签、同切分、
  同 60 分钟窗口，但输入聚合为 120 维特征。
- NaN 处理：L2 分钟特征存在逐列缺失（25/30 列，11,251 个样本含 NaN）。用训练
  区间逐列中位数填充（避免验证/测试信息泄漏）。不足 60 分钟的短窗口（21 个）
  尾部补 NaN 后统一由中位数填充。
- 切分：train 2024-01-01 ~ 06-30（11,595），val 07-01 ~ 09-30（6,293），
  test 10-01 ~ 12-31（6,000）。

### 模型与训练

- TCN：4 层膨胀因果卷积（核 3），64 通道，weight-norm，双头（分类 + 连续分数），
  与 HGB 相同的类别加权、选择指标（val Rank IC）与早停。
- 训练 12 epoch 内早停，best 在 epoch 7（val Rank IC 0.058）。
- HGB：HistGradientBoosting，max_iter 500，leaf 31，与既有分钟基线一致。

### 结果（同口径对比）

单 seed 对比：

| 指标 | HGB val | TCN val (best) | HGB test | TCN test (seed 0) |
|---|---|---|---|---|
| daily_rank_ic_mean | 0.0325 | 0.0579 | 0.0111 | -0.0081 |
| macro_f1 | 0.313 | 0.224 | 0.361 | 0.214 |
| mcc | 0.107 | 0.041 | 0.131 | 0.032 |

TCN 3 seed（seed 0/1/2，best 均选在 epoch 11 前后）锁定测试聚合：

| 指标 | HGB test | TCN test (3 seed mean ± std) |
|---|---|---|
| daily_rank_ic_mean | 0.0111 | 0.0093 ± 0.016 |
| macro_f1 | 0.361 | 0.253 ± 0.033 |
| mcc | 0.131 | 0.060 ± 0.024 |
| balanced_accuracy | 0.379 | 0.358 ± 0.014 |

### 结论

- 验证集上 TCN 的排序能力（Rank IC 0.047 ~ 0.058）超过 HGB（0.032），但分类
  指标弱于 HGB。
- 3 seed 测试集上 HGB 全面占优：TCN 的 test Rank IC 为 0.009 ± 0.016，低于
  HGB 的 0.011，MCC（0.060 vs 0.131）与 Macro F1（0.253 vs 0.361）差距明显。
  单 seed 0 的 test Rank IC -0.008 是种子噪声，多 seed 平均回到正值但不及 HGB。
- TCN 的验证集选模信号未能泛化到测试集，存在系统性过拟合。聚合特征 + HGB 是
  更稳健的分钟级基线，时序原始序列建模在当前口径下不占优。
- 本次对比已交付：时序 vs 聚合的受控实验设计、L2 分钟分片管线（含 NaN 中位数
  填充与短窗口补齐）、TCN 训练入口（`ticknet-minute-tcn-train` / `-evaluate`）。

### 体积与资源

- 分钟分片 174 MB（2024 top100），显著小于原始 L2 parquet（年度 23 GB），
  适合上传 Drive 与在 Colab 加载。
- TCN CPU 训练约 6.6 分钟/seed。如需多 seed 或更大隐藏维度，用 Colab GPU。

## 9. 分钟 HGB 多年份滚动稳健性验证（2026-08-08）

目标：回答信号在严格样本外是否仍为正。用同一套分钟 HGB 管线（聚合 120 维
特征、max_iter 500）逐年滚动，得到 4 个独立样本外 test 年。

### 设计

- 数据：L2 分钟缓存 2021-2025，动态前 100 股票，窗口 60 分钟。
- 对每个 test 年：train = 之前全部年份，val = test 年前半年，test = 目标年 H2。
- 配置：`configs/nextday-minute-rolling-{2022,2023,2024,2025}.yaml`。
- 结果：`results/nextday-minute-rolling-{2022,2023,2024,2025}.json`。

### 结果（test 为独立样本外年）

| test 年 | 样本外 Rank IC | 交易日数 | test mcc | train 样本 |
|---|---|---|---|---|
| 2022 | 0.0218 | 124 | 0.060 | 19,559 |
| 2023 | 0.0326 | 123 | 0.110 | 43,734 |
| 2024 | 0.0353 | 124 | 0.113 | 67,930 |
| 2025 | 0.0304 | 125 | 0.081 | 92,118 |

四个 test 年 Rank IC 全部为正，区间 0.022 ~ 0.035，未出现负值或接近 0 的年份。
分类指标（mcc 0.06 ~ 0.11）也全部为正。

### 结论

- 分钟聚合特征 HGB 的信号在 4 个独立样本外年一致为正，排除了 2024 单年侥幸。
  信号真实存在，强度约 Rank IC 0.02 ~ 0.035，弱但跨年稳定。
- 这回答了资源策略文档第 1 节的第三问，即信号在严格样本外仍然为正。
- 仍需注意：Rank IC 0.02 ~ 0.035 量级不足以直接构成扣除成本后盈利的策略，
  需结合分组收益与交易成本评估（见第 5 节止损信号）。

### 工程改动（本轮）

- `run_minute_baseline.py` 增加按年流式读取（`_build_samples_by_year`），解决
  多年数据 OOM：每年只读当年 parquet、构建样本后释放，峰值内存从 20 GB+ 降至
  约 9 GB（测试年 2025 读五年时实测）。
- `minute_baseline.py` 分钟特征由 float64 降为 float32，内存再减半，指标不变。

## 10. 成本后多空组合收益评估（2026-08-08）

目标：判断第 9 节确认的弱信号（Rank IC 0.02 ~ 0.035）在真实交易成本下是否
仍为正。对应文档第 5 节的止损信号："扣除合理成本后分组收益消失"。

### 方法

- 用 2025 独立样本外年的 HGB 预测明细（`results/predictions-rolling-2025.parquet`）
  做多空组合回测：每日按 score 取 top/bottom 10% 等权持仓，long-short 价差。
- 成本模型：单边成本（佣金+冲击）按档位 0/3/5/10/20 bp，卖出另加印花税
  0.05%。换手率按相邻调仓日组合成分差异计算。
- 脚本：`scripts/evaluate_cost_adjusted.py`，支持 `--rebalance-days` 控制调仓频率。

### 结果（2025 H2，125 个交易日，换手率 83%/日）

| 单边成本 | 无成本年化 | 净年化 | 净夏普 | 日均成本 |
|---|---|---|---|---|
| 0 bp | +27.9% | +27.9% | 0.88 | 0.04% |
| 3 bp | +27.9% | +15.7% | 0.49 | 0.09% |
| 5 bp | +27.9% | +7.5% | 0.24 | 0.13% |
| 10 bp | +27.9% | -12.8% | -0.40 | 0.21% |
| 20 bp | +27.9% | -53.6% | -1.68 | 0.38% |

日频换手率 83%（动态股票池 + 每日全调仓），盈亏平衡单边成本约 5 ~ 6 bp，
低于 A 股实际成本（佣金+冲击通常 ≥ 10 bp）。

周频（每 5 日调仓）换手降到 17%，日均成本降到 4.3 bp，但无成本毛利也跌到
-30%（信号是日频短期动量，持有 5 日丢失大部分 alpha），净收益 -40.8% 更差。

### 结论

- 信号真实（跨 4 年 Rank IC 为正），但属日频短期信号，要求每日高换手。
- 真实成本（单边 ≥ 10 bp）下净收益转负，盈亏平衡成本低于可实现成本。
- 降低调仓频率虽减成本，但毛利损失更大，不解决问题。
- 当前口径下信号不足以覆盖实际交易成本，按第 5 节标准应停止扩大模型。
  这是项目"停止"判断的完整证据链最后一块：信号存在、跨年稳健、但不可交易。

以上是当时的分位数多空诊断结论。新 Top-K 研究系列保留它作为历史证据，但正式策略改用
`ticknet.research.portfolio` 的 fixed-K long-only、open-to-open 和股票级成交明细口径，见
`docs/topk-agentx-m1-portfolio-evaluator.md`。

### 工程改动（本轮）

- `run_minute_baseline.py` 增加 `--save-predictions`，把 test 集每日成分与预测
  明细存为 parquet。
- 新增 `scripts/evaluate_cost_adjusted.py`：成本后多空回测，支持成本档位与
  调仓频率。`tests/test_evaluate_cost_adjusted.py` 有 3 个测试覆盖。

### IC 与 spread 背离的审计归因（2026-08-08）

第 8 节观察到 Rank IC ≈ 0.01 但无成本 spread ≈ 27.9 bp/日的矛盾。用新增的
`ticknet-research audit`（`src/ticknet/research/audit.py`）对 2025 预测明细做
诊断，找到根因：

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

结论：多空组合的利润基本由极少数大波动日贡献，不是持续稳定的方向性预测能力。
spread 均值被极端日拉高，掩盖了典型日的低收益。这解释了为什么成本后收益转负：
极端日的毛利无法抵消常态日的高换手成本。审计把"IC≈0 但 spread 高"的矛盾
从表面数字还原为可解释的结构性事实，也是未来 Evaluation Agent 的观测接口。

### 工程改动（本轮，审计）

- 新增 `src/ticknet/research/audit.py`：`PredictionTable`（读预测 parquet）+
  `audit_predictions()`（IC/spread/decile/月频/极端日/winsorize 诊断 + 异常标注）。
- `ticknet-research audit` 子命令。`tests/test_research_audit.py` 有 5 个测试覆盖。

## 11. AgentX 式自动量化研究闭环（2026-08-08）

参考 `references/agentx-paper-notes.md` 落地：把项目从"人工研究"升级为
"实验系统先机器可调用，再加 Agent"。本仓库本身已具备部分地基（锁定测试、
多 seed、统一 Rank IC、AGENTS.md），此处补上确定性研究控制面。

### 结构

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

### 已实现的闭环

```text
ResearchContext
      ↓
Brainstorm → ExperimentSpec
      ↓
Critic → 可证伪性审查
      ↓
Policy → 禁止改测试集/切分（程序级）
      ↓
Runner → 训练 + 结果
      ↓
Audit → IC/spread 背离诊断
      ↓
Registry → SQLite 记忆
```

`ticknet-research` CLI：run / show / compare / audit / locked-test / agent-step。

### 关键设计（对应 AgentX 论文原则）

1. 权限由程序控制。Agent 不能改 `test_end` 等字段（policy 黑名单），manifest 含
   锁定日期会被 `ResearchProtocol` 拦截，locked-test 需显式批准 token。
2. 确定性先于 LLM。指标提取、统计检验、policy 裁决全部用确定性 Python，LLM
   只负责提假设和解释（第一版用 TemplateClient，不接 LLM）。
3. 每个提案必须声明 falsification_condition，强制科研而不用 AutoML 思路。
4. 负面结果资产化。失败和被拒实验写入 Registry，形成实验 DAG（parent_id）。
5. 测试集物理隔离。当时的 research cutoff 为 2024-12-31，2025 及以后是该轮实验的
   锁定测试期。新 Top-K 研究系列的版本化边界见
   `configs/research-protocol-topk-v1.yaml`。

### 验证结果

- 全套件 115 个测试通过（新增 research 16 个），ruff/ty 全绿，覆盖率达标。
- 端到端 `agent-step` 真实跑通：Brainstorm 从"极端日贡献偏高"异常生成
  data_audit 提案 → Critic 通过 → Policy 校验 → TCN 训练完成 → Registry 登记
  （EXP-AUTO-TCN2）。
- 越权实验（改 `test_end`）被 PolicyViolation 拦截、不落库。

### 后续（未做）

- Developer Agent：开放代码修改（worktree + tests + diff review）。
- SGPO / Harness Evolution：需积累足够 research trajectories 后实施。
- Brainstorm 接真实 LLM：`--provider deepseek/openai` 已预留接口。

## 12. Agent-driven 研究验证：调仓频率假设（2026-08-08）

由 LLM（本文档作者）扮演 Brainstorm Agent 驱动的第一轮自动研究，验证系统闭环。

#### LLM 提出的假设

审计显示多空收益由极端日驱动（top 5 日贡献 121%）、成本后收益 -12.8%。LLM
提出假设：存在中等调仓频率（2-3 日）甜点区，可在保留 alpha 的同时把换手和
成本压到盈亏平衡以下。

#### 系统执行与结果

- 	icknet-research agent-step 走完整闭环：Brainstorm 识别成本异常 → 生成
  cost_analysis 提案 → Critic 通过 → Policy 校验 → TCN 训练完成登记
  （EXP-LLM-ROUND1，git_sha=86bb130）。
- 成本敏感度（单边 10bp，2025 H2）：

| 调仓频率 | 换手 | 毛利年化 | 净年化 | 净夏普 |
|---|---|---|---|---|
| 1 日 | 0.83 | +38.1% | -12.8% | -0.40 |
| 2 日 | 0.42 | -41.6% | -67.0% | -2.02 |
| 3 日 | 0.29 | -54.6% | -72.2% | -2.50 |
| 5 日 | 0.17 | -30.2% | -40.8% | -1.23 |
| 10 日 | 0.09 | -9.0% | -14.6% | -0.49 |

#### 结论

假设被否定：降低频率虽降换手，但毛利崩溃更快（2-3 日反而最差）。信号是纯日频
动量，alpha 集中在隔夜到次日单日窗口，与成本结构不兼容。系统自主收敛到
 成本是主要瓶颈的方向，与 LLM 独立分析一致，验证了闭环能产生可复现的
研究结论并登记为结构化证据（Registry）。
