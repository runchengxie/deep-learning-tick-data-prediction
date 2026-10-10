# External L2 Research Project Comparison

Reading update, 2026-10-10. This record preserves the plans and evidence available at its research stage. References to upcoming seed-0 training, M3 materialization, gradient checks, or label-scale work are historical plans, not the current task list. Later experiments and formal diagnostics are recorded in [Project status](../project-status.md). Follow that entry point for current decisions and locked-period rules.

This page revisits an external project comparison received on 2026-08-19 against TickNet's current code, artifacts, and research protocol. The source offers useful research perspectives but also includes causal explanations that require experiments. Each statement is labeled by evidence type.

## Evidence categories

| Category | Meaning | Use |
|---|---|---|
| Repository fact | Verifiable from TickNet code, configuration, tests, or local artifacts | May be stated as current status |
| External project claim | Description of `l2_foundation`, StyleNet, or ABCM in the comparison | Context for another research path; not independently reproduced locally |
| Mechanism hypothesis | Explanation for metric differences | Test through ablation; do not state as cause before evidence |
| Research decision | Proposed execution order and stop gates based on current evidence | Track in the roadmap under experiment IDs |

This comparison concerns research methods and experiment design, not repository authorship or project ownership. The external code and full artifacts are not in this repository, so those descriptions remain attributed to the source document.

## Two research paths

| Dimension | Current TickNet path | External project's account |
|---|---|---|
| Input representation | Orders, trades, and snapshots merged by time into a causal event stream | Intraday factor sequences supplied to a multitask temporal model |
| Main model | Causal Transformer of about 100M parameters, with HGB and LambdaMART downstream models | StyleNet multitask model and ABCM portfolio model |
| Supervision | Next-event tasks plus continuous H5 excess return; H3 is monitored | Multi-horizon, multi-style, or risk-adjusted return targets |
| Model selection | Daily cross-sectional validation Rank IC | Source emphasizes IC, IR, turnover, and portfolio metrics |
| Portfolio evaluation | Top-K, turnover, fixed/dynamic costs, active return versus equal-weight universe | Source emphasizes signal smoothing, staggered holdings, and risk control |
| Experiment governance | Fingerprints, SHA-256, date permissions, Registry, audit, and locked period | Source mainly describes modeling/portfolio methods; governance was not verified locally |

The approaches can complement each other. TickNet has a more complete raw-event representation, experiment identity, and access boundary. The external document suggests useful tests for label scale, risk attribution, signal half-life, and position execution.

## Existing TickNet evidence

The recent fold trained August–October 2025, validated in November, and used December as OOS. Three seeds of the 100M event-stream model produced H5 validation Rank IC `0.07259 ± 0.02615` and OOS `0.04300 ± 0.01385`; all six values were positive.

On December OOS, combining frozen embeddings and minute features in HGB increased Rank IC from 0.04010 to 0.05701. Three-seed joint end-to-end training achieved OOS Rank IC `0.06398 ± 0.00785`. Top-100 cost-adjusted active return remained negative for both approaches; cross-sectional correlation has not yet converted reliably into tradable top-group returns.

The first additional window, `fold-54-oos-202511`, completed seed 0 using July–September 2025 training, October validation, and November OOS:

| Target | Validation Rank IC | OOS Rank IC | Validation extreme-group spread | OOS extreme-group spread |
|---|---:|---:|---:|---:|
| H5 | 0.08735 | 0.03305 | 0.01780 | -0.00512 |
| H3 monitor | 0.04840 | 0.04231 | 0.00575 | -0.00259 |

H3 and H5 Rank IC were positive on both validation and OOS, so the signal direction recurred in the adjacent window. The November extreme-group spread was negative, showing a gap between whole-cross-section correlation and top-group portfolio return. This supports studying holding periods, signal decay, and top-group ranking before increasing model capacity.

## Review of claims in the source

### Daily target

The event-stream H5 target is continuous stock return minus concurrent CSI All Share return. The same daily label supervises every valid event position. A source-code comment incorrectly described it as a risk-neutral residual-return z-score and has been corrected.

Winsorizing and standardizing the target cross-sectionally each day is a reasonable A/B test because it changes loss weighting across days and stocks. There is no evidence that target scale caused training failure, so this remains a hypothesis.

### Multi-task loss strength

The current loss covers next stream type, order type, continuous event attributes, and daily return. Scalar loss magnitudes do not directly show their impact on a shared backbone. Record task gradient norms and angles on shared parameters to assess whether the daily task is weak or conflicts with event tasks.

### Ranking objective

Standardizing a continuous target with MSE or SmoothL1 changes numerical scale but does not make it equivalent to Spearman Rank IC. A closer daily-ranking objective requires within-day ranks, pairwise loss, or listwise loss while preserving full cross-sectional grouping.

### Risk exposures

Industry, size, liquidity, and volatility attribution is incomplete. A dynamic Top-400 universe and equal-weight benchmark do not by themselves identify a size-style exposure. First measure prediction and return exposures; then decide whether to residualize labels or outputs. Premature neutralization may also remove tradable information.

### Seeds and time windows

Consistent signs across three seeds reduce initialization noise but do not replace validation across time windows. Seed 0 on `fold-54-oos-202511` is the first adjacent-window evidence; stability still needs earlier windows, monthly distributions, and portfolio results.

### Pretraining date boundaries

Label-independent pretraining still observes market structure. Pretraining on all 2021–2025 events and then backtesting an earlier period leaks future structure into that period. Acceptable options include expanding the pretraining cutoff by fold, or explicitly treating 2021–2025 as a fixed base corpus and reserving 2026 onward for true OOS evaluation. 2026 is locked and requires approval before access.

## Hypotheses to test

| ID | Hypothesis | Current clue | Minimum test |
|---|---|---|---|
| H1 | Daily-task gradient on shared backbone is weak or conflicting | Only scalar multitask losses recorded so far | Record task gradient norms and pairwise cosine similarities |
| H2 | Raw return scale overweights high-volatility samples | Daily target is raw continuous return | Compare raw and daily winsorized z labels with identical data, seed, and budget |
| H3 | H5 signal holding period mismatches daily full rebalancing | Positive Rank IC but weak cost-adjusted active return and extreme-group spread | Calculate `IC(1)`–`IC(10)` and entry-date-grouped portfolio returns |
| H4 | Signal mostly reflects industry, size, liquidity, or volatility exposure | Risk inputs still `unavailable` | Run exposure regression and grouped return attribution |
| H5 | Repeating daily-label supervision at every valid position dilutes close-time information | Evaluation uses stock-day scores; training covers all valid positions | Compare all-position, last-position, and tail-weighted supervision |
| H6 | Daily full rebalancing amplifies noise and costs | Candidate one-way turnover is about 50–61% | Compare five staggered H5 cohorts, rank smoothing, and rebalance thresholds |

## Test order

### P0: Reuse existing predictions first

These tests need no new 100M training and first ask how the signal converts into a portfolio:

1. `EVT-HALFLIFE-001`: calculate `IC(1)`–`IC(10)`, extreme-group spread, and entry-date-grouped returns for the recent and `fold-54-oos-202511` folds.
2. `TRD-STAGGERED-H5-001`: compare daily full rebalancing with five staggered cohorts, so only one cohort expires each day.
3. `TRD-RANK-EMA-001`: under the same cost matrix, compare raw rank, rank EMA, absolute entry thresholds, expected-return rebalancing thresholds, and cash.
4. `RISK-ATTR-001`: add industry, size, liquidity, and volatility inputs; report prediction exposures, return attribution, and post-neutralization sensitivity separately.

P0 reports Rank IC, `NDCG@100`, `Precision@100`, Top-K returns, turnover, cost-adjusted active return, monthly win rate, and extreme-date contribution. All strategies share universe, dates, return labels, and cost assumptions.

### P1: Test training mechanisms

1. `EVT-GRAD-AUDIT-001`: record gradient norms and angles for four tasks on fixed batches without changing model parameters or selection rules.
2. `EVT-LABEL-SCALE-001`: run a single-variable comparison of raw returns and daily cross-sectional winsorized z labels.
3. `EVT-SUPERVISION-POSITION-001`: compare supervision at every valid position, only the last position, and tail-weighted positions.
4. Add seeds 1 and 2 and more rolling windows only if seed 0 improves both validation and adjacent OOS.

### P2: Add training complexity only afterward

If P0/P1 show a repeatable gain, evaluate day-batched pairwise/listwise ranking, SWA, staged pretraining, and multi-anchor embeddings. Change one major mechanism per experiment and preserve the 100M baseline's fingerprint, dates, and budget.

Keep `probe150m` deferred. Capacity expansion requires at least one training or portfolio change to improve cost-adjusted active return across multiple windows, without concentrating gains in a few dates or one risk exposure.

## Decision gates

Before a candidate advances, require validation and adjacent OOS to agree in direction; positive active return at 10 bp one-way costs; positive results in most months rather than one best month determining the conclusion; bounded contribution from a small set of extreme dates, with results also reported after removing them; no unexplained large degradation in Rank IC, NDCG, or precision when turnover falls; and risk attribution that separates microstructure signal from common style exposure.

If P0 converts current signals into stable net returns, prioritize execution rules. If P0 keeps failing but P1 finds a clear gradient or label issue, change the training objective. If neither produces a cross-window gain, retain the 100M evidence and stop 150M.

## Related pages

- [Project status](../project-status.md)
- [Model catalog](../model-catalog.md)
- [H5 event-stream rolling roadmap](../nextday/h5-rolling-eventstream-roadmap.md)
- [AgentX research roadmap](topk-agentx-research-roadmap.md)
- [Experiment log](experiment-log.md)

## Appendix: original source

The source document `ticknet_项目对比分析.md`, received on 2026-08-19, is preserved verbatim below. Its SHA-256 is `c9b84597e6e94c2d7d44eaa51079f497cbc31a761fc31aa48c5972f1e4103c27`. Names, numbers, interpretations, and local paths are retained only to trace the external analysis and are not updated with later TickNet work. Current facts and decisions are in the preceding sections, project status, and experiment log.

<details>
<summary>Show original Chinese source</summary>

````text
<!-- archival-source:start -->
# TickNet项目对比分析

> 对象: `deep-learning-tick-data-prediction-main`（下称 ticknet）
> 参照: 本仓 `l2_foundation`（L2FM）、`factors_code/model/`（StyleNet/ABCM）、m4 生产框架
> 日期: 2026-08-19。所有 ticknet 数字取自其 `docs/` 实验记录与源码，本仓数字取自
> `ABCM_改进方案.md`、`StyleNet_ADWM_研报复刻诊断.md` 及 2026-08 实测。

---

## 0. TL;DR

1. **血缘**: ticknet 的 `eventstream` 包是本仓 `l2_foundation` 的直系分支——模型结构、
   80 维事件特征、损失权重 (`ce_stream + 0.5·ce_otype + reg + day`) 逐行同源。他在这个
   底座上补了我们没有的东西: 研究闭环基础设施与成本评估体系。
2. **他强在工程与研究纪律**: 锁定测试集由代码强制（四层拦截 + 一次性内容绑定 token）、
   全链路 SHA-256 指纹、64 组成本矩阵、自动审计异常标注、负结果登记。这些在一台
   i5-4690K + GTX 970 + Colab 的硬件上做出来，纪律性远超本仓的 tmp_ 脚本流。
3. **他卡在信号侧**: 所有链路 RankIC 为正（0.03~0.07）但成本后全军覆没
   （`NO_TRADEABLE_REGION`，盈亏平衡单边成本 ≤4.33bp < 10bp 决策成本）。卡点恰好是
   我们 2026 年在 ABCM/StyleNet 上蹚过的三件事: **标签量纲失衡、风格暴露失控、
   换手-半衰期错配**。§3 给出带推导的修复方案，多数改动他现有基础设施一天内能落地。
4. **反向借鉴**: 本仓应该抄他的实验登记/指纹/审计三件套（§5）。

---

## 1. 项目画像与血缘

### 1.1 四条链路与已落地数字

| 链路 | 输入 | 模型 | 关键数字 (OOS) |
|---|---|---|---|
| 分钟 HGB | 60 分钟窗 120 维聚合 | HistGradientBoosting | 2025H2 RankIC **0.0699**（124 日，ICIR 0.54）；2022-2025 逐年滚动 0.022/0.033/0.035/0.030 全正 |
| raw 盘口 | 14:55 前 200/1000 个十档快照 | 分块 DeepLOB+GRU | 1M/raw-200 最优 val 0.0375±0.0010；**容量主效应 −0.0069**（100M 反而差） |
| L2 事件流 | 尾盘 512 事件 ×80 维 | 因果 Transformer 100M（= L2FM 分支） | val 0.0726±0.026，OOS 0.0430±0.014 |
| 联合端到端 | 512 事件 + 120 分钟特征 | 事件流主干 + 分钟塔 | OOS **0.0640±0.008**；成本后 **−9.67±3.7bp/日** |

中间产物: 冻结 960 维尾盘 embedding + 分钟特征进 HGB，OOS 0.0401→**0.0570**
（配对增量 +0.0169，bootstrap 95% CI [0.006, 0.029]，21 日中 16 日占优）。
这条"表征 + GBDT"混合路线是他目前最干净的正增量，方向与我们
"L2FM embedding 喂下游"的设想一致，且他已做完对照实验。

### 1.2 与 l2_foundation 的差异清单

| 维度 | 本仓 L2FM | ticknet eventstream | 评 |
|---|---|---|---|
| 架构/特征/损失 | 同源 | 同源 + capacity100m (d960×9) | — |
| 窗口 | seq_len 4096 | **512**（Colab 显存约束） | 他被硬件压着 |
| 宇宙 | top-2000，**2025-09 冻结** | top-400，**滞后 20 日成交额逐日动态** | **他更规范**（无幸存者/新鲜度问题） |
| day 标签 | Barra 残差 5d 收益 **z 值** | T+1 open→H 收盘减指数，**原始小数** | 我们对（§3.1/3.2） |
| 训练区间 | 长窗预训练 | 每折仅 3 个月 | 各有约束（§3.7） |
| 折/评估 | 人工 | 3/1/1 月滚动 56 折计划 + purge 契约 + 指纹 | 他更规范 |
| 执行契约 | close_exec（剔涨停） | open→open + can_buy/can_sell 状态机 | 各对一半（§2.3） |

他的 `model.py` 注释仍写着 "day：日级信号（**Barra 风格中性残差收益 z 值**）"——
这是从我们代码继承的注释，**他的实现没有跟上注释**（标签实际只减了中证全指，
无风格中性化、无 z 标准化）。§3.1 与 §3.2 是这一行注释欠下的债。

---

## 2. 他做得好的（值得本仓借鉴）

### 2.1 锁定测试集由代码强制，不靠自觉

`ResearchProtocol` 四层拦截: manifest 最大日期扫描 → 预测 parquet 日期扫描（防止绕过
manifest 走私）→ 配置字段纯白名单（`test_*`/`evaluate_test` 直接列为禁用字段）→
locked test 只能凭一次性 token 执行，token 签发前提是实验 `KEEP`+`release`、
checkpoint 磁盘 SHA-256 与登记值一致，消费是 SQLite 原子单行更新防重放，
**审计失败也消费 token**，防止靠失败信息反复试探。2026 年数据整年锁定，
解锁门槛（≥120 交易日完整对齐）未满足前封存不评估。

对照本仓: 我们的"锁定"是口头纪律。ABCM 筛选阶段我们自己也犯过
在同一段 2026 数据上反复迭代配置的错（后靠 2023-2026 长回测纠偏）。

### 2.2 成本评估是一等公民，结论敢写负

64 组（成本 4 档 × K 4 档 × buffer 4 档）矩阵跑完，正式结论
`NO_TRADEABLE_REGION`: 最优绝对收益组合（K=100, buffer=50）日均净收益 +12.15bp、
净 Sharpe 1.41，但**相对 top-400 等权基准净主动 −4.75bp**；全部组合主动收益
盈亏平衡单边成本 ≤4.33bp。另有独立证据链: 多空组合日换手 83%，0bp 年化 +27.9%、
10bp 年化 −12.8%；**降频到 2/3 日更差**（毛利崩溃快于换手下降）。
审计还原过一次经典陷阱: RankIC 0.030 配 27.9bp/日 spread，top-5 天贡献 121%——
利润全由极端日驱动，自动 flag `tail_return_concentration`。

对照本仓: 我们 2026 年才补上这一课（四模型信号裸日频无成本回测 +74.91%，
逐项加入真实约束: 扣 13bp 成本 → +34.39%，剔涨停不可买 → +6.62%，
降至周频 → −6.27%；在产真实周频记录同期约 +30%）。他从 M1 就把
成本矩阵做成了流水线，且"甜点区"判定有六条硬条件（含 top-5 日贡献 ≤50%、
buffer 有效性验证），比我们的事后拆解系统得多。

### 2.3 交易可行性建模进了组合状态机

`portfolio.py`: 停牌/一字板转成 `can_buy`/`can_sell`，不可卖旧持仓强制保留、
`exit_buffer` 排名缓冲、`min_score_gap` 换仓门槛、不可交易约束下的
bounded equal weights（权重上下界内迭代逼近等权，不靠调权隐式成交）、
持仓收益漂移产生的再平衡交易也计入换手。逐行校验每个 label_date 恰好
400 个 in_universe 候选，调出股票保留状态行不得静默删除。

对照本仓: m4 生产有等价逻辑，但分散在多个脚本里、无合约校验；
他的实现是单一状态机 + schema metadata 合约 + validator，可测试可审计。

### 2.4 研究方法论三件套

- **3 seed 强制**，报均值±标准差，明令不得按测试挑种子;
- **DiD 消融**: 容量×窗口 2×2 固定同一批股票日样本、同一数据指纹，
  difference-in-differences 分解主效应（结论: 容量 −0.0069、窗口 +0.0010、交互 +0.0063——
  100M 无益，干脆暂缓 150M）;
- **负结果登记**: LLM 提出的"降频甜点区"假设被数据否定后原样登记；
  预测文件缺 `can_buy` 列被 validator 拒绝也记为确定性失败边界。

### 2.5 其他值得抄的细节

- **monitor 标签隔离**: H3 标签只监控不参与选型，杜绝"顺手看一眼测试"的污染;
- **实验签名**: resume 时校验配置+数据指纹是否与 checkpoint 一致，PR #80 里
  seed 0 的 SHA-256 误填为 seed 2，加载前被合同拒绝——这类事故我们只能靠肉眼;
- **审计异常自动标注**: `ic_spread_divergence` / `tail_return_concentration` /
  `weak_decile_monotonicity` / `winsorize_sensitivity` 四类 flag 直接写进 review;
- **动态滞后宇宙**: top-400 按 `[t-20, t)` 日均成交额逐日重算（严格不含当日），
  上市 ≥120 交易日——比本仓 L2FM 的"2025-09 冻结 top-2000"干净（后者有
  新鲜度衰减与轻微幸存者偏差，此前 L2FM 分析已指出）。

---

## 3. 我们的思路能帮他的（按优先级，含论证）

### 3.1 P0: day 标签量纲吃掉了两个数量级的梯度 —— 一行修复

**问题**。`eventstream/model.py::compute_loss` 第 195-198 行:

```python
day_err = F.smooth_l1_loss(out["day"], tgt_day[...], reduction="none")   # beta=1.0
total = ce_stream + 0.5 * ce_otype + 1.0 * reg_loss + 1.0 * day_loss
```

`tgt_day` 是**原始小数超额收益**（H5 截面标准差 σ_y ≈ 0.03~0.05），无 z 标准化。
SmoothL1 在 |x|<β=1 区域退化为 0.5x²，故 day 头收敛到均值附近时

\[
\mathcal{L}_{day} \approx \tfrac{1}{2}\,\mathrm{Var}(y) \approx \tfrac{1}{2}(0.04)^2 = 8\times10^{-4},
\]

而 `ce_stream` 初始 ≈ ln4 ≈ 1.39（训练后 ~0.5），`reg` 头目标（price_bps/dt_log/qty_log）
量级 O(1)，损失 O(0.1)。四项直接相加意味着 **day 任务在总损失里的份额约 0.1%**——
名义权重 1.0，有效权重被标签方差缩掉两个数量级以上。主干几乎完全被生成式任务塑形，
day 头只是在冻结表征上学了个线性读出。

**他自己的数据佐证**: 联合微调 3 seed 的 best epoch 是 2/1/1——day 方向的可学信号
一两个 epoch 就"榨干"了，之后 val IC 回落，这是梯度供给不足而非容量不足的典型形态；
冻结 embedding + HGB 反而拿到最稳的正增量（+0.0169），同样说明表征里有 day 信息
但端到端训练没有把它送进主干。

**修复**（与本仓 L2FM 的现行做法一致）: 标签生成时逐日截面 z 标准化
（先 5×MAD winsorize 再 z，见本仓 `tmp_build_barra_resid_label.py` 协议），
使 Var(y)=1，day 项瞬间回到与 ce/reg 同量级。改动只在
`horizon_labels.py` 的 sidecar 生成一处，模型代码零改动。z 标准化不改变
逐日 Spearman IC 的评估口径（秩不变）。
**预期**: 这是他所有改动里性价比最高的一个。可用他自己的 3-seed 协议直接证伪:
若 z 化后 H5 val RankIC 均值未升，即弃。

### 3.2 P0: 风格中性化 —— 他的成本失败很可能一半是风险暴露

**现状**。他的标签只减中证全指（市场中性），**不做风格/行业中性**；文档自己承认
"行业、规模、波动率、流动性风险暴露输入尚未提供，结果显式标记 `unavailable`"。

**为什么这直接连到他的成本困局**。设模型打分 \(s = \alpha + \sum_k \beta_k F_k\)
（F_k 为风格暴露）。Top-K 选择继承全部 β 倾斜，组合主动收益方差

\[
\mathrm{Var}(r_p - r_b) = \boldsymbol{\beta}^\top \Sigma_F \boldsymbol{\beta} + \sigma_\alpha^2,
\]

风格项不贡献可预期的 α 均值（他的 IC 里混着风格动量的顺周期部分），却把分母撑大、
把收益集中到少数风格暴发日。他审计里的 **top-5 天贡献 121%**、
Precision@100 在联合模型上反而下降（0.268→0.239）、月度主动六个月仅一正——
全部是"排名头部被风格倾斜占据"的指纹。风格倾斜还推高换手:
风格因子日间自相关低于真 α，追着风格调仓就是白付成本。

**我们的实证**（同为 A 股截面、可迁移）:

| 案例 | 措施 | 效果 |
|---|---|---|
| StyleNet r2 (2026, 139 日) | 输出对 9 风格逐日 OLS 残差化 | IC 3.96%→4.58%（**ΔIC +0.63%, t=4.67**），IR 9.6→11.1 |
| ABCM (2023-2026) | 同上 | ICIR 0.63→0.77，2026 年符号翻转消失 |
| StyleNet 标签已 Barra 中性 | 仍测得输出端风格 R² 26.9% | **标签中性化约束不住输出端**，两端都要做 |

**给他的落地路径**（他没有 Barra 数据也能做）:
1. **零成本起步**: 输出端事后残差化——逐日把预测秩 z 对 {log(20 日 ADV)、
   1/6/12 月动量、20 日已实现波动、行业哑变量(申万或按相关性聚类)} 回归取残差。
   四个代理因子他的日频宽表（2016-2026 open/close/volume 已就位）当天就能算。
   预期量级参照我们: ΔIC +0.3~0.6%，且 top-K 组合的极端日集中度应显著下降
   （正好用他的 `tail_return_concentration` flag 验证）。
2. **进阶**: 标签侧同样残差化后再 z（等价本仓 `label_barra_resid` 的简化版）。
   注意我们的教训: **两端都做**，且逐对相关惩罚压不住子空间暴露
   （ABCM §8 结论），直接投影残差化比加惩罚项干净。

### 3.3 P1: 训练目标与评估指标失配 —— 截面结构没进损失

他的评估是逐日截面 RankIC，但训练损失全是**逐样本**的:
nextday 链是 `CE(三分类) + 0.5·SmoothL1`，回归目标用**训练集全局** mean/std 标准化；
eventstream 链如 §3.1。同日截面比较结构（谁比谁强）只存在于标签生成
（分位打标）和评估两端，模型中间全靠自己悟。

我们在 ABCM 上的实测: 预测与标签同为截面 z 时 \(\mathrm{MSE} = 2(1-\mathrm{IC})\)，
即 MSE 直接等价 IC 损失；**不做最后一层截面标准化时，MSE 存在"把输出缩向 0"
的退化解**（实测 mse 长期卡 0.98、alpha 头几乎不学习——ABCM 踩过的原坑）。

**建议**: day-batch 组织（同日股票同 batch）+ 输出与标签同做截面 z + MSE，
或保守起步——先只把回归目标从"全局标准化"改为"逐日截面 z"（数据侧一行改动，
兼容他现有 per-sample pipeline）。他的分钟物化缓存本来就带 (day, symbol) 索引，
按日重组 batch 成本低。

### 3.4 P1: checkpoint 选择的 winner's curse —— 他已自我声明，给个量化与解法

他所有 checkpoint 按单一验证月的 argmax RankIC 选择，文档诚实标注
"带选模乐观偏差"。量化一下: 从 E 个 epoch 里挑 max，若逐 epoch 验证 IC 含
独立噪声 σ_e，则

\[
\mathbb{E}[\max_k \widehat{IC}_k] \approx \mu + \sigma_e\sqrt{2\ln E}.
\]

他的验证只有 **1 个月 ≈21 日**。日 IC 标准差取 σ≈0.13（他 HGB 链
ICIR 0.54 与 IC 0.070 隐含值），σ_e ≈ 0.13/√21 ≈ 0.028；epoch 间高度相关，
有效独立选择次数取 E_eff≈2~4 ⇒ 乐观偏差 ≈ σ_e·√(2 ln E_eff) ≈ **+0.03~0.05**——
与他 val 0.0726 → OOS 0.0430 的落差（−0.030）同量级。
即 val/OOS 落差可能大部分不是过拟合数据，而是**选择效应**，容量与正则都不背锅。

**建议**（本仓 L2FM 分析同款处方）: ① 验证期加长（他 56 折计划里可用相邻 2-3 折
联合验证）；② 权重平均替代 argmax（last-k SWA/EMA，消掉 σ_e 的 max 抬升）；
③ 用 H3 monitor 与 H5 的**一致性**做平选（他 monitor 基础设施是现成的，
只差把它用进选型规则）。

### 3.5 P1: 成本困局的出路是持有期 × 半衰期对齐，不是降频

他试过降频: 1/2/3/5/10 日调仓净年化 −12.8%/−67%/−72%/−41%/−15%，2-3 日最差。
这个结果形态我们很熟——**信号是短半衰期的，持有期拉长毛利先崩，
换手后降**。设信号对 τ 日后收益的 IC 衰减为 IC(τ)，H 日调仓的日均毛利
∝ (1/H)Σ_{τ≤H} IC(τ)·σ，换手 ∝ 1/H: 当 IC(τ) 半衰期 ≪ H 时毛利掉得比成本快，
净收益在中间 H 出现谷底——正是他 2-3 日最差的形状。

**两条正路**（对应我们生产里实际走通的）:
1. **换长半衰期标签重训，而不是拿短标签降频**。他主目标已是 H5，但降频实验的
   谷底形态提示模型学到的可能仍以隔夜/日内成分为主——先测 IC(τ) 谱验证:
   对 τ=1..10 逐日算已训模型对"第 τ 日单日收益"的 IC 衰减曲线（本仓
   `tmp_abcm_paired.py` 的逐日配对框架可直接套）。目标: 找到 IC(τ) 平台期
   ≥ 持有期的配置再谈组合。
2. **信号平滑 + 缓冲带在信号层做，不在组合层硬降频**: 分数 EMA
   （\(\tilde s_t = \lambda \tilde s_{t-1} + (1-\lambda)s_t\)) 直接提升自相关、
   压换手，我们 ADWM/生产侧实测换手减半而 IC 损失 <10%；他的
   `exit_buffer`/`min_score_gap` 状态机已经是现成的第二级。
   参照系: 本仓 m4 生产（周频持仓 + 日频 overlay，四模型合成）2026 实盘
   超额约 +30%，而同一信号的裸日频无成本回测虚报 +74.91%（见 §2.2 分解）——
   信号过成本线靠的是周频 + 缓冲带，不是更高的裸 IC。先在他 10bp 口径下
   把 EMA λ 扫一遍（他 64 格矩阵加一维就行），比继续调 K/buffer 有望得多。

另外一个组合层细节: 他的基准是 top-400 等权（已减指数后的二次超额），
这是个**很强的小市值/流动性基准**——若叠加 §3.2 的风格中性化，主动收益的
分母（风格方差）会显著缩小，同样的 α 更容易过成本线。

### 3.6 P2: 种子集成用秩，不用分数

他的 horizon 评估把 3 seed 的**原始分数**等权平均。不同 seed 的分数尺度/
分布形状无对齐保证，均值会被大尺度 seed 主导。本仓协议: 逐日先把各 seed
预测转截面秩（pct rank）再平均（`stylenet_rolling.rank_ensemble`），
等价 Copula 化后聚合，对单 seed 尺度漂移免疫。改动约五行。
（他冻结 embedding 线里"每 checkpoint 各训下游再平均预测"的做法方向已对，
补秩变换即可。）

### 3.7 P2: 预训练数据量 vs 容量——把 56 折的数据用在一次预训练上

他的 DiD 已证明容量↑无益（100M 差于 1M）。按 Chinchilla 直觉这不是容量的错，
是**数据的错**: 每折只用 3 个月 ≈ 1.2 亿事件 token，喂 100M 参数远欠拟合区。
但他折内数据被 fold 协议限死。解法是我们 L2FM 的分工:
**生成式任务（stream/otype/reg）没有标签泄漏问题**，可以在 2021-2025 全量
（56 折并集，~1.5T 原始/300GB packed）上预训练一次主干；day 头再按折微调。
预训练不看任何收益标签，不违反他的锁定契约（值得在他的 contract 里补一条
显式豁免）。他的 Colab 约束下可行性: 512 窗×分片流式他已趟通，
预训练一次 ~几十 GPU 时（A100），比 56 折各自从头训省一个量级。

### 3.8 小项

- **embargo**: 他的 purge（信号/建仓/收益终点同区间）等价隐式 H−1 天 gap，
  但相邻区间零间隔，特征窗口若未来引入多日聚合会静默泄漏——建议加显式
  embargo 参数（哪怕默认 0），把假设写进合约。
- **信号日停牌股进入标签集**（诊断链）: `labels.py` 提供 universe 时不查信号日
  bar 存在性。正式链有 can_buy 兜底，诊断链的 IC 会被不可交易样本轻微美化，
  建议诊断链同样过滤——对应我们 ABCM 的教训（宇宙过滤 +0.67% 真实 IC）。
- **三分类的 NEUTRAL 60% 太宽**: 20/80 分位打标下中性类占 60%，CE 的
  balanced 权重把 2/3 的梯度花在"判断是否平庸"上。若保留分类头，
  建议 30/70 或直接砍掉分类头（他的回归头本来就是排序主力）。

---

## 4. 给他的落地清单（按性价比）

| # | 改动 | 工作量 | 预期 | 证伪条件（按他的 gate 语言） |
|---|---|---|---|---|
| 1 | day 标签逐日 winsorize+z（§3.1） | 半天 | 端到端 day 学习真正启动 | 3-seed H5 val IC 均值不升即弃 |
| 2 | 输出端风格残差化（§3.2） | 一天 | ΔIC +0.3~0.6%，极端日集中度↓ | 配对 ΔIC t<2 即弃 |
| 3 | 秩集成替代分数集成（§3.6） | 五行 | 集成方差↓ | 无 |
| 4 | IC(τ) 衰减谱测量（§3.5 前置） | 半天 | 决定 5/6 走向 | 纯测量 |
| 5 | 分数 EMA 进 64 格矩阵（§3.5） | 一天 | 换手减半，博净主动转正 | 净主动仍 <0 即回退 |
| 6 | SWA/双 horizon 选型（§3.4） | 一天 | val→OOS 落差收窄 ~0.02 | OOS 不升即弃 |
| 7 | 标签侧风格中性 + 截面损失（§3.2/3.3） | 一周 | 结构性提升 | 同 2 |
| 8 | 全量生成式预训练 + 折内微调（§3.7） | 数周 | 容量真正兑现 | 同折对照不升即弃 |

1+2+3 合计两天工作量，全部可用他现有 3-seed + bootstrap 协议裁决，
且不触碰任何锁定数据。

## 5. 反向借鉴清单（本仓该抄他的）

1. **实验登记 + 数据指纹**: 本仓 tmp_ 实验全靠目录名和聊天记录追溯，至少给
   `factor_results/` 每组产物落一个 `manifest.json`（配置、数据 SHA、git rev）;
2. **成本矩阵流水线**: 把 m4 的成本/缓冲带逻辑抽成他这种 64 格 sweep +
   盈亏平衡成本反解，新信号准入前必跑（ABCM 若早有此关卡，74pp 水分不会晚发现半年）;
3. **审计异常四 flag**: 尤其 `tail_return_concentration`（top-N 日贡献占比），
   加进本仓 `tmp_abcm_compare.py` 一类的评估脚本;
4. **monitor 标签隔离**: 我们选型和监控用同一批指标，隐性污染与他划清的做法比高下立判;
5. **动态滞后宇宙**: L2FM 的冻结 top-2000 应改为逐日滞后 ADV 重算（他的实现可直接参考）。

## 6. 附录: 关键事实核对

- 血缘: `diff l2_foundation/model.py ticknet/.../eventstream/model.py` 差异仅为
  中文化注释、类型注解、新增 capacity100m/smoke 配置与命名规范化；损失函数逐字符同构。
- 他的数据: L2 原始 ~4TB（2021-2025 快照 884GiB + order 1.2TB + trades 1.5TB），
  2021-01~06 沪市委托缺失（101 个交易日仅深市），正式配置从 2021-07 起。
- 他的硬件: i5-4690K/31GiB/GTX 970（本地不可训），训练全部 Colab T4/A100，
  Google Drive 200GiB 上限决定了 512 窗口 + 物化缓存的设计。
- 本仓引用数字出处: StyleNet 残差化 +0.63%(t=4.67) 为 2026-08-19 实测
  （`tmp_stylenet_residualize.py`，139 日）; ABCM 数字见 `ABCM_改进方案.md` §7-9;
  m4 水分分解（74.91% → 34.39% → 6.62% → −6.27%，对应成本/涨停/调仓频率
  三项约 40/28/13pp）与在产真实 ~+30% 见 2026-08 会话实测。
<!-- archival-source:end -->
````

</details>
