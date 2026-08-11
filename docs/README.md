# 文档索引

本目录按主题分成三个子目录，加上复现与实验产物。每一篇都写清命令、口径和结论，方便直接引用。先看阅读路径，再按需取用。

## 阅读路径

新人从根目录 README 了解项目全貌后，按下面的顺序读，可以走通从数据到结论的完整链条。

1. [nextday/cross-sectional-prediction.md](nextday/cross-sectional-prediction.md) 定义主链路的口径，包括样本、标签、日期切分、训练评估入口和指标清单
2. [nextday/raw-200-end-to-end-pipeline.md](nextday/raw-200-end-to-end-pipeline.md) 给出从本地加工到 Colab 训练的操作顺序
3. [nextday/eventstream.md](nextday/eventstream.md) 说明 L2 逐笔事件流这条新增主线
4. [research/topk-agentx-research-roadmap.md](research/topk-agentx-research-roadmap.md) 汇总当前真实结论和 M0 到 M9 的执行状态
5. [research/topk-agentx-m0-research-contract.md](research/topk-agentx-m0-research-contract.md) 说明 2025 已见、2026 封存的数据边界，跑任何实验前都要看
6. [research/topk-agentx-m3-topk-diagnostics.md](research/topk-agentx-m3-topk-diagnostics.md) 记录正在推进的正式 Top-K 诊断进度
7. [dev/development-guide.md](dev/development-guide.md) 说明模块边界、测试范围和门禁命令

## 目录

### nextday

A 股次日横截面预测主线，覆盖样本口径、数据加工、训练入口和分阶段路线。

| 文档 | 内容 |
|---|---|
| [cross-sectional-prediction.md](nextday/cross-sectional-prediction.md) | 主链路总规范，数据格式、适配、切分、训练、评估 |
| [raw-200-end-to-end-pipeline.md](nextday/raw-200-end-to-end-pipeline.md) | 200 tick 原始盘口端到端操作流程 |
| [eventstream.md](nextday/eventstream.md) | L2 逐笔事件流无损打包、因果 Transformer 与预测导出 |
| [raw-data-expansion-roadmap.md](nextday/raw-data-expansion-roadmap.md) | 五年 raw-200 数据生成、审计、上传和扩展门槛 |
| [multi-horizon-data-expansion-roadmap.md](nextday/multi-horizon-data-expansion-roadmap.md) | 1/3/5 日标签、容量门槛、raw-1000 与全天 tick 路线 |
| [nextday-100m-raw1000-benchmark.md](nextday/nextday-100m-raw1000-benchmark.md) | 100M 参数容量基准与 A100 batch sweep 实测快照 |
| [h5-rolling-eventstream-roadmap.md](nextday/h5-rolling-eventstream-roadmap.md) | H5 Rank IC 主目标、3/1/1 滚动协议与全天事件数据 pilot |
| [hardware-constraints-and-experiment-roadmap.md](nextday/hardware-constraints-and-experiment-roadmap.md) | 硬件约束、统一研究口径和分阶段路线 |

### research

Top-K 可交易组合与 AgentX 自动量化研究闭环。

| 文档 | 内容 |
|---|---|
| [topk-agentx-research-roadmap.md](research/topk-agentx-research-roadmap.md) | 研究路线总览，真实结论、统一原则和 M0 到 M9 状态 |
| [topk-agentx-m0-research-contract.md](research/topk-agentx-m0-research-contract.md) | 研究契约，数据权限审计和交易口径 |
| [topk-agentx-m1-portfolio-evaluator.md](research/topk-agentx-m1-portfolio-evaluator.md) | Top-K long-only 组合评估内核的输入契约和成本公式 |
| [topk-agentx-m2a-deterministic-loop.md](research/topk-agentx-m2a-deterministic-loop.md) | ExperimentSpec v2 与确定性实验闭环 |
| [topk-agentx-m2b-locked-approval.md](research/topk-agentx-m2b-locked-approval.md) | locked test 的一次性人工签发与受控消费 |
| [topk-agentx-m2c-executors-comparison.md](research/topk-agentx-m2c-executors-comparison.md) | prediction 导出、多 seed 对比与 walk-forward 稳健性 |
| [topk-agentx-m2d-registry-context.md](research/topk-agentx-m2d-registry-context.md) | 由 Registry 构造的可重放 ResearchContext |
| [topk-agentx-m3-topk-diagnostics.md](research/topk-agentx-m3-topk-diagnostics.md) | 正式 Top-K 成本诊断的进度与判定门槛 |
| [resource-strategy-and-pilot-gates.md](research/resource-strategy-and-pilot-gates.md) | 有限算力下的资源策略与门槛式实验原则 |
| [experiment-log.md](research/experiment-log.md) | 带日期的历史实验记录，含 TCN 对比、滚动验证、成本评估、审计归因与 Agent 闭环 |

### dev

开发维护与运行基础设施。

| 文档 | 内容 |
|---|---|
| [development-guide.md](dev/development-guide.md) | 模块划分、测试范围、质量门禁与依赖管理 |
| [colab-cli-automation.md](dev/colab-cli-automation.md) | Colab 无人值守训练与评估的自动化入口 |

### 复现与产物

| 路径 | 内容 |
|---|---|
| [reproduction-audit.md](reproduction-audit.md) | DeepLOB 在 FI-2010 上的复现核对，对象已归档到 legacy |
| [baselines/topk-agentx-v1.json](baselines/topk-agentx-v1.json) | M0 冻结的基线 artifact 清单，含 SHA-256 与指标，冻结后不再更新 |
| [reports/multi-horizon-decision-2026-08-10](reports/multi-horizon-decision-2026-08-10/source-inspection.md) | 2026-08-10 多周期决策的历史快照，结论已被 multi-horizon 路线吸收 |

## 数据结论的存放约定

同一条实验结果会在多处出现，因为不同的文档面向不同的阅读场景。维护时遵循下面的约定，可以避免数值漂移。

- 实验结论只在三处保留完整记录，其他文档用一句话摘要并加链接
- [research/topk-agentx-research-roadmap.md](research/topk-agentx-research-roadmap.md) 的当前证据一节收录最新真实结论
- [nextday/multi-horizon-data-expansion-roadmap.md](nextday/multi-horizon-data-expansion-roadmap.md) 收录多周期与容量实验
- [research/experiment-log.md](research/experiment-log.md) 按日期收录带完整数字的历史实验记录

涉及资源容量、磁盘占用、Drive 套餐这类随时间变化的数字，写明年份并放在带日期的章节里，例如 2026-08-09 的资源盘点。历史产物（`reports/`、`baselines/`）是冻结快照，改动会破坏可核对性，不更新，只保留。

## 写作规范

中文正文用中文标点，保留命令、配置名、模块名和指标名的行内代码。不用双引号、加粗、分号、破折号和先否定再转折的句式。改动文档里的命令时，先用对应的 `--help` 核对参数名，再落笔。
