# M2d：Registry 驱动的 ResearchContext

M2d 不再让 `agent-step` 依赖手工传入的 anomaly 和 predictions 参数。Brainstorm 与 Critic 现在消费同一份由 Registry 构造的版本化上下文，其中包含基线、近期实验、失败原因、Evaluation 决策、Audit 异常、parent 关系、可用 executor、预算和数据权限。

这一步建立的是确定性反馈层，不代表已经具备 M8 的多候选研究智能。

## 上下文来源

`ResearchContextBuilder` 只读取 Registry，不扫描代码仓库、行情文件或 locked predictions。上下文包含：

- 当前基线的实验 ID、状态、决策、数据指纹、主要指标均值和全部已登记指标均值。
- 最近实验的 parent、假设、executor、状态、决策、错误和主要指标。
- 状态为 `failed`、`rejected` 或 Evaluation 为 `DISCARD` 的负面结果。
- `audit_anomalies` review 中的异常、来源实验和来源决策。
- 全历史 `novelty_signature`，用于阻止重复实验。
- 已实现 executor、允许的实验类型和本轮算力预算。
- Registry 中已见的数据指纹，以及固定为 false 的 locked-test 访问权限。

自动基线优先选择最新 `KEEP`，没有时选择最新 `EXTEND`。也可以通过 `--baseline-id` 显式指定，但该实验必须已经完成并具有 `KEEP` 或 `EXTEND` 决策。`DISCARD`、失败或仍在运行的实验不能成为可继承基线。

如果来源实验登记了唯一 seed 0 prediction artifact，上下文会保留其 Registry 实验 ID、seed、artifact 名称和路径。模板处理极端收益异常时先走 `export_predictions`，校验 Registry checksum 后再由 Runner 自动执行 Audit。路径本身不扩大权限。所有 prediction 输入仍受日期协议和 locked-test 检查，只有 Registry export 路径提供来源 checksum 绑定。

## 可重放指纹

ResearchContext 使用 schema version 1。所有字段按稳定 JSON 编码后计算 SHA-256，不包含当前时间等非确定性字段。同一个 Registry 状态、问题、基线选择和预算会生成相同指纹。

Orchestrator 在提案登记后写入 `research_context` review，其中保存完整上下文和指纹。Critic review、ExperimentSpec 和最终结果因此可以追溯到同一个输入快照。如果 Brainstorm 修改上下文，编排器会在 reserve 前拒绝本轮执行。未显式传入实验 ID 时，Orchestrator 从 Registry 选择首个未使用的 `EXP-AUTO-NNNN`，CLI 重启不会重新占用 `EXP-AUTO-0001`。

可以直接预览上下文：

```bash
ticknet-research \
  --registry results/registry.sqlite \
  context \
  --question "Top-K 缓冲能否覆盖单边 10bp 成本" \
  --baseline-id EXP-BASE \
  --recent-limit 10 \
  --compute-budget-hours 4
```

运行一轮 Agent 时使用相同参数：

```bash
ticknet-research \
  --registry results/registry.sqlite \
  agent-step \
  --question "Top-K 缓冲能否覆盖单边 10bp 成本" \
  --baseline-id EXP-BASE \
  --recent-limit 10 \
  --compute-budget-hours 4
```

输出包含 `context_fingerprint`。`--anomaly` 和 `--predictions` 不再由 CLI 手工注入，需要进入上下文的证据必须先登记到 Registry。

## Brainstorm 与 Critic 约束

Brainstorm 在模板和 LLM 两条路径上都检查全历史 novelty signature。重复候选会在 experiment reserve 前失败，不产生新的 Registry experiment。模板会跳过已经处理过的 anomaly signature。

Critic 独立执行第二层检查：

- experiment type 必须位于上下文允许动作中。
- executor 必须是当前已实现入口，`train_ranker` 不会被描述为可用。
- novelty signature 不能出现在历史中。
- ExperimentSpec timeout 不能超过上下文算力预算。
- 原有语义路由和 ExperimentSpec 校验继续执行。

Policy、Runner 和 locked approval 仍保留最终权限。ResearchContext 或 LLM 输出不能增加 executor、修改数据日期，也不能授予 locked-test token。

## 当前边界

- Registry 只保存 dataset fingerprint，不能据此推断真实起止日期，因此上下文不伪造已见日期。
- Audit anomaly 当前没有独立的 resolved 生命周期。已见 novelty signature 会阻止完全相同的处理实验重复执行，异常本身仍保留为历史证据。
- M8 的多候选生成、候选排序、相关历史检索和因果链评审尚未实现。
- `train_ranker` 继续推迟到 M4 选定 HGB 和 LambdaMART 口径与固定训练命令。

至此 M2 的确定性闭环完成。下一阶段进入 M3，用已有 prediction artifact 运行 Top-K、buffer 和成本敏感性诊断。
