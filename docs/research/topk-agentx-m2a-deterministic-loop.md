# M2a：AgentX 确定性实验闭环

## 本阶段完成了什么

M2a 把原先提案里携带任意入口、Runner 统一启动训练的结构，改成受控闭环：

```text
ExperimentSpec v2
  → reserve experiment
  → Critic / Policy
  → typed executor
  → artifact contract
  → prediction Audit
  → Registry v2
  → metric gates
  → KEEP / EXTEND / DISCARD
```

它解决的是实验语义与证据可信度，不代表模型已经获得新的收益结论。本文记录 M2a 完成时的状态。后续 M2b、M2c 和 M2d 已经完成，当前总进度见[研究路线总览](topk-agentx-research-roadmap.md)。

## ExperimentSpec v2

Spec 必须声明 `executor`、`inputs`、`primary_metrics`、结构化 `success_gates`、`artifact_contract`、`budget`、`parent_id`、`novelty_signature` 和 `stage`。YAML 与 LLM JSON 统一走严格解析，未知字段和旧版 `entry_point` 会直接失败。

示例：

```yaml
hypothesis: Top-50 缓冲能在 10bp 成本下保留正净收益
objective: 对现有预测运行固定组合成本扫描
experiment_type: cost_analysis
executor: topk_cost_sweep
inputs:
  predictions_path: results/predictions.parquet
  top_k: [50]
  exit_buffer: [20]
  cost_bps: [10]
seeds: [0]
primary_metrics:
  - topk.k50.buffer20.cost10.net.sharpe
success_gates:
  - metric: topk.k50.buffer20.cost10.net.sharpe
    operator: gt
    threshold: 0.0
artifact_contract:
  - resolved_spec
  - resolved_config
  - stdout
  - stderr
  - result
  - run_manifest
  - topk_sweep
budget:
  timeout_seconds: 600
  max_seeds: 1
rationale: 先验证组合规则，不重新训练模型
falsification_condition: 单边 10bp 下净 Sharpe 不大于 0 则否定
novelty_signature: top50-buffer20-cost10-v1
stage: screening
```

## 执行器

M2a 发布时可以运行：

- `train_nextday`：固定调用次日模型训练入口。
- `train_minute_tcn`：固定调用分钟 TCN 训练入口。
- `audit_predictions`：只审计预测文件，不启动训练。
- `topk_cost_sweep`：复用 M1 的 `evaluate_topk_portfolio()`，扫描 K、buffer 和成本。
- `compare_experiments`：从 Registry 读取已登记指标并生成基础对比 artifact。

后续 M2c 已经实现 `import_predictions`、`export_predictions`、`walk_forward_robustness` 和更完整的 `compare_experiments`。`train_ranker` 仍会明确报告尚未实现。当前执行器清单由 `ticknet.research.spec` 和 `ticknet.research.executors` 共同约束。

## Artifact 与 Registry v2

Runner 在 Critic 与 Policy 执行前 reserve 实验 ID。每个 experiment 和 seed 使用独立目录，训练 executor 会注入独立 checkpoint 目录与名称。核心 artifact 包括 resolved spec 和 config、环境与 git 状态、stdout、stderr、result、run manifest，以及 executor 声明的预测、checkpoint、Audit 或成本明细。

Registry 对 experiment、run、metric、review 和 artifact 实施唯一性约束，父实验必须存在。嵌套数值指标递归展开为点号路径，artifact 用流式 SHA-256 和文件大小登记。被策略拒绝、超时或执行失败的实验同样保留终态和原因。已有 artifact 目录或 experiment ID 时拒绝覆盖。

## 强制 Audit、门槛和 locked 边界

训练 executor 一旦返回 predictions artifact，Runner 会先检查预测日期没有进入协议 locked 区间，再强制运行预测 Audit。显式 `audit_predictions` 和 `topk_cost_sweep` 输入也走同一协议检查。Audit 嵌套指标和异常写回 Registry。

Evaluation 对每个 gate 计算多 seed 均值。缺失指标或任一 gate 失败为 `DISCARD`，screening 全部通过为 `EXTEND`，robustness 和 release 全部通过为 `KEEP`。自然语言 `falsification_condition` 继续用于解释，实际裁决只使用结构化 gates。

M2a 发布时 `APPROVED` 仍是静态口令，这一历史限制已经由 M2b 的内容绑定一次性批准替换。当前流程见 [topk-agentx-m2b-locked-approval.md](topk-agentx-m2b-locked-approval.md)。

## 运行与验收

```bash
ticknet-research run \
  --spec path/to/experiment.yaml \
  --id EXP-TOPK-001

ticknet-research show --id EXP-TOPK-001
ticknet-research compare --ids EXP-BASE-001 EXP-TOPK-001
```

合成端到端测试覆盖：成本分析不启动训练、训练预测自动 Audit、locked predictions 拒绝、递归指标与 artifact 登记、KEEP、EXTEND、DISCARD、重复 ID、任意入口和 artifact 冲突。

## 后续完成情况

1. M2b 完成内容绑定的一次性锁定测试批准。
2. M2c 完成预测导入导出、多 seed 对比和 walk-forward 汇总。
3. M2d 完成 Registry 到 ResearchContext 的确定性构造与回放。
4. `train_ranker` 的固定入口仍待实现，当前研究阶段不依赖它。
