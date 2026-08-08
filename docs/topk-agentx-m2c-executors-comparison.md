# M2c：Prediction 导出、实验比较与滚动稳健性

M2c 补齐三个确定性 executor：从 Registry 物化已登记预测、比较多 seed 实验，以及把多个
已完成实验聚合为 walk-forward 稳健性证据。它们只消费已有 artifact 和指标，不重新训练，
也不会扩大 locked 数据权限。

## Prediction export 的边界

`export_predictions` 不是任意 checkpoint 的通用推理入口。不同模型的输入、checkpoint 和
推理命令尚未形成统一契约，因此当前实现只允许物化 Registry 中已经登记的 prediction
artifact：

1. `source_experiment_id` 必须存在并处于 `completed`、`frozen` 或 `locked_tested`。
2. 指定 seed 和名称的 artifact 必须唯一，文件 SHA-256 必须与 Registry 一致。
3. Parquet 必须包含 `symbol`、`trading_date`、`label_date`、`target_return` 和 `score`。
4. 文件按原始字节复制到本次实验的独立 seed 目录，再次核对 SHA-256。
5. Runner 对复制后的 predictions 重新执行日期协议检查和 Audit，不能借导出绕过 locked
   test。

最小输入如下：

```yaml
experiment_type: prediction_export
executor: export_predictions
inputs:
  source_experiment_id: EXP-SOURCE
  source_seed: 0
  artifact_name: predictions
seeds: [0]
```

真正的 checkpoint 推理仍由固定训练/推理 executor 负责；等 M4 选定排序模型库和统一命令后，
再决定是否增加模型特定的导出入口。

## 多 seed 实验比较

`compare_experiments` 从 Registry 读取至少两个已完成实验，对每个声明指标输出：

- 各 seed 原值、均值、样本标准差、最小值和最大值；
- 相对 baseline 的原始均值差 `delta_vs_baseline_mean`；
- 按指标方向归一、正值恒表示更好的 `improvement_vs_baseline_mean`；
- 同 seed 配对数量、原始配对差和方向归一后的配对改善。

默认还要求所有待比较实验具有同一个非空 dataset fingerprint；跨时间窗口的实验应使用
`walk_forward_robustness`。确有诊断需求时可以显式设置 `require_same_fingerprint: false`，
此时输出仍保留每个来源的数据指纹，不能把结果解释为受控消融。

指标默认 `higher`，Brier、误差、回撤等越低越好的指标必须显式声明：

```yaml
experiment_type: comparison
executor: compare_experiments
inputs:
  experiment_ids: [EXP-BASE, EXP-CANDIDATE]
  baseline_id: EXP-BASE
  metrics:
    - validation.daily_rank_ic_mean
    - validation.brier_score
  metric_directions:
    validation.brier_score: lower
seeds: [0]
```

CLI 也支持同一语义：

```bash
ticknet-research compare \
  --ids EXP-BASE EXP-CANDIDATE \
  --baseline EXP-BASE \
  --metrics validation.daily_rank_ic_mean validation.brier_score \
  --lower-is-better validation.brier_score
```

输出保存为 `comparison.json`，并保留每个来源的状态、Evaluation 决策和数据指纹。结果自身的
dataset fingerprint 是来源实验 ID 与各自数据指纹的稳定聚合哈希，不能伪装成某一个来源实验。

## Walk-forward 稳健性

`walk_forward_robustness` 把每个来源实验视为一个时间窗口，先在窗口内汇总 seed，再跨窗口
报告均值、样本标准差、最小值、最大值和最差窗口。默认要求至少三个窗口且每个窗口具有不同、
非空的 dataset fingerprint，避免把同一切分重复登记后冒充滚动验证。

最差窗口尊重指标方向：Rank IC 取窗口均值最小者，Brier 取窗口均值最大者。用于 gate 时，
高优指标一般约束 `window_min`，低优指标一般约束 `window_max`。例如：

```yaml
experiment_type: robustness
executor: walk_forward_robustness
inputs:
  experiment_ids: [EXP-W22, EXP-W23, EXP-W24]
  metrics:
    - validation.daily_rank_ic_mean
    - validation.brier_score
  metric_directions:
    validation.brier_score: lower
  minimum_windows: 3
seeds: [0]
```

输出保存为 `walk-forward.json`。本 executor 只聚合已登记窗口，不自行生成时间切分；窗口边界、
purge 和标签协议仍由来源训练实验负责。

## 安全失败与当前剩余项

未知实验、未完成实验、缺指标、重复 ID、错误指标方向、窗口不足、重复数据指纹、artifact
缺失或 checksum 改变都会确定性失败，并由 Runner 记录失败 run。`train_ranker` 仍显式不支持，
不会回退为其他训练入口；其固定实现推迟到 M4 的 HGB/LambdaMART 选择完成后。

M2 的下一项是从 Registry 构建 ResearchContext，让 Audit 异常、失败原因和历史决策自动进入
下一轮 Agent 提案。
