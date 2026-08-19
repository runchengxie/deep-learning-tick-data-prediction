# 事件流多任务梯度审计

`EVT-GRAD-AUDIT-001` 用来判断日级收益任务能否有效更新事件流 Transformer 的共享主干。审计结果将决定下一项训练实验，避免同时改动标签尺度、监督位置和任务权重后无法解释增量来源。

## 要回答的问题

事件流模型同时学习下一事件流类型、下一订单类型、连续值回归和 H5 日级收益。当前日级损失在每个有效事件位置都参与训练，四项任务共用 Transformer 主干。审计主要回答两个问题：

1. 日级任务传到共享主干的梯度是否明显偏弱
2. 日级任务与三个生成任务的梯度方向是否持续冲突

梯度审计只说明当前训练机制中的更新强度和方向。它不代替 Rank IC、NDCG、Precision 或成本后收益评估。

## 固定口径

正式审计使用最近折和相邻折的 seed 0：

| 滚动折 | 审计分区 | 模型状态 |
|---|---|---|
| 最近折，2025 年 11 月 validation | validation | seed 0 初始化、seed 0 best checkpoint |
| `fold-54-oos-202511`，2025 年 10 月 validation | validation | seed 0 初始化、seed 0 best checkpoint |

每个滚动折从 validation 分区等距选择 16 个固定 batch，每个 batch 8 个样本，共 128 个样本。审计保存样本下标、数据集指纹和完整 batch 张量指纹。相同输入、权重和源码应产生相同结果。

审计只计算共享主干参数，排除 `head_stream`、`head_otype`、`head_reg` 和 `head_day`。四项损失沿用正式训练权重：

| 任务 | 内容 | 权重 |
|---|---|---:|
| `stream` | 下一事件流类型交叉熵 | 1.0 |
| `otype` | 下一订单类型交叉熵 | 0.5 |
| `reg` | 下一事件连续值 Smooth L1 | 1.0 |
| `day` | H5 日级收益 Smooth L1 | 1.0 |

每个 batch 记录以下内容：

- 四项未加权损失和加权损失
- 四项任务对共享主干的梯度范数
- 各任务梯度范数占比
- 日级梯度范数与三个生成任务梯度范数中位数的比值
- 六组两两梯度余弦相似度
- 样本数、有效事件位置数、有效日级标签数和交易日

汇总结果保留均值、标准差、分位数、极值和负余弦比例。初始化与 best checkpoint 使用同一批数据，并关闭 dropout 等训练期随机行为。

## 预注册决策门槛

两个滚动折都完成后，按下面的顺序选择下一项实验：

1. 两折 best checkpoint 的日级梯度比值中位数都不高于 0.1，进入 `EVT-LABEL-SCALE-001`
2. 两折存在相同的日级任务冲突对，且余弦中位数不高于 -0.1、负余弦比例不低于 75%，进入 `EVT-SUPERVISION-POSITION-001`，同时复核任务权重
3. 日级梯度强度正常且没有持续冲突，直接进入 `EVT-SUPERVISION-POSITION-001`

标签尺度实验只运行 seed 0，对比原始 H5 收益与每日截面去极值 z 标签。最近折和相邻折使用相同合同。只有两折的 validation、OOS 和头部指标同时改善，才补 seed 1、2。

## 正式运行

运行前要求当前 worktree 已提交且保持干净。最近折命令如下：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-gradient-audit \
  --session ticknet-gradient-audit-recent-seed0 \
  --gpu A100 \
  --seeds 0 \
  --audit-batches 16 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-gradient-audit/recent-seed0
```

相邻折命令如下：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-rolling-gradient-audit \
  --eventstream-fold-id fold-54-oos-202511 \
  --session ticknet-gradient-audit-fold54-seed0 \
  --gpu A100 \
  --seeds 0 \
  --audit-batches 16 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-gradient-audit/fold54-seed0
```

两项工作流只暂存 validation 分片和已登记 SHA-256 的 seed 0 best checkpoint。train、OOS、监控分区和 2026 锁定区不会进入 Colab 运行环境。

结果同步回本机后生成跨折决策：

```bash
ticknet-eventstream-gradient-audit decide \
  --audit artifacts/eventstream-gradient-audit/recent-seed0/gradient-audit.json \
  --audit artifacts/eventstream-gradient-audit/fold54-seed0/gradient-audit.json \
  --output artifacts/eventstream-gradient-audit/decision.json
```

正式数值和下一项实验会在两折审计完成后补入本页与[实验日志](experiment-log.md)。
