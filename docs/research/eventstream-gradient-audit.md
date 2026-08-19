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

## 正式结果

两折均使用 16 个固定 batch，每个 batch 8 个样本。审计源码 revision 为 `3e28f04755a881cb72697db2fc50bba031c9f5b0`。2026 锁定区和两折 OOS 都没有进入运行环境。

| 滚动折 | 初始化日级梯度比值中位数 | best checkpoint 日级梯度比值中位数 | best epoch | 持续负相关任务对 |
|---|---:|---:|---:|---|
| 最近折 | 0.61608 | 0.01969 | 4 | 无 |
| `fold-54-oos-202511` | 0.65568 | 0.03927 | 11 | `reg__day`、`stream__day` |

日级任务在初始化时与三个生成任务处于相近量级。训练到 best checkpoint 后，两折的日级梯度只剩生成任务中位数的约 2% 和 4%，都低于 0.1 门槛。相邻折的 `reg__day` 余弦中位数为 -0.34875，负值比例为 81.25%。`stream__day` 余弦中位数为 -0.10432，负值比例为 75%。最近折没有出现相同冲突，因此当前证据不支持先调整任务权重。

最近折和相邻折结果指纹分别为 `2fd3064238b10476a2ddb2a5e54a5155e77b78ab369b7126377866770eb28ccd` 和 `7ec93b77258d108b673992cd1776e28d652b146cb425a60c8be15e9181bcfe12`。跨折决策指纹为 `9bdef3aad8f9be28f80b0236bfc90f093ce1f3b509d03afaf486f136e1140bbf`。正式决定为 `day_gradient_weak`，下一实验是 `EVT-LABEL-SCALE-001`。

## 标签尺度实验合同

`ticknet-eventstream-target-overlay` 为原物化缓存生成轻量 train 标签覆盖层。事件张量、采样窗口、validation、OOS 和 H3 监控标签保持不变。覆盖层只替换训练批次中的 H5 日级目标，validation 与 OOS 继续使用原始 H5 收益评估。

每个有 H5 标签的训练日先按动态截面的全部有限标签计算中位数和原始 MAD，将标签裁剪到中位数加减 5 倍 MAD，再按裁剪后截面的均值和总体标准差转换成 z 标签。因分区边界而没有 H5 标签的样本继续由 `day_valid=0` 屏蔽。覆盖层绑定原物化数据指纹、H5 标签 SHA-256、逐日统计和每个月份文件的 SHA-256。最近折与相邻折分别使用独立覆盖层和训练输出目录。

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

最近折标签覆盖层生成命令如下：

```bash
ticknet-eventstream-target-overlay build \
  --config configs/eventstream-h5-recent-capacity100m.yaml \
  --storage-manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --materialized-root artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  --output artifacts/eventstream-label-scale/recent-seed0/target-overlay \
  --source-revision "$(git rev-parse HEAD)"
```

通过核对后上传轻量覆盖层：

```bash
rclone --config ~/.config/rclone/rclone.conf copy \
  artifacts/eventstream-label-scale/recent-seed0/target-overlay \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/eventstream-top400-h5-target-overlays/recent/seed0 \
  --checksum
```

随后运行一个不读取 OOS 的短恢复检查：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-label-scale-train \
  --session ticknet-label-scale-recent-seed0 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-label-scale/recent-seed0/training
```

短恢复检查通过后，将 `--training-epochs` 改为 `20` 并使用 `--evaluate-test`。相邻折改用 `eventstream-rolling-label-scale-train`，同时提供 `--eventstream-fold-id fold-54-oos-202511`。两个工作流使用独立 checkpoint 和结果目录。

标签尺度训练完成后，将在本页与[实验日志](experiment-log.md)补充两折 validation、OOS 和极端组收益差，并按门槛决定是否补 seed 1、2。
