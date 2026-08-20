# 事件流标签尺度实验

`EVT-LABEL-SCALE-001` 检查日级收益标签的数值尺度是否限制了 100M 事件流 Transformer 对 H5 横截面信号的学习。实验已经完成最近折和相邻折 seed 0，正式决定为 `EXTEND_TO_SUPERVISION_POSITION`。

## 实验问题

梯度审计发现，两折 best checkpoint 的日级任务梯度只相当于三个生成任务梯度中位数的约 2% 和 4%。本实验只替换训练分区的 H5 日级目标，检查每日截面标准化能否增强日级学习信号。

每个有 H5 标签的训练日先按中位数加减 5 倍原始 MAD 去极值，再按去极值后截面的均值和总体标准差转换为 z 标签。事件窗口、采样顺序、模型容量、任务权重、validation、OOS 和 H3 监控标签保持不变。分区边界处没有 H5 标签的样本继续由 `day_valid=0` 屏蔽。

## 输入与运行身份

| 滚动折 | 训练期 | validation | OOS | 数据指纹 | z 标签指纹 |
|---|---|---|---|---|---|
| 最近折 | 2025-08 至 2025-10 | 2025-11 | 2025-12 | `5a7d9216c7b4a8f680ef8a22ca760b482b6ccd38f6a8df587bd7deb44f445314` | `7f8223c0581e08115b18c19e36756e8431e2e2b395231e5a047860eb5ae53832` |
| `fold-54-oos-202511` | 2025-07 至 2025-09 | 2025-10 | 2025-11 | `596daa34cfe2a44ad94f884db95d9ce164fd6aff38e05fad61ce1869cc8e9403` | `faefc4f65fe97bdebd27be3dd77033976beab8fd94ef45ce9f5f80015f149e60` |

两折都使用 100,604,180 参数的 `capacity100m` 和 seed 0。训练实验身份绑定源码 revision `1b4c0f163d1f0aab2c930468f20eecaafe8b60f3`。相邻折恢复运行时使用 revision `35f90d722e6dacd98cd9d0608d6fa3c3c7737b3e` 的 Colab 调度器，checkpoint 继续校验原实验 revision。2026 锁定区没有进入训练或评估环境。

## 正式结果

| 滚动折 | 训练标签 | best epoch | validation Rank IC | OOS Rank IC | validation 极端组收益差 | OOS 极端组收益差 |
|---|---|---:|---:|---:|---:|---:|
| 最近折 | 原始 H5 收益 | 4 | 0.04345 | 0.05879 | -0.38744% | 0.34105% |
| 最近折 | 每日截面 z 标签 | 4 | 0.11747 | 0.07446 | 1.27275% | 0.08885% |
| `fold-54-oos-202511` | 原始 H5 收益 | 11 | 0.08735 | 0.03305 | 1.78031% | -0.51240% |
| `fold-54-oos-202511` | 每日截面 z 标签 | 8 | 0.13534 | 0.07755 | 2.68055% | 0.48367% |

两个窗口的 validation 和 OOS Rank IC 都有提高。相邻折的 validation 与 OOS 极端组收益差同时改善，最近折的 validation 极端组收益差由负转正，OOS 极端组收益差从 0.34105% 降到 0.08885%。

H3 只作监控，没有参与 checkpoint 选择。它的 Rank IC 结果如下：

| 滚动折 | 训练标签 | validation H3 Rank IC | OOS H3 Rank IC |
|---|---|---:|---:|
| 最近折 | 原始 H5 收益 | 0.04095 | 0.05101 |
| 最近折 | 每日截面 z 标签 | 0.09230 | 0.05928 |
| `fold-54-oos-202511` | 原始 H5 收益 | 0.04840 | 0.04231 |
| `fold-54-oos-202511` | 每日截面 z 标签 | 0.09066 | 0.07326 |

## 决策

预注册门槛要求两折的 validation、OOS 和极端组收益差同时改善。最近折的 OOS 极端组收益差没有改善，因此暂不补 seed 1、2。z 标签在四段评估区间都提高了 H5 Rank IC，并且四段 H3 Rank IC 也全部提高，说明它值得作为下一项训练机制实验的共同标签。

下一项实验为 `EVT-SUPERVISION-POSITION-001`。它使用现有 z 标签全部位置结果作为对照，只新增最后位置和线性尾部加权两种模式。

## 监督位置预注册合同

当前日级损失在每个有效事件位置计算，评估只读取最后一个有效位置。下一项实验比较以下模式：

| 模式 | 日级损失位置 |
|---|---|
| `all` | 所有有效位置，保持当前实现和现有结果 |
| `last` | 每个样本的最后一个有效位置 |
| `tail_weighted` | 所有有效位置，权重从序列首部到尾部线性增加 |

`tail_weighted` 对长度为 L 的有效序列使用位置权重 `(t + 1) / L`，其中 t 从 0 开始。padding 权重为 0，最终损失除以 batch 内有效权重总和。三项生成任务、日级任务权重、优化器、数据和选模指标保持不变。`day_supervision_mode` 和尾部权重版本必须写入 checkpoint 实验签名，三个模式使用独立输出目录。

训练入口已支持 `--day-supervision-mode all|last|tail_weighted`。Colab Runner 只允许标签尺度工作流使用新模式，并会将模式写入运行摘要。`last` 和 `tail_weighted` 的 checkpoint 名称、本地目录和 Drive 目录都相互独立。第一轮命令保持 OOS 关闭，例如：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-label-scale-train \
  --day-supervision-mode last \
  --session ticknet-supervision-recent-last-seed0 \
  --gpu A100 \
  --seeds 0 \
  --no-evaluate-test \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-supervision-position/recent-last-seed0
```

相邻折使用 `eventstream-rolling-label-scale-train`，增加 `--eventstream-fold-id fold-54-oos-202511`。将模式改为 `tail_weighted` 即可运行线性尾部加权对照。

现有 `all` z 标签结果不重复训练。`last` 和 `tail_weighted` 先在两个滚动折运行 seed 0，并使用 `--no-evaluate-test` 保持 OOS 关闭。候选需要同时满足以下 validation 门槛：

1. 两折 Rank IC 都高于同折 `all` 基线，两个增量的均值至少为 0.005
2. 两折极端组收益差都不低于同折 `all` 基线，并保持为正
3. 两个候选都通过时，选择两折 Rank IC 平均增量更高的方案。平均增量差小于 0.002 时选择 `tail_weighted`

只对入选方案开放两折 OOS。两折 OOS Rank IC 和极端组收益差都高于同折 `all` 基线后，才补 seed 1、2。没有方案通过时，下一步检查日级任务权重和成本感知排序目标。

## 产物

本地正式结果位于：

- `artifacts/eventstream-label-scale/recent-seed0/training`
- `artifacts/eventstream-label-scale/fold54-seed0/training`

Drive 使用对应的 `eventstream-top400-h5-capacity100m-*-label-z/training` 目录。每个目录包含 best、last、训练历史、结果、预检报告和 Colab 运行摘要。
