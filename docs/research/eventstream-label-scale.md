# 事件流标签尺度实验

`EVT-LABEL-SCALE-001` 检查日级收益标签的数值尺度是否限制了 100M 事件流 Transformer 对 H5 横截面信号的学习。实验已经完成最近折和相邻折 seed 0，并进一步完成 `EVT-SUPERVISION-POSITION-001`。最终决定为 `KEEP_ALL`。

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

## 监督位置正式结果

`last` 和 `tail_weighted` 都使用 seed 0、每日截面 z 标签和最近折数据。模型容量、三项生成任务、日级任务权重、优化器、选模指标和评估样本保持不变。两次训练都绑定源码 revision `41290ff056fb318d37ce44ba89bcbf31453c07f3`，数据指纹和 z 标签指纹与 `all` 基线一致。

| 模式 | best epoch | validation Rank IC | 相对 `all` | validation 极端组收益差 | 相对 `all` |
|---|---:|---:|---:|---:|---:|
| `all` | 4 | 0.11747 | 基线 | 1.27275% | 基线 |
| `last` | 7 | 0.07802 | -0.03945 | 0.92692% | -0.34583 个百分点 |
| `tail_weighted` | 7 | 0.11289 | -0.00458 | 1.10797% | -0.16478 个百分点 |

`last` 只在最后一个有效位置提供日级梯度，Rank IC 和极端组收益差都明显下降。`tail_weighted` 保留了全序列监督，结果更接近 `all`，两项主要指标仍然较低。H3 监控 Rank IC 分别为 0.06055 和 0.08667，也低于 `all` 的 0.09230。

预注册门槛要求候选方案在两折都超过 `all`。两个候选在最近折已经违反这项必要条件，相邻折结果无法改变正式判定。因此实验按门槛提前停止，未运行相邻折，也未开放任何 OOS。这一停止不依赖 OOS 结果，也不改变原有选择条件。

正式决定为 `KEEP_ALL`。日级标签继续在所有有效位置计算损失，不补监督位置的 seed 1、2，也不启动 `probe150m`。下一步先检查日级任务在多任务总损失中的权重，再根据结果决定是否实现更贴近选股和交易成本的排序目标。

## 产物

本地正式结果位于：

- `artifacts/eventstream-label-scale/recent-seed0/training`
- `artifacts/eventstream-label-scale/fold54-seed0/training`
- `artifacts/eventstream-supervision-position/recent-last-seed0`
- `artifacts/eventstream-supervision-position/recent-tail-weighted-seed0`

Drive 使用对应的 `eventstream-top400-h5-capacity100m-*-label-z/training`、`eventstream-top400-h5-capacity100m-recent-label-z-day-last/training` 和 `eventstream-top400-h5-capacity100m-recent-label-z-day-tail-weighted/training` 目录。每个目录包含 best、last、训练历史、结果、预检报告和 Colab 运行摘要。

`last` 的 best、last 和结果 JSON SHA-256 分别为 `6148eb9d4b8d83134c625e7af0570069f733e852ffd5bbc33dddcb7aecf26b5b`、`eb14826d9080cff3c466334735fc584adb921af17faa66c93a0b382fe7050f7e` 和 `28d1420ae275ee28c02d255afa03fbfd5bb6495d2710363bb34adea7622a0c04`。`tail_weighted` 对应为 `3f6a4a5956631d85f74aae43ea7a3b13a61015bcca8e86306ca8879418a0a819`、`b74b141603667680afe8717e874b5a2fd0f08f3025eb6e4cf0e37028b250b2e2` 和 `b463d172fc470abf63eb3dd4fedd6c41c3f01d219925e3786069fe9ffb5e4fdd`。两个 Drive 目录与本机都有 6 个正式文件，文件名和大小一致。
