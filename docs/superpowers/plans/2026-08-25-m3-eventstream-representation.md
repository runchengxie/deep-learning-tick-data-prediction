# M3-inspired EventStream Representation Implementation Plan

目标：在不改变旧配置默认行为的前提下，为现有事件流 Transformer 增加可选的 LOB prefix、固定因果 session anchor 和 Hybrid VQ 表征，并把它们纳入实验身份与物化合同。

架构约束：继续使用 80 维事件张量和现有 dataset tuple 合同。Prefix 占用序列中的一个输入位置。VQ 只在模型侧作为连续事件 embedding 的残差分支。撮合引擎与闭环市场模拟不属于本 PR。

技术栈：Python 3.10+、NumPy、PyTorch、PyArrow、pytest、YAML。

设计文档：`docs/superpowers/specs/2026-08-25-m3-eventstream-representation-design.md`

## 全局约束

- 所有新开关关闭时，保留旧样本形状、模型参数量和 checkpoint 行为。
- Prefix 和 session anchor 不能读取窗口边界之后的快照或成交。
- `N_FEATURES=80`、`N_STREAMS=4`、`N_ORDER_TYPES=12` 保持不变。
- 物化数组 schema 保持不变。
- 不读取 2026 锁定区，也不加入任何真实数据性能结论。

## 任务一：LOB prefix 与固定 session anchor

涉及文件：

- `src/ticknet/eventstream/dataset.py`
- `tests/test_eventstream_m3_repr.py`

实施步骤：

- [x] 先写失败测试，覆盖 prefix 张量形状、目标对齐、严格使用边界前快照和 anchor 因果性。
- [x] 确认 RED：旧实现因缺少 `use_lob_prefix`、`ORDER_TYPE_LOB_PREFIX` 和 prefix 构造接口而失败。
- [x] 实现 `ORDER_TYPE_LOB_PREFIX = 11`，保持公开 tensor shape 不变。
- [x] Prefix 从窗口开始前最后一个 snapshot 构造，不读取未来 snapshot。
- [x] 固定 session anchor 只从边界前已经出现的 trade 或 snapshot last 中选择最早价格。同时间 trade 优先。首次出现后，后续窗口的 anchor 不再变化。
- [x] Anchor 尚未出现时用昨收作为数值 fallback，并把可用标记设为 0。

## 任务二：配置与物化合同

涉及文件：

- `src/ticknet/eventstream/train.py`
- `src/ticknet/eventstream/materialized.py`
- `tests/test_eventstream_materialized.py`
- `configs/eventstream.yaml`

实施步骤：

- [x] `EventstreamConfig` 增加 `use_lob_prefix`、`use_session_anchors`、`use_vq`、`vq_codebook_size`、`vq_dim` 和 `vq_loss_weight`。
- [x] `use_session_anchors=True` 时要求同时启用 LOB prefix。
- [x] 原始训练、validation、OOS 和 monitor dataset 统一接收表征开关。
- [x] 物化合同记录 prefix 与 anchor 开关。Prefix 开启时使用 `seeded_fixed_window_v2`，旧合同继续使用 v1。
- [x] 在 prefix 模式下增加 source 与 materialized 样本逐张量一致测试。
- [x] 更新示例 YAML。

## 任务三：Hybrid VQ

涉及文件：

- `src/ticknet/eventstream/model.py`
- `src/ticknet/eventstream/train.py`
- `tests/test_eventstream_vq.py`

实施步骤：

- [x] 先写失败测试，覆盖关闭 VQ 时参数形状兼容、code 输出、prefix/pad 屏蔽和 VQ loss 权重。
- [x] 确认 RED：旧模型构造器和 loss 接口不接受 VQ 参数。
- [x] 用 `[dt_log, price_bps, qty_log, side, is_cancel]` 编码核心事件行为。
- [x] 用最近邻 codebook、straight-through estimator、codebook loss 和 commitment loss 实现量化。
- [x] 量化结果投影到 `d_model` 后作为连续 embedding 的残差。`sid==0` 的 pad 与 prefix 不参与 VQ。
- [x] `compute_loss_components` 保留四项原任务合同，VQ 正则只在 `compute_loss` 中加入。

## 任务四：兼容链

涉及文件：

- `src/ticknet/eventstream/train.py`
- 冻结 embedding、固定窗口缓存和其他 checkpoint 消费端
- 相关测试

实施步骤：

- [x] legacy checkpoint 缺少新字段时按默认关闭配置归一化。
- [x] 检查所有 `build_eventstream_model` checkpoint 消费端，确保 VQ checkpoint 可以按实验签名重建模型。
- [x] 检查固定尾盘窗口与 prefix 模式的输入合同，必要时绑定 representation identity。
- [x] 保留旧 checkpoint、旧 materialized manifest 和旧 close cache 的默认兼容路径。

## 任务五：文档与验证

涉及文件：

- `docs/nextday/eventstream.md`
- `docs/model-catalog.md`
- PR 描述

实施步骤：

- [x] 记录新配置、因果边界、VQ 角色和实验解释。
- [x] 明确说明新表征尚未经过真实滚动窗口性能验证。
- [x] 运行 PR CI，并确认 Python 3.10、Python 3.12、ruff、format、ty、pytest、coverage 和依赖审计通过。
- [x] 审查最终 diff，确认没有 matching engine、simulator raw-ID 合同、2026 锁定数据访问或未经验证的性能声明。
- [x] 把 draft PR 更新为可审查状态，并写明 RED 与 GREEN 验证证据。
