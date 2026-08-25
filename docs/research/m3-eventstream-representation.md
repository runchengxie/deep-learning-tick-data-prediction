# M3-inspired 事件流表征实验

本页记录从 M3 市场微观结构生成模型中借鉴到 TickNet 的表征改造。当前目标仍然是次日横截面排序，不把项目改造成闭环市场模拟器，也不把尚未训练的结构改动写成性能提升。

## 为什么做这组改动

现有事件流模型会从日内任意位置截取固定长度窗口。窗口可能从订单或成交开始，因此模型未必在第一步就看到窗口边界处的完整十档盘口。另一方面，事件价格一直使用滚动中间价做局部坐标，这对预测任务有价值，但没有单独提供一个固定的日内价格坐标。M3 的盘口前缀和固定价格锚点正好对应这两个表示问题。

VQ 属于第三个独立假设。现有模型分别预测事件类型、订单类型、价格、时间间隔和数量。Hybrid VQ 把核心行为字段压成离散 code，再作为连续事件 embedding 的残差，让模型可以同时保留精细连续信息和重复出现的典型订单行为模式。

## 三个可选机制

### LOB prefix

`use_lob_prefix: true` 时，每个窗口的第一个输入位置变成一个特殊盘口状态 token。它来自窗口开始之前最后一个 snapshot，绝不读取边界后的快照。真实事件仍按原顺序跟在 prefix 后面，公开 tensor shape 维持不变。

Prefix 使用 `stream_id=0` 和保留的 `order_type_id=11`。模型因此可以区分它与普通 pad。十档价量、order count、spread、imbalance、weighted bid/ask 等字段继续复用原来的 80 维布局。

### 固定 session anchor

`use_session_anchors: true` 依赖 LOB prefix。它不会替换当前事件的 rolling-mid 归一化，而是在 prefix 中补充一个固定的日内坐标。

Anchor 只使用窗口边界前已经观察到的成交价或 snapshot last price，并选择全日最早出现的有效价格。同时间戳时成交优先。一旦这个最早价格已经出现，后续窗口都会得到相同 anchor。此前只能使用昨收作为数值 fallback，并把 anchor availability 标为 0。

这样同时保留两种信息：rolling mid 描述订单离当前盘口有多远，session anchor 描述当前盘口相对当天早期固定基准走了多远。

### Hybrid VQ

`use_vq: true` 时，模型从 `dt_log`、`price_bps`、`qty_log`、`side` 和 `is_cancel` 五个核心行为字段生成低维向量，再分配到最近的可学习 codebook 项。量化后的向量投影回 `d_model`，作为原连续事件 embedding 的残差。

Pad 和 LOB prefix 的 `stream_id` 都为 0，因此不会参与 VQ 正则。训练损失额外加入 codebook 与 commitment loss，权重由 `vq_loss_weight` 控制。VQ 关闭时不会创建对应参数，旧模型 state dict 和参数量保持不变。

## 配置

示例配置新增：

```yaml
use_lob_prefix: false
use_session_anchors: false
use_vq: false
vq_codebook_size: 1024
vq_dim: 64
vq_loss_weight: 0.25
```

所有字段默认关闭或保持固定默认值。推荐把三项机制拆成受控实验，而不是一次全开后只看一个 Rank IC 数字。

## 实验顺序

第一阶段只比较 LOB prefix。第二阶段在 prefix 基础上加入 session anchor。第三阶段单独检查 Hybrid VQ，必要时再做组合。每个实验继续沿用现有的时间外验证、相邻滚动折、成本后组合指标和锁定区规则。

模型规模暂不因为 M3 的 scaling law 扩大。TickNet 目前的主要瓶颈仍是排序信号向成本后收益的转换，扩大参数量没有现成证据能解决这个问题。

## 兼容与数据合同

Prefix 和 session anchor 会改变固定窗口的实际输入内容，因此写入 materialized manifest 和 close-cache contract。旧缓存缺少这些字段时按 `false` 解释，继续使用原 v1 采样合同。Prefix 缓存使用 v2 合同，消费端会拒绝表征身份不一致的 checkpoint 或缓存。

VQ 只改变模型结构，不改变物化数组，因此不进入张量 schema，但会进入 checkpoint 实验身份。冻结 embedding、物化预测和梯度审计会按 checkpoint 中的 VQ 参数重建模型。

现有联合微调缓存链暂未扩展到 M3-inspired 表征。联合微调对带 prefix、session anchor 或 VQ 的预训练 checkpoint 会明确拒绝，避免使用旧缓存合同产生静默语义错位。

## 明确不在本次范围内的内容

本次没有实现 matching engine、AI 订单自回归 rollout 或大单冲击模拟。当前 prediction pack 在提炼撤单年龄等字段后会丢弃原始 OrderID、DealID、BuyID 和 SellID，而严格撮合需要订单身份与时间优先队列。完整市场模拟应单独设计 simulator data contract，再实现重放一致性测试和撮合引擎。

## 当前证据状态

这组改动目前只有合成数据单元测试、兼容性测试和仓库 CI 证据。尚未运行真实 A 股滚动窗口训练，因此不能声称它提高了 Rank IC、NDCG、成本后收益或任何实盘指标。真实实验结果只有在按现有研究协议跑完后才能写入项目现状。
