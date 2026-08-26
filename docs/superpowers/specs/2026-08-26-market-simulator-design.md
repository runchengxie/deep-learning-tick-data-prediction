# 市场模拟器子系统设计（第二阶段）

日期：2026-08-26
状态：Draft（实现中）
关联：第一阶段 `agent/m3-eventstream-representation`（PR #96，已合并）

## 目标

在 deep-learning-tick-data-prediction 中新增一个独立子系统 `src/ticknet/simulator/`，实现：

1. 确定性撮合引擎（matching engine），能够消费保留原始 OrderID 的订单流并维护十档订单簿
2. 生成式市场模拟器：以真实初始盘口为 prefix，用 eventstream Transformer 生成背景订单流，插入外部干预订单（如 TWAP 子单），经撮合引擎回放产出合成市场轨迹
3. 冲击成本估计接口：把候选执行单放入模拟器，估计真实冲击 / slippage

## 非目标（本阶段不做）

- 不修改 `eventstream` 预测主链路（第一阶段已完成 LOB prefix / 双锚点 / VQ）
- 不做跨资产、跨市场相关性联动
- 不扩模型到 150M/1B（Scaling 实验单独排期）
- 不直接替换 next-day Alpha 信号，模拟器先作为独立成本估计 / 沙盒工具

## 为什么需要新数据契约

`eventstream/config.py` 明确说明：原始 OrderID / DealID / BuyID / SellID 已丢弃，关联信息提炼为撤单年龄等派生特征。预测任务下这是合理取舍。但撮合引擎需要精确识别撤单对应的挂单（含时间优先队列），没有原始 ID 无法重建。因此 simulator 必须读取或重建一套保留 ID 的 simulator pack，与预测用的 eventstream pack 解耦，互不污染。

## 模块边界（遵循 AGENTS.md）

- `src/ticknet/simulator/` 新模块，负责撮合引擎、生成式回放、冲击估计
- 不直接 import `nextday` 实现，通过 CLI / 配置与 `eventstream` 解耦，可复用其 tokenizer / model 加载
- 测试用合成数据，不依赖真实 L2 全量数据

## 实施顺序

1. `simulator/pack.py`：从原始 L2 解析保留 OrderID 的 simulator pack（RED test 先行）
2. `simulator/matching.py`：撮合引擎，消费 order/cancel 事件，维护十档 LOB
3. `simulator/engine_correctness_test`：用真实初始盘口 + 真实 order stream 回放，重建盘口须与真实 snapshot 对上（correctness gate）
4. `simulator/generator.py`：加载 eventstream Transformer，生成背景订单 token
5. `simulator/replay.py`：闭环回放（prefix + 背景流 + 外部干预 → 轨迹）
6. `simulator/impact.py`：冲击成本估计接口
7. 文档与 CLI 入口

## 验收门槛

- matching engine 对合成序列的撮合结果与手算一致
- engine correctness test：重建盘口与真实 snapshot 在容差内一致
- 冲击曲线在双对数坐标下可观测，数值仅作研究用途，不宣称实盘因果
