# L2 逐笔事件流主线

逐笔委托、成交和快照三条原始流经过无损打包，喂给因果 Transformer 做下一事件预测和日级信号输出。这条主线在 2026-08 前后随真实 L2 数据到位而落地，对应 [development-guide](../dev/development-guide.md) 里的 `ticknet.eventstream` 模块。

## 数据契约

打包入口 `ticknet-eventstream-pack` 把每个交易日的三条流整理成无损整数镜像，相关 ID 在打包时就地解析，之后读取不再回链原始文件。每天产出四类文件，布局见 `ticknet.eventstream.config`：

- `orders_{day}.bin` 按股票、时间和 OrderID 排序
- `trades_{day}.bin` 按股票、时间和 DealID 排序
- `snaps_{day}.bin` 按股票和时间排序
- `index_{day}.npz` 记录每只股票的流偏移、长度和昨收

撤单事件回链原始订单，得到撤单年龄和原始量。成交按买卖方 ID 回链挂单到达时间，得到双方挂单年龄。无法解析的关联记 `AGE_UNKNOWN_MS = -1`。打包产物保留原始字段的整数镜像，归一化全部在数据加载器里完成，改动特征口径时无需重新打包。

## 数据集与特征

`ticknet.eventstream.dataset` 把三流按时间归并成事件序列，每个样本是某股票某日连续时间窗内的事件串。特征共 80 维，包括距上一事件的时间对数、相对滚动中间价的基点偏移、量对数、买卖方向、撤单与挂单年龄、L1 价差与失衡、十档价量，以及成交额对数、时间相位和竞价标记。

目标有三组：

- 下一事件流类型，取值 pad、snapshot、order、trade
- 下一订单类型，来自原始 OrderType 的词汇表
- 日级信号，用外部标签表按股票和日期提供，可为空

## 模型与训练

`ticknet.eventstream.model` 提供带旋转位置编码的因果 Transformer，尺寸配置包括 smoke、probe25m、probe50m、capacity100m 和 probe150m。`capacity100m` 有 100,604,180 个参数，用于客户容量门槛和首个 H5 fold。训练入口为 `ticknet-eventstream-train`。每 epoch 在训练窗口上做多任务下一事件预测，验证集按日算日头的 Rank IC，按 `selection_metric` 早停，保存 best 和 last 检查点并写历史 JSON。恢复训练时会校验实验签名和数据集指纹。

`ticknet-eventstream-prepare-horizon-labels` 把 nextday 多周期长表转换成事件流数据集读取的 H3/H5 宽表。转换时强制 `trading_date`、`entry_date` 和 `return_end_date` 同属 train、validation 或 OOS，跨边界标签不会进入训练或评估。`ticknet-eventstream-benchmark` 在真实 pack 上执行前向、反向和 AdamW 更新，输出吞吐、显存和单 seed 耗时，不读取 validation 和 OOS。

## 预测导出

`ticknet-eventstream-export-predictions` 把日头分数与正式 open-to-following-open 收益、可交易状态和动态股票池合并，产出符合 `ticknet.research.prediction_contract` 的预测 Parquet，可以直接被 `import_predictions` 登记，或被 `topk_cost_sweep` 消费。候选行分数来自模型，状态行分数记 0.0，只用于持仓可交易跟踪。

## 配置示例

基础示例见 `configs/eventstream.yaml`。2021 基础设施 fold 使用
`configs/eventstream-h5-fold0-capacity100m.yaml`。正式 recent fold 使用
`configs/eventstream-h5-recent-capacity100m.yaml`：train 为 2025-08 至 2025-10，validation 为
2025-11，OOS 为 2025-12；2026 保持 locked，配置和标签均不读取它。
