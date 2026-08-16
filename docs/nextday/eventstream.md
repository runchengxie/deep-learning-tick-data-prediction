# L2 逐笔事件流主线

这条链路把逐笔委托、成交和快照无损打包，再用因果 Transformer 完成下一事件任务和日级信号输出。代码位于 `ticknet.eventstream`。截至 2026-08-16，数据打包和输入基准已经完成，正式模型训练尚未开始。

## 数据契约

`ticknet-eventstream-pack` 把每个交易日的三条流整理成整数镜像，并在打包时解析关联 ID。后续读取无需回查原始文件。每天产生四类文件：

- `orders_{day}.bin` 按股票、时间和 OrderID 排序
- `trades_{day}.bin` 按股票、时间和 DealID 排序
- `snaps_{day}.bin` 按股票和时间排序
- `index_{day}.npz` 记录每只股票的流偏移、长度和昨收

撤单会回查原始订单，得到撤单年龄和原始量。成交会按买卖方 ID 回查挂单到达时间，得到双方挂单年龄。无法解析的关联记为 `AGE_UNKNOWN_MS = -1`。打包产物保留原始字段的整数值，归一化在数据加载器中完成，调整特征口径时无需重新打包。

## 数据集与特征

`ticknet.eventstream.dataset` 按时间合并三条流。每个样本是某只股票在一个交易日内的连续事件窗口。80 维特征包括事件间隔、相对滚动中间价的基点变化、数量、买卖方向、撤单与挂单年龄、L1 价差与失衡、十档价量、成交额、时间相位和竞价标记。

目标分为三组：

- 下一事件流类型，包括 pad、snapshot、order 和 trade
- 下一订单类型，取值来自原始 OrderType 词表
- 日级信号，由外部标签表按股票和日期提供，允许为空

## 模型与训练

`ticknet.eventstream.model` 提供带旋转位置编码的因果 Transformer，预设尺寸包括 smoke、probe25m、probe50m、capacity100m 和 probe150m。`capacity100m` 使用 960 维隐藏层、9 个 Transformer block、15 个 attention head 和 3,840 维 FFN，共 100,604,180 个参数。attention 使用 PyTorch scaled-dot-product attention。

训练入口为 `ticknet-eventstream-train`。每个 epoch 在训练窗口上完成多任务下一事件预测，验证阶段按日计算日级输出的 Rank IC。训练按 `selection_metric` 早停，保存 best 和 last checkpoint，并写出历史 JSON。恢复训练时会校验实验签名和数据集指纹。

`ticknet-eventstream-prepare-horizon-labels` 把 nextday 多周期长表转换成 H3 和 H5 宽表。转换时要求 `trading_date`、`entry_date` 和 `return_end_date` 同属 train、validation 或 OOS，跨边界标签会被清除。`ticknet-eventstream-benchmark` 在真实 pack 上执行前向、反向和 AdamW 更新，输出吞吐、显存和单 seed 耗时。基准不会读取 validation 和 OOS。

## 预测导出

`ticknet-eventstream-export-predictions` 把日级分数与正式 open-to-following-open 收益、可交易状态和动态股票池合并，生成符合 `ticknet.research.prediction_contract` 的预测 Parquet。结果可以交给 `import_predictions` 登记，也可以由 `topk_cost_sweep` 直接消费。候选行使用模型分数，状态行的分数固定为 0.0，只用于跟踪已有持仓的可交易状态。

## 最近折配置

基础示例位于 `configs/eventstream.yaml`。2021 基础设施折使用 `configs/eventstream-h5-fold0-capacity100m.yaml`。最近折使用 `configs/eventstream-h5-recent-capacity100m.yaml`，日期如下：

```text
train       2025-08 至 2025-10
validation  2025-11
OOS         2025-12
locked      2026 起
```

2025 年 8 月至 12 月共 103 个交易日，已经全部打包，产物约为 313.11 GiB。目录中没有遗留的 partial 文件。2026 保持锁定，配置和标签均不读取该区间。

## 输入基准

A100 容量基准完成后，使用同一个 2025 年 8 月 pack 扫描物理 batch 8、16、32 和 64。有效 batch 固定为 64：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-batch-size-sweep \
  --session ticknet-eventstream-h5-recent-sweep-a100 \
  --gpu A100 \
  --batch-sizes 8 16 32 64 \
  --effective-batch-size 64 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/batch-size-sweep/a100
```

每档会独立记录吞吐和显存，单档 OOM 不影响后续档位。基准只访问训练 pack。

输入分析可以分别测量 Dataset、预加载 GPU batch 和端到端吞吐：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-input-profile \
  --session ticknet-eventstream-h5-recent-input-a100 \
  --gpu A100 \
  --num-workers 2 4 8 16 \
  --effective-batch-size 64 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/input-profile/a100
```

2026-08-12 的 A100 实测确认旧 Dataset 是主要输入瓶颈。预加载 GPU batch 的吞吐为 238.79 samples/s，旧 Dataset 使用 16 个 worker 时端到端吞吐为 18.19 samples/s。旧实现每取 512 个事件，都会重新合并并处理该股票全天的事件。

优化后的实现先用时间二分定位目标事件，再稳定合并窗口附近的 513 个事件。真实数据的 9 个窗口与旧实现逐元素一致，单样本构造速度提高 15.6 至 18.9 倍。8 个 worker 的端到端吞吐达到 149.40 samples/s，16 个 worker 受到 A100 运行环境 12 个 CPU core 的限制，吞吐降至 140.73 samples/s。因此最近折配置固定 `num_workers: 8`。

按 120,000 个训练样本和 20 个 epoch 外推，每个 epoch 约 13.39 分钟，每个 seed 上限约 4.46 小时，三个 seed 串行上限约 13.39 小时。这个估算不含 validation 和 checkpoint I/O，正式耗时以训练日志和早停结果为准。

下一步先运行最近折正式 seed 0。H5 用于选择 checkpoint，H3 只作监控。seed 0 通过预先写明的门槛后，再补 seed 1 和 2。
