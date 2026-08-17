# L2 逐笔事件流主线

这条链路把逐笔委托、成交和快照无损打包，再用因果 Transformer 完成下一事件任务和日级信号输出。代码位于 `ticknet.eventstream`。截至 2026-08-17，100M 最近折三 seed 已完成，当前进入冻结 embedding 的下游增量检验。

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

## 正式训练结果与扩容门槛

截至 2026-08-17，本机没有可用 CUDA GPU。Google Drive 总额为 200 GiB，剩余约 21.4 GiB。完整五个月 pack 约为 313.11 GiB，无法同时放入现有 Drive 或 Colab 临时盘。

正式训练改用固定窗口物化方案。训练窗口在本地按 seed 一次性确定，保存模型实际读取的 80 维特征、下一事件目标、日级标签和有效位置。物化清单绑定五个月源清单、源码 revision、日期、seed、采样参数和每个张量文件的 SHA-256。训练前逐文件复核，内容漂移、错误 seed、错误日期或错误源码 revision 都会停止运行。

`eventstream-recent-train` 工作流会下载单个 seed 的物化训练集，恢复已有 checkpoint，核对允许访问的文件，再启动 100M 训练。训练成功或失败都会回传 best、last、history、result、预检报告和运行摘要。短恢复验证只下载 train、validation 和 H3 validation 分片，训练一个 epoch，OOS 文件不会进入运行环境。随后使用相同源码 revision 恢复到正式 epoch 上限，此时才下载并评估 OOS。

当前代码和合成数据覆盖物化前后逐张量一致、篡改拒绝和 1 至 2 epoch 恢复。真实 seed 0、1、2 均已完成源清单核对、固定窗口物化、远端训练、checkpoint 回传和 OOS 评估。

| seed | 最佳 epoch | H5 validation Rank IC | H5 OOS Rank IC | 训练时间 |
|---:|---:|---:|---:|---:|
| 0 | 4 | 0.04345 | 0.05879 | 82.9 分钟 |
| 1 | 6 | 0.09403 | 0.03730 | 116.3 分钟 |
| 2 | 5 | 0.08029 | 0.03291 | 102.7 分钟 |

validation 均值为 0.07259，OOS 均值为 0.04300。三组结果的方向全部为正，100M 信号门槛已经通过。H3 监控的 validation 与 OOS 也全部为正。2026 数据没有进入训练或评估。

### 存储清单与预检

`ticknet-eventstream-storage-readiness` 提供源数据审计基础。清单生成器读取五个月按日股票池，把每个交易日固定到 train、validation 或 OOS，逐项记录 412 个 pack 文件及标签产物的字节数、MD5 和 SHA-256。股票池触及 2026、日期没有落入唯一分区、pack 缺失或源数据指纹不一致时会停止生成。

在保存本地真实产物的主工作区执行：

```bash
ticknet-eventstream-storage-readiness build \
  --config configs/eventstream-h5-recent-capacity100m.yaml \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1 \
  --universe artifacts/eventstream-h5-recent-fold/202508/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202509/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202510/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202511/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202512/universe.json \
  --artifact fold-labels/manifest.json=artifacts/eventstream-h5-recent-fold/fold-labels/manifest.json \
  --artifact fold-labels/h3.parquet=artifacts/eventstream-h5-recent-fold/fold-labels/h3.parquet \
  --artifact fold-labels/h5.parquet=artifacts/eventstream-h5-recent-fold/fold-labels/h5.parquet \
  --output artifacts/eventstream-h5-recent-fold/storage-manifest.json
```

生成过程会顺序读取完整 pack 计算内容哈希，只需在数据定版后执行一次。输出清单只含文件路径、大小、哈希、日期合同和聚合统计，不含股票列表或行情内容。

存储清单还保留了完整 pack 直传远端时的核对命令，供 benchmark pack 和将来的存储迁移使用。远端必须通过 rclone 提供 MD5 或 SHA-256 中的至少一种：

```bash
rclone lsjson remote:ticknet-data/eventstream-h5-recent \
  --recursive --files-only --hash \
  > artifacts/eventstream-h5-recent-fold/remote-listing.json

ticknet-eventstream-storage-readiness verify-direct-remote \
  --manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --listing artifacts/eventstream-h5-recent-fold/remote-listing.json
```

完整 pack 复制方案还要在运行环境中检查数据、临时文件和 checkpoint 空间。默认给数据体积留出 5% 余量，并额外保留 20 GiB：

```bash
ticknet-eventstream-storage-readiness check-full-copy-capacity \
  --manifest /content/storage-manifest.json \
  --path /content
```

数据落盘后再做一次逐文件内容核对：

```bash
ticknet-eventstream-storage-readiness verify-staged \
  --manifest /content/storage-manifest.json \
  --root /content/ticknet-eventstream/top400-h5-recent
```

上述完整 pack 命令用于审计和 benchmark，正式 100M 训练使用下文的固定窗口缓存。物化器按月原子落盘并支持恢复，正式训练工作流会串联缓存清单核对、checkpoint 恢复、训练和产物回传。

最近折以 H5 选择 checkpoint，H3 只作监控。三 seed 已经满足以下门槛：

- validation 与 OOS 的 H5 每日 Rank IC 均为正
- 数据指纹、训练历史、best 与 last checkpoint、validation 和 OOS 评估产物完整
- 训练与评估没有读取 2026，OOS 结果不用于修改本轮配置

冻结表征对照使用次日 open-to-following-open 下游标签。事件流 H5 标签负责训练编码器，下游继续回答项目当前的日频 Top-K 交易问题。HGB 与 LambdaMART 分别比较分钟特征、冻结 embedding、二者组合。

`probe150m` 当前只是代码中的模型预设。冻结 embedding 已经出现 HGB Rank IC 增量，成本后主动收益和跨窗口稳定性仍待确认。下一步先补风险暴露和联合端到端小实验，再决定是否补充 150M 正式配置、参数量测试和预算。第一轮 150M 实验只改变模型容量，先完成 benchmark 和 seed 0，再决定是否增加重复实验。原始盘口的容量与窗口矩阵已经停止，本路线不重新启动 raw-200 或 raw-1000 扩容。

### 固定窗口物化与正式训练

源清单生成完成后，在保存完整 pack 的本地主机物化 seed 0：

```bash
ticknet-eventstream-materialize build \
  --config configs/eventstream-h5-recent-capacity100m.yaml \
  --storage-manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --output artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  --source-revision "$(git rev-parse HEAD)"
```

物化支持按月恢复。已有分片会先复核 SHA-256，再跳过。完成后执行完整核对：

```bash
ticknet-eventstream-materialize verify \
  --root artifacts/eventstream-h5-recent-fold/materialized/seed0
```

把通过核对的目录上传到固定 seed 路径：

```bash
rclone --config ~/.config/rclone/rclone.conf copy \
  artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/eventstream-top400-h5-recent-materialized/seed0 \
  --checksum
```

第一次远端运行只完成一个正式 epoch，不读取 OOS，用于验证 checkpoint 回传和恢复：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-train \
  --session ticknet-eventstream-h5-recent-seed0-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/training/seed0
```

### 冻结 embedding 与下游对照

三组 checkpoint 共用一份尾盘窗口缓存。缓存按股票日保存收盘前最后 512 个事件，只包含模型输入和股票日键。它不保存随机训练窗口，也不随 seed 改变：

```bash
ticknet-eventstream-close-cache build \
  --storage-manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1 \
  --output artifacts/eventstream-h5-recent-fold/daily-close-cache \
  --seq-len 512 \
  --min-events 256 \
  --source-revision "$(git rev-parse HEAD)"

ticknet-eventstream-close-cache verify \
  --root artifacts/eventstream-h5-recent-fold/daily-close-cache
```

共享缓存已完成本地生成、全量核对和远端上传，包含 39,903 个股票日、5 个分片，共 6,619,831,094 字节，约 6.17 GiB。数据指纹为 `59577182c8124c312de0591059c67e55d472511ca77753403ce77afbf8f109f4`。远端对每个 seed 分别载入对应训练缓存 manifest 和 checkpoint，导出 960 维向量。完整导出会读取已经批准评估的 2025 年 12 月 OOS，因此调度器会保留显式 OOS 授权。每次只处理一个 seed：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-export-embeddings \
  --session ticknet-eventstream-embedding-seed0 \
  --gpu A100 \
  --seeds 0 \
  --embedding-batch-size 16 \
  --local-output-dir artifacts/eventstream-h5-recent-fold/embeddings/seed0
```

调度器只下载共享尾盘缓存、对应 seed 的训练 manifest 和 best checkpoint。任务完成后会回传 embedding、manifest、运行摘要和 Colab 执行记录，并核对 seed、源码 revision、OOS 状态和 2026 隔离状态。直接在已有 CUDA 环境执行时可使用底层入口：

```bash
ticknet-eventstream-export-embeddings \
  --close-cache artifacts/eventstream-h5-recent-fold/daily-close-cache \
  --checkpoint artifacts/eventstream-h5-recent-fold/training/seed0/eventstream-top400-h5-capacity100m-recent.seed0.best.pt \
  --training-manifest-root artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  --model capacity100m \
  --device cuda \
  --allow-oos \
  --output artifacts/eventstream-h5-recent-fold/embeddings/seed0 \
  --source-revision "$(git rev-parse HEAD)"
```

三个 seed 已按源码 revision `449b843c83d7494ae7a396d658792eaa664ab2eb` 完成导出。本地 manifest 全量校验与 Drive 逐文件核对均通过，每组 39,903 行，股票日主键和顺序完全一致。

| seed | embedding 数据指纹 | 文件字节数 |
|---:|---|---:|
| 0 | `a4d67c5f06a3147d036a43700bcc88bd2e5b47b74c934a4255192255f0435b36` | 146,103,015 |
| 1 | `850ed79795d34b8e040bacad174abc3ba4b4942f865b6b9f560439fcef78530a` | 145,899,441 |
| 2 | `c51bceff90a52982b57fd1c9c4999fed4e27285993a8c00a38379e444f61e43a` | 145,939,261 |

重复执行 seed 1 和 2 后，运行下游对照：

```bash
ticknet-embedding-compare \
  --minute-config configs/nextday-minute-formal-2025-v2.yaml \
  --minute-features results/m3-formal-minute-features-v2-202107 \
  --comparison-config configs/embedding-frozen-recent-2025.yaml \
  --embedding artifacts/eventstream-h5-recent-fold/embeddings/seed0 \
  --embedding artifacts/eventstream-h5-recent-fold/embeddings/seed1 \
  --embedding artifacts/eventstream-h5-recent-fold/embeddings/seed2 \
  --output results/embedding-frozen-recent-2025
```

结果包含 HGB 与 LambdaMART 的分钟特征、embedding、组合特征三组对照。主要指标为 Rank IC、`NDCG@50/100`、`Precision@50/100`、Top-K 成本后收益、换手率和月度稳定性。风险暴露诊断需要额外提供含 `trading_date`、`symbol`、`industry`、`size`、`liquidity`、`volatility` 的 Parquet。缺少该文件时结果会明确记录 `unavailable`。

### 冻结表征正式结果

`FEAT-EMB-FROZEN-001` 使用 22,409 个训练样本、6,963 个 validation 样本和 8,125 个 OOS 样本。事件流对最近折分钟候选的覆盖率为 96.69%。validation 有 18 个评估日，OOS 有 21 个评估日。表中的 E1 和 E2 使用三个 seed 各自训练下游模型，再平均预测分数。

| 下游模型 | 输入 | validation Rank IC | OOS Rank IC | OOS `NDCG@100` | OOS `Precision@100` | OOS Top-100 日均成本后主动收益 | OOS 日均单边换手 |
|---|---|---:|---:|---:|---:|---:|---:|
| HGB | E0 分钟特征 | 0.01808 | 0.04010 | 0.53424 | 0.26810 | -11.51bp | 62.89% |
| HGB | E1 embedding | 0.02647 | 0.01966 | 0.52349 | 0.25905 | -13.14bp | 64.30% |
| HGB | E2 组合 | 0.02833 | 0.05701 | 0.54450 | 0.26667 | -4.80bp | 60.91% |
| LambdaMART | E0 分钟特征 | -0.04334 | 0.00766 | 0.52153 | 0.30143 | -0.73bp | 48.95% |
| LambdaMART | E1 embedding | -0.01117 | 0.03414 | 0.53030 | 0.29286 | 15.53bp | 52.57% |
| LambdaMART | E2 组合 | -0.05081 | 0.01389 | 0.52695 | 0.31143 | 6.91bp | 52.32% |

HGB E2 的单 seed OOS Rank IC 为 0.04333、0.05644、0.05912，全部高于 E0 的 0.04010。三 seed 预测均值的 OOS 配对增量为 0.01691，21 天中有 16 天优于 E0，逐日 bootstrap 95% 区间为 0.00596 至 0.02851。validation 的配对增量为 0.01025，区间仍跨过零。HGB E2 已形成当前折内较稳定的表征增量，成本后主动收益仍为负。

LambdaMART E2 在三个 seed 和两个月之间波动较大。E1 在 OOS 的成本后主动收益为正，validation 为 -15.34bp，暂时只保留为待复核线索。风险暴露输入尚未提供，行业、规模、波动率和流动性诊断均记录为 `unavailable`。结果文件为 `results/embedding-frozen-recent-2025/comparison.json`，数据指纹为 `56a7689048e539963a217c92221e8cddf1ce472526115411d5478a4a6d18dc00`。

当前决策保留 frozen E2 和 HGB 作为候选，允许一个 seed 的联合端到端小实验。该小实验需要固定相同股票日、标签和评估口径，并以当前 E2 为直接对照。最近折只有一个 validation 月和一个 OOS 月，联合实验完成后仍需额外时间窗口复核。150M 继续等待这些结果。

正式训练使用相同 revision 恢复 checkpoint，并在训练结束后评估 2025 年 12 月 OOS。以下命令保留为复现实验入口：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-train \
  --session ticknet-eventstream-h5-recent-seed0-formal-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 20 \
  --evaluate-test \
  --timeout 21600 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/training/seed0
```
