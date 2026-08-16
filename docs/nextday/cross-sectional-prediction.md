# 次日横截面预测

## 研究问题

这条实验链路与 FI-2010 论文复现相互独立。原始盘口支持 raw-200 和 raw-1000 两种窗口。下面以当前候选 raw-200 为例：

```text
一只股票 × 一个输入交易日
  当天信号时点前最后 200 个十档盘口事件
    2 个 100-event 块
      DeepLOB 编码每个块
        GRU 汇总当天块序列
          连续超额收益分数 + 下跌/中性/上涨概率
```

默认标签按下一交易日收益横截面排序，最低 20% 记为 0，中间 60% 记为 1，最高 20% 记为 2。分位切点处的并列收益保留为中性，避免用股票代码任意拆开同收益样本。也可以改用固定收益阈值。传入基准日收益后，标签和 Rank IC 都使用超额收益。

每个股票交易日只生成一个样本。代码不会把同一天的每个 tick 复制成共享一个标签的独立样本。

端到端主线直接训练原始 tick 模型。raw-200 让五年、动态 Top-400 的 float16 工作集保持在约 8 GB。raw-1000 与 100M 参数模型也已经实现并完成受控比较，当前证据仍支持较小的 `1M/raw-200`。分钟微观结构模型保留为低成本对照。资源预算和扩展门槛见[硬件约束与分阶段实验路线](hardware-constraints-and-experiment-roadmap.md)。

raw-200 默认结构为 16 个卷积通道、每个 Inception 分支 32 个通道、64 维日内 embedding 和 64 维日级 GRU，共 86,775 个参数。`configs/nextday-raw-1m-pilot.yaml` 提供独立的容量实验配置，对应宽度为 32、64、320 和 192，共 1,033,383 个参数。新配置使用不同的 checkpoint 目录，不会覆盖默认基线。训练和推理 checkpoint 都记录完整结构签名。

## 与论文复现的边界

FI-2010 论文复现，包括 `ticknet-fi2010-train` 入口、`FI2010WindowDataset`、Setup 1 和 Setup 2，已归档到 `legacy/`，不再属于主链路。新链路位于 `ticknet.nextday`，训练命令为 `ticknet-nextday-train`。FI-2010 缺少可靠的股票和交易日边界，不能用来生成这里的次日标签。

## 输入特征

上游事件数组用 `float32` 计算，最终训练分片可以保存为 `float16` 或 `float32`，形状为 `events × 40`。数据集会在送入模型前转回 `float32`。40 列必须按十档盘口顺序排列：

```text
ask_price_1, ask_size_1, bid_price_1, bid_size_1,
ask_price_2, ask_size_2, bid_price_2, bid_size_2,
...
ask_price_10, ask_size_10, bid_price_10, bid_size_10
```

事件数组必须已经完成以下处理：

- 只保留信号时点以前的数据
- 价格、数量单位和盘口档位保持一致
- 不包含 NaN 和无穷值
- 需要拟合的标准化参数只使用训练日期
- 不使用测试期复权信息或全样本统计量

现有真实数据适配器使用固定、无需拟合的变换。价格转换为相对选中窗口第一个中间价的基点变化并除以 100，数量转换为 `log1p(size) / 16`，最后裁剪到固定范围。这样既保留窗口内价格变化，也不会用到验证期或测试期统计量。变换参数会写入 manifest。

## 原始文件适配

本机已有沪深月度 snapshot Parquet 时，直接运行：

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw.yaml
```

适配器会做以下事情：

- 用前一日以前的 20 日成交额代理，生成历史动态流动性前 400 股票池
- 使用 14:30 至 14:55 的候选快照，保留最后 200 个有效十档盘口状态
- 计算下一交易日个股开盘到收盘收益减中证全指同期收益
- 按月利用 ticker row-group 范围跳过无关股票
- 输出 float16 大分片、manifest 和 `data-audit.json`

以上是原始序列模型保留的历史诊断口径。M3 Top-K 的正式 HGB 输入使用独立配置 `configs/nextday-minute-formal-2025-v2.yaml`，目标改为 T 日信号对应 T+1 open 至 T+2 open 的持有收益，并按同期中证全指 open-to-open 收益形成分类监督。动态 Top-400 只使用 T 日以前的成交额。T+1 停牌、一字涨停和一字跌停分别写入交易状态，调出但尚不可卖的持仓继续写 `in_universe=false` 状态行。正式切分还要求 `return_end_date` 落在同一时间段内。

```bash
uv run python scripts/materialize_minute_features.py \
  --config configs/nextday-minute-formal-2025-v2.yaml \
  --output results/m3-formal-minute-features-v2-202107

uv run python scripts/run_minute_baseline.py \
  --config configs/nextday-minute-formal-2025-v2.yaml \
  --materialized-features results/m3-formal-minute-features-v2-202107 \
  --evaluate-test \
  --save-predictions results/predictions-hgb-top400-open2open-2025-v2.parquet \
  --output results/nextday-minute-formal-2025-v2.json
```

第一条命令按月原子写入 Parquet，记录源文件身份、目标键、每月资源统计和分片 SHA-256。中断后重复执行即可从完整月份继续。第二条命令只接受覆盖全部目标月份的 manifest，输出正式数据指纹和 prediction metadata，供 `import_predictions` 登记。缺少 L2 分钟窗口的候选使用 HGB 原生支持的全 NaN 特征并保留在 Top-400 中，不能静默删除。

M3 v1 已物化 14/60 个月。2025 年 7 月至 12 月共 49,600 行，其中 192 行缺少分钟特征，真实特征覆盖率为 99.61%。恢复处理 2021 年数据后，发现 1 月至 5 月约有 48% 至 50% 的候选缺少全部三模态特征。逐日原始委托文件确认 2021 年 1 月 4 日至 6 月 4 日只覆盖深市，沪市从 6 月 7 日才开始出现。v1 已停止并保留，v2 从首个完整月份 `2021-07` 开始重新物化。

连续数组抽取器在 2025 年 7 月生成的分片与旧版 SHA-256 完全一致，耗时从 100.3 秒降至 72.4 秒，峰值内存从约 2.26 GB 降至 1.69 GB。其余五个月耗时在 62.0 至 82.6 秒之间，批次峰值不超过 1.72 GB。这组结果只验证全量执行的资源、恢复和数值一致性边界，不代表预测效果。

修正后的本机冒烟 v2 覆盖 2024-01-02 至 2024-01-12 的动态前 20 股票。180 个目标中写出 178 个，两个股票日缺少可用 snapshot。178 个写出样本均有完整 200 个有效 tick。430 个 row group 中读取 44 个并跳过 386 个。候选时段共剔除 1,017 行无效盘口。该结果只验证数据链路，不代表模型有效。

如果行情商格式不同，也可以继续使用通用事件清单入口。`scripts/prepare_nextday.py` 使用三个必需输入。

日线 CSV：

```csv
symbol,trading_date,open,close
000001.SZ,2024-01-02,9.41,9.53
000001.SZ,2024-01-03,9.55,9.49
```

交易日历文本，每行一个日期：

```text
2024-01-02
2024-01-03
```

事件 JSONL 清单：

```json
{"symbol":"000001.SZ","trading_date":"2024-01-02","features_path":"events/000001.SZ-2024-01-02.npy","last_event_timestamp":"2024-01-02T14:54:59+08:00","signal_timestamp":"2024-01-02T14:55:00+08:00"}
```

`features_path` 相对 JSONL 所在目录解析。这里刻意不绑定某家行情商的原始格式。上游适配器可以从按月 Parquet、数据库或对象存储生成股票日数组和 JSONL 清单。

可选基准 CSV 包含 `trading_date,return`。收益率使用小数，例如 `0.005` 表示 0.5%。

生成 2 个 100-event 块：

```powershell
python scripts/prepare_nextday.py `
  --daily-bars data/daily-bars.csv `
  --calendar data/calendar.txt `
  --events-manifest data/events.jsonl `
  --benchmark data/benchmark.csv `
  --output-dir data/nextday `
  --label-method cross_sectional `
  --min-cross-section 100 `
  --chunks-per-sample 2 `
  --chunk-size 100 `
  --samples-per-shard 512
```

通用事件清单入口在事件不足 200 条时会左侧重复当天第一条有效盘口，清单中的 `valid_events` 保留真实事件数。原始 snapshot 主配置把 `min_valid_events` 设为 200，先剔除无效盘口再取最后 200 条，不足完整窗口的股票日不会写入。

输出结构：

```text
data/nextday/
  manifest.json
  shards/
    part-00000.npy
    part-00001.npy
```

每个分片形状为 `samples × chunks × time × 40`。NPY 可以内存映射，训练不需要一次把全部数据载入内存。`manifest.json` 保存股票、输入日、标签日、收益、标签、信号时间、有效事件数和分片位置。每个分片记录文件大小和 SHA-256，清单记录覆盖全部样本元数据与分片哈希的 `dataset_fingerprint`。

## 日期切分和泄漏控制

配置使用三个互不重叠的完整日期区间。正式工作集为 2021 至 2023 年训练、2024 年验证、2025 年锁定测试：

```yaml
train_start: "2021-01-01"
train_end: "2023-12-31"
val_start: "2024-01-01"
val_end: "2024-12-31"
test_start: "2025-01-01"
test_end: "2025-12-31"
```

数据集要求输入日和标签日同时落入同一区间。训练期最后一天对应的标签若进入验证期，该样本会被自动清除。某个交易日的全部股票使用同一个日期分配规则，不会随机拆到不同集合。

还需要由上游股票池保证历史时点可得性，包括上市日期、停牌、涨跌停、退市和流动性筛选。当前代码不会从今天的股票列表反推历史股票池。

## 训练和基线

编辑 `configs/nextday.yaml` 后运行：

```powershell
ticknet-nextday-train --config configs/nextday.yaml
```

检查点和训练历史写入 `checkpoint_dir`。恢复训练会核对日期、模型、学习率和 `dataset_fingerprint` 等实验签名，配置或数据冲突时停止。训练开始时默认顺序校验全部分片 SHA-256。

正式配置的 `evaluate_test` 默认为 `false`。此时训练只计算验证指标，结果 JSON 中的 `test` 为 `null`。验证期模型、配置和随机种子集合冻结后，用纯评估入口一次读取全部 best checkpoint：

```bash
ticknet-nextday-evaluate \
  --seeds 0 1 2 3 4 \
  --config configs/nextday.yaml
```

纯评估入口会在计算测试指标前确认全部 checkpoint 存在，并核对日期、数据指纹、模型和训练配置。它不创建优化器，不读取 last checkpoint，也不会进入训练循环。结果写入 `locked_test.<checkpoint_name>.seeds0-1-2-3-4.json`，包含每个种子的指标以及跨种子的均值和样本标准差。不能按测试结果挑选种子。

训练入口保留 `evaluate_test` 供冒烟闭环使用。正式 locked test 用上面的纯评估入口，避免恢复尚未达到 `epochs` 上限的训练任务。

训练后可以直接把一只股票信号时点前的原始 `N × 40` snapshot NPY 转成次日信号：

```bash
ticknet-nextday-predict \
  --checkpoint checkpoints-nextday/raw-200-dual-head.seed0.best.pt \
  --manifest data/nextday-raw-200/manifest.json \
  --events-npy data/today-000001.npy \
  --input-format raw \
  --device cpu
```

输出包含标准化连续 `score`、映射回收益尺度的 `expected_excess_return`、三个类别概率和方向编号。横截面交易优先使用同一天股票的 `score` 排序。单只股票的收益数值需要另做校准，不能直接解释为承诺收益率。

先运行聚合特征 Logistic Regression 基线：

```powershell
python scripts/run_nextday_baseline.py `
  --config configs/nextday.yaml `
  --output results/nextday-baseline.json
```

基线同样遵守 `evaluate_test`。正式配置冻结后才用 `--evaluate-test` 生成测试指标。

基线对每个股票日计算 40 列事件特征的均值、标准差、最后值和首尾变化，再用只在训练集拟合的 `StandardScaler` 与类别平衡 Logistic Regression 训练。它是判断原始序列模型是否提供额外价值的最低对照，替代不了更完整的微观结构特征或 LightGBM 基线。

## 评估口径

双头模型用分类交叉熵和标准化连续收益的 Smooth L1 联合训练。训练结果包含：

- Accuracy、Macro F1 和 Weighted F1
- Balanced Accuracy 和 MCC
- 每类 Precision 和 Recall
- 多分类 Brier 分数
- 每日 Rank IC 的均值、标准差和 ICIR
- 按预测分数最高组减最低组计算的无成本日均收益差

多空收益差没有包含手续费、滑点、涨跌停、冲击成本和做空约束，不能当成可交易回测。模型选择默认用验证期日均 Rank IC。正式测试期由 `ticknet-nextday-evaluate` 显式解锁，只在训练和模型选择结束后评估固定 best checkpoint。

## 数据量和 Colab

使用 float16 时，400 只股票、五年约 1,250 个交易日、每股票日 200 个事件的纯特征约 8 GB，同样范围的 500 和 1,000 个事件约为 20 GB 和 40 GB。分片方案让 Colab 可以逐批读取，但不会让原始数据自动变小。

建议把原始逐笔和完整盘口保存在移动硬盘、NAS 或对象存储，只生成当前研究需要的紧凑分片。Colab 运行时把当前实验所需分片顺序复制到 `/content`，Drive 保存工作集、checkpoint 和结果。不要通过 Drive 挂载点直接随机训练。可运行入口见 [`notebooks/nextday_end_to_end_colab.ipynb`](../../notebooks/nextday_end_to_end_colab.ipynb)。

12 个交易日适合验证工程链路。初步横截面研究仍建议至少 120 至 250 个交易日，正式样本外实验建议覆盖两至三年和多个市场状态。

本机 CPU 用于一次性提取原始 tick 分片和数据审计，Colab GPU 用于端到端训练。现有 `level2_minute_cache` 用来建立低成本对照，判断原始 tick 是否真的带来额外价值。

2024 受控 pilot 使用动态前 100 股票和最后 200 个盘口事件：

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw-pilot.yaml
python scripts/run_nextday_baseline.py --config configs/nextday-pilot.yaml \
  --output results/nextday-pilot-baseline.json
```

`configs/nextday-pilot.yaml` 使用 2024H1 训练、2024Q3 验证和 2024Q4 测试。默认 `evaluate_test: false`，因此基线和 DeepLOB 选模阶段都不会输出测试指标。日度横截面指标至少要求 80 只股票，低于动态股票池 80% 覆盖的日期不进入日度指标。百万参数开发实验可改用：

```bash
ticknet-nextday-train --config configs/nextday-raw-1m-pilot.yaml --seed 0
```

分别运行 seed 0、1、2，并且只用验证期比较容量增量。2024Q4 已经用于既有 pilot 结果，不能重新作为未见测试集选择百万参数模型。规范月文件读取损坏时，转换器只会在目标交易日的逐日 snapshot 备份完整时回退，并在 `data-audit.json` 记录回退月份和文件数。逐日备份不完整时会停止运行。

## 当前状态和下一步

当前版本已经实现真实月度 snapshot 适配、双头分块编码、AMP、梯度累积、断点恢复和 Colab 交接。五年 raw-200 工作集、raw-1000 Top-100 工作集、多周期评估和四格三 seed 容量矩阵均已完成，详见 [raw-data-expansion-roadmap.md](raw-data-expansion-roadmap.md) 和 [multi-horizon-data-expansion-roadmap.md](multi-horizon-data-expansion-roadmap.md)。硬盘中的逐笔委托和成交由事件流主线承接，说明见 [eventstream.md](eventstream.md)。

当前只保留 `1M/raw-200` 作为候选。下一步需要先冻结三 seed 聚合方式、checkpoint 和一次性测试通过条件，再决定是否解锁 2025 测试。多日版本可以缓存每日 embedding 后再训练，无需反复编码原始 tick。
