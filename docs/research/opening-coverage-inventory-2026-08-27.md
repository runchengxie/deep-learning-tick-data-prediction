# raw L2 盘前覆盖清单

## 审计目标

本轮建立交易日和股票粒度的 raw L2 覆盖扫描器，区分盘前文件存在、股票有盘前订单、关联文件存在、股票在关联文件中出现以及开盘成交是否存在。扫描器只做数据审计，不改变撮合器和沪市默认 lag。

## 统计口径

盘前订单量统计 order_preopen 中非撤单记录的行数和 Volume 之和。开盘成交定义为 trades 中 time_ms <= 0 的记录，开盘成交量为这些记录的 Volume 之和。snapshot 支持按日文件、按月文件和带连字符的日文件名。

batch 使用盘前文件所在的月份目录作为当前可复现的原始文件批次代理。数据湖目前没有单独的批次元数据，因此这个字段不宣称代表供应商批次。

## 运行方式

使用项目虚拟环境运行：

    PYTHONPATH=src .venv/bin/python scripts/audit_opening_coverage.py \
      --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
      --json-output /tmp/opening-coverage.json \
      --csv-output /tmp/opening-coverage.csv

扫描可用 --limit-days 分段执行。输出文件建议放在 /tmp 或硬盘盒上的非仓库目录，不提交到 Git。

全量扫描建议指定 `--index-path` 保存覆盖索引。首次扫描仍会读取关联的原始文件，后续运行会按文件大小和修改时间复用未变化的交易日。需要强制重扫时增加 `--refresh-index`。

    PYTHONPATH=src .venv/bin/python scripts/audit_opening_coverage.py \
      --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
      --index-path /mnt/data/hdd6t/quant-data-lake/projects/level2-coverage-index.json \
      --json-output /tmp/opening-coverage.json \
      --csv-output /tmp/opening-coverage.csv

## 首个真实 smoke 结果

使用 6TB raw L2 的 2021-01-04 盘前文件运行 --limit-days 1：

| 指标 | 数量 |
|---|---:|
| 盘前股票日 | 2335 |
| 有开盘成交的股票日 | 2283 |
| 三类关联文件都存在 | 2335 |
| 三类文件中股票都出现 | 2283 |
| 盘前委托记录 | 1112321 |
| 盘前委托量 | 4675764535 |
| 开盘成交记录 | 200882 |
| 开盘成交量 | 451384210 |

这个日文件的扫描耗时约 26 秒。主要耗时来自关联的大型订单、成交和月度 snapshot 文件。全量 1282 个盘前日文件应按日期分段运行，避免单次任务长时间占用资源。覆盖索引可以把这次读取成本摊到后续审计中，但不会减少首次读取原始 orders 和 trades 的成本。

## 代码边界

ticknet.simulator.coverage 按日读取关联文件。订单 ticker 存在性只扫描目标盘前股票，找到当日全部目标后提前停止。成交和 snapshot 按目标股票及交易日过滤后聚合，因此不会为每个股票重复读取同一日文件。

报告中的完整文件不代表完整身份链。股票同时出现在三类文件中，也不能证明每笔开盘订单都具备可回链的订单身份。后续仍需在覆盖清单上抽取沪市样本，调用 lag 扫描和订单级 trace，分析年份、月份、文件批次和股票的 lag 分布。

## 下一步

先完成全量覆盖清单，再按有盘前股票记录的沪市股票日运行 -200ms 至 200ms 的 lag 扫描。lag 只有在跨日期稳定、十档匹配明显改善且 Volume 和 DealNum 同时一致时，才具备进入市场配置的依据。
