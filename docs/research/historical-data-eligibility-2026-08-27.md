# 历史 raw L2 数据准入边界

## 当前决定

2026 年数据暂不进入历史事件流重建和模型数据集。2021 至 2025 年建立股票日准入清单，深市合格股票日作为主数据，沪市合格股票日作为研究数据。

主数据和研究数据都要求 order_preopen、order、trades、snapshot 文件存在，且目标股票出现在三类关联文件中，并且存在盘前委托。深市主数据使用已验证的 +140ms 审计配置。沪市只保留覆盖资格，不写入固定 lag，仍需股票日级时间审计。

## 当前数据证据

2026 年共有 70 个 order_preopen 日文件，其中 55 个日期没有对应 order 和 trades 文件，但有 snapshot。这是数据湖分区缺失，不是撮合规则问题。2026 年剩余 15 个日期的三类文件路径齐全。

2021-01-04 的真实 smoke 扫描包含 2335 个深市股票日，其中 2283 个同时有开盘成交，2283 个通过当前主数据准入，52 个因关联股票记录缺失被排除。

沪市样本的最佳 lag 不是市场常量。已观察到同一市场和不同股票日出现 0、20、90、150、240、280、370、380ms。600000 在 2023-05-12 使用 240ms、2024-12-13 使用 280ms 后十档精确匹配。

## 这是 A 股普遍问题还是当前数据问题

A 股 Level-2 研究普遍需要处理集合竞价、连续竞价、快照和逐笔事件之间的时间与语义契约。深交所公开说明 Level-2 同时包含十档快照和逐笔行情，交易规则还区分开盘集合竞价、连续竞价和收盘集合竞价。上交所行情网关接口也明确区分快照和逐笔行情，并定义逐笔委托或成交的行情生成时间。

因此，开盘阶段需要审计是市场数据研究的普遍工程要求。它不等于所有 A 股 tick 数据都同样脏。

本数据集的特有问题包括：

- 2026 年存在整段 order 和 trades 文件缺失。
- 2021 年早期沪市盘前订单覆盖不足。
- 深市存在跨样本稳定的 +140ms 偏移，说明供应链至少做过统一时钟转换。
- 沪市出现股票日级甚至股票级 lag 差异，说明不能直接复用深市配置。
- 部分 snapshot 的 Volume 和 DealNum 与盘口最佳 lag 不能由同一个事件边界完全解释。

当前证据没有显示 2021 至 2025 年 orders 和 trades 在已检查样本中普遍随机损坏。更接近的判断是文件覆盖、时间标签和统计窗口没有形成统一的数据契约。

## 事件排序边界

真实 order Parquet 若保留 `ChannelNo` 与 `ApplSeqNum`、`BizIndex` 等 exchange sequence 字段，simulator 会把这些字段保留到 `SimulatorEvent`。同一 `time_ms` 内，只有事件都具备 sequence 且属于单一 channel 时才按 sequence 重排；snapshot 仍放在同毫秒 order/cancel 之后。

如果同毫秒出现多个 channel，simulator 保留文件 source order，并在 `SimulatorPack.ordering_provenance` 中记录 `cross_channel_total_order=false`。如果数据源没有 sequence，则明确记录 `timestamp_fallback`。因此时间戳始终是跨 channel 时间坐标，sequence 只在有证据支持的范围内增强局部顺序，不能把供应商文件推断成交易所全局总序。

## 使用边界

深市 2021 至 2025 的合格股票日可用于主事件流研究，但必须使用准入清单和已验证的时间配置。沪市 2022 至 2025 的合格股票日可用于 lag 和数据契约研究，不能直接与深市共用撮合配置。2026 年先排除，待 order 和 trades 补齐后重新建清单。

## 运行入口

    PYTHONPATH=src .venv/bin/python scripts/build_historical_data_manifest.py \\
      --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \\
      --json-output /tmp/historical-manifest.json \\
      --csv-output /tmp/historical-manifest.csv

准入规则实现于 src/ticknet/simulator/eligibility.py。覆盖扫描实现于 src/ticknet/simulator/coverage.py。

## 外部规则参考

- [深交所投资者服务：交易时间和集合竞价](https://www.szse.cn/www/investor/knowledge/stock/deal/t20190626_568129.html)
- [深交所新一代交易系统 FAQ：Level-2 快照和逐笔行情](https://www.szse.cn/www/marketServices/technicalservice/introduce/P020180328467244590967.pdf)
- [上交所 IS120 行情网关 STEP 接口规范](https://www.sse.com.cn/services/tradingtech/development/c/10816478/files/51a3e4c6b92345689c682448582c019d.pdf)
