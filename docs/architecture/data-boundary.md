# TickNet 数据边界

本仓库是 `market-data-platform` 已发布市场数据的模型消费者，不是通用市场数据资产 owner。

## Owner 分工

`market-data-platform` 负责：

- raw 数据接入、字段标准化和 canonical schema；
- 重复、缺失、乱序、异常值等通用质量检查；
- 数据版本、provenance、质量 receipt 和 published asset。

本仓库的 `ticknet` 负责：

- 将 canonical L2/eventstream 通过显式 adapter 转成模型输入；
- 模型专属的盘口归一化、window 切分和特征 embedding；
- horizon label、leakage 检查、训练/验证切分；
- tensor materialization、模型训练、replay 和评估。

## 依赖方向

```text
market-data-platform
  raw -> normalized -> canonical -> quality/provenance -> published asset

deep-learning-tick-data-prediction
  published asset -> adapter -> model window/features/labels -> train/evaluate
```

`ticknet` 不应 import `market_data_platform` 的业务实现。跨仓输入应通过已发布文件、schema、receipt 或本仓的显式 adapter 接入。只有当某个清洗规则被多个消费者复用时，才应回到 `market-data-platform`；仅服务于模型输入的归一化和窗口逻辑留在本仓。

当前对应实现包括 `ticknet.eventstream.canonical_adapter`、`ticknet.nextday.snapshot_features` 和 `ticknet.nextday.snapshot_io`。它们负责消费和转换，不重新定义平台的 raw/canonical 数据资产。
