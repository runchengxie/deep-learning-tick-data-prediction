# TickNet 数据边界

本仓库消费 `market-data-platform` 已发布的市场数据。通用市场数据资产由 `market-data-platform` 持有。

## Owner 分工

`market-data-platform` 负责：

- provider API 接入、raw landing、字段标准化和 canonical schema。
- 重复、缺失、乱序、异常值等通用质量检查。
- 数据版本、provenance、质量 receipt 和 published asset。

本仓库的 `ticknet` 负责：

- 将 canonical L2/eventstream 通过显式 adapter 转成模型输入。
- 模型专属的盘口归一化、window 切分和特征 embedding。
- horizon label、leakage 检查、训练与验证切分。
- tensor materialization、模型训练、replay 和评估。

## 依赖方向

```text
market-data-platform
  provider API -> ingest -> raw -> standardize -> canonical -> quality/provenance -> published asset

deep-learning-tick-data-prediction
  published asset -> adapter -> model window/features/labels -> train/evaluate
```

`ticknet` 通过已发布文件、schema、receipt 或本仓显式 adapter 接入跨仓输入。可复用的通用清洗规则回到 `market-data-platform`。只服务于模型输入的归一化、窗口、特征和标签逻辑留在本仓。

当前对应实现包括 `ticknet.eventstream.canonical_adapter`、`ticknet.nextday.snapshot_features` 和 `ticknet.nextday.snapshot_io`。这些模块消费和转换平台资产，不重新定义平台 raw/canonical 资产的 owner。

## 可执行门禁

`tests/test_data_boundary.py` 把上述依赖方向纳入 pytest 门禁。`scripts/check.py` 会运行完整 pytest，因此本地 pre-push 和 CI 都会执行这些检查。

门禁包含两层：

- `src/ticknet` 禁止直接 import `market_data_platform`、`tushare` 和 `rqdatac`。
- 项目 runtime dependencies 禁止声明 `market-data-platform`、`tushare` 和 `rqdatac`。

第一层防止模型代码直接依赖平台实现或 provider SDK。第二层防止 provider runtime 先进入模型项目依赖，再逐步形成新的数据接入职责。未来若出现独立的 schema-only distribution，应单独评审并显式调整白名单。
