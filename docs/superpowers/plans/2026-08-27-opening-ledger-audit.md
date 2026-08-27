# Opening Ledger Audit Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal：建立可复跑的盘前订单身份账本审计，跨股票和交易日验证盘前委托、成交、撤单能否重建开盘前十档盘口。

Architecture：在 `simulator` 下新增纯函数审计模块，先从盘前订单生成订单级剩余量，再应用盘前成交和撤单，最后聚合前十档并与第一张完整连续竞价快照比较。CLI 只负责从 raw L2 数据读取样本、输出 JSON 摘要，不把未经验证的账本直接接入撮合回放。

Tech Stack：Python 3.10、dataclasses、PyArrow Parquet、pytest、argparse。

Spec：`docs/nextday/eventstream.md` 与 `src/ticknet/eventstream/config.py` 中的 raw L2 文件契约。

## Global Constraints

- 测试只使用合成数据，不依赖 6TB 硬盘或外部服务。
- 价格按 raw L2 已缩放的整数分处理，数量按股处理。
- 盘前账本只使用 `time_ms < 0` 的委托、成交和撤单，`time_ms == 0` 的连续竞价事件留给后续回放。
- 审计结果必须区分精确匹配、不可比较、未知成交身份、未知撤单身份和数量不一致。
- 不改变现有 `day_input_files()` 返回结构，不修改 eventstream 打包格式。

### Task 1: 订单级盘前账本纯函数

Files：
- Create: `src/ticknet/simulator/opening_ledger.py`
- Test: `tests/test_opening_ledger.py`

- [x] 写测试：盘前买卖委托经过成交和部分撤单后，按价格聚合出正确剩余量和前十档。
- [x] 写测试：未知成交 ID、未知撤单 ID、超出剩余量的成交分别被记录，不静默吞掉。
- [x] 运行测试确认在模块不存在时按预期失败。
- [x] 实现 `audit_opening_ledger(orders, trades, cancels, snapshot_levels)`，返回不可变审计结果。
- [x] 运行测试通过并保持字段含义与 raw L2 单位一致。

### Task 2: 真实 Parquet 样本读取

Files：
- Modify: `src/ticknet/simulator/opening_ledger.py`
- Test: `tests/test_opening_ledger.py`

- [x] 写测试：从合成 `order_preopen`、`trades` 和 `snapshot` Parquet 读取指定股票日，并选出首个完整非负快照。
- [x] 运行测试确认读取接口尚未实现时失败。
- [x] 实现 `audit_opening_day(day, ticker, raw_root)`，使用 `day_preopen_file()`、`day_input_files()` 和现有 snapshot 解析约定。
- [x] 对盘前交易按 `BuyID`、`SellID` 扣减对应订单，对盘前撤单按 `OrderID` 扣减，保留未知身份计数和数量。
- [x] 运行测试通过，验证股票过滤、日期过滤、价格单位和快照缺档状态。

### Task 3: 跨股票跨日期 CLI 与摘要

Files：
- Create: `scripts/audit_opening_ledger.py`
- Test: `tests/test_opening_ledger_cli.py`

- [x] 写测试：CLI 接受多个 `--sample YYYYMMDD:TICKER`，输出 JSON，并汇总 exact、mismatch、not_comparable 和身份缺口。
- [x] 运行测试确认 CLI 在实现前失败。
- [x] 实现 CLI 的显式样本模式和 `--raw-root` 参数，避免默认扫描数 TB 数据。
- [x] 输出每个样本的十档价格差、数量差、未知 ID 和订单覆盖率，汇总只对可比较样本计算比例。
- [x] 用 2021、2023、2025 三个年份的深市样本和至少一个沪市样本执行真实审计。

### Task 4: 文档、质量门禁与 PR

Files：
- Modify: `docs/project-status.md`
- Modify: `docs/nextday/eventstream.md`

- [x] 将真实审计命令和结果写入现状文档，明确样本范围、数据日期和限制。
- [x] 运行 `pre-commit run --all-files`。
- [x] 运行 `python scripts/check.py`。
- [ ] 提交、推送、创建 PR，合并到 `main`。
- [ ] 合并后删除分支和 worktree，确认 `main` 与 `origin/main` 同步且干净。
