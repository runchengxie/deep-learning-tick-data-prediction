# Shanghai Contract Audit Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal：建立跨股票、跨日期的开盘数据契约审计，定位沪市事件时钟、盘前订单覆盖和开盘十档数量差异。

Architecture：在现有 `opening_ledger` 之上增加样本覆盖状态、候选事件 lag 扫描和订单级差异追踪。报告入口接受显式样本列表，输出 JSON 和 CSV，不自动修改撮合器，也不把沪市 lag 写入默认配置。

Tech Stack：Python 3.10、dataclasses、PyArrow Parquet、pytest、argparse、csv 和 json。

Spec：`docs/nextday/eventstream.md`、`src/ticknet/eventstream/config.py` 和当前 raw L2 文件布局。

## Global Constraints

- 测试使用合成 Parquet，不依赖 6TB 硬盘。
- 价格单位为 raw L2 已缩放的整数分，数量单位为股。
- 候选 lag 使用闭区间 `-200ms` 至 `200ms`，默认步长为 `10ms`。
- `not_comparable` 不计入可比较样本的匹配率。
- 不改变撮合器、事件流打包格式和沪市默认 lag。

### Task 1: 覆盖状态和候选 lag 纯函数

Files：
- Modify: `src/ticknet/simulator/opening_ledger.py`
- Test: `tests/test_opening_ledger.py`

- [x] 写测试：区分盘前文件缺失、文件存在但股票无记录、股票有盘前记录和订单/成交都可闭合。
- [x] 写测试：在候选 lag 中选出十档匹配且身份缺口最少的最佳 lag，并在并列时选择绝对值更小者。
- [x] 运行测试确认新接口按预期失败。
- [x] 实现覆盖状态、候选 lag 扫描和结果摘要，不改变现有 `audit_opening_day` 默认行为。
- [x] 运行测试通过。

### Task 2: 沪市样本报告和差异追踪

Files：
- Modify: `src/ticknet/simulator/opening_ledger.py`
- Create: `scripts/audit_shanghai_contract.py`
- Test: `tests/test_shanghai_contract_cli.py`

- [x] 写测试：报告保存样本日期、股票、盘前覆盖、最佳 lag、匹配状态、每档数量差和身份缺口。
- [x] 写测试：指定价格档能列出贡献该档的订单剩余量、成交量和撤单量。
- [x] 运行测试确认 CLI 在实现前失败。
- [x] 实现显式 `--sample`、`--raw-root`、`--lag-min`、`--lag-max`、`--lag-step`、`--json-output` 和 `--csv-output` 参数。
- [x] 对不可比较样本保留原因，不把它们强行归入撮合器失败。
- [x] 运行测试通过并核对 `--help`。

### Task 3: 真实数据探索

Files：
- Create: `docs/research/shanghai-opening-contract-audit-2026-08-27.md`

- [x] 运行跨年份深市、沪市样本，记录盘前股票覆盖率和最佳 lag 分布。
- [x] 运行 `600000 @ 20220615` 的价格 777 档追踪，区分成交边界、撤单语义和原始遗漏。
- [x] 将真实结果、样本列表、命令和限制写入研究文档。

### Task 4: 质量门禁与 PR 收口

Files：
- Modify: `docs/project-status.md`
- Modify: `docs/nextday/eventstream.md`

- [x] 更新现状和事件流文档中的审计入口与最新结论。
- [x] 运行 `pre-commit run --all-files`。
- [x] 运行 `python scripts/check.py`。
- [ ] 提交、推送、创建 PR 并合并到 `main`。
- [ ] 合并后删除本地和远程分支与 worktree，确认 `main` 干净且同步。
