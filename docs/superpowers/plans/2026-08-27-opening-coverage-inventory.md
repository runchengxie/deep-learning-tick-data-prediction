# Opening Coverage Inventory Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal：建立跨交易日、股票和数据批次的 raw L2 盘前覆盖清单，并为沪市 lag 分层和剩余失败样本提供可复用的报告输入。

Architecture：新增独立覆盖扫描模块，逐个读取 `order_preopen` 日文件，按股票聚合盘前委托，再关联 order、trades 和 snapshot。报告输出 JSON、CSV 和按年份、月份、市场、批次聚合的摘要。生产撮合器和沪市默认 lag 保持不变。

Tech Stack：Python 3.10、PyArrow Parquet、dataclasses、argparse、csv、json、pytest。

Spec：`docs/research/shanghai-opening-contract-audit-2026-08-27.md` 和 raw L2 文件布局。

## Global Constraints

- 真实数据扫描只读 raw L2，不把数据产物提交到 Git。
- 合成测试使用临时 Parquet，不依赖硬盘盒数据。
- 开盘成交定义为 `time_ms <= 0` 的 trades 行。
- 覆盖状态区分文件存在、股票存在和开盘成交是否存在。
- 不改变撮合器、事件流打包格式和沪市默认 lag。

### Task 1: 覆盖清单纯函数和 Parquet 扫描接口

Files：
- Create: `src/ticknet/simulator/coverage.py`
- Test: `tests/test_opening_coverage.py`

- [ ] 先写测试，验证盘前文件、股票记录、三类关联文件和 `time_ms <= 0` 成交统计彼此独立。
- [ ] 运行 focused pytest，确认测试因模块不存在而失败。
- [ ] 实现 `CoverageRow`、`scan_preopen_coverage` 和 `summarize_coverage`，使用批量 Parquet 读取，支持 snapshot 日文件和月文件。
- [ ] 运行 focused pytest，确认合成 Parquet 测试通过。
- [ ] 提交 `feat: add raw opening coverage scanner`。

### Task 2: 覆盖报告 CLI 和分层摘要

Files：
- Create: `scripts/audit_opening_coverage.py`
- Modify: `tests/test_opening_coverage.py`
- Modify: `docs/README.md`

- [ ] 先写 CLI JSON、CSV 和 year、month、market、batch 摘要测试并确认失败。
- [ ] 实现 `--raw-root`、`--json-output`、`--csv-output` 和可选的 `--limit-days`。
- [ ] 运行 focused pytest 和 `--help`。
- [ ] 提交 `feat: add opening coverage report CLI`。

### Task 3: 真实数据扫描和研究记录

Files：
- Create: `docs/research/opening-coverage-inventory-2026-08-27.md`
- Modify: `docs/project-status.md`

- [ ] 用 6TB raw root 生成报告到 `/tmp`，不把 CSV 或 JSON 加入 Git。
- [ ] 记录总日文件、stock-day、缺文件、缺股票、开盘成交缺口、盘前委托量和各层 lag 输入范围。
- [ ] 明确这是覆盖审计，不等于十档身份账本已经被证明完整。
- [ ] 提交 `docs: record raw opening coverage inventory`。

### Task 4: 质量门禁和 PR 收口

- [ ] 运行 focused pytest、`pre-commit run --all-files`、`python scripts/check.py` 和 `git diff --check`。
- [ ] 推送 `agent/opening-coverage-inventory`，创建 PR，合并到 `main`。
- [ ] 删除合并后的本地和远程分支及 worktree，prune 后确认 `main` 干净且同步。
