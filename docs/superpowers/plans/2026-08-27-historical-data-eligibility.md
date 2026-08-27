# Historical Data Eligibility Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (recommended). Steps use checkbox syntax for tracking.

Goal：建立 2021 至 2025 历史 raw L2 股票日准入清单，冻结 2026，并区分深市主数据和沪市研究数据。

Architecture：在现有 CoverageRow 之上增加纯数据准入模块。准入只依据文件覆盖、股票覆盖、盘前委托和交易日范围，不把未经证明的沪市 lag 写入规则。CLI 输出 JSON、CSV 和分层计数。

Tech Stack：Python 3.10、dataclasses、argparse、csv、json、pytest。

Spec：docs/research/opening-coverage-inventory-2026-08-27.md。

## Global Constraints

- 2026 年数据不进入历史准入清单。
- 深市主数据要求三类文件存在、股票出现在三类文件、存在盘前订单。
- 沪市研究数据使用同样的文件完整性条件，但必须单独标记 lag 未校准。
- 不改变撮合器、事件流打包格式和沪市默认 lag。
- 合成测试不依赖 6TB 硬盘。

### Task 1: 准入判定

Files：
- Create: src/ticknet/simulator/eligibility.py
- Test: tests/test_historical_eligibility.py

- [ ] 写测试覆盖深市主数据、沪市研究数据、缺文件、缺股票、2026 排除和无盘前订单。
- [ ] 运行测试确认模块不存在导致失败。
- [ ] 实现 EligibilityRow、classify_coverage 和 summarize_eligibility。
- [ ] 运行 focused pytest。
- [ ] 提交 feat: add historical data eligibility rules。

### Task 2: 准入清单 CLI

Files：
- Create: scripts/build_historical_data_manifest.py
- Modify: tests/test_historical_eligibility.py
- Modify: docs/README.md

- [ ] 写 JSON、CSV 和分层摘要测试并确认失败。
- [ ] 实现 --raw-root、--json-output、--csv-output、--start-year、--end-year 和 --limit-days。
- [ ] 运行 focused pytest 和 --help。
- [ ] 提交 feat: add historical data manifest CLI。

### Task 3: 研究记录和质量门禁

Files：
- Create: docs/research/historical-data-eligibility-2026-08-27.md
- Modify: docs/project-status.md

- [ ] 记录 2021 至 2025 的准入定义、深市主数据和沪市研究数据边界。
- [ ] 记录交易所公开接口资料支持的普遍性背景和本数据集特有的缺口。
- [ ] 运行 focused pytest、pre-commit、scripts/check.py、git diff --check。
- [ ] 推送、创建 PR、合并 main，删除分支和 worktree，验证 main 同步。
