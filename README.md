# 深度学习 tick 数据预测

本项目用 A 股逐笔行情研究下一交易日的横截面排序信号。项目从 DeepLOB 论文复现起步，现在主要维护真实数据训练、成本评估和受控实验研究三类能力。FI-2010 复现已归档到 `legacy/`，论文和阅读笔记放在 `docs/references/`。

## 这个项目做什么

项目维护四条相互配合的链路：

1. 原始盘口模型读取信号时点前的十档快照
2. 分钟模型用较低成本检验聚合量价特征
3. 事件流 Transformer 直接编码委托、成交和快照
4. 研究闭环管理实验身份、成本评估、审计和锁定数据访问

所有主线都按交易日切分，在时间外数据上比较股票的横截面顺序。项目同时检查换手和交易成本，Rank IC 为正只代表模型捕捉到排序信号，还不能直接说明策略可交易。

## 当前判断

截至 2026-08-22，100M 事件流模型已经在最近折三 seed 和第一个相邻滚动折 seed 0 中得到正的 H5 样本外 Rank IC。冻结 embedding 与分钟特征组合、联合端到端训练也得到正的排序增量。每日截面 z 标签的 seed 0 两折实验进一步提高了 validation 和 OOS Rank IC。

现有候选仍未稳定覆盖单边 10bp 交易成本，头部收益也会随月份变化。信号半衰期、交易规则、已具备数据的风险暴露、多任务梯度、标签尺度和监督位置已经检查完成。最后位置和尾部加权都没有超过全位置基线，下一步检查日级任务权重，再决定是否实现成本感知排序目标。150M 容量实验继续暂缓。

分钟模型、原始盘口模型和 AgentX 成本矩阵都已经形成阶段结论。完整数字、数据限制和下一步统一记录在[项目现状](docs/project-status.md)和[实验日志](docs/research/experiment-log.md)。模型原理与取舍见[模型清单](docs/model-catalog.md)，外部研究路线带来的改进计划见[外部 L2 研究项目对比](docs/research/external-l2-research-comparison.md)。

## 快速验证

以下步骤不需要真实行情数据，支持 Python 3.10 及以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
python scripts/check.py
```

`scripts/check.py` 会运行 Ruff、格式检查、ty、带覆盖率的 pytest 和 FI-2010 兼容模型冒烟检查。

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活环境。修改 `pyproject.toml` 中的命令入口后，需要重新执行可编辑安装，让 `.venv/bin/` 或 Windows 的 `.venv\Scripts\` 生成新入口。

## 从哪里开始

- 想了解项目全貌和当前进度，阅读[项目现状](docs/project-status.md)
- 想选择模型，阅读[模型清单](docs/model-catalog.md)
- 想准备真实数据或运行训练，阅读[次日横截面预测规范](docs/nextday/cross-sectional-prediction.md)和[事件流说明](docs/nextday/eventstream.md)
- 想了解实验边界与后续计划，阅读[AgentX 研究路线](docs/research/topk-agentx-research-roadmap.md)
- 想确认数据清洗与模型输入的归属，阅读[数据边界](docs/architecture/data-boundary.md)
- 想查找其他专题说明和命令，使用[文档索引](docs/README.md)

## 项目结构

```text
src/ticknet/            共享训练工具和兼容模型
src/ticknet/nextday     次日标签、分片、分钟模型和原始盘口模型
src/ticknet/eventstream L2 事件流打包、因果 Transformer 和预测导出
src/ticknet/research    实验提案、执行、审计、Registry 和研究 Agent
scripts/                数据准备、基线和本地检查入口
tests/                  不依赖真实行情的主链路自动化测试
configs/                本地与 Colab 配置
docs/                   当前说明、路线图和实验记录
docs/references/        论文与阅读笔记归档
legacy/                 FI-2010 复现归档
legacy/notebooks/       已退休 Colab 流程的 Python 快照
```

维护约定见 [AGENTS.md](AGENTS.md)。FI-2010 的数据格式和复现边界见[复现核对](docs/reproduction-audit.md)。
