# 深度学习 tick 数据预测

本项目用 A 股逐笔行情研究下一交易日的横截面排序信号。项目从 DeepLOB 论文复现起步，现在主要维护真实数据训练、成本评估和受控实验研究三类能力。FI-2010 复现已归档到 `legacy/`，论文和阅读笔记放在 `references/`。

## 项目能力

当前代码包含四条可以独立运行的链路：

1. 原始盘口链路读取信号时点前最后 200 或 1,000 个十档快照，由分块 DeepLOB 和 GRU 编码，输出连续分数与三分类概率
2. 分钟聚合链路提供 HGB、TCN 和 GRU，用较低成本检验分钟量价特征
3. L2 事件流链路无损打包委托、成交和快照，由因果 Transformer 完成事件任务与日级信号输出
4. 研究闭环用 ExperimentSpec、白名单执行器、Registry、预测审计和锁定测试审批管理实验

这些链路共用按交易日切分、时间外评估和横截面排序的基本原则。不同研究阶段使用的锁定区间有所区别，开始实验前请查看[项目现状](docs/project-status.md)和[研究契约](docs/research/topk-agentx-m0-research-contract.md)。

## 当前结论

截至 2026-08-16，已经落地的主要结论如下：

- 分钟 HGB 在 2022 至 2025 四个历史滚动样本外年份的每日 Rank IC 均为正，约为 0.02 至 0.035。2021 年上半年委托源缺少沪市记录，这组结果保留为带数据限制的历史基线。信号强度不足以覆盖现实交易成本
- 分钟 TCN 的验证集排序能力高于 HGB，优势未延续到测试集
- 原始盘口 Top-100 的三 seed 受控矩阵已经完成。`1M/raw-200` 的验证 Rank IC 为 `0.03748 ± 0.00096`，是四格中最稳的候选。扩大到 100M 参数或把窗口增至 raw-1000 都没有形成稳定增益
- L2 事件流最近折已完成 2025 年 8 月至 12 月的 103 个交易日打包和 A100 输入基准，正式训练尚未开始，因此目前只有工程结论
- AgentX 研究闭环的 M0 至 M2 已完成。M3 v1 在物化 14/60 个月后发现 2021 年上半年委托源缺少沪市记录，已经停止。M3 v2 改从 2021 年 7 月开始物化 54 个月，正式 Top-K 预测和成本诊断仍待执行

完整状态、数据权限和下一步见[项目现状](docs/project-status.md)。带日期和数字的研究记录见[实验日志](docs/research/experiment-log.md)。

## 快速开始

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

## 运行真实数据链路

在已经准备好沪深月度快照 Parquet 的机器上，可以先生成分片，再启动训练：

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw.yaml
ticknet-nextday-train --config configs/nextday.yaml
```

训练完成后，可以把单只股票的原始事件转成次日信号：

```bash
ticknet-nextday-predict \
  --checkpoint checkpoints-nextday/raw-200-dual-head.seed0.best.pt \
  --manifest data/nextday-raw-200/manifest.json \
  --events-npy data/today-000001.npy \
  --input-format raw \
  --device cpu
```

输出包含连续分数、映射回收益尺度的预期超额收益、三类概率和方向编号。横截面交易使用同一天全部股票的分数排序。

分钟模型使用 `ticknet-minute-gru-train` 和 `ticknet-minute-tcn-train`。事件流使用 `ticknet-eventstream-pack`、`ticknet-eventstream-storage-readiness`、`ticknet-eventstream-train` 和 `ticknet-eventstream-export-predictions`。研究闭环统一从 `ticknet-research` 进入。完整命令见[文档索引](docs/README.md)。

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
references/             论文与阅读笔记
legacy/                 FI-2010 复现归档
notebooks/              Colab 入口笔记本
```

维护约定见 [AGENTS.md](AGENTS.md)。FI-2010 的数据格式和复现边界见[复现核对](docs/reproduction-audit.md)。
