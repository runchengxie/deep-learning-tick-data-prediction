# 深度学习 tick 数据预测

用 A 股交易所逐笔数据训练模型，预测下一交易日的横截面涨跌位置。项目从复现 DeepLOB 论文起步，现在以真实业务的端到端链路为主，论文复现已经归档到 `legacy/`，只作临摹参考。

论文原文和阅读笔记放在 `references/`，见该目录下的 `README.md`。

## 项目在做什么

核心问题：一只股票当天盘口和成交的表现，能不能用来判断它下一个交易日相对全市场的涨跌位置。数据只使用信号时点之前的信息，时间按完整交易日切分，训练、验证、测试三段互不重叠。

当前有三条模型主线，共用同一套横截面标签和评估口径：

1. 原始盘口主线，输入信号时点前最后 200 个十档 snapshot 事件，切成两个 100 事件块，用共享权重的 DeepLOB 编码器提特征，GRU 汇总成向量，输出连续超额收益分数和下跌、中性、上涨三分类概率
2. 分钟聚合主线，把分钟级量价序列分别喂给 HGB 树模型、TCN 和 GRU 三种模型，用作低成本对照
3. L2 逐笔事件流主线，把委托、成交、快照三流无损打包，喂给因果 Transformer 做下一事件预测和日级信号输出，详见 [docs/nextday/eventstream.md](docs/nextday/eventstream.md)

配套的实验研究闭环参考 AgentX 论文实现，把提案、训练、审计、锁定测试串成确定性流程，负责可交易 Top-K 组合的评估与诊断。

## 当前结论

分钟级特征携带真实的次日信息，但强度不足以覆盖交易成本。HGB 在 2022 至 2025 四个独立样本外年份的每日 Rank IC 全部为正，约 0.02 至 0.035。日频换手约 83%，单边 10 个基点成本下净年化为负，盈亏平衡成本约 5 至 6 个基点，低于实际可实现水平。多空组合的收益还集中在少数极端交易日。

分钟 TCN 在验证集上的排序能力强于 HGB，但优势没有泛化到测试集。原始盘口和逐笔事件流主线的正式结论仍在推进中。详细结果和证据链见 [docs/research/topk-agentx-research-roadmap.md](docs/research/topk-agentx-research-roadmap.md)。

## 快速开始

以下步骤不需要真实行情数据，可以确认代码链路正常。环境为 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
python scripts/check.py
```

`scripts/check.py` 依次跑 Ruff、格式检查、ty 类型检查、带覆盖率的 pytest 和冒烟脚本。也可以单独运行 `python scripts/smoke_test.py`、`python -m pytest -q`、`ruff check .`、`ty check`。Linux 和 macOS 使用对应的虚拟环境激活命令。

## 用真实数据跑通主链路

在已有沪深月度 snapshot Parquet 的机器上，先准备数据再训练。

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw.yaml
ticknet-nextday-train --config configs/nextday.yaml
```

训练完成后用推理 CLI 把单只股票信号时点前的原始事件转成次日信号：

```bash
ticknet-nextday-predict \
  --checkpoint checkpoints-nextday/raw-200-dual-head.seed0.best.pt \
  --manifest data/nextday-raw-200/manifest.json \
  --events-npy data/today-000001.npy \
  --input-format raw \
  --device cpu
```

输出包含连续分数、映射回收益尺度的预期超额收益、三类概率和方向编号。横截面交易优先用同一天股票的分数排序。

分钟线用 `ticknet-minute-gru-train` 和 `ticknet-minute-tcn-train`，逐笔事件流用 `ticknet-eventstream-pack` 和 `ticknet-eventstream-train`，完整命令见对应文档。

## 文档导航

技术细节全部放在 `docs/`，按主题分目录，从 [docs/README.md](docs/README.md) 进入。建议阅读顺序是主链路规范、端到端流程、研究路线图、开发指南。

维护者和代码代理的使用约定见 [AGENTS.md](AGENTS.md)。

## 项目结构

```text
src/ticknet/            模型、数据集和训练逻辑
src/ticknet/nextday     次日标签、分片数据集、分块模型、指标和训练入口
src/ticknet/eventstream L2 逐笔事件流打包、因果 Transformer、预测导出
src/ticknet/research    实验研究闭环，提案、审计、Registry 和 Agent 框架
scripts/                数据准备、基线和冒烟检查等人工执行入口
tests/                  不依赖真实数据的自动化测试
configs/                本地和 Colab 配置
docs/                   技术文档，按主题分目录
references/             论文原文和阅读笔记
legacy/                 FI-2010 复现归档，不参与主链路开发
notebooks/              Colab 入口笔记本
```

## FI-2010 数据格式

官方文本文件形状为 `149 × N`，每列一个样本。第 0 至 39 行是十档买卖盘的价格和数量，即 DeepLOB 输入。第 40 至 143 行是 104 个手工特征，转换后保留但不进入模型。第 144 至 148 行是五个预测标签，列索引对应跨度 10、20、30、50、100。项目使用论文采用的 NoAuction 和 z-score 版本。完整说明见 [docs/reproduction-audit.md](docs/reproduction-audit.md)。
