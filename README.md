# 深度学习 tick 数据预测

本项目从复现论文 DeepLOB 起步，目标是逐步长出一个面向真实业务的 A 股次日方向预测模型。
论文复现目前是临摹阶段，用来对齐方法和验证代码，后续会随着业务需求推进而逐步退出主线。
现在的重点是端到端链路：用 A 股十档盘口数据训练模型，根据交易日下午的盘口状态预测下一
交易日的横截面方向。

如果你是第一次接触这个项目，建议直接看下面的端到端主线，论文复现只是验证代码正确的起点。

- 论文：[arXiv 1808.03668](https://arxiv.org/abs/1808.03668)
- 作者公开实现：[GitHub 仓库](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

## 端到端主线：A 股次日方向预测

核心问题很简单：一只股票在某天下午收盘前的盘口长什么样，能否用来判断它下一个交易日相对
全市场的涨跌位置。

### 思路

把一只股票一个交易日的盘口演变切成固定长度的块，每块用 DeepLOB 编码，再用 GRU 把当天的
块汇总成一个向量，最后同时输出两个结果：

- 连续超额收益分数，用于横截面排序
- 下跌、中性、上涨三类概率

标签按下一交易日个股开盘到收盘收益减去中证全指同期收益计算。同一天所有股票按这个超额
收益排序，最低的 20% 标为下跌，中间的 60% 标为中性，最高的 20% 标为上涨。每个股票每个
交易日只生成一个样本，不会出现同一天多个 tick 共享一个标签的情况。

### 输入和模型

输入是信号时点前最后 200 个十档盘口事件，每个事件 40 个原始盘口特征（十档买卖价与量），
形状为 `200 × 40`。数据切成 2 个 100 事件块，分别经过共享权重的 DeepLOB 编码器，再由 GRU
汇总。模型支持混合精度、梯度累积和断点恢复。

价格先转成相对窗口首笔中间价的基点变化，数量做对数缩放，全程只用训练期信息，避免未来
函数。

### 训练和评估

时间按完整交易日切分，训练、验证、测试三段互不重叠，跨边界的标签样本会被清除。当前主线
配置使用 2021 至 2023 训练、2024 验证、2025 锁定测试，每天动态筛选成交额前 400 只股票。

评估以验证期日均 Rank IC 为主要模型选择指标，正式测试期只在模型和训练设置冻结后解锁一次，
用多随机种子跑固定 best checkpoint，不按测试结果挑种子。报告口径包括 Macro F1、MCC、Brier、
每日 Rank IC 和分组收益差。多空收益差不含手续费和冲击成本，不能直接当作可交易回测。

### 当前进度

端到端链路已经能训练并产出信号，数据准备、分块编码、双头训练、断点恢复、推理 CLI 和自动化
测试都已就绪。项目围绕两个问题逐步推进，都已跑出真实结论。

第一个问题是分钟级特征是否有稳定的次日信息。用聚合特征 HGB 在 2022 到 2025 四个独立样本外
年份做滚动验证，每日 Rank IC 全部为正，区间约 0.02 到 0.035。信号真实存在，但强度弱。进一步
做成本后回测发现，日频换手率约 83%，单边 10 个基点的成本下净年化收益为负，盈亏平衡成本约
5 到 6 个基点，低于实际可实现的成本。审计还发现多空组合的收益集中在少数极端交易日，前 5 天
贡献超过全部收益。结论是信号跨年稳健但不足以覆盖真实交易成本，当前口径下停止扩大模型。

第二个问题是原始盘口序列相对聚合特征是否有增量。分钟级 TCN 与 HGB 的同口径对比在验证集上
显示 TCN 排序能力更强，但测试集上 HGB 全面占优，TCN 的验证集优势没有泛化到样本外。

当前主链路使用 2021 至 2023 训练、2024 验证、2025 锁定测试，每天动态筛选成交额前 400 只股票。
正式锁定测试仍需在模型冻结后评估。硬盘里的逐笔委托和成交数据尚未进入模型，只用了 snapshot
十档盘口。

项目还实现了参考 AgentX 论文的实验研究闭环。ExperimentSpec v2 通过白名单 typed executor、
结构化指标门槛、独立 artifact、完整 Registry 和强制预测审计，把提案到评估的确定性部分串联
起来；locked test 使用绑定 spec、checkpoint、预测和数据指纹的一次性批准。详见
[docs/topk-agentx-m2a-deterministic-loop.md](docs/topk-agentx-m2a-deterministic-loop.md) 和
[docs/topk-agentx-m2b-locked-approval.md](docs/topk-agentx-m2b-locked-approval.md)。Registry prediction
导出、多 seed 对比和 walk-forward 聚合见
[docs/topk-agentx-m2c-executors-comparison.md](docs/topk-agentx-m2c-executors-comparison.md)。

完整的研究问题、数据资源、硬件预算和分阶段实验路线见
[docs/hardware-constraints-and-experiment-roadmap.md](docs/hardware-constraints-and-experiment-roadmap.md)。数据格式、适配命令、泄漏控制和评估口径见
[docs/nextday-cross-sectional-prediction.md](docs/nextday-cross-sectional-prediction.md)。

## 第一次运行

以下步骤都不需要真实行情数据，可以先确认代码链路正常：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
python scripts/check.py
```

`scripts/check.py` 会依次跑 Ruff、格式检查、ty 类型检查、带覆盖率的 pytest 和冒烟脚本。
也可以单独运行 `python scripts/smoke_test.py`、`python -m pytest -q`、`ruff check .`、
`ty check`。

Linux、macOS 和 Colab 使用对应环境的虚拟环境激活命令。

## 用 A 股数据跑通端到端

本机已有沪深月度 snapshot Parquet 时，先准备数据：

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw.yaml
```

这一步会按动态前 400 股票池筛选、截取 14:30 至 14:55 的最后 200 个有效盘口、计算下一交易日
超额收益标签，写出 float16 分片和 manifest。然后用训练配置启动：

```powershell
ticknet-nextday-train --config configs/nextday.yaml
```

训练完成后把一只股票信号时点前的原始 `N × 40` snapshot 转成次日信号：

```bash
ticknet-nextday-predict \
  --checkpoint checkpoints-nextday/raw-200-dual-head.seed0.best.pt \
  --manifest data/nextday-raw-200/manifest.json \
  --events-npy data/today-000001.npy \
  --input-format raw \
  --device cpu
```

输出包含连续分数、映射回收益尺度的预期超额收益、三类概率和方向编号。横截面交易优先用同一天
股票的分数排序。

数据格式、对照基线、Colab 端到端笔记本和当前限制见
[docs/nextday-cross-sectional-prediction.md](docs/nextday-cross-sectional-prediction.md)。

## 论文复现：临摹起点

FI-2010 复现用于对齐论文结构和训练协议，是验证代码正确的基线，不回答 A 股次日方向问题。
这部分已随真实业务需求推进而退出主线，完整复现链路（含 `ticknet-fi2010-train` 入口、
Setup 1 和 Setup 2）已归档到 `legacy/`，不再参与主链路开发与质量门禁，仅作临摹参考。

模型输入为最近 100 个订单簿状态，每个状态 40 个原始特征，经过三个卷积块、三分支 Inception、
64 单元 LSTM，输出三分类。预测跨度为 `10/20/30/50/100`，Adam 使用 `lr=0.01` 和 `eps=1`，
mini-batch 为 32，验证准确率连续 20 个 epoch 未提升时早停。

仓库暂未提交真实 FI-2010 训练结果，因此数值复现仍待完成。数据格式、转换命令、实验协议和
核对结果见 [docs/reproduction-audit.md](docs/reproduction-audit.md)。

## 文档导航

细节都收录在 `docs/`：

- [docs/nextday-cross-sectional-prediction.md](docs/nextday-cross-sectional-prediction.md)：次日预测的数据格式、适配、切分、训练和评估
- [docs/hardware-constraints-and-experiment-roadmap.md](docs/hardware-constraints-and-experiment-roadmap.md)：研究问题、数据资源、硬件预算和分阶段路线
- [docs/topk-agentx-research-roadmap.md](docs/topk-agentx-research-roadmap.md)：Top-K 可交易目标、隐藏表征、多日模型与 AgentX 的逐项执行路线
- [docs/topk-agentx-m0-research-contract.md](docs/topk-agentx-m0-research-contract.md)：新研究系列的数据权限、交易口径、2026 可用性审计与基线冻结
- [docs/topk-agentx-m1-portfolio-evaluator.md](docs/topk-agentx-m1-portfolio-evaluator.md)：fixed-K long-only 选股、缓冲、成交成本和评估 artifact 契约
- [docs/topk-agentx-m2a-deterministic-loop.md](docs/topk-agentx-m2a-deterministic-loop.md)：ExperimentSpec v2、typed executor、Registry v2、强制 Audit 与剩余安全边界
- [docs/topk-agentx-m2b-locked-approval.md](docs/topk-agentx-m2b-locked-approval.md)：内容绑定、不可重放的 locked-test 人工批准与消费流程
- [docs/topk-agentx-m2c-executors-comparison.md](docs/topk-agentx-m2c-executors-comparison.md)：Registry prediction 导出、多 seed 对比与 walk-forward 稳健性聚合
- [docs/resource-strategy-and-pilot-gates.md](docs/resource-strategy-and-pilot-gates.md)：分钟信号验证、成本评估和实验研究闭环
- [docs/raw-200-end-to-end-pipeline.md](docs/raw-200-end-to-end-pipeline.md)：原始盘口主线从本地加工到训练的门槛式推进
- [docs/reproduction-audit.md](docs/reproduction-audit.md)：FI-2010 复现的模型、协议和核对结论
- [docs/development-guide.md](docs/development-guide.md)：模块划分、测试范围和质量门禁

维护者和代码代理的使用约定见 [AGENTS.md](AGENTS.md)。

## 项目结构

```text
src/ticknet/        模型、数据集和训练逻辑（FI-2010 临摹）
src/ticknet/nextday 次日标签、分片数据集、分块模型、指标和训练逻辑（A 股主线）
src/ticknet/research  实验研究闭环：提案、策略校验、实验登记、预测审计和 Agent 框架
scripts/            数据准备、基线和冒烟检查等人工执行入口
tests/              不依赖真实数据的自动化测试
configs/            本地和 Colab 配置
docs/               复现核对、次日预测、实验路线与开发维护说明
references/         论文原文和阅读笔记
legacy/             FI-2010 复现归档，不再参与主链路开发
notebooks/          Colab 入口笔记本
```

## FI-2010 数据格式

官方文本文件形状为 `149 × N`，每列一个样本。第 0 至 39 行是十档买卖盘的价格和数量，即
DeepLOB 输入。第 40 至 143 行是 104 个手工特征，转换后保留但不进入模型。第 144 至 148 行
是五个预测标签，列索引对应跨度 `10/20/30/50/100`。本项目使用论文采用的 `NoAuction` 和
`z-score` 版本。

数据转换、论文实验协议、Colab 用法、检查点与训练记录等完整说明见
[docs/reproduction-audit.md](docs/reproduction-audit.md) 和 [docs/development-guide.md](docs/development-guide.md)。
