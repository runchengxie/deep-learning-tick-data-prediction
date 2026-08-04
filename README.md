# DeepLOB 复现项目

本项目使用 PyTorch 复现论文 DeepLOB: Deep Convolutional Neural Networks for
Limit Order Books 的 FI-2010 实验，并提供独立的 tick/LOB 到次日横截面方向研究链路。

- 论文：[arXiv 1808.03668](https://arxiv.org/abs/1808.03668)
- 作者公开实现：[GitHub 仓库](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
- 数据集：[FI-2010 官方页面](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)

## 当前状态

项目已经具备可运行的模型、数据转换、论文实验协议、训练恢复、指标汇总和测试。
仓库暂未提交真实 FI-2010 的训练结果，因此还没有完成论文数值复现。当前代码可以用于
启动 Table I 和 Table II 实验，训练完成后仍需核对各预测跨度的 Accuracy、Precision、
Recall 和 F1。

已经对齐的内容包括：

- 输入窗口为最近 100 个订单簿状态，每个状态使用前 40 个原始订单簿特征
- 三个卷积块、三分支 Inception 模块、64 单元 LSTM 和三分类输出
- 模型参数量约 6 万
- 预测跨度为 `10/20/30/50/100`
- Adam 参数为 `lr=0.01` 和 `eps=1`
- mini-batch 大小为 32
- 验证准确率连续 20 个 epoch 未提升时早停
- FI-2010 Setup 1 和 Setup 2 的官方 `CF` 文件口径

详细核对结果见 [docs/reproduction-audit.md](docs/reproduction-audit.md)。

## 次日横截面预测

新链路不使用 FI-2010 标签，也不改变论文复现入口。它把每个股票交易日的最后若干个
十档盘口事件切成固定长度块，用 DeepLOB 编码每个块，再由 GRU 汇总并同时预测下一交易日
开盘到收盘的连续超额收益分数与横截面方向。

已经实现：

- 按交易日历生成次日开盘到收盘收益和横截面三分类标签
- 每个股票交易日一个样本，信号时点和标签日期泄漏检查
- float16/float32 NPY 大分片、内存映射和训练前 float32 转换
- 完整交易日 walk-forward 切分和边界标签 purge
- 分块 DeepLOB、日内 GRU、连续分数/三分类双头、AMP、梯度累积和恢复训练
- Macro F1、MCC、Brier、每日 Rank IC 和分组收益差
- 聚合日内特征加 Logistic Regression 基线
- 现有沪深月度 snapshot Parquet 的动态股票池适配器
- 可复制 Drive 分片并断点续训的 Colab notebook
- 分片 SHA-256、数据清单指纹和可切换的 locked-test 评估
- 从原始 `N × 40` snapshot NPY 返回连续分数和方向概率的推理 CLI

本机客户交付主线使用 `configs/nextday-raw.yaml` 生成 200-tick 数据，再用
`configs/nextday.yaml` 训练。数据格式、转换命令、Colab 存储方式和研究限制见
[docs/nextday-cross-sectional-prediction.md](docs/nextday-cross-sectional-prediction.md)。本机资源预算、端到端交付主线、
内部对照和扩展门槛见
[docs/hardware-constraints-and-experiment-roadmap.md](docs/hardware-constraints-and-experiment-roadmap.md)。

## 安装

项目要求 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
```

Linux、macOS 和 Colab 使用对应环境的虚拟环境激活命令。

## 本地快速检查

以下命令都不需要真实数据：

```powershell
python scripts/check.py
```

`pre-commit install` 会同时安装提交前和推送前 hook。每次 `git push` 前自动执行完整
本地质量门禁。确需临时跳过时可以显式使用 `git push --no-verify`。

也可以单独运行：

```powershell
python scripts/smoke_test.py
deeplob-train --config configs/base.yaml
python -m pytest -q
ruff check .
ruff format --check .
ty check
```

合成数据训练只验证代码链路，不产生可与论文比较的指标。

## FI-2010 数据格式

官方文本文件的形状为 `149 × N`，每列代表一个样本：

- 第 0 至 39 行是十档买卖盘的价格和数量，也是 DeepLOB 的模型输入
- 第 40 至 143 行是 104 个手工特征，本项目在转换文件中保留这些列
- 第 144 至 148 行是五个预测标签

标签列映射如下：

| 预测跨度 | 转置后的列索引 |
|---:|---:|
| 10 | 144 |
| 20 | 145 |
| 30 | 146 |
| 50 | 147 |
| 100 | 148 |

本项目使用论文采用的 `NoAuction` 和 `z-score` 版本。请使用官方
`BenchmarkDatasets.zip`，第三方 CSV 镜像可能采用不同的列布局。

## 数据转换

解压官方数据后运行：

```powershell
python scripts/convert_fi2010.py `
  --base-dir C:\path\to\BenchmarkDatasets\BenchmarkDatasets `
  --auction NoAuction `
  --norm z-score `
  --folds 1 2 3 4 5 6 7 8 9 `
  --out FI2010_normalised.npy
```

转换脚本会生成：

```text
FI2010_normalised.npy
FI2010_normalised_meta.json
```

NPY 文件使用 `float32`，形状为 `N × 149`。元数据记录每个 Training 和
Testing 源文件在 NPY 中的行范围。训练时必须同时提供两个文件。

转换过程逐个读取源文件并写入磁盘映射数组，内存占用主要取决于单个源文件，
不会同时保留全部 18 个矩阵。

## 论文实验协议

### Setup 1，对应 Table I

FI-2010 的 `CF_1` 至 `CF_9` 已经是锚定前向切分。第 `i` 次实验使用
`Train_*_CF_i.txt` 训练，并使用 `Test_*_CF_i.txt` 测试。训练代码不会把
其余八个 Training 文件并入当前训练集。

```powershell
deeplob-train `
  --config configs/colab.yaml `
  --data-path path\to\FI2010_normalised.npy `
  --meta-path path\to\FI2010_normalised_meta.json `
  --protocol setup1 `
  --k 10 `
  --device cuda
```

命令会依次运行九个 `CF`。调试时可以用
`--setup1-cfs 7 8 9` 限制运行范围。

### Setup 2，对应 Table II

Setup 2 使用 `CF_7` 的 Training 文件训练，并把 `CF_7`、`CF_8` 和
`CF_9` 的 Testing 文件作为测试集。

```powershell
deeplob-train `
  --config configs/colab.yaml `
  --data-path path\to\FI2010_normalised.npy `
  --meta-path path\to\FI2010_normalised_meta.json `
  --protocol setup2 `
  --k 10 `
  --device cuda
```

每个预测跨度需要单独运行一次。

训练段最后 20% 用于验证。训练窗口和验证窗口之间留出 99 行间隔，因此两组窗口
不会共享原始行。测试窗口也不会跨越三个 Testing 源文件的拼接位置。官方矩阵没有
提供文件内部的股票和日期边界，本项目无法识别这类内部边界，详见复现核对文档。

## Colab

把以下文件上传到 Google Drive：

```text
MyDrive/DeepLOB/data/FI2010_normalised.npy
MyDrive/DeepLOB/data/FI2010_normalised_meta.json
```

`scripts/run_colab.py` 也兼容已有的 `FI2010_meta.json` 文件名。

在 Colab 主内核中运行：

```python
from google.colab import drive

drive.mount("/content/drive")
```

随后克隆仓库并启动训练：

```python
!git clone https://github.com/runchengxie/deeplob-reproduction.git
%cd deeplob-reproduction
!python scripts/run_colab.py --protocol setup2 --k 10
```

`scripts/run_colab.py` 会安装当前项目，检查数据和元数据，选择 CUDA，并把检查点
保存到 `MyDrive/DeepLOB/checkpoints/`。脚本默认先把数据和元数据顺序复制到 Colab
本地目录 `/content/DeepLOB/data/`，训练期间不再通过挂载的 Google Drive 反复读取。
同一运行时内再次启动会复用未变化的本地副本。Colab 运行时重置后，本地副本会消失，
脚本会重新复制，Drive 中的检查点不受影响。

本地盘空间不足或需要对比 Drive 读取性能时，可以关闭复制：

```python
!python scripts/run_colab.py --protocol setup2 --k 10 --no-local-copy
```

## 检查点和训练记录

每次实验会生成四类文件：

- `*.last.pt` 保存最近一个 epoch 的模型、优化器、早停状态和训练历史
- `*.best.pt` 保存验证准确率最高的状态
- `train_history.*.json` 供查看和绘制训练曲线
- `result.*.json` 保存配置、环境、耗时和测试指标

训练记录还包含每个 epoch 的训练秒数、验证秒数和训练吞吐量，便于判断 GPU 是否在
等待数据。复制数据时会显示文件大小、耗时和平均吞吐量。

开启 `resume` 后，训练从 `*.last.pt` 继续。实验配置与检查点不一致时会直接报错，
避免误用其他预测跨度或协议的状态。

绘制曲线：

```powershell
python scripts/plot_curves.py `
  --checkpoint-dir ./checkpoints `
  --out results/train_curves.png
```

## 项目结构

```text
src/deeplob/        模型、数据集和训练逻辑
src/deeplob/nextday 次日标签、分片数据集、分块模型、指标和训练逻辑
scripts/            数据转换、Colab 入口、冒烟检查和绘图
tests/              不依赖真实数据的自动化测试
configs/            本地和 Colab 配置
docs/               复现核对与开发维护说明
references/         论文原文和阅读笔记
notebooks/          Colab 入口笔记本
```

开发约定和质量门禁见 [docs/development-guide.md](docs/development-guide.md)。
