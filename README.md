# DeepLOB 复现项目

这是论文 DeepLOB: Deep Convolutional Neural Networks for Limit Order Books（Zhang、Zohren、Roberts，2018）的 PyTorch 复现骨架。

- 论文地址：https://arxiv.org/abs/1808.03668
- 参考实现：https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

## 论文资料

论文原文和中文整理笔记放在 `references/` 目录：

- `references/1808.03668v6.pdf`：论文原文（arXiv v6）
- `references/DeepLOB-论文整理.md`：结构化中文读书笔记（模型架构、数据处理、实验结果）
- `references/README_论文资料.md`：资料索引

## 项目范围

本仓库围绕论文的官方模型结构，提供一个轻量且可验证的训练与评估框架。目前包含：

- `src/model.py`：DeepLOB 模型（CNN、Inception、LSTM、softmax），使用 PyTorch。输入特征维度可配置，官方 FI-2010 默认 144。
- `src/dataset.py`：`RandomLOBDataset`（无需数据的冒烟测试）和 `FI2010WindowDataset`（读取处理好的 `(N,149)` float32 .npy，含 144 个特征和 5 个标签列，对应 k=10/20/50/100 以及一个额外时间跨度）。
- `src/train.py`：训练循环，固定随机种子，Adam 优化器（lr=0.01，eps=1），输出 macro/weighted F1，支持早停和断点续训。
- `convert_fi2010.py`：本地脚本，把官方 `.txt` 转换成干净的 `.npy`。
- `smoke_test.py`：本地检查，验证前向形状、softmax 求和为 1、梯度可回流。
- `run_colab.py`：Colab GPU 入口。

## 本地快速体验（无需数据）

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/model.py        # 打印参数量和形状追踪
python smoke_test.py       # 前向与梯度冒烟检查
python src/train.py        # 用合成数据做 3 个 epoch 的冒烟训练
```

## 标签列的关键说明（容易出错）

FI-2010 的预测时间跨度 k = 10、20、50、100 是归一化文件里标签列的索引，不是原始事件个数。官方数据中，最后 5 列是这 4 个 k 值的三分类标签（第 144/145/146/147 列）加上一个额外时间跨度。取标签时统一用 `dataset.K_TO_LABEL_COLUMN[k]`，选错列会静默地训练出一个不同的任务。

第三方镜像（例如 Hugging Face 上的 `shanehans/FI2010` CSV）没有采用官方布局。那个 CSV 有 150 列，包含 130 个特征、15 个空列，以及标签值被污染的 4 个标签列（行号混进了标签列）。复现论文请只用官方数据。

## 数据准备（本地转好再上传到 Drive）

官方 FI-2010 数据以 `.txt` 文件发布（Ntakaris 等人，2017，arXiv:1705.03233），托管在芬兰 FAIR 数据平台：

- https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- 下载 `BenchmarkDatasets.zip`（1.74 GB），解压两次到达 `BenchmarkDatasets/BenchmarkDatasets/`

文件格式要点（已在真实下载中验证）：

- 压缩包内的路径形如 `BenchmarkDatasets/{NoAuction,Auction}/<i>.<Auction>_<Norm>/<Auction>_<Norm>_{Training,Testing}/<Split>_Dst_<Auction>_<Norm>_CF_<N>.txt`，例如 `NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt`。
- 这里的 `<i>` 是归一化方法的索引（1=Zscore，2=MinMax，3=DecPre），不是折数。每个归一化文件夹里都包含全部 9 个折的文件（CF_1 到 CF_9）。
- 每个 `.txt` 是一个矩阵，按（行=通道）乘（列=样本）存储：共 149 行（144 个特征行加 5 个标签行），标签取值为 1、2、3。`convert_fi2010.py` 在读取时会转置，使生成的 .npy 形状为 (N_samples, 149)。

在本地转换（你的电脑有内存和完整的 Python 环境可以核对结构，Colab 不需要接触原始 `.txt`）：

```bash
pip install numpy
python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets/BenchmarkDatasets \
    --auction NoAuction --norm z-score --folds 1 2 3 4 5 6 7 8 9 \
    --out FI2010_normalised.npy
# 该脚本同时写出 FI2010_folds.npy（每行一个折编号 0..8），供 9 折交叉验证使用
# 想先确认真实形状（预期为 (N, 149)），可以只做检查：
python convert_fi2010.py --base_dir /path/to/BenchmarkDatasets/BenchmarkDatasets --inspect-only
```

转换产物 `FI2010_normalised.npy` 的形状为 `(N, 149)` float32（NoAuction Z-score 全 9 折约 204 万行）：

- 第 `[0:144]` 列是 LOB 特征
- 第 `[144:149]` 列是 5 个标签列，k=10/20/50/100 分别对应第 144/145/146/147 列

把 `FI2010_normalised.npy` 上传到 Google Drive：

```
MyDrive/DeepLOB/data/FI2010_normalised.npy
```

Colab 里的 `run_colab.py` 会在此处找到它，跳过下载。

## 工作流（本地开发 → Colab GPU）

```
本地修改 → GitHub 推送 → Colab 拉取 → GPU 训练
```

代码放在 GitHub，数据和模型权重放在 Google Drive。多 GB 的 FI-2010 数据不要提交进 git。

在 Colab 里执行 `git pull`（或重新下载压缩包）后运行：

```python
!python deeplob-reproduction/run_colab.py
```

它会挂载 Drive、找到处理好的 `.npy`、安装依赖并开始训练。

## 复现论文数字（对照 Table II 前先读这段）

DeepLOB 论文报告的 F1 基于 FI-2010 的 9 折锚定交叉验证协议（论文中的 Setup 2）：对每一折，用其余 8 折训练、在该折上测试，最后对 9 折取平均。

本仓库已经实现这套协议。`convert_fi2010.py` 写出 `FI2010_folds.npy`（每行一个折编号 0..8）。`train.py` 加 `--cv` 会跑 9 折循环并打印 `mean_macro_f1 ± std_macro_f1`（含每一折的结果）。`run_colab.py` 在 Drive 上检测到 `FI2010_folds.npy` 时会自动切换到交叉验证模式。带上折编号文件得到的数字才可与论文 Table II 比较。

如果 `FI2010_folds.npy` 缺失，`run_colab.py` 会退回到简单的 70/15/15 切分，得到的数字只是流程自查，不能与论文比较。

重新上传提醒：你之前上传的 `FI2010_normalised.npy` 不带折编号。要用交叉验证，需要重新跑一次 `convert_fi2010.py`（现在会同时生成 `FI2010_folds.npy`），并把两个文件都上传到 `MyDrive/DeepLOB/data/`。

## 断点续训（应对 Colab 断连）

训练可以续训：每一折的完整状态（模型、优化器、epoch）会保存到检查点目录下的 `best.fold<N>.pt`。Colab 断连后重跑 `run_colab.py` 会从上次的 epoch 继续（默认 `--resume`）。切走标签页或断连都不会丢失进度。

## 暂未包含的内容

- 加密数据、Transformer 基线、回测（第二阶段，等基线跑通后再做）
- 多随机种子运行（默认单种子，以后可加 `--seed` 循环）

## 配置说明

`train.py` 的命令行参数和 `configs/base.yaml` 都来自同一个 `Config` 数据类，字段只定义一次。可以用 yaml 集中管理参数，再用命令行覆盖：

```bash
python src/train.py --config configs/base.yaml --epochs 100 --device cuda
```

不加 `--config` 时按命令行默认值运行（默认 `dataset=random`，本地冒烟）。

Colab 入口 `run_colab.py` 使用 `configs/colab.yaml`（epochs=100、device=cuda），再在运行时覆盖 `data_path`、`device`、`checkpoint_dir` 等动态值。本地调试可复制一份改参数。

pandas 和 matplotlib 目前代码未用到，仅作后续绘图准备（见 `plot_curves.py`）。

## 开发与质量门禁

项目用 ruff（lint）、ty（类型检查）、pytest（测试）三道门禁，团队 PR 前自动跑。

### 安装开发依赖

```bash
pip install -r requirements.txt   # 含 pytest / ruff / ty
```

### 本地跑检查

```bash
ruff check .          # lint，应全过
ty check              # 类型检查，应全过
python -m pytest -q   # 测试，10 个用例全过（无需真实数据）
```

`pyproject.toml` 里集中放了这三项的配置（ruff 规则、ty 检查范围、pytest 的 pythonpath 与 testpaths）。

### 提交前自动卡质量

```bash
pip install pre-commit
pre-commit install    # 安装 git 钩子，每次 commit 自动跑 ruff + ty
```

之后每次 `git commit` 会先执行钩子，ruff 和 ty 都通过才允许提交。CI 上可用 `pre-commit run --all-files` 复跑同一份配置。

### 目录约定

- `tests/` 放结构化测试（pytest 发现），全部用合成数据，不依赖真实 FI-2010。
- 根目录 `smoke_test.py` 是独立入口，不依赖 pytest 也能 `python smoke_test.py` 直接跑。
- 训练曲线由 `plot_curves.py` 读取 `train_history*.json` 生成（见上文配置说明）。

