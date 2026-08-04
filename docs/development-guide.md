# 开发与维护

## 模块划分

核心包采用标准 `src` 布局：

| 模块 | 职责 |
|---|---|
| `deeplob.model` | 网络结构和模型工厂 |
| `deeplob.dataset` | 数据校验、实验选段、标签映射和窗口索引 |
| `deeplob.train` | 配置合并、训练、评估、早停、检查点和实验汇总 |
| `deeplob.nextday` | 次日标签、日期切分、分片读取、分块模型、横截面指标和训练 |

脚本层只处理人工入口：

| 脚本 | 用途 |
|---|---|
| `convert_fi2010.py` | 官方文本转 NPY 和元数据 |
| `run_colab.py` | Colab 环境检查和训练启动 |
| `smoke_test.py` | 无真实数据的快速链路检查 |
| `plot_curves.py` | 读取 JSON 训练历史并绘图 |
| `prepare_nextday.py` | 股票日日线、事件清单转次日预测 NPY 分片 |
| `run_nextday_baseline.py` | 聚合日内特征的 Logistic Regression 对照 |

旧版按 `folds.npy` 排除某一折训练的兼容路径已经移除。该路径与 FI-2010 预制切分的
含义不符，也增加了配置分支和泄漏风险。真实数据缺少元数据时，训练会直接停止。

`run_colab.py` 默认把 Drive 中的数据原子复制到 `/content/DeepLOB/data/`。完整且与源文件
大小、修改时间一致的本地副本会被复用，检查点始终保存在 Drive。临时副本使用
`.copying` 后缀，复制成功后才替换目标文件，避免中断后误用不完整数据。

## 配置

`Config` 数据类保存默认值。YAML 先覆盖默认值，命令行再覆盖 YAML。YAML 出现未知字段
时会报错，避免拼写错误被静默忽略。

命令行使用连字符，例如 `--data-path`。YAML 使用下划线，例如 `data_path`。

查看完整参数：

```powershell
deeplob-train --help
python scripts/convert_fi2010.py --help
python scripts/run_colab.py --help
```

## 测试范围

测试分为两条链路：FI-2010 论文复现和次日横截面研究。两套测试都用合成数据，不依赖
真实行情、Google Drive 或完整 FI-2010。

FI-2010 复现链路覆盖：

- 五个预测跨度和标签列映射
- 40 特征输入和论文规模的模型结构
- Setup 1 的同折 Training 与 Testing 选择
- Setup 2 的 `CF_7` 训练与三个 Testing 文件选择
- 训练集和验证集的原始行隔离
- 测试窗口不跨源文件
- 文本转换、分段元数据和流式 NPY 写入
- 最近检查点恢复和实验配置冲突
- 训练曲线文件读取

次日横截面链路覆盖：

- 交易日历、相邻交易日标签和横截面三分类切点
- 信号时点与标签日泄漏检查、跨边界样本 purge
- 分块 DeepLOB、日内 GRU、双头输出和分片数据集索引
- 连续分数与三分类的 Macro F1、MCC、Brier、每日 Rank IC 等指标
- 梯度累积、AMP、检查点恢复和实验签名冲突
- 从原始 `N × 40` snapshot 返回分数与方向概率的推理入口
- 聚合日内特征的 Logistic Regression 基线
- 沪深月度 snapshot Parquet 适配器：动态股票池、候选时段筛选、分片指纹和 row-group 跳过
- YAML 与命令行覆盖、CPU 设备选择

冒烟脚本补充检查模型前向传播、softmax、梯度、参数量和数据窗口。冒烟脚本由人工或
本地 pre-push hook 单独执行，不参与 pytest 收集。

## 质量门禁

本地运行：

```powershell
python scripts/check.py
```

脚本依次运行以下检查：

```powershell
ruff check .
ruff format --check .
ty check
python -m pytest --cov --cov-report=term-missing
python scripts/smoke_test.py
```

Ruff 检查 pycodestyle、Pyflakes、导入顺序、现代语法、常见缺陷、推导式、
pytest 写法和简化规则。`RUF001`、`RUF002`、`RUF003` 只因中文字符串、文档字符串和
注释需要中文标点而关闭。

`ty` 的全局未解析导入忽略已经移除。Colab 笔记本保留单独覆盖，因为本地环境通常
没有 `google.colab`。

pre-commit 在提交前运行 Ruff 自动修复、Ruff 格式化和 ty。运行 `pre-commit install`
会同时安装 pre-push hook，在每次推送前调用 `scripts/check.py`。完整门禁包含格式检查、
静态检查、类型检查、带覆盖率的测试和冒烟检查。核心包分支覆盖率低于 80% 时测试失败。

## 依赖

运行依赖和开发依赖统一写在 `pyproject.toml`。使用以下命令安装开发环境：

```powershell
python -m pip install -e ".[dev]"
```

项目提交 `uv.lock`。修改依赖后运行：

```powershell
uv lock
```

## 后续重构顺序

当前最值得继续投入的工作按优先级排列：

1. 增加小型真实数据夹具，验证官方文件的特征顺序和标签值
2. 扩充早停边界和检查点写入中断测试
3. 把指标结果整理为可直接对照论文表格的报告
4. 真实训练稳定后，再评估多随机种子调度和超参数搜索
5. 为长时间训练增加结构化日志和运行状态监控

核心模块当前规模适中。`train.py` 集中了训练流程和命令行配置，后续加入多种模型、
多数据集或分布式训练时，可以把检查点和实验汇总拆成独立模块。现阶段提前拆分会增加
接口数量，收益有限。
