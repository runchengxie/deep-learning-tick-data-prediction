# 开发与维护

## 模块划分

核心包采用标准 `src` 布局：

| 模块 | 职责 |
|---|---|
| `ticknet.model` | 网络结构和模型工厂 |
| `ticknet.dataset` | 共享张量形状常量与合成数据工具 |
| `ticknet.train` | 主链路与 FI-2010 复现共用的训练工具（`set_seed` / `resolve_device` / `f1_metrics`） |
| `ticknet.nextday` | 次日标签、日期切分、分片读取、分块模型、横截面指标和训练 |
| `ticknet.research` | 实验研究闭环，包括 ExperimentSpec v2、typed executor、策略与锁定期隔离、Registry、预测审计和确定性 Evaluation |

FI-2010 论文复现（DeepLOB 在 FI-2010 上的训练、评估、Colab 入口、文本转换和绘图）已
归档到 `legacy/`，不再参与主链路开发与质量门禁。如需运行，参见 `legacy/` 下的对应脚本
和测试。

脚本层只处理人工入口：

| 脚本 | 用途 |
|---|---|
| `smoke_test.py` | 无真实数据的快速链路检查 |
| `prepare_nextday.py` | 股票日日线、事件清单转次日预测 NPY 分片 |
| `run_nextday_baseline.py` | 聚合日内特征的 Logistic Regression 对照 |
| `run_minute_baseline.py` | 分钟级聚合特征的 HGB 基线，支持多年滚动验证和预测明细导出 |
| `prepare_minute_shards.py` | 分钟序列切分为 `samples x time x features` 分片，供时序模型使用 |
| `evaluate_cost_adjusted.py` | Top-K long-only 成本评估薄入口；兼容历史分位数多空诊断 |

旧版按 `folds.npy` 排除某一折训练的兼容路径已经移除。该路径与 FI-2010 预制切分的
含义不符，也增加了配置分支和泄漏风险。真实数据缺少元数据时，训练会直接停止。

## 配置

`Config` 数据类保存默认值。YAML 先覆盖默认值，命令行再覆盖 YAML。YAML 出现未知字段
时会报错，避免拼写错误被静默忽略。

命令行使用连字符，例如 `--data-path`。YAML 使用下划线，例如 `data_path`。

查看完整参数：

```powershell
ticknet-fi2010-train --help
python legacy/scripts/convert_fi2010.py --help
python legacy/scripts/run_colab.py --help
```

## 测试范围

测试分为两条链路：次日横截面研究（主链路，纳入 pytest 与覆盖率门禁）和 FI-2010 论文
复现（已归档到 `legacy/tests/`，需手动运行，不参与主链路门禁）。两套测试都用合成数据，
不依赖真实行情、Google Drive 或完整 FI-2010。

FI-2010 复现链路（归档）覆盖：

- 五个预测跨度和标签列映射
- 40 特征输入和论文规模的模型结构
- Setup 1 的同折 Training 与 Testing 选择
- Setup 2 的 `CF_7` 训练与三个 Testing 文件选择
- 训练集和验证集的原始行隔离
- 文本转换、分段元数据和流式 NPY 写入
- 训练曲线文件读取

次日横截面链路（主链路）覆盖：

- 交易日历、相邻交易日标签和横截面三分类切点
- 信号时点与标签日泄漏检查、跨边界样本 purge
- 分块 DeepLOB、日内 GRU、双头输出和分片数据集索引
- 连续分数与三分类的 Macro F1、MCC、Brier、每日 Rank IC 等指标
- 梯度累积、AMP、检查点恢复和实验签名冲突
- 从原始 `N × 40` snapshot 返回分数与方向概率的推理入口
- 聚合日内特征的 Logistic Regression 基线
- 沪深月度 snapshot Parquet 适配器：动态股票池、候选时段筛选、分片指纹和 row-group 跳过
- YAML 与命令行覆盖、CPU 设备选择

分钟链路覆盖：

- 分钟级 HGB 基线的数据管线、L2 与 tushare 两种特征源
- 分钟序列分片的 NaN 中位数填充、短窗口补齐和分片校验
- 分钟 TCN 模型、分片数据集、训练入口和成本后回测
- 预测审计的 IC、decile、极端日贡献和 winsorize 诊断

研究闭环覆盖：

- ExperimentSpec v2 严格解析、白名单 executor、结构化 metric gates 和 artifact contract
- 锁定测试期隔离：manifest、显式 predictions 输入和训练产生的 predictions 都受程序级拦截
- SQLite Registry v2 的递归指标、唯一性、父实验、失败状态和 artifact SHA-256
- Brainstorm、Critic、编排器、强制 Audit 与 KEEP/EXTEND/DISCARD 的确定性执行
- fixed-K long-only、排名缓冲、不可交易约束、权重漂移、成本和明细 artifact

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

1. 拆分 `train.py` 与 `train_tcn.py` 共用的训练循环，抽取共享的 epoch 训练与评估逻辑
2. 把 `run_minute_baseline.py` 与 `prepare_minute_shards.py` 重复的配置读取逻辑收敛为共享函数
3. 为长时间训练增加结构化日志和运行状态监控
4. 研究闭环接入真实 LLM 后，补足 Brainstorm 证据检索与实验去重

核心模块当前规模适中。FI-2010 复现的训练流程、命令行配置和实验汇总已拆到
`legacy/fi2010_train.py`。主链路 `ticknet.train` 只保留被次日预测复用的共享工具
（`set_seed` / `resolve_device` / `f1_metrics`）。`ticknet.research` 通过 CLI 入口名和
YAML 配置与 `nextday` 解耦，不直接 import 其实现。
