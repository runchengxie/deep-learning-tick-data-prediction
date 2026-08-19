# 开发与维护

## 模块划分

核心包采用标准 `src` 布局：

| 模块 | 职责 |
|---|---|
| `ticknet.model` | FI-2010 兼容 DeepLOB 网络和模型工厂 |
| `ticknet.dataset` | FI-2010 兼容张量常量与合成数据工具 |
| `ticknet.train` | 主链路与 FI-2010 复现共用的训练工具（`set_seed`、`resolve_device`、`f1_metrics`） |
| `ticknet.nextday` | 次日标签、日期切分、分片读取、分块模型、横截面指标和训练，含分钟 HGB、TCN、GRU 基线与原始盘口主线 |
| `ticknet.eventstream` | L2 逐笔事件流的无损打包、因果 Transformer 训练和预测导出 |
| `ticknet.research` | 实验研究闭环，包括 ExperimentSpec v2、typed executor、策略与锁定期隔离、Registry、预测审计和确定性 Evaluation |

FI-2010 论文复现（DeepLOB 在 FI-2010 上的训练、评估、Colab 入口、文本转换和绘图）已归档到 `legacy/`，不再参与主链路开发与质量门禁。如需运行，参见 `legacy/` 下的对应脚本和测试。

`scripts/` 主要放人工入口和运行编排。可复用的数据协议、模型计算和评估逻辑应放在 `src/ticknet/`。当前仍有少数脚本承担较长的 Colab 任务编排，后续重构需要逐步把可复用部分移回核心包。

常用脚本如下：

| 脚本 | 用途 |
|---|---|
| `smoke_test.py` | FI-2010 兼容 DeepLOB 的快速模型检查 |
| `prepare_nextday.py` | 股票日日线、事件清单转次日预测 NPY 分片 |
| `run_nextday_baseline.py` | 聚合日内特征的 Logistic Regression 对照 |
| `run_minute_baseline.py` | 分钟级聚合特征的 HGB 基线，支持多年滚动验证和预测明细导出 |
| `prepare_minute_shards.py` | 分钟序列切分为 `samples x time x features` 分片，供时序模型使用 |
| `materialize_minute_features.py` | 按月原子物化正式分钟聚合特征 |
| `evaluate_cost_adjusted.py` | Top-K long-only 成本评估薄入口，兼容历史分位数多空诊断 |

## Colab 与 notebook 边界

现行 Colab 任务全部使用 Python 入口。顶层 `notebooks/` 已移除，两个旧 notebook 保存在 `legacy/notebooks/`，只用于追溯早期交互流程。

| 旧 notebook 能力 | 现行 Python 入口 |
|---|---|
| 次日模型训练、恢复和锁定评估 | `ticknet.nextday.train`、`ticknet-nextday-train`、`ticknet-nextday-evaluate` |
| 多周期 validation 评估 | `ticknet.nextday.horizon_cli`、`ticknet-nextday-evaluate-horizons` |
| Colab 会话、数据暂存和产物回传 | `scripts/run_colab_nextday.py` |
| Colab 远端任务执行 | `scripts/colab_multi_horizon_job.py` |

旧 notebook 的生成脚本和专属结构测试已经删除。现行入口继续由 CLI 契约测试、`tests/test_horizon_cli.py` 和 `tests/test_colab_nextday.py` 覆盖。主代码和文档不得重新依赖 `legacy/notebooks/`。

旧版按 `folds.npy` 排除某一折训练的兼容路径已经移除。该路径与 FI-2010 预制切分的含义不符，也增加了配置分支和泄漏风险。真实数据缺少元数据时，训练会直接停止。

## 配置

`Config` 数据类保存默认值。YAML 先覆盖默认值，命令行再覆盖 YAML。YAML 出现未知字段时会报错，避免拼写错误被静默忽略。

命令行使用连字符，例如 `--data-path`。YAML 使用下划线，例如 `data_path`。

查看完整参数：

```powershell
ticknet-nextday-train --help
ticknet-minute-tcn-train --help
ticknet-research --help
```

## 测试范围

测试按链路组织，都使用合成数据，不依赖真实行情、Google Drive 或完整 FI-2010。

FI-2010 复现链路已归档到 `legacy/tests/`，需手动运行，不参与主链路门禁。覆盖五个预测跨度和标签列映射、40 特征输入和论文规模的模型结构、Setup 1 与 Setup 2 的选段协议、训练集和验证集的原始行隔离，以及文本转换、分段元数据和流式 NPY 写入。

次日横截面主链路覆盖：

- 交易日历、相邻交易日标签和横截面三分类切点
- 信号时点与标签日泄漏检查、跨边界样本 purge
- 分块 DeepLOB、日内 GRU、双头输出和分片数据集索引
- 连续分数与三分类的 Macro F1、MCC、Brier、每日 Rank IC 等指标
- 梯度累积、AMP、检查点恢复和实验签名冲突
- 从原始 `N × 40` snapshot 返回分数与方向概率的推理入口
- 聚合日内特征的 Logistic Regression 基线
- 沪深月度 snapshot Parquet 适配器，包括动态股票池、候选时段筛选、分片指纹和 row-group 跳过
- YAML 与命令行覆盖、CPU 设备选择
- 多周期标签侧车生成与收益结束日防泄漏

分钟链路覆盖：

- 分钟级 HGB 基线的数据管线、L2 与 tushare 两种特征源
- 分钟序列分片的 NaN 中位数填充、短窗口补齐和分片校验
- 分钟 TCN 与 GRU 模型、分片数据集、训练入口和成本后回测
- 预测审计的 IC、decile、极端日贡献和 winsorize 诊断

eventstream 链路覆盖：

- 三条原始流的无损打包、ID 关联解析和逐日索引
- 事件窗口采样、多任务预测头和日级信号头
- 训练、断点恢复、数据集指纹与预测导出契约

CLI 契约测试读取 `pyproject.toml` 的全部命令声明，逐个导入目标函数并运行 `--help`。新增、删除或移动入口时，测试会直接反映声明与代码是否一致。

文档测试覆盖根目录 README、AGENTS 和 `docs/` 下的 Markdown 文件。它会检查内部链接目标，并检查中文正文是否误用双引号、强调、分号、破折号、半角括号和先否定再转折的句式。`docs/reports/` 是冻结产物，不参加文风检查。

研究闭环覆盖：

- ExperimentSpec v2 严格解析、白名单 executor、结构化 metric gates 和 artifact contract
- 锁定测试期隔离，manifest、显式 predictions 输入和训练产生的 predictions 都受程序级拦截
- locked approval 两步签发和消费、内容 SHA-256 绑定、原始 token 不落库和重放拒绝
- SQLite Registry v2 的递归指标、唯一性、父实验、失败状态和 artifact SHA-256
- Brainstorm、Critic、编排器、强制 Audit 与 KEEP、EXTEND、DISCARD 的确定性执行
- fixed-K long-only、排名缓冲、不可交易约束、权重漂移、成本和明细 artifact
- prediction artifact checksum 物化、多 seed 基线差值与按方向归一的配对改善
- 不同数据指纹窗口的 walk-forward 聚合、指标方向和最差窗口选择
- Registry 到 ResearchContext 的基线选择、失败与 Audit 回流、稳定指纹和 novelty replay 拒绝
- Brainstorm 与 Critic 共用上下文、预算和 executor 限制，以及 context review 快照

冒烟脚本检查 FI-2010 兼容 DeepLOB 的前向传播、softmax、梯度和参数量。它不读取真实数据，也不覆盖次日、分钟、事件流或研究闭环。冒烟脚本由 `scripts/check.py` 和本地 pre-push hook 调用，不参与 pytest 收集。

## 质量门禁

本地运行：

```bash
python scripts/check.py
```

脚本依次运行以下检查：

```bash
ruff check .
ruff format --check .
ty check
python -m pytest --cov --cov-report=term-missing
python scripts/smoke_test.py
```

Ruff 检查 pycodestyle、Pyflakes、导入顺序、现代语法、常见缺陷、推导式、pytest 写法和简化规则。`RUF001`、`RUF002`、`RUF003` 只因中文字符串、文档字符串和注释需要中文标点而关闭。

`ty` 的全局未解析导入忽略已经移除。Colab 笔记本保留单独覆盖，因为本地环境通常没有 `google.colab`。

pre-commit 在提交前运行 Ruff 自动修复、Ruff 格式化和 ty。运行 `pre-commit install` 会同时安装 pre-push hook，在每次推送前调用 `scripts/check.py`。完整门禁包含格式检查、静态检查、类型检查、带覆盖率的测试和冒烟检查。核心包分支覆盖率低于 80% 时测试失败。

## 依赖

运行依赖和开发依赖统一写在 `pyproject.toml`。使用以下命令安装开发环境：

```bash
python -m pip install -e ".[dev]"
```

项目提交 `uv.lock`。修改依赖后运行：

```bash
uv lock
```

修改 `pyproject.toml` 中的 CLI 入口后，应重新运行可编辑安装。已有 `.venv` 不会自动生成新增的命令脚本。

## 后续重构顺序

当前最值得继续投入的工作按优先级排列：

1. 拆分 `train.py`、`train_tcn.py` 与 `train_gru.py` 共用的训练循环，抽取共享的 epoch 训练与评估逻辑
2. 按校验、持久化和评估职责拆分 `ticknet.research` 中较长的 Registry、Spec 和组合评估模块
3. 把跨模块复用的日期、指标和原子写入工具整理为公开接口，减少对其他模块私有函数的依赖
4. 收窄 `run_colab_nextday.py` 和 `colab_multi_horizon_job.py` 的任务编排职责，把可测试的 job spec 与执行逻辑移入核心包
5. 为长时间训练增加结构化日志和运行状态监控

核心模块的规模目前不均衡。训练器之间存在较多重复循环，研究闭环和部分编排脚本也已经承担多项职责。后续按上述顺序逐步拆分，每次保留稳定的 CLI、配置和 artifact 契约，并在对应重构完成后移除复杂度忽略项。

FI-2010 复现的训练流程、命令行配置和实验汇总已拆到 `legacy/fi2010_train.py`。主链路 `ticknet.train` 只保留被次日预测复用的共享工具（`set_seed`、`resolve_device`、`f1_metrics`）。`ticknet.research` 通过 CLI 入口名和 YAML 配置与 `nextday` 解耦，不直接 import 其实现。
