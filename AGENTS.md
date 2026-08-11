# 项目协作说明

本文件供维护者和代码代理使用。改动代码前先读根目录 README 和 [docs/README.md](docs/README.md)，涉及实验口径时再核对仓库内论文原文。

## 项目目标

项目包含两条独立链路：复现 DeepLOB 在 FI-2010 上的 Table I 和 Table II，以及研究 tick 和 LOB 数据对次日横截面方向的预测。复现已归档到 `legacy/`，只作临摹参考。代码和文档应清楚区分：

- 已经由自动化测试验证的工程事实
- 由论文或官方数据格式支持的实验设定
- 需要真实训练结果才能确认的数值结论

`ticknet.nextday` 的结果与论文复现无关。FI-2010 不能用来证明次日方向有效。新链路的结果必须使用真实带股票、交易日和时间戳的数据，并报告时间外检验。真实结果尚未产生时，不得使用已经复现、严格复现或结果一致这类表述。

## 代码边界

- `src/ticknet/model.py`、`dataset.py`、`train.py` 只承载 FI-2010 复现逻辑，随 legacy 归档
- `src/ticknet/nextday/` 负责次日标签、分片数据集、分块模型、横截面指标和训练，包含分钟 HGB、TCN、GRU 三套基线和原始盘口主线
- `src/ticknet/eventstream/` 负责 L2 逐笔事件流的无损打包、因果 Transformer 训练和预测导出，数据契约见 `config.py`
- `src/ticknet/research/` 负责实验研究闭环，包括提案定义、策略校验、锁定测试隔离、实验登记、预测审计和研究 Agent 框架。它通过 CLI 入口名和 YAML 配置与 `nextday` 解耦，不直接 import `nextday` 的实现
- `scripts/` 放人工执行入口，不承载可复用的核心业务逻辑
- `tests/` 使用合成数据，不依赖 Google Drive 或完整 FI-2010
- `legacy/` 是 FI-2010 复现归档，不参与主链路质量门禁，主代码不得反向依赖它

新增功能优先放入现有边界。一个模块同时承担数据发现、模型计算和实验调度时，先拆分职责。`ticknet.research` 的锁定测试隔离由代码强制，涉及测试集或日期切分的改动必须先过 `ResearchProtocol` 校验。

## 关键事实

- 模型输入使用 FI-2010 前 40 个原始订单簿特征
- 第 40 至 143 列的手工特征不进入模型
- 标签列映射为 10、20、30、50、100，对应列 144 至 148
- setup1 使用同一 CF 的 Training 和 Testing 文件
- setup2 使用 CF_7 Training 和 CF_7、CF_8、CF_9 三个 Testing 文件
- 真实数据训练必须提供转换脚本生成的 NPY 和元数据文件
- 缺少元数据时应停止运行，不得静默改用随机切分

## 修改要求

完成代码修改后运行：

```powershell
ruff format .
ruff check .
ty check
python -m pytest -q
python scripts/smoke_test.py
```

涉及数据协议时，增加能识别错误选段、标签错列或跨文件窗口的回归测试。涉及检查点时，测试恢复位置和配置冲突。涉及文档命令时，用 `--help` 核对参数名再落笔。

## 分支与合并

每次改动都在独立的 worktree 上进行，避免多个代理同时改同一份文件互相竞争。

1. 从 main 新建 worktree 和分支，命名见历史约定，例如 `feat/next-round`
2. 在 worktree 里实现改动，跑通上面的门禁
3. 提交、推送、开 PR 并把 PR 合并到 main
4. 删除已合并的分支和对应的 worktree，拉取最新 main

worktree 里没有 `.venv`，跑 ty 前先建立软链：

```bash
ln -sfn ~/code/deep-learning-tick-data-prediction/.venv <worktree>/.venv
```

提交前删掉这个软链，它会被 Git 当成未跟踪文件。

## 文档语言

中文文档使用自然、直接的表达，保留路径、参数、类名和指标名等必要的英文标识。中文正文用中文标点，包括全角括号。不用双引号、加粗、分号、破折号和先否定再转折的句式。研究限制直接说明已知范围、证据和影响。

## 数据和产物

FI-2010 数据、检查点、日志和图片不提交到 Git。`uv.lock` 应提交，用于固定开发和实验依赖。修改 `pyproject.toml` 后运行 `uv lock` 并检查锁文件变化。

不要修改 `references/1808.03668v6.pdf`。阅读笔记可以修正文句和事实，但保留原论文的章节结构与引用信息。
