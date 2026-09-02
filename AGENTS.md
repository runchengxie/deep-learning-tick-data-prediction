# 项目协作说明

本文件供维护者和代码代理使用。改动代码前先读根目录 README 和 [docs/README.md](docs/README.md)，涉及实验口径时再核对仓库内论文原文。

## 项目目标

项目保留 DeepLOB 在 FI-2010 上的复现归档，并持续研究 tick 和 LOB 数据对次日横截面方向的预测。复现代码位于 `legacy/`，只作临摹参考。代码和文档应清楚区分：

- 已经由自动化测试验证的工程事实
- 由论文或官方数据格式支持的实验设定
- 需要真实训练结果才能确认的数值结论

`ticknet.nextday` 的结果与论文复现无关。FI-2010 不能用来证明次日方向有效。新链路的结果必须使用真实带股票、交易日和时间戳的数据，并报告时间外检验。真实结果尚未产生时，不得使用已经复现、严格复现或结果一致这类表述。

## 代码边界

- `src/ticknet/model.py` 和 `dataset.py` 保留兼容模型与合成数据工具。`src/ticknet/train.py` 提供主链路复用的随机种子、设备选择和分类指标工具
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
- 早期原始盘口容量系列锁定 2025，当前 AgentX 与事件流系列把 2025 作为开发区并锁定 2026
- 当前功能和研究进度统一维护在 `docs/project-status.md`，阶段路线图不能代替现状页

## 修改要求

完成修改后运行：

```bash
pre-commit run --all-files
python scripts/check.py
```

`pre-commit` 检查基础文件卫生、Ruff、ty 和 notebook。`scripts/check.py` 运行 Ruff 静态检查、Ruff 格式检查、ty、带覆盖率的 pytest 和冒烟检查。

公开合成数据质量门禁由 GitHub Actions PR workflow 运行，本地 `pre-commit` 与 `scripts/check.py` 继续用于提交前反馈。Python 3.10 兼容性和依赖安全审计不由该轻量 workflow 覆盖，涉及相关风险时手动用 Python 3.10 环境验证一次。

涉及数据协议时，增加能识别错误选段、标签错列或跨文件窗口的回归测试。涉及检查点时，测试恢复位置和配置冲突。涉及文档命令时，用 `--help` 核对参数名再落笔。

## GitHub Actions 策略

工作区统一采用以下默认规则：

- public 仓库默认启用 GitHub Actions，用于拉取请求的轻量自动检查。
- private 仓库默认关闭 GitHub Actions，避免持续占用私有仓库的 Actions 额度。
- private 仓库如需启用远端 CI，应在仓库文档中记录原因、检查范围和资源成本，并由维护者明确批准。
- 本地完整门禁继续由仓库自身检查和工作区共享 `pre-push` 承担。

本仓库是 public 仓库，`.github/workflows/ci.yml` 运行轻量 PR 检查。远端 CI 提供快速反馈，本地检查继续覆盖完整训练环境之外的工程门禁。

## 分支与合并

每个改动都在独立 worktree 上进行。多 agent 并行修改项目时，这是强制约束：每个 agent 独占一个 worktree 和一条 `agent/<topic>` 分支，绝不在主工作区或其他 agent 的 worktree 里直接改文件，避免多个 agent 竞争同一份文件或互相覆盖未提交的改动。

1. 从最新 `main` 新建 worktree 和 `agent/<topic>` 分支
2. 在该 worktree 里独立完成改动，跑通上面的门禁
3. 提交、推送到 `origin`、开 PR 并合并到 `main`
4. 合并后删除该分支和对应的 worktree，拉取最新 `main` 再启动下一个任务

合并顺序由 PR 评审决定，不要为了抢合并而跳过门禁。若多个 agent 的 PR 出现冲突，后合并的一方负责把自己 worktree 的 `main` 更新到最新后解决冲突，再重新推送。

worktree 里没有 `.venv`，跑 ty 前先建立软链：

在仓库外的 worktree 中建立指向主工作区 `.venv` 的软链即可。提交前删除这个软链，它会被 Git 当成未跟踪文件。

文档改动也要运行 pytest。文档测试会检查内部 Markdown 链接和本文件约定的中文文风。修改文档里的命令时，先用对应的 `--help` 核对参数名。

## 文档语言

中文文档使用自然、直接的表达，保留路径、参数、类名和指标名等必要的英文标识。中文正文用中文标点，包括全角括号。不用双引号、加粗、分号、破折号和先否定再转折的句式。研究限制直接说明已知范围、证据和影响。

## 数据和产物

FI-2010 数据、检查点、日志和图片不提交到 Git。`uv.lock` 应提交，用于固定开发和实验依赖。修改 `pyproject.toml` 后运行 `uv lock` 并检查锁文件变化。

不要修改 `docs/references/1808.03668v6.pdf`。阅读笔记可以修正文句和事实，但保留原论文的章节结构与引用信息。

## Worktree-first 目录规范

开发和实验使用 `/home/richard/code/.worktrees/` 下的独立 worktree。生产或定时任务使用
`/home/richard/code/production/` 下固定版本目录，不直接使用实验 worktree。训练数据、检查点、
日志和大体量产物放在仓库外的专用数据目录。清理 worktree 前必须确认没有定时任务引用该路径。
