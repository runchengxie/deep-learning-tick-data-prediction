# Colab CLI 无人值守运行

Linux 开发机负责代码、数据和实验产物的调度，Colab 提供临时 GPU 算力。正式入口是 Python CLI。旧 notebook 已转换为 `legacy/notebooks/` 下的 Python 快照，只用于追溯早期交互流程。

## 边界

- Linux 保存 Git worktree、raw-200、本地归档、Colab OAuth 和 rclone OAuth。
- Colab VM 只保存一次 session 所需的数据、wheel、临时凭据和运行输出。
- Google Drive 保存训练数据、checkpoint 和跨设备实验产物。
- rclone.conf 包含刷新凭据，只允许保存在仓库外路径。
- 早期原始盘口容量系列继续锁定 2025 test。当前事件流和 AgentX 系列把 2025 用作开发区，并由协议锁定 2026。自动化入口支持多周期评估、独立 H=5、容量矩阵和事件流基准。

截至 2026-08-19，原始盘口四格三 seed 矩阵、事件流输入基准、最近折正式训练三 seed 和相邻折 seed 0 已经完成。本文保留对应命令用于复现。当前新增的正式工作流是两折多任务梯度审计。

## Linux 主机安装

Colab CLI 0.6.0 的 PyPI 元数据没有锁住 Google 的 kernel client fork。安装后需要用官方 lockfile 对应版本替换 PyPI 的 1.x 包，否则 `colab exec` 会找不到 `KernelClient`：

    uv tool install --force google-colab-cli==0.6.0 \
      --with 'jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git@f18e982c3265df5e923aa9def101ab3fd737e139'

检查 OAuth 和 rclone，这两条命令都应当无需人工输入：

    colab --auth=oauth2 sessions
    rclone --config /home/richard/.config/rclone/rclone.conf \
      lsjson gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-200/manifest.json

runner 也会在非登录 SSH 环境中查找 `/home/richard/.local/bin`，不依赖交互式 shell 的 `PATH`。Colab 的 Ubuntu apt 源目前提供较老的 rclone，因此 VM 内命令只使用兼容参数。

## Python 入口

固定三个 best checkpoint，只计算 2024 validation：

    ticknet-nextday-evaluate-horizons \
      --config configs/nextday-raw-200-capacity-1m.yaml \
      --sidecar /content/nextday-raw-200-targets-v1/horizon-labels.json \
      --output-dir /content/ticknet-results/multi-horizon-validation-2024 \
      --seeds 0 1 2 \
      --horizons 1 3 5 \
      --source-revision "$(git rev-parse HEAD)"

配置中的 `manifest_path` 和 `checkpoint_dir` 与原始 checkpoint 签名保持一致。runner 会通过 rclone 把数据同步到这些路径，不修改 checkpoint 匹配规则。

## Linux 调度入口

先做不申请 GPU 的 dry run：

    python scripts/run_colab_nextday.py \
      --dry-run \
      --session ticknet-multi-horizon \
      --gpu T4 \
      --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/latest

确认命令后正式运行：

    python scripts/run_colab_nextday.py \
      --session ticknet-multi-horizon \
      --gpu T4 \
      --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/$(date +%Y%m%d-%H%M%S)

Stage C 的独立 H=5 seed 0 训练不需要 notebook：

    python scripts/run_colab_nextday.py \
      --workflow h5-train \
      --seeds 0 \
      --keep-on-failure \
      --session ticknet-h5-seed0 \
      --gpu T4 \
      --local-output-dir artifacts/raw-200-capacity_1m-h5/seed0

`h5-train` 默认读取 `configs/nextday-raw-200-capacity-1m-h5.yaml`。它会把已有同名 checkpoint 从 Drive 恢复到固定路径，因此命令中断后可用相同 seed 继续。每次训练结束或失败都会尽力把 checkpoint、history、result 和 `colab-run-summary.json` 同步回 Drive。

五年 raw-1000 Top-100 的 100M 训练使用 A100 与 batch 32。训练只用 2021 至 2023，checkpoint 只由 2024 validation 选择，2025 test 不评估。三个 seed 都已经完成，下面保留 seed 0 的复现命令：

    python scripts/run_colab_nextday.py \
      --workflow raw1000-train \
      --seeds 0 \
      --keep-on-failure \
      --session ticknet-100m-raw1000-seed0 \
      --gpu A100 \
      --local-output-dir artifacts/raw-1000-top100-capacity_100m/training

`raw1000-train` 默认读取 `configs/nextday-raw-1000-top100-capacity-100m.yaml`，从 Drive 下载完整的 8.84 GiB 工作集，并在启动前恢复同一训练目录已有的 checkpoint。seed 1 和 2 使用相同命令修改 `--seeds`。三个 seed 的选择都没有读取 2025 test。

容量与窗口 2×2 矩阵共用上面的 Top-100 工作集。raw-200 格通过 `input_last_chunks: 2` 只读取每个样本最后两个 100-event chunk，股票日、标签和底层数据指纹均不变，也不复制分片。四格三 seed 均已完成。下面保留同口径 `1M/raw-200` seed 0 的复现命令：

    python scripts/run_colab_nextday.py \
      --workflow capacity-matrix-train \
      --matrix-cell 1m-raw200 \
      --seeds 0 \
      --keep-on-failure \
      --session ticknet-matrix-1m-raw200-seed0 \
      --gpu A100 \
      --local-output-dir artifacts/raw-1000-top100-capacity-matrix/1m-raw200

`--matrix-cell` 还接受 `1m-raw1000` 和 `100m-raw200`。三格分别读取 `configs/nextday-capacity-matrix-1m-raw200.yaml`、`configs/nextday-capacity-matrix-1m-raw1000.yaml` 和 `configs/nextday-capacity-matrix-100m-raw200.yaml`。它们固定使用 batch 32、学习率 0.0001、patience 8 和 2024 validation 选模，2025 test 保持锁定。每格使用独立 Drive 目录并支持断点恢复。

100M benchmark 先用相同 revision、相同 raw-1000 单月 preflight 分别测 T4 和 A100：

    python scripts/run_colab_nextday.py \
      --workflow capacity-benchmark \
      --session ticknet-100m-raw1000-t4 \
      --gpu T4 \
      --benchmark-batches 100 \
      --warmup-batches 5 \
      --local-output-dir artifacts/raw-1000-top100-capacity_100m/benchmarks/t4

把 `--gpu`、session 和输出末级目录改成 `A100` 和 `a100` 即可得到可比结果。默认配置是 `configs/nextday-raw-1000-top100-capacity-100m-benchmark.yaml`，精确参数量为 100,817,575。benchmark 会执行 AMP 前向、反向与 AdamW 更新，不访问 validation 和 test。早期基准按 75,000 个训练样本外推，数据完成后已用实际的 70,805 个样本重算。

首个 Top-400 全天事件流 H5 fold 使用独立工作流。Drive 只需要预先放入 2021-01 benchmark pack 和 fold 级 H5 标签，A100 会运行精确 100,604,180 参数的事件流模型：

    python scripts/run_colab_nextday.py \
      --workflow eventstream-capacity-benchmark \
      --session ticknet-eventstream-h5-100m-a100 \
      --gpu A100 \
      --benchmark-batches 100 \
      --warmup-batches 5 \
      --keep-on-failure \
      --local-output-dir artifacts/eventstream-h5-fold0/benchmarks/a100

默认配置是 `configs/eventstream-h5-fold0-capacity100m-colab.yaml`。该工作流只构造 2021-01 训练集，不读取 2021-04 validation 或 2021-05 OOS。

2021 结果只作基础设施吞吐基线。正式 recent fold 应上传 2025-08 pack，并补一次相同口径 benchmark：

    python scripts/run_colab_nextday.py \
      --workflow eventstream-recent-capacity-benchmark \
      --session ticknet-eventstream-h5-recent-100m-a100 \
      --gpu A100 \
      --benchmark-batches 100 \
      --warmup-batches 5 \
      --keep-on-failure \
      --local-output-dir artifacts/eventstream-h5-recent-fold/benchmarks/a100

最近折工作流默认使用 `configs/eventstream-h5-recent-capacity100m-colab.yaml`，只访问 2025 年 8 月训练 pack。2025 年 11 月 validation、2025 年 12 月 OOS 和 2026 locked 均不参与 benchmark。

最近折正式训练使用固定窗口物化目录。每次只运行一个 seed，第一次用一个 epoch 验证恢复，不读取 OOS：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-train \
  --session ticknet-eventstream-h5-recent-seed0-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/training/seed0
```

工作流在训练前核对物化清单和允许访问分片的全部 SHA-256，从 Drive 恢复同一 seed 的 checkpoint。短任务会排除 OOS 和 H3 OOS 分片。成功或失败都会回传 checkpoint、history、result、物化预检和 `colab-run-summary.json`。短任务通过后把 `--training-epochs` 改为 `20`，使用新的 session 名并加 `--evaluate-test`。

额外滚动折使用独立路径，并把折标识写入运行摘要。下面的命令启动 `fold-54-oos-202511` 的 seed 0 短恢复验证：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-rolling-train \
  --eventstream-fold-id fold-54-oos-202511 \
  --session ticknet-eventstream-fold54-seed0-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-fold54/training/seed0
```

默认配置按折标识解析。新折需要先提交对应的固定日期配置，Runner 会拒绝路径分隔符和不符合 `fold-NN-oos-YYYYMM` 格式的标识。

多任务梯度审计只读取 validation。最近折和相邻折都固定使用 seed 0、16 个 batch 和已登记 SHA-256 的 best checkpoint。最近折命令如下：

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-gradient-audit \
  --session ticknet-gradient-audit-recent-seed0 \
  --gpu A100 \
  --seeds 0 \
  --audit-batches 16 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-gradient-audit/recent-seed0
```

相邻折把 workflow 改为 `eventstream-rolling-gradient-audit`，并增加 `--eventstream-fold-id fold-54-oos-202511`。工作流只暂存 validation 分片和一个 checkpoint，train、OOS、监控分区与 2026 锁定区不会进入 Colab。审计门槛见[事件流多任务梯度审计](../research/eventstream-gradient-audit.md)。

如果 batch sweep 的吞吐没有随物理 batch 增长，可以使用相同的 2025 年 8 月 pack 分别测量 DataLoader 和 GPU，并扫描 worker 数：

    python scripts/run_colab_nextday.py \
      --workflow eventstream-recent-input-profile \
      --session ticknet-eventstream-h5-recent-input-a100 \
      --gpu A100 \
      --num-workers 2 4 8 16 \
      --effective-batch-size 64 \
      --benchmark-batches 50 \
      --warmup-batches 5 \
      --keep-on-failure \
      --local-output-dir artifacts/eventstream-h5-recent-fold/input-profile/a100

输出分别记录只运行 DataLoader、预加载 batch 的纯 GPU 和真实端到端吞吐。worker 数按端到端吞吐选择。该工作流不读取 validation、OOS 或 2026 locked 数据。

2026-08-12 的优化后实测选择 8 个 worker。DataLoader-only 为 140.48 samples/s，端到端为 149.40 samples/s，GPU-only 为 238.79 samples/s。按 120,000 个样本和 20 个 epoch 外推，每个 seed 约为 4.46 小时。正式 recent 配置使用 `num_workers: 8`。

事件流 recent sweep 复用同一份 2025 年 8 月暂存数据，在一个 A100 session 内测 batch 8、16、32 和 64，并按完整三个月的 120,000 个训练样本外推：

    python scripts/run_colab_nextday.py \
      --workflow eventstream-recent-batch-size-sweep \
      --session ticknet-eventstream-h5-recent-sweep-a100 \
      --gpu A100 \
      --batch-sizes 8 16 32 64 \
      --effective-batch-size 64 \
      --benchmark-batches 50 \
      --warmup-batches 5 \
      --keep-on-failure \
      --local-output-dir artifacts/eventstream-h5-recent-fold/batch-size-sweep/a100

正式训练前可在单个 A100 session 内扫描物理 batch 2、4、8、16 和 32。命令固定 effective batch 为 32，每档执行 5 个 warmup batch 和 50 个 measured batch。单档 OOM 会留下记录并继续，最终按成功档位的 samples/s 选择最佳值：

    python scripts/run_colab_nextday.py \
      --workflow batch-size-sweep \
      --session ticknet-100m-batch-sweep-a100 \
      --gpu A100 \
      --batch-sizes 2 4 8 16 32 \
      --effective-batch-size 32 \
      --benchmark-batches 50 \
      --warmup-batches 5 \
      --keep-on-failure \
      --local-output-dir artifacts/raw-1000-top100-capacity_100m/batch-size-sweep/a100

runner 会执行：

1. 要求当前 worktree 已提交且干净，并记录精确 commit。
2. 查询同名 session。默认要求它不存在，只有显式传入 `--reuse-session` 才允许复用。
3. 用 `git archive` 在临时目录构建该 commit 的 wheel，不污染当前 worktree。需要新 session 时才创建命名的 Colab GPU runtime。
4. 上传 wheel、固定训练配置、job spec 和临时 rclone.conf。
5. Colab 从 Drive 下载工作流所需数据。多周期和 H=5 使用 raw-200 与侧车标签。100M benchmark 下载 raw-1000 preflight、2021 年 1 月事件流 pack 或 2025 年 8 月 recent pack。
6. 执行多周期 validation、独立 H=5 训练、raw-1000 正式训练或对应的100M容量 benchmark。
7. 将 JSON 和 Parquet 结果同步回 Drive，再同步到 Linux artifact 目录。
8. 导出 CLI execution notebook 并删除临时 rclone 配置，再按生命周期策略处理 session。

## Session 生命周期

- 默认是 ephemeral，新建 session，成功或失败后都关闭。
- `--keep-on-failure`：成功后关闭，失败时保留 session 供排查。
- `--keep-session`：成功或失败都保留本次新建的 session。
- `--reuse-session`：要求同名 session 已存在并复用它，runner 永远不负责关闭该 session。

`--keep-session` 与 `--keep-on-failure` 互斥。同名 session 已存在但没有传 `--reuse-session` 时，runner 会在上传文件前拒绝执行。传了 `--reuse-session` 但 session 不存在时也会拒绝执行。所有模式都会删除本次上传的临时 rclone 配置。

失败时保留 session：

    python scripts/run_colab_nextday.py \
      --keep-on-failure \
      --session ticknet-multi-horizon \
      --gpu T4 \
      --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/debug

重复使用已经保留的 runtime：

    python scripts/run_colab_nextday.py \
      --reuse-session \
      --session ticknet-multi-horizon \
      --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/reuse

保留的 VM 会继续消耗 compute units，确认不再使用后显式运行：

    colab --auth=oauth2 stop -s ticknet-multi-horizon

## 为什么不直接上传 raw-200

colab upload 适合 wheel、YAML 和 JSON 等小文件。它通过 Jupyter Contents API 发送文件，二进制内容需要 base64。raw-200 当前约 7.2GB，直接上传会增加内存、请求体和失败重试成本。rclone 在 Colab 内直接从 Drive 下载，支持校验、并行传输和重复执行。

## 凭据安全

- 不提交 rclone.conf、Colab token 或 session metadata。
- runner 拒绝使用位于 Git 仓库内的 rclone 配置。
- job spec 只记录远端名称和路径，不记录 token 内容。
- Colab 内的 rclone 配置权限固定为 600，job 结束立即删除。
- runtime 停止后临时磁盘由 Colab 回收。

## 运行产物

Drive：

    deep-learning-tick-data-prediction/
      ticknet-runs/raw-200-capacity_1m/multi-horizon-validation-2024/
      ticknet-runs/raw-200-capacity_1m-h5/

Linux 的 `--local-output-dir`：

    multi_horizon_validation_2024.json
    daily_rank_ic_2024.parquet
    validation_scores_2024.parquet
    execution.ipynb

H=5 训练目录还包含每个 seed 的 last 和 best checkpoint、history、result 和 `colab-run-summary.json`。

100M benchmark 的 GPU 独立目录包含 `capacity-benchmark.json`、`colab-run-summary.json` 和 `execution.ipynb`。JSON 记录实际 GPU、精确参数量、数据指纹、吞吐、峰值显存以及 75,000 个训练样本的单 seed 和三 seed 外推。

batch-size sweep 目录包含每档的 `batch-NN.json`、汇总
`batch-size-sweep.json`、`colab-run-summary.json` 和 `execution.ipynb`。汇总文件记录每档梯度
累积、吞吐、显存、相对最小成功 batch 的加速比和最终选择。

事件流输入分析目录包含 `gpu-only.json`、每个 worker 的 data-only 与 end-to-end JSON、汇总 `input-profile.json`、`colab-run-summary.json` 和 `execution.ipynb`。

执行历史由官方 colab log 生成，因此不需要人工打开或保存 notebook。

## 事件流 VQ 续训

现有 eventstream checkpoint 训练时未启用 VQ。`EventstreamConfig` 的 `init_checkpoint` 支持把这类 checkpoint 热启动为启用 VQ 的新实验：主干权重原样加载，VQ 模块（vq_encoder、vector_quantizer、vq_proj）随机初始化，训练配置里同时设置 `use_vq: true`。

热启动与 `resume` 的关系：本实验自己的 last checkpoint 存在时优先 resume，否则才读 `init_checkpoint`。两者都不存在时从零训练。

便宜路线的推荐步骤：先用小模型配置和几天打包数据在本地或免费档 GPU 验证整条管线，确认 vq_loss 收敛后再用 A100 跑 capacity100m 配置。A100 消耗约每小时 15 个 compute unit。
