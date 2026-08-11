# Colab CLI 无人值守运行

Linux 开发机负责代码、数据和实验产物的调度，Colab 提供临时 GPU 算力。正式入口是 Python CLI，notebook 只保留为历史实验快照和人工查看入口。

## 边界

- Linux 保存 Git worktree、raw-200、本地归档、Colab OAuth 和 rclone OAuth。
- Colab VM 只保存一次 session 所需的数据、wheel、临时凭据和运行输出。
- Google Drive 保存训练数据、checkpoint 和跨设备实验产物。
- rclone.conf 包含刷新凭据，只允许保存在仓库外路径。
- 2025 test 仍由代码锁定。自动化入口支持 2024 多周期 validation、独立 H=5 训练，以及 raw-1000 Top-100 上的 100M 参数训练吞吐 benchmark。

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
      --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/raw-200-capacity_1m/cli-runs/latest

确认命令后正式运行：

    python scripts/run_colab_nextday.py \
      --session ticknet-multi-horizon \
      --gpu T4 \
      --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/raw-200-capacity_1m/cli-runs/$(date +%Y%m%d-%H%M%S)

Stage C 的独立 H=5 seed 0 训练不需要 notebook：

    python scripts/run_colab_nextday.py \
      --workflow h5-train \
      --seeds 0 \
      --keep-on-failure \
      --session ticknet-h5-seed0 \
      --gpu T4 \
      --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/raw-200-capacity_1m-h5/seed0

`h5-train` 默认读取 `configs/nextday-raw-200-capacity-1m-h5.yaml`。它会把已有同名 checkpoint 从 Drive 恢复到固定路径，因此命令中断后可用相同 seed 继续。每次训练结束或失败都会尽力把 checkpoint、history、result 和 `colab-run-summary.json` 同步回 Drive。

100M benchmark 先用相同 revision、相同 raw-1000 单月 preflight 分别测 T4 和 A100：

    python scripts/run_colab_nextday.py \
      --workflow capacity-benchmark \
      --session ticknet-100m-raw1000-t4 \
      --gpu T4 \
      --benchmark-batches 100 \
      --warmup-batches 5 \
      --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/raw-1000-top100-capacity_100m/benchmarks/t4

把 `--gpu`、session 和输出末级目录改成 `A100` 和 `a100` 即可得到可比结果。默认配置是 `configs/nextday-raw-1000-top100-capacity-100m-benchmark.yaml`，精确参数量为 100,817,575。benchmark 会执行 AMP 前向、反向与 AdamW 更新，不访问 validation 和 test。正式 Top-100 train 样本数在数据完成前按 75,000 外推，完成后应使用实际样本数重算。

runner 会执行：

1. 要求当前 worktree 已提交且干净，并记录精确 commit。
2. 查询同名 session。默认要求它不存在，只有显式传入 `--reuse-session` 才允许复用。
3. 用 `git archive` 在临时目录构建该 commit 的 wheel，不污染当前 worktree。需要新 session 时才创建命名的 Colab GPU runtime。
4. 上传 wheel、固定训练配置、job spec 和临时 rclone.conf。
5. Colab 从 Drive 下载 workflow 所需数据。多周期和 H=5 使用 raw-200 与侧车标签，100M benchmark 只下载 raw-1000 preflight。
6. 执行多周期 validation、独立 H=5 训练或 100M 容量 benchmark。
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
      --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/raw-200-capacity_1m/cli-runs/debug

重复使用已经保留的 runtime：

    python scripts/run_colab_nextday.py \
      --reuse-session \
      --session ticknet-multi-horizon \
      --local-output-dir /home/richard/code/.artifacts/deep-learning-tick-data-prediction/raw-200-capacity_1m/cli-runs/reuse

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

执行历史由官方 colab log 生成，因此不需要人工打开或保存 notebook。
