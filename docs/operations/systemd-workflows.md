# 历史 systemd 工作流记录

> 本文记录 2026-09-02 清理前存在于本机用户级 systemd 中的 ticknet 工作流。相关 unit 已删除。本文仅作为维护和迁移记录，不代表这些任务仍应恢复。

## 作用范围

这些任务曾由用户级 systemd 管理，unit 文件位于 `~/.config/systemd/user/`，并直接引用旧项目路径：

```text
/home/richard/code/deep-learning-tick-data-prediction
```

当时的正式代码位置已经统一为：

```text
/home/richard/code/research-workspace/deep-learning-tick-data-prediction
```

大型数据和运行产物位于：

```text
/home/richard/data/deep-learning-tick-data-prediction
```

项目代码目录中的 `artifacts/`、`data/`、`results/`、`logs/`、`checkpoints*/` 和 `.venv` 是指向上述外部数据目录的本地软链接。

## 历史 unit 清单

### 数据准备、打包和审计

| Unit | 原用途 |
| --- | --- |
| `ticknet-eventstream-202101-top400.service` | 2021-01 Top400 eventstream 数据准备/打包任务 |
| `ticknet-eventstream-202102-202105-top400.service` | 2021-02 至 2021-05 Top400 eventstream 数据准备/打包任务 |
| `ticknet-eventstream-202508-top400.service` | 2025-08 Top400 eventstream 数据打包任务 |
| `ticknet-eventstream-202509-202512-top400.service` | 2025-09 至 2025-12 Top400 eventstream 数据打包任务 |
| `ticknet-eventstream-202101-audit.service` | 2021-01 eventstream 完整性/审计任务 |
| `ticknet-eventstream-202101-audit.path` | 监听审计相关路径并触发审计 service 的 path watcher |
| `ticknet-eventstream-top400-preflight.service` | Top400 eventstream 任务的前置检查 |

### Benchmark、扫描和任务编排

| Unit | 原用途 |
| --- | --- |
| `ticknet-eventstream-h5-a100-benchmark.service` | H5 eventstream 的 A100 benchmark |
| `ticknet-eventstream-h5-recent-a100-benchmark.service` | 最近数据集的 A100 benchmark |
| `ticknet-eventstream-h5-recent-a100-sweep.service` | 最近数据集的 A100 batch-size sweep |
| `ticknet-eventstream-h5-recent-chain.service` | 等待前序任务结束后串联启动后续任务 |

### 上传

| Unit | 原用途 |
| --- | --- |
| `ticknet-eventstream-h5-benchmark-upload.service` | 将 benchmark 产物上传到远端存储 |
| `ticknet-eventstream-h5-recent-upload.service` | 将最近数据集的 pack、label 和 manifest 上传到远端存储 |

## 清理记录

清理前检查结果：

- 没有活动的 `ticknet-*` user service。
- 所有 service 都是 `static`，没有启用状态。
- `ticknet-eventstream-202101-audit.path` 是 `disabled`。
- 旧项目目录中的 `.pid` 文件对应进程均已不存在。

随后删除了上述 `ticknet-*.service`、`ticknet-*.path` 和该项目遗留的 `.pid` 文件，并执行了以下命令：

```bash
systemctl --user daemon-reload
```

日志和实验产物本身没有因本次 systemd 清理删除，仍保存在外部数据目录中。

## 后续约定

- 不再把实验或数据处理任务默认注册为常驻 systemd user service。
- 需要恢复自动化时，应先明确任务的输入、输出、重试策略、资源占用和停止方式，再单独设计新的调度配置。
- 运行入口使用 workspace 下的正式子模块。数据、checkpoint、日志和产物继续放在仓库外。
- 清理旧任务时，应同时检查 `~/.config/systemd/user/`、用户 crontab、PID 文件和项目文档中的旧路径引用。
