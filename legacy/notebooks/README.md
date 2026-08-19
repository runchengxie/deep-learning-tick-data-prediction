# 历史 notebook

本目录保存已经退出主链路的 Colab notebook。文件保留早期交互流程和参数拼装方式，不再作为项目运行入口。

| 文件 | 历史用途 | 现行替代入口 |
|---|---|---|
| `nextday_end_to_end_colab.ipynb` | raw-200 pilot 训练、恢复和锁定评估 | `ticknet-nextday-train`、`ticknet-nextday-evaluate`、`scripts/run_colab_nextday.py` |
| `nextday_multi_horizon_validation_colab.ipynb` | 2024 validation 多周期评估和图表展示 | `ticknet-nextday-evaluate-horizons`、`scripts/run_colab_nextday.py --workflow multi-horizon-validation` |
| `colab.ipynb` | FI-2010 复现 | `legacy/scripts/run_colab.py` |

训练、多周期评估、日期权限、数据暂存和产物回传已经由 Python 模块与自动化测试承接。需要复现当前流程时，请从[开发指南](../../docs/dev/development-guide.md)和[Colab CLI 自动化](../../docs/dev/colab-cli-automation.md)进入。
