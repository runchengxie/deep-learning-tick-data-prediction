# 100M 参数 × raw-1000 Top-100 容量基准

## 结论

截至 2026-08-11，精确 100,817,575 参数的 `ChunkedDeepLOB` 已在同一份 raw-1000 Top-100 preflight 上完成 T4 与 A100 训练 benchmark。模型在两张卡上都只占约 2.4 GiB reserved 显存，容量可行，主要约束是训练吞吐。

A100 达到 80.23 samples/s，是 T4 的 3.80 倍。按五年 Top-100 的 2021 至 2023 train 暂估 75,000 个样本、30 epochs 外推，A100 约 7.79 小时每 seed，T4 约 29.58 小时每 seed。后续完整 100M 训练应优先使用 A100，并保留 checkpoint 和断点续训。T4 只适合冒烟或更小模型。基准摘要与后续路线见 [multi-horizon-data-expansion-roadmap.md](multi-horizon-data-expansion-roadmap.md)。

## 可比结果

| 指标 | T4 | A100 |
|---|---:|---:|
| 实际 GPU | Tesla T4 | NVIDIA A100-SXM4-40GB |
| measured batches / samples | 100 / 200 | 100 / 200 |
| samples/s | 21.13 | 80.23 |
| 100 batch 纯训练秒数 | 9.47 | 2.49 |
| 峰值 allocated GiB | 2.19 | 2.18 |
| 峰值 reserved GiB | 2.40 | 2.35 |
| GPU 总显存 GiB | 14.56 | 39.49 |
| 75k 样本 epoch 外推分钟 | 59.16 | 15.58 |
| 30 epochs 单 seed 外推小时 | 29.58 | 7.79 |
| 三 seed 外推 GPU 小时 | 88.73 | 23.37 |

两次运行固定：

- source revision：`09988b7cd3ff722ff075bef241e570f9178684e7`
- dataset fingerprint：`5b7ec4f0aac03847a29a43ac8266c60b16d076e940e0e30cf8dc3227af947406`
- 参数量：100,817,575
- 输入：每样本 `10 × 100 × 40`，共 1000 个 snapshot
- physical batch 为 2，gradient accumulation 为 16，effective batch 为 32
- AMP、分类损失、回归损失、反向传播和 AdamW 更新全部启用
- 2025 locked test：未访问

## 数据 preflight

源 snapshot 约每 3 秒一条。原 raw-200 的 14:30 至 14:55 扫描窗中位数为 500、最大为 503，不足以构造 raw-1000。扫描起点因此改为 13:30，写出时仍严格截取 14:55 前最后 1000 个有效事件。

修正后的 2021-01 Top-100 preflight：

- 请求目标 2,000 个，写出 1,947 个。缺 snapshot 16 个，事件不足 37 个
- 每个已写样本的 `valid_events` 最小值和最大值均为 1000
- 三分类样本数为 390、1,170、387
- 严格落在 1 月 train 的样本为 1,848 个，月底跨界标签按切分合同 purge
- 一个 float16 NPY 分片，共 155,760,128 bytes，目录约 150 MiB
- 本地完整 SHA-256 与 Drive `rclone check --checksum` 均通过

## 解释边界

30 epochs 外推采用数据完成前的 75,000 train 样本假设，不包含每轮 validation、checkpoint 写入、Colab staging 或资源等待，也没有扣除 early stopping。五年 Top-100 pilot 完成后必须用实际 train 样本数重算。当前 benchmark 的 physical batch 为 2，显存余量很大。正式训练前应当再做 batch-size sweep，A100 很可能还能通过更大 micro-batch 提高利用率并缩短 7.79 小时外推。

原始 JSON、execution notebook 和完成标记保存在 Drive 与 Linux 实验产物目录，不提交逐样本数据或凭据。
