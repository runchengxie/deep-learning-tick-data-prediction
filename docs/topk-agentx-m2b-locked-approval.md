# M2b：内容绑定的一次性 locked approval

## 目的

M2b 移除了任何人都能重复使用的静态 `APPROVED` 字符串，把最终 locked test 拆成两个权限
阶段：人工签发和受控消费。Brainstorm、Critic、Runner 及其白名单 executor 都没有签发动作，
研究闭环不能通过 ExperimentSpec 请求 locked 权限。

```text
completed + release + KEEP
  → 人工 approve-locked-test
  → 冻结 spec / checkpoint / predictions / dataset fingerprint
  → 一次性 bearer token
  → locked-test 原子消费
  → RECORDED 或 FAILED，均不可重放
```

## 签发前提

`approve-locked-test` 只有在以下条件全部满足时才签发：

- 实验已登记，状态为 `completed`，阶段为 `release`。
- 确定性 Evaluation 决策为 `KEEP`。
- 实验已经登记非空 `dataset_fingerprint`。
- Registry 至少存在一个 `best_checkpoint` artifact，且磁盘文件 SHA-256 与登记值一致。
- locked predictions 文件已经存在，可计算稳定 SHA-256。
- `reason` 和 `approved_by` 由人工入口显式提供。

签发会原子地把实验改为 `frozen`，写入唯一 approval 和 review。随机 token 只在命令输出中
显示一次，SQLite 仅保存 token 的 SHA-256，不保存可执行的原始 bearer token。同一个实验不能
签发第二份批准。

批准绑定以下内容：

- Registry 中 canonical ExperimentSpec 的 SHA-256。
- 指定 artifact 名称下全部 seed checkpoint 的路径、SHA-256、大小和 bundle SHA-256。
- locked predictions 的绝对路径和 SHA-256。
- 训练/研究数据的 `dataset_fingerprint`。

## 两步命令

人工评审者在模型、预测和理由冻结后签发：

```bash
ticknet-research \
  --registry results/registry.sqlite \
  approve-locked-test \
  --id EXP-RELEASE-001 \
  --predictions locked/predictions.parquet \
  --checkpoint-artifact-name best_checkpoint \
  --reason "正式发布前的唯一最终确认" \
  --approved-by "risk-reviewer"
```

保存输出中的 `token`。执行方只能消费该 token，不能修改任何绑定内容：

```bash
ticknet-research \
  --registry results/registry.sqlite \
  locked-test \
  --id EXP-RELEASE-001 \
  --predictions locked/predictions.parquet \
  --token "<one-time-token>"
```

## 消费与失败语义

执行前会重新计算 spec、checkpoint bundle、predictions 和数据指纹。任何差异都会在 token
消费前拒绝，允许人工调查文件为什么变化。全部绑定一致后，Registry 使用带
`status='issued'` 条件的单行更新原子消费批准，然后才读取 locked 结果。

审计成功时实验进入 `locked_tested`，结果 review 保存批准 ID、消费时间、全部绑定摘要和 Audit。
审计本身失败时 token 仍保持已消费，实验进入 `locked_test_failed`，失败原因写入 result review，
防止根据失败信息反复试探 locked 数据。已消费 token 的任何重放都会拒绝。

## 权限边界

内容绑定与一次性消费解决的是误用、内容替换和重放，不替代真实人员认证。`approved_by` 是审计
标签，不是数字签名。生产运行时应把 `approve-locked-test` 放在 Agent 无法调用的 OS 账号、CI
environment approval 或独立服务中；Agent 只获得签发后的单次 token。即使签发入口被隔离，
Registry 和 locked 文件仍应使用最小文件权限。

## 验收覆盖

合成测试验证：

- 静态 `APPROVED` 没有已签发记录时无效。
- 原始 token 不出现在 SQLite 文件中。
- predictions 或任一 checkpoint 改变时批准失效。
- 非 release、非 KEEP、缺数据指纹或缺 checkpoint 的实验不能签发。
- 成功消费后不可重放；审计失败同样消费并登记失败状态。
- CLI 的签发和消费可以端到端完成。
