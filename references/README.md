# 论文资料

本目录保存项目涉及论文的原文和中文阅读笔记。

## DeepLOB

- 标题：DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
- 作者：Zihao Zhang、Stefan Zohren、Stephen Roberts
- 单位：牛津大学工程科学系 Oxford-Man Institute of Quantitative Finance
- 链接：[arXiv 1808.03668](https://arxiv.org/abs/1808.03668)
- 笔记：`deeplob-paper-notes.md`

论文提出 CNN、Inception 和 LSTM 组合模型，从限价订单簿原始价格和数量中提取空间与
时间特征。论文包含 FI-2010、伦敦证券交易所数据、跨股票泛化、简单交易模拟和 LIME
解释实验。

代码实现与论文设定的逐项核对放在 `docs/reproduction-audit.md`。阅读笔记用于理解论文，
不替代项目当前实现说明。

## AgentX

- 标题：AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems
- 作者：AgentX Team（快手）
- 链接：[arXiv 2606.26859](https://arxiv.org/abs/2606.26859)
- 笔记：`agentx-paper-notes.md`

论文提出生产级多智能体系统，用闭环把推荐系统研发从"人工 idea-to-launch 串行链"
改写成"可复合、可演化的自动循环"：Brainstorm Agent 生成证据支撑的提案，Developing
Agent 把提案变成生产代码，Evaluation Agent 做护栏否决式 A/B 判断并资产化负面结果，
SGPO 从执行轨迹持续改进 Agent 自身。三周生产验证：374 想法 → 10 可上线结果，吞吐
与线上增益约一个数量级提升。

本仓库参考该论文思路推进"自动量化研究闭环"，落地路线见 `agentx-paper-notes.md`
末节的"对本项目的启示"。

## 文件

| 文件 | 说明 |
|---|---|
| `1808.03668v6.pdf` | DeepLOB arXiv v6 论文原文 |
| `deeplob-paper-notes.md` | DeepLOB 按论文章节整理的中文笔记 |
| `2606.26859v2.pdf` | AgentX arXiv v2 论文原文 |
| `agentx-paper-notes.md` | AgentX 按论文章节整理的中文笔记 |
| `README.md` | 本索引 |
