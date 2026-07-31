# DeepLOB 论文资料

本目录保存 DeepLOB 论文原文和中文阅读笔记。

## 论文信息

- 标题：DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
- 作者：Zihao Zhang、Stefan Zohren、Stephen Roberts
- 单位：牛津大学工程科学系 Oxford-Man Institute of Quantitative Finance
- 链接：[arXiv 1808.03668](https://arxiv.org/abs/1808.03668)

## 文件

| 文件 | 说明 |
|---|---|
| `1808.03668v6.pdf` | arXiv v6 论文原文 |
| `DeepLOB-论文整理.md` | 按论文章节整理的中文笔记 |
| `README_论文资料.md` | 本索引 |

论文提出 CNN、Inception 和 LSTM 组合模型，从限价订单簿原始价格和数量中提取空间与
时间特征。论文包含 FI-2010、伦敦证券交易所数据、跨股票泛化、简单交易模拟和 LIME
解释实验。

代码实现与论文设定的逐项核对放在 `docs/复现核对.md`。阅读笔记用于理解论文，
不替代项目当前实现说明。
