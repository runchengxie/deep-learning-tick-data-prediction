# DeepLOB 复现项目

本项目是对论文 **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books** 的复现与整理。

## 论文

- **标题**：DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
- **作者**：Zihao Zhang, Stefan Zohren, Stephen Roberts（Oxford-Man Institute of Quantitative Finance, University of Oxford）
- **链接**：https://arxiv.org/abs/1808.03668

## 关于本项目

本目录包含上述论文的原文 PDF（`1808.03668v6.pdf`）以及一份整理好的中文 Markdown 笔记（`DeepLOB-论文整理.md`），便于快速理解论文的模型架构、数据处理方式与实验结果。

### 论文核心内容

提出一种结合 **CNN + Inception Module + LSTM** 的混合深度神经网络，直接从原始限价订单簿（LOB）数据预测短期价格走势：

- **卷积层 / Inception** 自动提取订单簿的空间结构特征（无需手工特征）；
- **LSTM** 捕捉时序依赖；
- 在基准数据集 **FI-2010** 上超越当时所有 SOTA 方法；
- 在 **伦敦证券交易所（LSE）一整年**数据上验证，并展示了向训练集外股票的**迁移能力**（通用特征）；
- 用 **LIME** 做可解释性分析，缓解"黑盒"问题。

## 目录结构

| 文件 | 说明 |
|---|---|
| `1808.03668v6.pdf` | 论文原文（arXiv v6） |
| `DeepLOB-论文整理.md` | 论文的结构化中文整理笔记 |
| `README.md` | 本文件 |
