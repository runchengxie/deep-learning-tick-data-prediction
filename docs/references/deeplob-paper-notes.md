# DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

> 论文信息
> - 作者：Zihao Zhang, Stefan Zohren, Stephen Roberts
> - 单位：Oxford-Man Institute of Quantitative Finance, Department of Engineering Science, University of Oxford
> - arXiv：[1808.03668v6](https://arxiv.org/abs/1808.03668) [q-fin.CP], 2020-01-23
> - 代码：https://github.com/zcakhaa

---

## 摘要（Abstract）

本文提出一种大规模深度学习模型，用于根据限价订单簿（Limit Order Book, LOB）数据预测现金股票的未来价格走势。模型架构使用：

- 卷积滤波器（convolutional filters）捕捉订单簿的空间结构，
- LSTM 模块捕捉长时间依赖。

主要贡献与结论：

1. 在基准数据集 FI-2010 上超越了当时所有最先进的算法，
2. 在更真实的设定下，用伦敦证券交易所（LSE）一整年的行情数据做测试，对多种标的都取得了非常稳健的样本外预测准确率，
3. 模型能很好地迁移到训练集中未出现的标的，说明它提取了通用特征（universal features），
4. 通过敏感性分析（LIME）理解预测背后的逻辑，揭示订单簿中哪些成分最相关，超越了黑盒模型。

---

## I. 引言（Introduction）

- 当今超过一半的市场使用电子限价订单簿（LOB）记录交易。LOB 按价格分为多个档位（levels），其随时间演化是一个多维问题，涉及买卖两侧多档的价格与数量。
- LOB 是复杂、高维、动态的环境，传统方法（如 VAR、ARIMA 等马尔可夫类模型）依赖手工特征，难以应对。
- 金融时间序列非平稳、噪声大，越深的档位越容易被挂单/撤单预期行为污染。
- 本文贡献：设计结合 CNN + LSTM 的新架构预测高频 LOB 数据下的价格走势。相比已有研究，本模型能从高噪声数据中为多种股票提取代表性特征。
- 使用 Inception Module 包裹卷积与池化层，推断不同时域的局部交互，再将特征图送入 LSTM 捕捉动态时序行为。
- 在公开基准 FI-2010 上超越当时的对比方法。该数据集只有 10 个连续交易日，并经过降采样和预归一化，市场流动性也较低，无法充分验证稳健性。
- 进一步用 LSE 5 只股票一整年数据验证，并在验证集上谨慎调参以避免过拟合，测试期 3 个月，跨股票表现稳健。
- 还测试了训练集之外的股票（时间和数据流双重样本外），仍取得良好结果，表明订单簿中存在调制股票供需与价格的通用特征。
- 用简单交易模拟验证实用性（假设中间价成交、比较手续费前毛利），模型在较小风险下取得显著正收益。
- 用 LIME 方法解释模型可解释性：被关注的区域符合价格与成交量在订单簿中合理（虽有些反常）的活动模式。

论文结构：
- Section II：背景与相关工作
- Section III：数据、归一化与标注
- Section IV：网络架构及各部分依据
- Section V：与大量主流方法的对比实验
- Section VI：总结与未来工作

---

## II. 背景与相关工作（Background and Related Work）

- 关于股票市场可预测性的研究历史悠久，主流分为两类：统计参数模型与数据驱动的机器学习方法。
- 近期大量工作用机器学习预测 LOB 数据。常用特征提取：PCA、LDA 等静态预处理，BoF（Bag-of-Features）被表达为神经网络层并端到端训练，效果显著提升。
- 深度学习的关键贡献是把特征提取与表示作为可学习模型的一部分。CNN 的滤波器组自动调优到整体网络目标，已成功用于目标跟踪、检测、分割等。将 CNN 用于金融微观结构数据的工作较少且架构简单：本文证明精心的网络设计（类比 AlexNet→VGGNet）能带来更好结果。
- LSTM 用于解决 RNN 的梯度消失问题，已被广泛用于语言建模、seq2seq，以及金融数据分析（多篇文献）。其中 [20] 用 1000 只股票的 LOB 数据测试四层 LSTM，样本外准确率随时间稳定。
- 本文定位：据作者所知，这是首个结合 CNN 与 LSTM 预测股票价格走势的研究，也是首次把嵌套 CNN-LSTM 和 Inception 模块用于原始市场数据。

---

## III. 数据、归一化与标注（Data, Normalisation and Labelling）

### A. 限价订单簿（Limit Order Books）

- LOB 含两类订单：bid（买）与 ask（卖）。bid 是在指定价格或更低买入，ask 是在指定价格或更高卖出。
- 买单有价格 `Pb(t)` 与数量 `Vb(t)`，卖单有 `Pa(t)` 与 `Va(t)`。`P(t)`、`V(t)` 为不同档位取值的向量。
- 用 `pb(1)(t)` 表示最优买价（第一买档），`pa(1)(t)` 表示最优卖价（第一卖档）。
- 图 1 展示了 t 与 t+1 时刻的 LOB 切片及一笔市价买单吃掉第一、二卖档使 `pa(1)` 从 20.6 移动到 20.8 的过程。

### B. 输入数据（Input Data）

本文在两个数据集上测试：

1. FI-2010：首个公开的高频限价订单簿基准，取自 Nasdaq Nordic 5 只股票、连续 10 个交易日。很多既有算法在此测试，便于公平对比。10 天数据量不足以充分验证稳健性和泛化能力，也容易出现回测过拟合。
2. LSE 数据集（本文使用）：取 Lloyds Bank (LLOY)、Barclays (BARC)、Tesco (TSCO)、BT、Vodafone (VOD) 五只 LSE 最流动性股票的一整年数据（2017-01-03 至 2017-12-24，每日 08:30 至 16:00 正常交易时段，无竞价）。
   - 每侧 10 档，每档含价格与数量 → 每时刻 40 个特征。
   - 共 12 个月、超过 1.34 亿样本，平均约 15 万事件/天/股，事件间隔不规则，平均 `0.192` 秒。
   - 划分：前 6 个月训练，中间 3 个月验证，最后 3 个月测试（高频下 3 个月测试对应数百万观测）。
   - 仅输入原始订单簿信息，不做额外处理（FI-2010 则对每 10 个非重叠事件块做了降采样）。

### C. 数据归一化与标注（Data Normalisation and Labelling）

- FI-2010 提供三种归一化（z-score、min-max、decimal precision），本文用 z-score 且无修改，另两种差异很小。
- LSE 数据集同样用 z-score 标准化，但用前 5 天的均值/标准差归一化当天数据（各标的分开）。原因：金融时间序列存在机制切换（regime shifts），静态归一化不适合一年长度的数据，动态归一化使数据落在合理范围。
- 以 LOB 最近 100 个状态 为模型输入。单个输入定义：

  ```
  X = [x1, x2, ..., xt, ..., x100]^T ∈ R^{100×40}
  xt = [pa(i)(t), va(i)(t), pb(i)(t), vb(i)(t)]_{i=1}^{n=10}
  ```

  其中 `p(i)`、`v(i)` 分别为第 i 档的价格与数量。

- 用中间价（mid-price） 生成方向标签：

  (1)  `p(t) = (pa(1)(t) + pb(1)(t)) / 2`

  由于金融数据高度随机，直接比较 `pt` 与 `pt+k` 会得到噪声标签。论文采用两种平滑标注法（令 `m-` 为前 k 个中间价均值，`m+` 为后 k 个中间价均值）：

  (2)  `m-(t) = (1/k) Σ_{i=0}^{k} p_{t-i}`
  (3)  `m+(t) = (1/k) Σ_{i=1}^{k} p_{t+i}`
  (4)  `l_t = (m+(t) - p_t) / p_t`        （[1] 的方法）
  (5)  `l_t = (m+(t) - m-(t)) / m-(t)`     （[26] 的方法）

- 根据百分比变化 `l_t` 与阈值 `θ` 决定标签：
  - `l_t > θ` → 涨 (+1)
  - `l_t < -θ` → 跌 (-1)
  - 否则 → 平稳 (0)

- 选择：FI-2010 采用式 (4)（仅对未来价格平滑，更贴近真实价格但信号一致性差），本文 LSE 数据实验发现式 (4) 产生的标签更随机，故 LSE 采用式 (5) 以产生更一致的信号（图 2 给出两种方法对比示意）。

---

## IV. 模型架构（Model Architecture）

### A. 概述（Overview）

网络由三大模块组成（图 3）：
1. 标准卷积层（standard convolutional layers）
2. Inception Module
3. LSTM 层

动机：金融数据噪声大、信噪比低，用 CNN 与 Inception 自动完成特征提取，省去手工技术指标（如 MACD、RSI）与 PCA 等预处理。权重在推断中学习，特征是数据自适应的。LSTM 层负责捕捉特征间额外的时序依赖（极短时依赖已由卷积层对时空图像的扫描捕捉）。

### B. 各组件细节（Details of Each Component）

#### a) 卷积层（Convolutional Layer）

- 高频算法常短时间内大量挂/撤单（>90% 订单以撤单告终），且最优买卖档（L1）对价格发现贡献最大（约 80%），深层档位贡献很小。因此不应把所有档位都喂给网络，需对深层信息做平滑/汇总。
- 卷积滤波器本质为有限冲激响应（FIR）滤波器，是常用的去噪平滑技术。CNN 把滤波器系数设为可学习参数，并根据网络目标完成训练。
- 输入尺寸 `(100×40)`，40 个特征按式 (6) 组织：

  (6)  `{pa(i)(t), va(i)(t), pb(i)(t), vb(i)(t)}_{i=1}^{n=10}`

- 第一层卷积核大小 `(1×2)`、步长 `(1×2)`。步长必要：价格与数量动态行为不同，若不用步长会让 `{p(i), v(i)}` 与 `{v(i), p(i+1)}` 共享参数（错误）。第一层在每个订单簿档位内汇总价格与数量信息。
- 再用一层核 `(1×2)`、步长 `(1×2)` 的卷积跨档位整合，实际形成 [55] 定义的微价（micro-price）：

  (7)  `p_micro = I·pa(1) + (1-I)·pb(1)`，其中 `I = v_b(1) / (v_a(1) + v_b(1))`

  `I` 称为失衡（imbalance），是预测下一价格走势的强指标。本文用卷积为 LOB 所有档位构造微价，两层 stride 后特征图尺寸 `(100, 10)`，最后用大核 `(1×10)` 整合全部信息，送入 Inception 前特征图尺寸 `(100, 1)`。
- 每层做 zero padding 保持时间维度不变，激活函数用 Leaky-ReLU（负区间小梯度设为 `0.01`，由验证集网格搜索确定）。
- 卷积具有平移等变性（equivariance to translation），对时序数据重要：能在不同时间点提取同一通用特征。
- 不使用池化层（Inception 内部除外）：池化虽提供平移不变性，但其平滑特性会导致 LOB 数据欠拟合，时间序列中特征位置很重要。

#### b) Inception Module（图 4）

- 标准卷积层滤波器大小固定（如 `(4×1)` 只能捕捉 4 个时间步的局部交互）。Inception Module 包裹多个卷积以捕捉多时间尺度动态行为，带来性能提升。
- 可类比技术分析中的不同衰减权重移动平均：大衰减得到平滑长期趋势但丢失高频变化。Inception 让权重在反向传播中自动学习。
- 本文做法：先用 `1×1` 卷积把输入降为低维表示，再用 `3×1` 与 `5×1` 卷积变换，最后合并输出，模块内含 stride 1、zero-padding 的 max-pooling。`1×1` 卷积构成 Network-in-Network 方法，用小网络捕捉非线性，带来准确率提升。Inception@32 表示模块内所有卷积层均 32 个滤波器。

#### c) LSTM 模块与输出（LSTM Module and Output）

- 通常分类用全连接层，但全连接默认各输入独立。Inception 后特征很多，单全连接层（64 单元）参数将超 63 万。
- 改用 64 个 LSTM 单元捕捉提取特征的时序关系，参数约 6 万（减少约 10 倍）。
- 输出层用 softmax，最终输出为各价格走势类别在每个时间步的概率。
- 因 LSTM 自带反馈与记忆，可建模特征的时序动态。

---

## V. 实验结果（Experimental Results）

### A. 实验设置（Experiments Settings）

- 所有实验使用同一架构，模型记为 DeepLOB。
- 损失：分类交叉熵（categorical cross-entropy），优化器：Adam（epsilon=1，学习率=0.01）。
- 早停：验证准确率连续 20 个 epoch 不提升则停止（FI-2010 约 100 epoch，LSE 约 40 epoch）。
- mini-batch = 32（小批量倾向于收敛到更宽更浅的最小值，泛化更好）。
- 基于 Keras + TensorFlow，单张 NVIDIA Tesla P100 GPU 训练。

### B. FI-2010 数据集实验

两种设定（Setup 1 / Setup 2）：

- Setup 1：按天分 9 折（anchored forward split），第 i 折用前 i 天训练、第 i+1 天测试（i=1..9）。前几折训练数据仅 1 至 2 天，深度学习表现差，随训练数据增长表现显著提升（表 I）。
- Setup 2：前 7 天训练、后 3 天测试（深度学习常用设定）。DeepLOB 大幅领先（表 II）。

> 注：FI-2010 不平衡，[1] 建议以 F1 分数为公平对比指标。

表 I：Setup 1（FI-2010）结果（节选，F1 %）

| Model | k=10 | k=50 | k=100 |
|---|---|---|---|
| RR [1] | 41.00 | 68.84 | 41.60 |
| LDA [22] | 36.28 | 74.32 | 41.00 |
| MDA [22] | 46.06 | 未报告 | 未报告 |
| MTR [22] | 40.14 | 未报告 | 未报告 |
| WMTR [22] | 47.87 | 未报告 | 未报告 |
| BoF [24] | 36.28 | 39.56 | 40.84 |
| B(TABL) [25] | 67.12 | 68.84 | 68.86 |
| C(TABL) [25] | 72.84 | 74.32 | 73.52 |
| DeepLOB | 77.66 | 74.96 | 76.58 |

表 II：Setup 2（FI-2010）结果（节选，Accuracy% / F1%）

| Model | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|
| SVM [28] | 44.92 / 35.88 | 84.47 / 43.20 | 70.80 / 49.42 | 未报告 |
| MLP [28] | 60.78 / 48.27 | 65.20 / 51.12 | 73.74 / 55.95 | 未报告 |
| CNN-I [26] | 39.62 / 55.21 | 67.38 / 59.17 | 74.85 / 59.44 | 47.00 / 47.00 |
| LSTM [28] | 47.81 / 66.33 | 70.52 / 62.37 | 68.58 / 61.43 | 未报告 |
| B(TABL) [25] | 60.77 / 69.20 | 51.33 / 62.22 | 73.09 / 73.64 | 未报告 |
| C(TABL) [25] | 56.00 / 77.63 | 54.79 / 66.93 | 46.05 / 78.44 | 未报告 |
| DeepLOB | 78.91 / 83.40 | 84.70 / 76.95 | 59.60 / 72.82 | 55.21 / 80.35 |

表 III：计算时间对比（前向传播 ms）与参数量

| Models | Forward (ms) | 参数量 |
|---|---|---|
| BoF [24] | 0.972 | 86k |
| N-BoF [24] | 0.524 | 12k |
| CNN-I [26] | 0.025 | 768k |
| LSTM [28] | 0.061 | 未报告 |
| C(TABL) [25] | 0.229 | 未报告 |
| DeepLOB | 0.253 | 60k |

> 虽然层数更多，但因用 LSTM 代替全连接，DeepLOB 参数量远小于 CNN-I（768k 对 60k），且前向速度快，适合高频交易。

### C. 伦敦证券交易所（LSE）实验

- 训练 5 只股票：LLOY、BARC、TSCO、BT、VOD，测试期 3 个月。
- 迁移学习（transfer learning）：直接在 5 只未参与训练的高流动性股票上应用模型：HSBC、Glencore (GLEN)、Centrica (CNA)、BP、ITV，测试期相同 3 个月，类别大致平衡。

表 IV：LSE 数据集结果

| 设定 | 预测步长 k | Accuracy % | Precision % | Recall % | F1 % |
|---|---|---|---|---|---|
| 训练内股票 | k=20 | 70.17 | 70.17 | 70.17 | 70.15 |
| (LLOY,BARC,TSCO,BT,VOD) | k=50 | 63.93 | 63.43 | 63.93 | 63.49 |
| | k=100 | 61.52 | 60.73 | 61.52 | 60.65 |
| 迁移学习 | k=20 | 68.62 | 68.64 | 68.63 | 68.48 |
| (GLEN,HSBC,CNA,BP,ITV) | k=50 | 63.44 | 62.81 | 63.45 | 62.84 |
| | k=100 | 61.46 | 60.68 | 61.46 | 60.77 |

- 图 5 混淆矩阵、图 6 每日准确率箱线图显示：所有股票在整个测试期表现一致且稳健（IQR 窄、离群点少）。
- 关键观察：模型能泛化到训练集外数据，说明 CNN 模块从 LOB 提取了与价格形成机制相关的通用模式（universal patterns）。

### D. 简单交易模拟（Simple Trading Simulation）

- 设定每笔股数 `n=1`（最小化市场冲击，保证以最优价成交）。
- 规则：每步模型输出信号 `(-1, 0, +1)` 对应动作（卖出、等待、买入）。预测 `+1` 则在 `t+5` 买入 `n` 股，直到出现 `-1` 卖出（出现 `0` 不动），做空同理，每日收盘前平仓，不在竞价时段交易。
- 假设：以中间价成交、无交易成本（作为模型可预测性的相对度量，非独立交易策略）。
- 图 7（归一化日收益箱线图与 t 统计量）与图 8（累计收益）显示：所有股票、各预测步长下收益一致且 t 值显著。虽长预测步长准确率略低，但因信号更稳健，累计收益反而更高。

### E. 敏感性分析（Sensitivity Analysis）

- 金融应用中信任与风险至关重要，需理解预测背后的原因。深度网络常被视为黑盒。
- 使用 LIME（Local Interpretable Model-agnostic Explanations）：通过局部扰动输入、观察预测变化，给出输入重要性与敏感性度量。
- 图 9 示例：展示 DeepLOB 与 CNN-I [26] 对给定输入的反应。CNN-I 大部分输入区域不活跃，原因是两个最大池化层和较大的首层滤波器让深层表示覆盖了较大范围。DeepLOB 显示出更丰富、符合计量经济学直觉的活跃区域，可以看到不同档位和时刻的价格、数量如何影响预测。

---

## VI. 结论（Conclusion）

- 本文提出首个混合深度神经网络（CNN + Inception + LSTM）用于高频 LOB 数据预测股票价格走势，自动完成特征提取与时序依赖建模，无需手工特征。
- 在 FI-2010 基准上超越所有对比技术，在 LSE 一整年数据（测试期 3 个月）上表现稳健。
- 重要发现：模型泛化到训练集外标的，提示订单簿中存在对价格形成具有信息量的通用特征，且本模型能从中大数据集中学习到这些特征。
- 简单交易模拟取得统计显著的收益。
- 用 LIME 做敏感性分析，揭示输入各成分对预测的贡献，符合计量经济学理解，缓解了黑盒质疑。
- 后续工作：
  - 扩展为贝叶斯神经网络（Bayesian neural networks）[69]，提供输出不确定性度量，可用于头寸加仓，
  - 研究更细致的交易策略，结合强化学习（Reinforcement Learning）。

---

## 致谢（Acknowledgements）

感谢牛津大学机器学习研究组、Oxford-Man Institute of Quantitative Finance（提供 LOB 数据）、Arcus Phase B / JADE HPC / Hartree 计算设施、英国皇家工程院（Royal Academy of Engineering）的支持。

---

## 参考文献（References，节选关键条目）

> 完整参考文献见原文。以下列出正文重点引用的文献。

- [1] A. Ntakaris et al., Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods, *Journal of Forecasting*, 2018.（FI-2010 数据集）
- [2] C. A. Parlour, D. J. Seppi, Limit order markets: A survey, 2008.
- [4] E. Zivot, J. Wang, Vector autoregressive models for multivariate time series, 2006.
- [5] A. A. Ariyo et al., Stock price prediction using the ARIMA model, 2014.
- [7] M. D. Gould et al., Limit order books, *Quantitative Finance*, 2013.
- [9] C. Szegedy et al., Going deeper with convolutions (Inception), CVPR 2015.
- [10] M. T. Ribeiro et al., Why should I trust you?（LIME）, KDD 2016.
- [20] 使用 1000 只股票 LOB 数据的四层 LSTM 研究。
- [22] LDA / MDA / MTR / WMTR 等基准方法。
- [24] BoF / N-BoF（Bag-of-Features）。
- [25] B(TABL) / C(TABL)（双线性网络）。
- [26] A. Tsantekidis et al., CNN-I，CI 2017.
- [27] A. Tsantekidis et al., 利用平稳 LOB 特征做价格预测, arXiv:1810.09965.
- [28] A. Tsantekidis et al., 用深度学习检测价格变化指示, 信号处理会议.
- [38] LSTM 原始论文（Hochreiter & Schmidhuber）。
- [64] Network-in-Network.
- [65] Adam 优化器.
- [69] 贝叶斯神经网络扩展（作者后续工作，用于不确定性度量与头寸加仓）.

---

## 附：核心架构速查

```
输入 X ∈ R^{100×40}  (100 个最近 LOB 状态 × 40 特征)
   │
   ├─ Conv (1×2, stride 1×2) ×若干    ← 价格/数量配对 + 跨档位微价
   ├─ Conv (1×10) 整合
   │
   ├─ Inception Module (3×1, 5×1, 1×1, max-pool) @32
   │
   ├─ LSTM @64 units
   │
   └─ Softmax → P(Down) / P(Stationary) / P(Up)
```

关键设计点：
- 卷积自动提取空间特征，避免手工特征，
- Inception 捕捉多时间尺度，
- LSTM 捕捉时序依赖，参数量远小于全连接，
- 动态 z-score 归一化（LSE 用前 5 天统计量），
- 平滑标注（式 4 / 式 5）+ 阈值 θ 三分类。
