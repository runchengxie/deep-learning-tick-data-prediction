# DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

> Paper information
> - Authors: Zihao Zhang, Stefan Zohren, and Stephen Roberts
> - Affiliation: Oxford-Man Institute of Quantitative Finance, Department of Engineering Science, University of Oxford
> - arXiv: [1808.03668v6](https://arxiv.org/abs/1808.03668), q-fin.CP, 2020-01-23
> - Code: https://github.com/zcakhaa

---

## Abstract

The paper proposes a deep-learning model for forecasting future price movements of cash equities from limit order book (LOB) data. Convolutional filters capture the book's spatial structure, and an LSTM captures longer-term temporal dependencies.

The authors report that the model outperformed contemporaneous methods on FI-2010, achieved robust OOS accuracy across multiple instruments using a year of London Stock Exchange data, and generalized to instruments absent from training. They use LIME sensitivity analysis to examine which book components drive predictions, aiming to make the model more interpretable than a black box.

---

## I. Introduction

- More than half of markets use electronic limit order books to record trades. A LOB is organized into price levels and evolves as a multidimensional process involving prices and quantities on both sides.
- LOB data are complex, high-dimensional, and dynamic. Traditional methods such as VAR and ARIMA rely on hand-crafted features and can be inadequate.
- Financial time series are non-stationary and noisy. Deeper book levels are increasingly affected by anticipated order placement and cancellation behavior.
- The paper's main contribution is a CNN+LSTM architecture for price movement prediction from high-frequency LOB data. The authors argue that carefully designed networks can extract representative features from noisy data across stocks.
- An Inception module wraps convolution and pooling to infer local interactions at multiple time scales. Its feature maps are then passed to an LSTM to capture temporal behavior.
- On the public FI-2010 benchmark, the model beats the comparison methods reported at the time. FI-2010 has only ten consecutive trading days, uses downsampling and pre-normalization, and represents a relatively illiquid market, so it cannot establish robustness on its own.
- The authors also test one year of LSE data across five stocks. They tune cautiously on validation data to limit overfitting and use a three-month test period.
- The model is also evaluated on stocks absent from training, making the test out of sample both in time and in data stream. The authors interpret the results as evidence that LOB data contain common patterns related to supply, demand, and price formation.
- A simple trading simulation, assuming midpoint execution and measuring gross returns before fees, reports positive returns with relatively low risk.
- LIME explanations highlight price and volume patterns in the book that the authors consider economically plausible, though sometimes counterintuitive.

The paper is organized as follows: Section II reviews background and related work; Section III describes data, normalization, and labels; Section IV presents the network and its motivation; Section V reports comparisons; and Section VI concludes and discusses future work.

## II. Background and related work

Research on stock-market predictability has a long history and broadly includes statistical parametric models and data-driven machine learning. Recent LOB studies use machine learning, often with static preprocessing such as PCA or LDA. Bag-of-Features (BoF) can also be represented as neural-network layers and trained end to end, with reported improvements.

An important contribution of deep learning is to make feature extraction and representation learnable parts of the model. CNN filter banks can be optimized for the overall objective and have succeeded in tracking, detection, and segmentation. Earlier CNN work on financial microstructure was less common and used simpler architectures. The paper argues, by analogy with the progression from AlexNet to VGGNet, that a careful network design can improve results.

LSTMs address vanishing gradients in RNNs and have been used in language modeling, sequence-to-sequence work, and financial analysis. Reference [20], for example, tested a four-layer LSTM on LOB data from 1,000 stocks and reported stable OOS accuracy over time.

The authors describe this as the first study, to their knowledge, to combine CNN and LSTM for stock-price-movement prediction and the first to apply a nested CNN-LSTM with Inception modules to raw market data.

## III. Data, normalization, and labels

### A. Limit order books

A LOB contains bid and ask orders. A bid offers to buy at its specified price or lower; an ask offers to sell at its specified price or higher. Bid prices and quantities are `Pb(t)` and `Vb(t)`, while ask prices and quantities are `Pa(t)` and `Va(t)`. `P(t)` and `V(t)` denote vectors of values across levels. `pb(1)(t)` is the best bid, and `pa(1)(t)` is the best ask.

Figure 1 shows book snapshots at times `t` and `t+1` and a market buy order consuming the first and second ask levels, moving `pa(1)` from 20.6 to 20.8.

### B. Input data

The paper evaluates two datasets:

1. **FI-2010:** an early public high-frequency LOB benchmark from five Nasdaq Nordic stocks over ten consecutive trading days. It supports comparisons with prior methods, but ten days are insufficient to establish robustness or generalization and invite backtest overfitting.
2. **LSE dataset used in the paper:** one year of data for five liquid LSE stocks: Lloyds Bank (LLOY), Barclays (BARC), Tesco (TSCO), BT, and Vodafone (VOD). The dates are 2017-01-03 to 2017-12-24, during regular trading hours from 08:30 to 16:00, excluding auctions.
   - Ten levels on each side, with price and quantity at each level, yield 40 features per event.
   - The data cover 12 months and more than 134 million samples, with around 150,000 events per day per stock on average. Event intervals are irregular and average 0.192 seconds.
   - The split is six months for training, three for validation, and the final three for testing. At high frequency, three test months contain millions of observations.
   - Only raw order-book data are used. FI-2010, by contrast, is downsampled by taking each tenth non-overlapping event block.

### C. Data normalization and labeling

FI-2010 provides z-score, min-max, and decimal-precision normalization. The paper uses z-score without modification; the other methods reportedly differ little.

The LSE data also use z-scores, but each day's values are normalized with statistics from the preceding five days, separately for each stock. The motivation is regime change: static normalization may be unsuitable for a year of financial time series, while dynamic normalization keeps values in a reasonable range.

The model input is the latest 100 LOB states:

```text
X = [x1, x2, ..., xt, ..., x100]^T ∈ R^{100×40}
xt = [pa(i)(t), va(i)(t), pb(i)(t), vb(i)(t)]_{i=1}^{n=10}
```

Here `p(i)` and `v(i)` are the price and quantity at level `i`.

Direction labels are derived from the midpoint price:

```text
(1)  p(t) = (pa(1)(t) + pb(1)(t)) / 2
```

Because financial data are noisy, directly comparing `p_t` with `p_{t+k}` produces noisy labels. The paper smooths prices on either side of time `t`, where `m-` is the mean of the preceding `k` mid-prices and `m+` is the mean of the following `k` mid-prices:

```text
(2)  m-(t) = (1/k) Σ_{i=0}^{k} p_{t-i}
(3)  m+(t) = (1/k) Σ_{i=1}^{k} p_{t+i}
(4)  l_t = (m+(t) - p_t) / p_t       ([1]'s method)
(5)  l_t = (m+(t) - m-(t)) / m-(t)  ([26]'s method)
```

The percentage change `l_t` and threshold `θ` define the label: `l_t > θ` is up (`+1`), `l_t < -θ` is down (`-1`), and otherwise the label is stationary (`0`).

FI-2010 uses Equation (4), which smooths only future prices and is closer to a realized price but less consistent as a signal. In the LSE experiments, Equation (4) produced noisier labels, so the authors use Equation (5) for a more consistent signal. Figure 2 illustrates the two methods.

## IV. Model architecture

### A. Overview

The network has three main components (Figure 3): standard convolutional layers, an Inception module, and an LSTM. The authors argue that noisy, low signal-to-noise financial data benefit from CNN and Inception feature extraction instead of hand-crafted technical indicators such as MACD and RSI or preprocessing such as PCA. The weights learn from the task, making features adaptive to data. The LSTM models additional temporal dependencies between the extracted features; convolutional scans already capture very short-term dependencies.

### B. Component details

#### a. Convolutional layers

- High-frequency strategies may place and cancel many orders over short intervals (the paper cites over 90% of orders ending in cancellation). The best bid and ask levels (L1) contribute most to price discovery (the paper cites about 80%), while deeper levels contribute less. The authors therefore argue against feeding all levels directly without smoothing or aggregation.
- A convolution filter is a finite impulse response (FIR) filter, commonly used for smoothing and denoising. CNNs make the filter coefficients learnable for the network objective.
- The input has shape `(100×40)`, with 40 features organized as:

  ```text
  (6)  {pa(i)(t), va(i)(t), pb(i)(t), vb(i)(t)}_{i=1}^{n=10}
  ```

- The first convolution uses a `(1×2)` kernel and `(1×2)` stride. The stride matters because prices and quantities behave differently. Without it, the filter would share parameters between `{p(i), v(i)}` and `{v(i), p(i+1)}`, which the authors consider incorrect. This layer aggregates price and quantity within each book level.
- A second `(1×2)` convolution with `(1×2)` stride combines levels and approximates the micro-price in [55]:

  ```text
  (7)  p_micro = I·pa(1) + (1-I)·pb(1)
       I = v_b(1) / (v_a(1) + v_b(1))
  ```

  `I`, the imbalance, is described as a strong predictor of the next price move. The paper uses convolution to construct micro-price-like features across all LOB levels. After two strided layers, the feature map has shape `(100, 10)`. A `(1×10)` kernel then combines all levels, yielding shape `(100, 1)` before Inception.
- Each layer uses zero padding to preserve the time dimension. The activation is Leaky ReLU with negative-side slope `0.01`, selected by validation-grid search.
- Convolutions are translation equivariant, which matters for time series because the same feature can be detected at different times.
- Pooling is not used except inside Inception. Although pooling provides translation invariance, its smoothing can underfit LOB data, where feature position in the sequence matters.

#### b. Inception module (Figure 4)

A standard convolution has a fixed filter size; for example, `(4×1)` captures only local interactions over four time steps. Inception wraps several convolutions to capture dynamics at multiple time scales. The paper compares this with moving averages that use different decay weights: stronger decay smooths long-term trends but loses high-frequency changes. Inception lets backpropagation learn the relevant weights.

The implementation first uses a `1×1` convolution to reduce input dimensions, then `3×1` and `5×1` convolutions, and finally concatenates the outputs. The module includes stride-1, zero-padded max pooling. The `1×1` convolutions apply the Network-in-Network idea: a small network captures nonlinearities, which the authors report improves accuracy. `Inception@32` means all convolution layers in the module have 32 filters.

#### c. LSTM module and output

Classification often uses a fully connected layer, which treats inputs as independent. After Inception, a single 64-unit fully connected layer would have more than 630,000 parameters. The paper instead uses 64 LSTM units to capture temporal relations between extracted features, with roughly 60,000 parameters, about one-tenth as many. A softmax output gives the probability of each price-movement class at each time step. The LSTM's feedback and memory provide temporal modeling.

## V. Experimental results

### A. Experimental settings

- All experiments use the same architecture, named DeepLOB.
- Loss: categorical cross-entropy. Optimizer: Adam (`epsilon=1`, learning rate `0.01`).
- Early stopping: stop when validation accuracy fails to improve for 20 epochs (about 100 epochs on FI-2010 and 40 on LSE).
- Mini-batch size: 32. The paper argues that small batches tend toward wider and shallower minima with better generalization.
- Implementation: Keras and TensorFlow, trained on one NVIDIA Tesla P100 GPU.

### B. FI-2010 experiments

The paper uses Setup 1 and Setup 2:

- **Setup 1:** nine day-based anchored forward folds. Fold `i` trains on the first `i` days and tests on day `i+1`, for `i=1..9`. Early folds have only one or two training days and perform poorly with deep learning; results improve as training data grow (Table I).
- **Setup 2:** train on the first seven days and test on the final three, a common deep-learning setup. DeepLOB reports a substantial lead (Table II).

> FI-2010 is imbalanced. Reference [1] recommends F1 as a fair comparison metric.

Table I: Setup 1 FI-2010 results (selected rows, F1 %)

| Model | k=10 | k=50 | k=100 |
|---|---:|---:|---:|
| RR [1] | 41.00 | 68.84 | 41.60 |
| LDA [22] | 36.28 | 74.32 | 41.00 |
| MDA [22] | 46.06 | Not reported | Not reported |
| MTR [22] | 40.14 | Not reported | Not reported |
| WMTR [22] | 47.87 | Not reported | Not reported |
| BoF [24] | 36.28 | 39.56 | 40.84 |
| B(TABL) [25] | 67.12 | 68.84 | 68.86 |
| C(TABL) [25] | 72.84 | 74.32 | 73.52 |
| DeepLOB | 77.66 | 74.96 | 76.58 |

Table II: Setup 2 FI-2010 results (selected rows, Accuracy % / F1 %)

| Model | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|
| SVM [28] | 44.92 / 35.88 | 84.47 / 43.20 | 70.80 / 49.42 | Not reported |
| MLP [28] | 60.78 / 48.27 | 65.20 / 51.12 | 73.74 / 55.95 | Not reported |
| CNN-I [26] | 39.62 / 55.21 | 67.38 / 59.17 | 74.85 / 59.44 | 47.00 / 47.00 |
| LSTM [28] | 47.81 / 66.33 | 70.52 / 62.37 | 68.58 / 61.43 | Not reported |
| B(TABL) [25] | 60.77 / 69.20 | 51.33 / 62.22 | 73.09 / 73.64 | Not reported |
| C(TABL) [25] | 56.00 / 77.63 | 54.79 / 66.93 | 46.05 / 78.44 | Not reported |
| DeepLOB | 78.91 / 83.40 | 84.70 / 76.95 | 59.60 / 72.82 | 55.21 / 80.35 |

Table III: forward-pass time and parameter count

| Model | Forward pass (ms) | Parameters |
|---|---:|---:|
| BoF [24] | 0.972 | 86k |
| N-BoF [24] | 0.524 | 12k |
| CNN-I [26] | 0.025 | 768k |
| LSTM [28] | 0.061 | Not reported |
| C(TABL) [25] | 0.229 | Not reported |
| DeepLOB | 0.253 | 60k |

Despite having more layers, DeepLOB uses an LSTM instead of a fully connected layer and has far fewer parameters than CNN-I (60k versus 768k). Its forward pass is also fast, which the paper presents as suitable for high-frequency trading.

### C. London Stock Exchange experiments

The model is trained on LLOY, BARC, TSCO, BT, and VOD, with a three-month test period. Transfer learning applies the trained model directly to five unseen liquid stocks: HSBC, Glencore (GLEN), Centrica (CNA), BP, and ITV, using the same three-month test period. The classes are approximately balanced.

Table IV: LSE dataset results

| Setting | Horizon k | Accuracy % | Precision % | Recall % | F1 % |
|---|---:|---:|---:|---:|---:|
| Training stocks (LLOY, BARC, TSCO, BT, VOD) | 20 | 70.17 | 70.17 | 70.17 | 70.15 |
| | 50 | 63.93 | 63.43 | 63.93 | 63.49 |
| | 100 | 61.52 | 60.73 | 61.52 | 60.65 |
| Transfer stocks (GLEN, HSBC, CNA, BP, ITV) | 20 | 68.62 | 68.64 | 68.63 | 68.48 |
| | 50 | 63.44 | 62.81 | 63.45 | 62.84 |
| | 100 | 61.46 | 60.68 | 61.46 | 60.77 |

Figures 5 and 6 show confusion matrices and daily-accuracy box plots. The authors report consistent performance across stocks over the test period, with narrow interquartile ranges and few outliers. They interpret the unseen-stock results as evidence that the CNN learns common LOB patterns related to price formation.

### D. Simple trading simulation

- Position size is `n=1` share to minimize market impact and assume execution at the best price.
- At each step, the model's `(-1, 0, +1)` signal means sell, wait, or buy. A `+1` signal buys `n` shares at `t+5` and holds until a `-1` signal; `0` leaves the position unchanged. Shorting follows the corresponding inverse rule. Positions are closed before each day's end, and auction periods are excluded.
- The simulation assumes midpoint execution and no transaction costs. It is intended as a relative measure of predictability, not an independent trading strategy.
- Figures 7 and 8 show normalized daily-return box plots, t-statistics, and cumulative returns. The paper reports consistent returns and significant t-statistics across stocks and horizons. Longer horizons have slightly lower accuracy but more stable signals, so cumulative returns are higher.

### E. Sensitivity analysis

The paper notes that trust and risk are important in finance, while deep networks are often treated as black boxes. It uses LIME (Local Interpretable Model-agnostic Explanations), perturbing inputs locally and observing prediction changes to estimate input importance and sensitivity.

Figure 9 compares DeepLOB and CNN-I [26] on one input. Most CNN-I regions are inactive, which the authors attribute to two max-pooling layers and a large first-layer filter covering broad regions of the deeper representation. DeepLOB shows richer active regions that the authors consider more consistent with econometric intuition, illustrating how prices and quantities at different levels and times affect predictions.

## VI. Conclusion

- The paper proposes a hybrid CNN+Inception+LSTM for predicting price movements from high-frequency LOB data. It learns features and temporal dependencies without manual features.
- It outperforms the comparison methods on FI-2010 and reports robust performance on one year of LSE data with a three-month test period.
- Generalization to unseen stocks suggests that LOB data contain informative common patterns in price formation and that the model can learn them from a larger dataset.
- The paper's simple trading simulation reports statistically significant returns.
- LIME sensitivity analysis attributes predictions to input components in ways the authors consider economically interpretable, addressing some black-box concerns.
- Future work includes Bayesian neural networks [69] for output uncertainty that could inform position sizing, and more detailed trading strategies combined with reinforcement learning.

## Acknowledgements

The paper acknowledges the University of Oxford Machine Learning Research Group, the Oxford-Man Institute of Quantitative Finance (for LOB data), Arcus Phase B, JADE HPC, Hartree computing facilities, and the Royal Academy of Engineering.

## Selected references

The full bibliography is in the paper. Selected works cited in these notes:

- [1] A. Ntakaris et al., “Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods,” *Journal of Forecasting*, 2018 (FI-2010 dataset).
- [2] C. A. Parlour and D. J. Seppi, “Limit order markets: A survey,” 2008.
- [4] E. Zivot and J. Wang, “Vector autoregressive models for multivariate time series,” 2006.
- [5] A. A. Ariyo et al., “Stock price prediction using the ARIMA model,” 2014.
- [7] M. D. Gould et al., “Limit order books,” *Quantitative Finance*, 2013.
- [9] C. Szegedy et al., “Going deeper with convolutions” (Inception), CVPR 2015.
- [10] M. T. Ribeiro et al., “Why should I trust you?” (LIME), KDD 2016.
- [20] Four-layer LSTM study using LOB data from 1,000 stocks.
- [22] Baseline methods including LDA, MDA, MTR, and WMTR.
- [24] BoF and N-BoF (Bag-of-Features).
- [25] B(TABL) and C(TABL) (bilinear networks).
- [26] A. Tsantekidis et al., CNN-I, CI 2017.
- [27] A. Tsantekidis et al., stationary LOB features for price prediction, arXiv:1810.09965.
- [28] A. Tsantekidis et al., detecting price-movement indicators with deep learning, signal-processing conference.
- [38] Original LSTM paper by Hochreiter and Schmidhuber.
- [64] Network-in-Network.
- [65] Adam optimizer.
- [69] The authors' later Bayesian neural network work for uncertainty and position sizing.

## Architecture reference

```text
Input X ∈ R^{100×40}  (100 recent LOB states × 40 features)
   │
   ├─ Conv (1×2, stride 1×2) × several  ← pair price/quantity, then combine levels
   ├─ Conv (1×10) to aggregate
   │
   ├─ Inception Module (3×1, 5×1, 1×1, max-pool) @32
   │
   ├─ LSTM @64 units
   │
   └─ Softmax → P(Down) / P(Stationary) / P(Up)
```

Key design choices:

- Convolution learns spatial features rather than requiring manual features.
- Inception captures multiple time scales.
- LSTM captures temporal dependencies with far fewer parameters than a fully connected layer.
- Dynamic z-score normalization uses the preceding five days of statistics for LSE data.
- Smoothed labels (Equation 4 or 5) and threshold `θ` define three classes.
