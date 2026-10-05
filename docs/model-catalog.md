# Model Catalog and Selection Guide

This page summarizes implemented models and explicitly planned research. It describes their inputs, methods, intended use, and current evidence. Metrics change as experiments progress. Refer to [Project status](project-status.md) and the [experiment log](research/experiment-log.md) for complete figures.

## At a glance

| Model | Input | Deep learning | Status | Current assessment |
|---|---|---:|---|---|
| Logistic Regression | Daily statistics of raw order-book inputs | No | Implemented | Low-cost minimum baseline for testing whether sequence models add information |
| HGB | Minute aggregates of orders, trades, and snapshots | No | Formal M3 experiment complete | Most stable minute baseline; its prediction signal has not produced stable active returns that cover 10 bp one-way cost |
| Minute TCN | Minute features ordered by time | Yes | Controlled three-seed comparison complete | Validation result was stronger than HGB, but test results were weaker and indicate overfitting |
| Minute GRU | Minute features ordered by time | Yes | Training and evaluation implemented | Tests whether a recurrent model adds to the TCN comparison; no separate formal result yet |
| Chunked DeepLOB | Last 200 or 1,000 ten-level snapshots | Yes | Four-cell, three-seed matrix complete | `1M/raw-200` is most stable; longer windows and more parameters have no stable gain |
| L2 event-stream Transformer | Losslessly merged orders, trades, and snapshots, optionally combined with minute features | Yes | 100M, frozen-representation, and joint-training three-seed work complete | Joint Rank IC is consistently positive; net active returns after costs still need improvement |
| FI-2010 DeepLOB | Ten-level FI-2010 order-book windows | Yes | Archived under `legacy/` | Used for paper-reproduction and compatibility checks; does not show next-day A-share prediction performance |
| LambdaMART | Cross-sectional features grouped by trading day | No | Recent-fold M4 comparison complete | Embeddings alone show OOS signal, but combined gains lack cross-seed and cross-month stability |

## Logistic Regression

**How it works:** Compress each stock-day order-book sequence into per-feature mean, standard deviation, last value, and first-to-last change. Standardize the features and learn a weight for each one. Convert the weighted output to down, neutral, and up probabilities.

**Why use it:** It trains quickly, uses few resources, and is easy to interpret. It provides a clear minimum baseline for more complex models.

**Limitations:** It represents only simple linear relationships. Aggregation discards event order and does not automatically learn complex feature interactions.

Entry point: `scripts/run_nextday_baseline.py`.

## HGB

HGB is the histogram gradient-boosting tree model `HistGradientBoostingClassifier`, a conventional machine-learning model.

**How it works:** Train a sequence of shallow decision trees. Each new tree focuses on samples that earlier trees still classify poorly, and the tree outputs are summed. Continuous features are binned to reduce computation. The current pipeline uses recent minute-feature aggregates and retains missing-value paths.

**Why use it:** It handles tabular features and nonlinear relationships, trains inexpensively on CPU, supports missing values, and has been stable at the current data scale. M3 v2 reported a 2025 H2 Rank IC of 0.06994, with positive monthly IC in all six months.

**Limitations:** The model sees manually aggregated daily features and cannot read tick-level order directly. The 64-cell M3 cost matrix also shows that ranking correlation has not yet produced stable active returns that cover 10 bp one-way costs.

Entry points: `scripts/run_minute_baseline.py` and `ticknet.nextday.minute_baseline`.

## Minute TCN

TCN is a temporal convolutional network and a deep-learning model.

**How it works:** Apply one-dimensional convolutions along time. Causal convolutions ensure that each step uses only data available at that time. Dilated convolutions let deeper layers see a longer history with fewer layers. Residual connections add each layer's new information to its input. The final time-step representation feeds classification and continuous-score heads.

**Why use it:** It computes multiple time positions in parallel and is often faster to train than a recurrent model. Different dilation rates can capture short changes and longer intraday patterns.

**Limitations:** Layer count, kernel size, and dilation predefine its receptive field. In the current three-seed experiment, the validation advantage did not carry through to test, where formal results were weaker than HGB.

Code: `ticknet.nextday.minute_tcn`. Training entry point: `ticknet-minute-tcn-train`.

## Minute GRU

GRU is a gated recurrent unit and a deep-learning model.

**How it works:** Read features in minute order while maintaining a hidden state. The update gate controls how much history to retain, and the reset gate controls how the current input combines with history. The final hidden state feeds classification and continuous-score heads.

**Why use it:** Its structure is straightforward and naturally represents ordered sequences. It is a lower-cost recurrent comparison for the TCN.

**Limitations:** Time steps are usually computed sequentially, so parallelism is lower than in TCNs and Transformers. Long histories must be compressed into a fixed-size hidden state. The repository has training, recovery, and locked-test workflows, but no separate formal experiment shows that it outperforms HGB or TCN.

Code: `ticknet.nextday.minute_gru`. Training entry point: `ticknet-minute-gru-train`.

## Chunked DeepLOB

Chunked DeepLOB is the project's raw-order-book next-day model.

**How it works:** Split the last 200 or 1,000 ten-level order-book snapshots into chunks of 100 events. Convolution layers extract price-level and local temporal patterns from each chunk. A multi-branch Inception module observes several time scales, then an LSTM compresses each chunk into one vector. A daily GRU summarizes the vectors in order. A classification head predicts three directions, while a score head produces a continuous value for cross-sectional ranking.

**Why use it:** It reads the ten-level sequence directly and retains local structures that minute aggregation discards. Chunking limits individual sequence length and reuses the same DeepLOB encoder.

**Limitations:** Data preparation and GPU training cost substantially more than for minute models. The controlled Top-100 four-cell, three-seed matrix found `1M/raw-200` most stable. Neither 100M parameters nor raw-1000 produced stable gains, so further expansion is paused.

Code: `ticknet.nextday.model`. See the [raw-order-book end-to-end pipeline](nextday/raw-200-end-to-end-pipeline.md).

## L2 Event-Stream Transformer

The event-stream model is a causal Transformer with rotary position embeddings. Presets include smoke, 25M, 50M, 100M, and 150M parameters. The `capacity100m` preset has 100,604,180 parameters.

**How it works:** Merge order, trade, and snapshot events in their true event order. Encode each event using numerical features, stream type, and order type. Self-attention lets each position inspect earlier visible events; the causal mask blocks future information, while rotary embeddings encode relative event position. The model jointly learns the next event type, next order type, next event values, and daily return signal. The shared trunk compresses fine-grained market behavior into a reusable representation.

**Why use it:** It can use long-range dependencies and connect different event types. Multi-task training uses abundant event-level supervision, and hidden states can be exported as frozen embeddings for downstream ranking models.

**Limitations:** Training requires a remote GPU, and storage and input throughput are expensive. The recent-fold 100M H5 runs had mean Rank IC of 0.07259 on validation and 0.04300 OOS across three seeds, all with positive direction. After adding frozen representations to HGB, mean OOS Rank IC rose from the minute baseline's 0.04010 to 0.05701, while validation rose from 0.01808 to 0.02833. Joint-training three-seed validation Rank IC was `0.05917 ± 0.01400`, and OOS Rank IC was `0.06398 ± 0.00785`. OOS `Precision@100` was `0.23921 ± 0.01611`, while mean daily Top-100 active return after costs was `-9.67 ± 3.71bp`. Current evidence covers only one validation month and one OOS month.

Frozen representations use the 960-dimensional hidden state of each stock's last valid event in the closing window. Three checkpoints use the same closing-window cache but retain separate coordinate spaces. HGB and LambdaMART are trained separately for each checkpoint; metrics are summarized or prediction scores averaged afterward. This avoids directly mixing embedding dimensions whose meaning may differ across seeds.

Code: `ticknet.eventstream.model`. See the [event-stream guide](nextday/eventstream.md).

The trunk also has default-off, M3-inspired representation options for order-book prefixes, fixed session anchors, and Hybrid VQ. They are representation experiments, not completed real rolling-window training, and do not change the formal performance conclusions here. See the [M3-inspired event-stream representation study](research/m3-eventstream-representation.md) for its design and contract.

Joint end-to-end experiments reuse the same 100M trunk and closing-window cache. Each sample combines the last 512 events before close with 120-dimensional minute aggregates. The model takes the last valid event's hidden state, concatenates the minute-feature tower output, and uses a three-class head to produce the up-probability minus down-probability ranking score. Training first freezes the trunk so the added feature tower and classifier can adapt, then jointly updates all parameters at a lower learning rate. Entry points: `ticknet-eventstream-joint-cache` and `ticknet-eventstream-joint-train`.

For formal seeds 0, 1, and 2, the best epochs were 2, 1, and 1. Later epochs declined quickly and triggered early stopping. Joint-model OOS `NDCG@100` was `0.54507 ± 0.00450`, close to the frozen-HGB three-seed mean. Mean daily one-way turnover was `49.74% ± 5.79%`, and all three seeds had negative active returns after costs. Further work must track Rank IC, precision, and trading costs together; Rank IC alone must not determine capacity expansion.

## FI-2010 DeepLOB

This is a compatibility implementation of the DeepLOB paper.

**How it works:** Three convolution blocks extract patterns from ten-level prices, volumes, and local time. Multi-branch Inception combines several time scales, a 64-unit LSTM summarizes the sequence, and a linear layer outputs three directional classes.

It is useful for checking the paper's architecture, FI-2010 conversion, and training process. Its market, labels, and task differ from current A-share next-day prediction, so its results apply only to reproduction. The implementation and training scripts are under `legacy/`. The main quality gate retains smoke checks for forward pass, gradients, and parameter count.

## LambdaMART

LambdaMART is a gradient-boosted ranking model. This repository uses LightGBM's `LGBMRanker` for the M4 comparison.

**How it works:** Group samples by trading day and compare stock orderings within each day. Training focuses on inversions that affect ranking metrics such as NDCG. The final score sums the outputs of multiple trees.

**Why use it:** Its objective is closer to daily Top-K selection while retaining the strengths of tree models on tabular data, nonlinear relationships, and CPU training.

**Limitations:** Date grouping must be exact, and features must not leak future information. A better-aligned ranking objective cannot add information that is absent from the input. The current implementation groups each trading day, maps future returns to five non-negative relevance levels, and shares stock-days and evaluation rules with HGB. In the recent fold, the three-seed mean for embedding-only predictions had OOS Rank IC 0.03414 and daily Top-100 active return after costs of 15.53 bp. Validation active return was -15.34 bp. The combined-input results also disagreed in direction between validation and OOS, so LambdaMART has not been promoted to the main candidate.

Entry point: `ticknet-embedding-compare`.

## Should the project add more Transformers?

The project already has a complete Transformer implementation and has completed the following recent-fold work:

1. Three `capacity100m` runs passed the recent-fold signal gate.
2. The shared closing-window cache and three checkpoints' 960-dimensional embeddings were generated and verified file by file.
3. HGB and LambdaMART comparisons covered minute features, embeddings, and their combination. HGB with combined inputs produced a consistent Rank IC gain across seeds, but active returns after costs remained negative.
4. Three joint end-to-end seeds completed with OOS Rank IC `0.06398 ± 0.00785`. Top-group precision and active returns after costs did not improve at the same time.
5. Seed 0 completed on an adjacent rolling fold with positive H5 validation and OOS Rank IC. The first checks of signal decay, trading conversion, and available-data risk exposures are complete.
6. Daily cross-sectional z-score labels improved validation and OOS Rank IC on both the recent and adjacent folds. The recent fold's OOS extreme-group return spread did not improve.
7. Recent-fold validation Rank IC for `last` and `tail_weighted` supervision was 0.07802 and 0.11289, below 0.11747 for `all`. Keep all-position supervision.
8. Next, test daily task weights before deciding whether to implement a cost-aware ranking objective.
9. Defer `probe150m` until training changes show gains across windows.

There is no current need to add Hugging Face `transformers`. The existing model already uses causal attention, rotary position embeddings, and PyTorch efficient-attention APIs. The identified bottlenecks are data reading, remote storage, and training automation. Reconsider the dependency if the project needs public pretrained models, a standard model-publishing format, or a mature Trainer ecosystem.

Do not add a separate Transformer to the minute pipeline yet. Joint end-to-end experiments already reuse the event-stream Transformer and feed minute features and daily hidden representations to one prediction head. All three seeds had positive OOS Rank IC but negative active returns after costs. The TCN validation gain did not generalize consistently, and M3, frozen E2, and joint training have not produced stable after-cost active returns across months. Keep the existing all-position supervision. Current focus is daily task weighting and training objectives, not model capacity.
