# M3-Inspired Event-Stream Representation Experiments

This page records representation changes inspired by the M3 market-microstructure generative model. The target remains next-day cross-sectional ranking. The project is not being turned into a closed-loop market simulator, and untrained architecture changes are not described as performance improvements.

## Motivation

The current event-stream model samples fixed-length windows from arbitrary points during the day. A window can begin with an order or trade, so the model may not see the complete ten-level book at its boundary. Event prices also use a rolling-midpoint local coordinate, which is useful for prediction but does not provide a separate fixed intraday coordinate. M3's book prefix and fixed price anchors address these two representation questions.

VQ tests a third, independent hypothesis. The current model predicts event type, order type, price, time interval, and quantity separately. Hybrid VQ compresses core behavior fields into discrete codes and adds them as a residual to continuous event embeddings. This may preserve fine continuous information while representing recurring order behaviors.

## Three optional mechanisms

### LOB prefix

With `use_lob_prefix: true`, the first input position becomes a special book-state token. It is constructed from the last snapshot before the window start; it never reads a snapshot after the boundary. Real events follow in their original order, and the public tensor shape remains unchanged.

The prefix uses `stream_id=0` and the reserved `order_type_id=11`, distinguishing it from ordinary padding. Ten-level prices and sizes, order count, spread, imbalance, weighted bid/ask, and other fields reuse the existing 80-dimensional layout.

### Fixed session anchor

`use_session_anchors: true` depends on the LOB prefix. It does not replace rolling-midpoint normalization for current events. Instead, it adds a fixed intraday coordinate to the prefix.

The anchor uses only a valid trade price or snapshot last price observed before the window boundary, selecting the earliest valid price of the day. Trades take precedence when timestamps tie. Once this first valid price has appeared, later windows share the same anchor. Previously, the only numeric fallback was prior close, with anchor availability marked as 0.

The two coordinates retain different information: rolling midpoint says how far an order is from the current book; the session anchor says how far the current book has moved from a fixed early-day reference.

### Hybrid VQ

With `use_vq: true`, the model maps five core behavior fields—`dt_log`, `price_bps`, `qty_log`, `side`, and `is_cancel`—to a low-dimensional vector and assigns it to the nearest learnable codebook entry. The quantized vector is projected to `d_model` and added as a residual to the original continuous event embedding.

Padding and LOB-prefix tokens have `stream_id=0` and do not participate in VQ regularization. Training adds codebook and commitment losses, weighted by `vq_loss_weight`. When VQ is disabled, no VQ parameters are created, so existing state dictionaries and parameter counts remain unchanged.

## Configuration

Example configuration:

```yaml
use_lob_prefix: false
use_session_anchors: false
use_vq: false
vq_codebook_size: 1024
vq_dim: 64
vq_loss_weight: 0.25
```

All options default to disabled or fixed defaults. Test these mechanisms as controlled experiments rather than enabling all of them and judging from a single Rank IC value.

## Experiment order

First compare LOB prefix alone. Next add session anchors on top of the prefix. Then test Hybrid VQ independently and combine mechanisms only if warranted. Each experiment follows the existing time-based validation, adjacent rolling folds, cost-adjusted portfolio metrics, and locked-period rules.

Do not increase model size because of M3 scaling laws alone. TickNet's current bottleneck is converting ranking signals into cost-adjusted returns; there is no evidence that more parameters solve it.

## Compatibility and data contracts

Prefix and session anchors change the actual content of fixed windows, so they are recorded in materialized manifests and close-cache contracts. Old caches without these fields are interpreted as `false` and continue using the original v1 sampling contract. Prefix caches use v2; consumers reject checkpoints or caches whose representation identities differ.

VQ changes model structure but not materialized arrays. It therefore does not enter the tensor schema but is part of checkpoint experiment identity. Frozen-embedding export, materialized prediction, and gradient audit rebuild the model from VQ parameters stored in the checkpoint.

The existing joint fine-tuning cache chain has not been extended for M3-inspired representations. Joint fine-tuning explicitly rejects pretrained checkpoints with prefix, session anchor, or VQ so an old cache contract cannot silently change meaning.

## Out of scope

This work does not implement a matching engine, autoregressive AI order rollout, or large-order market-impact simulation. The current prediction pack discards raw OrderID, DealID, BuyID, and SellID after deriving fields such as cancel age. Strict matching requires order identity and a time-priority queue. A full market simulator needs a separate data contract, replay-consistency tests, and matching engine.

## Evidence status

So far, evidence consists only of synthetic-data unit tests, compatibility tests, and repository CI. No real A-share rolling-window training has run. Do not claim gains in Rank IC, NDCG, cost-adjusted returns, or live metrics until experiments complete under the existing research protocol.
