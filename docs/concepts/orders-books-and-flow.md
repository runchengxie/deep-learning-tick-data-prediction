# From order events to order books and order flow

An **order message** records an instruction or an order update; a **trade message** records an execution; an **order book** is the remaining visible state; **order flow** describes events over time. These are connected views of a market, not four interchangeable datasets.

Use the [interactive concept explorer](https://runchengxie.github.io/quant-deep-learning/concepts/order-book/) for synthetic replay and charts. The examples establish bookkeeping behavior, not actual exchange feed coverage or a profitable signal.

## Start with events, process and state

```text
Order additions + executions + cancellations
                     |
               ordered event flow
                     |
        remaining quantity for each order (MBO)
                     |
          aggregate quantity by price (MBP)
```

| Concept | What it records | What it cannot establish by itself |
|---|---|---|
| Tick-by-tick order messages | Published additions or other order events, as defined by the feed | Every original instruction, the final investor, or a hidden parent order |
| Tick-by-tick executions | Executed price and quantity, sometimes linked order IDs | Unexecuted additions and withdrawals |
| Order-book snapshot | Visible resting quantities at the snapshot boundary | The unique path between snapshots or all order identities |
| Order flow | An event sequence or an explicitly defined derived measurement | One universal vendor-independent metric |

“Tick-by-tick” describes record granularity. “Order” and “trade” describe event meaning. A message carried in an execution stream can be a cancellation rather than a trade. Every trade has both a buyer and a seller: buyer-initiated versus seller-initiated identifies the aggressive side, not the number of buyers versus sellers. An aggressive order can be a marketable limit order, not only a market order.

## A visible order lifecycle

All quantities below are shares. Prices are synthetic; the explorer represents prices in integer cents. Events are already normalized and ordered, and there are no missing messages or hidden orders.

| Step | Normalized event at ask 10.01 | S1 remaining | S2 remaining | S3 remaining | MBP ask quantity | Cumulative buyer-initiated volume |
|---|---|---:|---:|---:|---:|---:|
| 0 | Empty book | 0 | 0 | 0 | 0 | 0 |
| 1 | Add S1, 500 | 500 | 0 | 0 | 500 | 0 |
| 2 | Add S2, 500 | 500 | 500 | 0 | 1,000 | 0 |
| 3 | Execute 400 against S1 | 100 | 500 | 0 | 600 | 400 |
| 4 | Cancel all of S2 | 100 | 0 | 0 | 100 | 400 |
| 5 | Add S3, 200 | 100 | 0 | 200 | 300 | 400 |

S1's partial execution preserves its identity. S2's cancellation removes liquidity without executing it. The cancellation example removes an entire remaining order; it does not assert that arbitrary partial cancellation is permitted by a venue. The diagram is bookkeeping, not a full matching-engine emulator: trade events already specify the resting order consumed.

**MBP (Market by Price)** aggregates resting quantity at each side and price. **MBO (Market by Order)** retains individual visible orders. A ten-level snapshot is bounded MBP coverage; a queue of the first few orders at the best price is bounded order detail. Neither guarantees a complete MBO book, full depth or complete queue history. Order counts are not order identities.

## The same final book can hide different flow

Begin with 10,000 bid shares and 10,000 ask shares at unchanged prices.

- Path A: add 5,000 bid shares, then cancel those 5,000. Final bid quantity: 10,000. Trade Delta: 0.
- Path B: seller-initiated executions consume 3,000 bid shares, then a new bid replenishes 3,000. Final bid quantity: 10,000. Trade Delta: −3,000.

Both final snapshots match. Their execution histories do not. The two-path chart exposes the intermediate states. A model given only the final snapshot cannot uniquely recover the event path.

In a second example, add 8,000 bid shares, cancel them, then add 3,000. Final bid depth is 13,000 while Trade Delta remains zero. This is why a book measurement and a trade-flow measurement can disagree without either being wrong.

## Broad and narrow order flow

Broad usage includes additions, cancellations and executions: who joins the queue, who leaves, and who completes a transaction. Narrow usage usually means signed executed volume: completed transactions and the initiating side. An order-flow display based only on trades cannot reveal unexecuted withdrawals. Check the vendor's inputs and formula rather than relying on the label.

## Measurements: OBI, OFI, Delta and CVD

State the level scope, units, window and reset policy before comparing values.

| Measurement | Definition / scope | Interpretation boundary |
|---|---|---|
| OBI | `(Q_bid − Q_ask) / (Q_bid + Q_ask)` at a specified level scope | Standing visible depth; undefined when both sides are empty. Combining levels needs an explicit weighting rule. |
| Trade Delta | Buyer-initiated execution volume minus seller-initiated execution volume in a window | Only executions; unknown aggressors must remain unknown rather than being assigned a direction silently. |
| CVD | Cumulative Trade Delta from an explicit reset point | Depends on reset point, included trades and signing method. It is not itself an exchange book. |
| Best-level OFI | Sum of signed changes in the best bid/ask price and queue size | Quote-price changes matter. A naive sum of all added/cancelled shares across every level is not this definition. |

In Cont, Kukanov and Stoikov's best-level formulation, the contribution between observations n−1 and n is:

```text
e_n = 1[bid_n >= bid_(n-1)] * bid_qty_n
    - 1[bid_n <= bid_(n-1)] * bid_qty_(n-1)
    - 1[ask_n <= ask_(n-1)] * ask_qty_n
    + 1[ask_n >= ask_(n-1)] * ask_qty_(n-1)
OFI(window) = sum(e_n)
```

The indicator is 1 when its condition holds. At unchanged best prices this simplifies to bid-quantity change minus ask-quantity change. With changing prices, those indicators prevent an incorrect interpretation of queue replacement. The paper studies price impact in its own sample; an explanatory contemporaneous relationship is not automatically a forecast or an A-share trading result. See [the original paper](https://arxiv.org/abs/1011.6402).

The explorer displays OBI and cumulative Trade Delta. It deliberately does not label general book-quantity changes as universal OFI. Its single-price-per-side examples cannot teach multi-level queue transitions, hidden liquidity or ambiguous aggressor classification.

## Shanghai and Shenzhen: respect the source contract

These are **version-scoped examples**, not a promise about every contemporary or historical feed. Verify the actual date, instrument, trading phase, product version and vendor mapping used by a dataset.

| Boundary | Shanghai example | Shenzhen example |
|---|---|---|
| Published event meaning | The cited LDDS 2.0.8 order-message specification publishes new orders after initial matching. Continuous-auction resting quantity therefore must not be treated as every original order's submitted quantity. Auction/suspension phases have separate rules. | The cited Binary interface distinguishes execution type `F` (trade) from `4` (cancellation) within tick-by-tick execution messages. Filter by semantic event type before computing volume. |
| Linking and order | The cited LDDS order ID links to execution-side IDs; `BizIndex` order is scoped by channel. | Check order references and the documented sequence scope before merging order and execution streams. |
| Historical availability | SSE Info records order-message launch in May 2021. Historical interface 1.1.2 states that combined `StockTick.csv` starts on 27 November 2023. | Older SZSE technical specifications already describe tick order/execution messages; do not assume recent vendor access marks the first exchange availability. |

Sources: [SSE Info product history](https://www.sseinfo.com/aboutus/events/), [Shanghai LDDS 2.0.8](https://www.ciis.com.hk/hongkong/tc/uploadfiles/202308/31/2023083114092093975509.pdf), [Shanghai historical interface 1.1.2](https://bsp.sseinfo.com/admin/static/public/2024-06-07/0017f8fa1c1a47cb997d433c6d05fc11/%E4%B8%8A%E6%B5%B7%E8%AF%81%E5%88%B8%E4%BA%A4%E6%98%93%E6%95%B0%E6%8D%AE%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E%E4%B9%A6.pdf), [SZSE Binary market-data specification](https://www.szse.cn/marketServices/technicalservice/interface/P020230904554302750296.pdf). Sources checked on 2026-10-10. Some official PDF endpoints are intermittently unavailable; the definitions here are tied to the linked versions, not asserted as the latest interface.

The launch date, a combined-file format start date, a vendor archive's earliest date and a particular user's access date are different facts. A unified file is an event container, not a newly complete MBO dataset.

## Reconstruction is a contract, not just a sort

1. Establish an initial state and the instrument/session boundary.
2. Preserve venue, channel, exchange sequence and order references. Do not invent a global sequence across independent channels.
3. Separate source event time, publication/availability time and local receive time. A timestamp tie is not proof of a known event order.
4. Apply phase-specific addition, execution, cancellation and instrument-status semantics. Handle duplicates according to source identity, not arbitrary row removal.
5. Reject unexplained over-consumption and missing references; report gaps and incomplete coverage. Synthetic replay fails closed on unknown references and negative remaining quantity.
6. Compare reconstructed MBP with properly aligned independent snapshots. Record completeness and timing uncertainty, not just a match percentage.

The project's [Shanghai opening contract audit](../research/shanghai-opening-contract-audit-2026-08-27.md) shows why timing boundaries matter. It is a dated sample, not proof of full-market completeness. Do not generalize a sample's best timestamp offset to every venue and trading date.

## Splitting is not partial execution

A parent trading requirement can be split by an execution algorithm into independent child orders. A single child order can then execute in several pieces. Aggregating several orders at a price is a third operation. Similar sizes and timestamps do not prove a common parent or investor: parent-order detection is an inference with uncertainty, not direct observation from public order IDs. A vendor indicator called “split orders” may use a different definition.

## Connect the concepts to model inputs

| Representation | Information available | Information compressed or absent |
|---|---|---|
| Raw book snapshots | Sampled depth and price states | Unobserved paths, events between samples, usually complete order identities |
| Minute or other aggregates | Chosen price/volume/flow summaries | Within-window order and queue detail |
| Event sequences | Available ordered messages and causal context | Hidden events, missing source coverage, events outside context |

Compare the [raw-book path](../nextday/raw-200-end-to-end-pipeline.md) and [event-stream path](../nextday/eventstream.md) using the same information cutoff, target, out-of-time split and cost assumptions. More detailed inputs may increase compute or sensitivity to data defects without improving prediction. FI-2010 reproduction is a separate research track and does not establish that these A-share representations are profitable.
