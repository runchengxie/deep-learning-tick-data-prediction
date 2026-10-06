# Market Simulator Subsystem Design (Phase 2)

Date: 2026-08-26
Status: Draft (implementation in progress)
Related work: Phase 1, `agent/m3-eventstream-representation` (PR #96, merged)

## Goal

Add an independent `src/ticknet/simulator/` subsystem to the former `deep-learning-tick-data-prediction` project. It will implement:

1. A deterministic matching engine that consumes order flows with original `OrderID` values and maintains a ten-level limit order book.
2. A generative market simulator that uses a real initial book as a prefix, generates background order flows with the event-stream Transformer, inserts external intervention orders such as TWAP child orders, and replays them through the matching engine to produce synthetic market trajectories.
3. An interface for estimating market impact by placing candidate execution orders into the simulator and estimating realized impact and slippage.

## Out of Scope for This Phase

- Changing the primary `eventstream` prediction path (Phase 1 completed LOB prefixes, dual anchors, and VQ).
- Modeling cross-asset or cross-market correlations.
- Scaling the model to 150M or 1B parameters; scaling experiments are scheduled separately.
- Directly replacing next-day alpha signals. Initially, the simulator is a standalone cost-estimation and sandbox tool.

## Why a New Data Contract Is Needed

`eventstream/config.py` documents that raw `OrderID`, `DealID`, `BuyID`, and `SellID` values are discarded, while their relationships are distilled into derived features such as cancel age. This is a reasonable trade-off for prediction. A matching engine, however, must identify the exact resting order associated with a cancel, including its time-priority queue position. That cannot be reconstructed without the original identifiers.

The simulator must therefore read or rebuild a simulator pack that preserves IDs. This pack remains separate from the prediction event-stream pack so the two contracts do not affect each other.

## Module Boundaries (Following AGENTS.md)

- The new `src/ticknet/simulator/` module owns the matching engine, generative replay, and impact estimation.
- Do not directly import `nextday` implementations. Use CLI/configuration to decouple from `eventstream`, while allowing reuse of its tokenizer and model loader.
- Tests use synthetic data and do not require the full raw L2 dataset.

## Implementation Order

1. `simulator/pack.py`: parse raw L2 while preserving `OrderID` values into a simulator pack (write a failing test first).
2. `simulator/matching.py`: build a matching engine that processes order and cancel events and maintains the ten-level book.
3. `simulator/engine_correctness_test`: replay a real initial book and real order stream and require the reconstructed book to match the real snapshot (correctness gate).
4. `simulator/generator.py`: load the event-stream Transformer and generate background order tokens.
5. `simulator/replay.py`: run closed-loop replay (prefix + background flow + external intervention → trajectory).
6. `simulator/impact.py`: add the market-impact estimation interface.
7. Add documentation and CLI entry points.

## Acceptance Criteria

- Matching results for synthetic sequences agree with hand calculations.
- The engine correctness test reconstructs the real snapshot within tolerance.
- The impact curve is observable on log-log axes. Values are for research use only and do not establish live-trading causality.
