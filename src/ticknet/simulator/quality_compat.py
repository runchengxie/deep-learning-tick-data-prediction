"""Compatibility bridge from deep-learning to the shared data platform profiler."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def profile_parquet_compat(
    path: str | Path,
    *,
    batch_size: int = 262_144,
    max_tracked_ids: int = 1_000_000,
) -> dict[str, Any]:
    """Use the platform profiler when installed, otherwise use the local implementation."""
    try:
        from market_data_platform.quality import profile_parquet
    except ModuleNotFoundError as error:
        if error.name != "market_data_platform":
            raise
        from ticknet.simulator.quality import profile_parquet

    return profile_parquet(
        path,
        batch_size=batch_size,
        max_tracked_ids=max_tracked_ids,
    )


def audit_opening_ledger_compat(
    orders: Sequence[Any],
    trades: Sequence[Any],
    cancels: Sequence[Any],
    *,
    expected_bid_levels: Sequence[tuple[int, int]] | None,
    expected_ask_levels: Sequence[tuple[int, int]] | None,
    depth: int = 10,
) -> Any:
    """Use the platform accounting core while preserving the local result type."""
    try:
        from market_data_platform.quality_opening import audit_opening_ledger
    except ModuleNotFoundError as error:
        if error.name != "market_data_platform":
            raise
        from ticknet.simulator.opening_ledger import _audit_opening_ledger_local

        return _audit_opening_ledger_local(
            orders,
            trades,
            cancels,
            expected_bid_levels=expected_bid_levels,
            expected_ask_levels=expected_ask_levels,
            depth=depth,
        )

    result = audit_opening_ledger(
        orders,
        trades,
        cancels,
        expected_bid_levels=expected_bid_levels,
        expected_ask_levels=expected_ask_levels,
        depth=depth,
    )
    from ticknet.simulator.opening_ledger import OpeningLedgerAudit

    return OpeningLedgerAudit(
        status=result.status,
        bid_levels=result.bid_levels,
        ask_levels=result.ask_levels,
        expected_bid_levels=result.expected_bid_levels,
        expected_ask_levels=result.expected_ask_levels,
        unknown_trade_count=result.unknown_trade_count,
        unknown_trade_volume=result.unknown_trade_volume,
        unknown_cancel_count=result.unknown_cancel_count,
        unknown_cancel_volume=result.unknown_cancel_volume,
        overdrawn_count=result.overdrawn_count,
        overdrawn_volume=result.overdrawn_volume,
    )
