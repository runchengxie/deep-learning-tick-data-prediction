from ticknet.simulator.ordering import (
    detect_ordering_columns,
    ordering_provenance,
    sort_simulator_events,
)
from ticknet.simulator.pack import SimulatorEvent, build_simulator_pack


def test_detects_channel_and_sequence_aliases() -> None:
    assert detect_ordering_columns(["ticker", "time_ms", "ChannelNo", "ApplSeqNum"]) == {
        "channel_column": "ChannelNo",
        "sequence_column": "ApplSeqNum",
        "exchange_sequence_available": True,
    }


def test_same_timestamp_single_channel_uses_exchange_sequence() -> None:
    events = [
        SimulatorEvent(100, "order", order_id="later", channel="7", sequence=12, source_index=0),
        SimulatorEvent(100, "order", order_id="earlier", channel="7", sequence=11, source_index=1),
        SimulatorEvent(100, "snapshot", source_index=2),
    ]

    ordered = sort_simulator_events(events)

    assert [event.order_id for event in ordered[:2]] == ["earlier", "later"]
    assert ordered[-1].kind == "snapshot"


def test_multiple_channels_do_not_invent_cross_channel_sequence_order() -> None:
    events = [
        SimulatorEvent(100, "order", order_id="a", channel="1", sequence=20, source_index=0),
        SimulatorEvent(100, "order", order_id="b", channel="2", sequence=1, source_index=1),
    ]

    ordered = sort_simulator_events(events)
    provenance = ordering_provenance(
        events,
        channel_column="ChannelNo",
        sequence_column="ApplSeqNum",
    )

    assert [event.order_id for event in ordered] == ["a", "b"]
    assert provenance["ordering_mode"] == "timestamp_then_source_order_cross_channel"
    assert provenance["cross_channel_total_order"] is False


def test_missing_sequence_uses_source_order_fallback() -> None:
    events = [
        SimulatorEvent(100, "cancel", order_id="c", source_index=0),
        SimulatorEvent(100, "order", order_id="o", source_index=1),
    ]

    assert [event.order_id for event in sort_simulator_events(events)] == ["c", "o"]
    assert ordering_provenance(events)["ordering_mode"] == "timestamp_fallback"


def test_build_pack_preserves_optional_sequence_metadata() -> None:
    pack = build_simulator_pack(
        [
            {
                "time_ms": 100,
                "order_id": "later",
                "side": 1,
                "price": 100,
                "volume": 10,
                "channel": "7",
                "sequence": 12,
            },
            {
                "time_ms": 100,
                "order_id": "earlier",
                "side": 1,
                "price": 100,
                "volume": 10,
                "channel": "7",
                "sequence": 11,
            },
        ],
        [],
        [],
    )

    assert [event.order_id for event in pack.events] == ["earlier", "later"]
    assert pack.ordering_provenance["ordering_mode"] == "timestamp_then_channel_sequence"
