from ticknet.nextday.minute_materialization import _peak_rss_mb


def test_materialization_memory_receipt_is_available_on_this_platform():
    assert _peak_rss_mb() > 0
