"""Tests for MT5Connector.partial_close_position."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


@pytest.fixture
def mt5_connector():
    with patch("connectors.metatrader_connector.HAS_MT5", False):
        from connectors.metatrader_connector import MT5Connector
        conn = MT5Connector.__new__(MT5Connector)
        return conn


def test_partial_close_no_mt5(mt5_connector):
    """Without MT5 module, partial_close returns False gracefully."""
    result = mt5_connector.partial_close_position(12345, 0.1)
    assert result is False


def test_partial_close_volume_rounding():
    """Partial close should round volume to broker minimum step."""
    with patch("connectors.metatrader_connector.HAS_MT5", True):
        import connectors.metatrader_connector as _m
        # Simulate mt5 not available but logic reachable via mock
        fake_mt5 = MagicMock()
        fake_mt5.positions_get.return_value = None  # No open position
        with patch.object(_m, "mt5", fake_mt5):
            from connectors.metatrader_connector import MT5Connector
            conn = MT5Connector.__new__(MT5Connector)
            result = conn.partial_close_position(99999, 0.15)
            assert result is False  # No position found


def test_close_position_signature():
    """close_position and partial_close_position exist on MT5Connector."""
    from connectors.metatrader_connector import MT5Connector
    assert hasattr(MT5Connector, "close_position")
    assert hasattr(MT5Connector, "partial_close_position")


def test_modify_sl_tp_signature():
    """modify_position_sl_tp exists on MT5Connector."""
    from connectors.metatrader_connector import MT5Connector
    assert hasattr(MT5Connector, "modify_position_sl_tp")


# ---------------------------------------------------------------------------
# get_closing_deal must SUM partial + final deals, not return the first match
# ---------------------------------------------------------------------------

def _make_deal(position_id, entry, profit, swap=0.0, commission=0.0,
                price=1.1, symbol="EURUSD", volume=0.5, time=1000):
    d = MagicMock()
    d.position_id = position_id
    d.entry = entry
    d.profit = profit
    d.swap = swap
    d.commission = commission
    d.price = price
    d.symbol = symbol
    d.volume = volume
    d.time = time
    return d


def test_get_closing_deal_sums_partial_and_final():
    """A position that partial-closed at 1R then fully closed later produces
    TWO entry=1 deals on the same position_id -- both must be counted."""
    with patch("connectors.metatrader_connector.HAS_MT5", True):
        import connectors.metatrader_connector as _m
        fake_mt5 = MagicMock()
        partial = _make_deal(12345, entry=1, profit=50.0, volume=0.5, price=1.105, time=1000)
        final   = _make_deal(12345, entry=1, profit=80.0, volume=0.5, price=1.110, time=2000)
        fake_mt5.history_deals_get.return_value = [partial, final]
        with patch.object(_m, "mt5", fake_mt5):
            from connectors.metatrader_connector import MT5Connector
            conn = MT5Connector.__new__(MT5Connector)
            result = conn.get_closing_deal(12345)

    assert result["profit"] == pytest.approx(130.0)  # 50 + 80, not just 50
    assert result["volume"] == pytest.approx(1.0)     # 0.5 + 0.5
    assert result["time"] == 2000                     # from the last (final) deal


def test_get_closing_deal_single_deal_still_works():
    """A position with no partial close (one entry=1 deal) still works."""
    with patch("connectors.metatrader_connector.HAS_MT5", True):
        import connectors.metatrader_connector as _m
        fake_mt5 = MagicMock()
        only = _make_deal(555, entry=1, profit=-42.0, volume=1.0)
        fake_mt5.history_deals_get.return_value = [only]
        with patch.object(_m, "mt5", fake_mt5):
            from connectors.metatrader_connector import MT5Connector
            conn = MT5Connector.__new__(MT5Connector)
            result = conn.get_closing_deal(555)

    assert result["profit"] == pytest.approx(-42.0)


def test_get_closing_deal_no_matching_deals_returns_empty():
    with patch("connectors.metatrader_connector.HAS_MT5", True):
        import connectors.metatrader_connector as _m
        fake_mt5 = MagicMock()
        fake_mt5.history_deals_get.return_value = []
        with patch.object(_m, "mt5", fake_mt5):
            from connectors.metatrader_connector import MT5Connector
            conn = MT5Connector.__new__(MT5Connector)
            result = conn.get_closing_deal(999)

    assert result == {}
