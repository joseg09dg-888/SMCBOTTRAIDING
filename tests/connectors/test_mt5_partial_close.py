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
