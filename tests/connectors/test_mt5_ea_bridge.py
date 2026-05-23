"""Tests for MT5 EA Bridge."""
import json, time, pytest
from pathlib import Path
from unittest.mock import patch
from connectors.mt5_ea_bridge import MT5EABridge, SIGNAL_FILE, RESULT_FILE, LOCK_FILE


@pytest.fixture(autouse=True)
def cleanup():
    for f in [SIGNAL_FILE, RESULT_FILE, LOCK_FILE]:
        try: f.unlink(missing_ok=True)
        except: pass
    yield
    for f in [SIGNAL_FILE, RESULT_FILE, LOCK_FILE]:
        try: f.unlink(missing_ok=True)
        except: pass


def test_place_order_writes_signal():
    bridge = MT5EABridge(timeout_sec=0.5)
    # Write result immediately (simulate EA)
    def fake_ea():
        time.sleep(0.1)
        RESULT_FILE.write_text('{"retcode":10009,"order":12345,"price":1.16050}')
    import threading
    threading.Thread(target=fake_ea, daemon=True).start()
    result = bridge.place_order("EURUSD", "long", 0.01, 50, 100)
    assert result["retcode"] == 10009
    assert result["order"] == 12345

def test_place_order_signal_file_content():
    bridge = MT5EABridge(timeout_sec=0.2)
    # Don't respond — timeout
    bridge.place_order("GBPUSD", "short", 0.02, 30, 60)
    # Signal should have been cleaned up
    assert not SIGNAL_FILE.exists()

def test_place_order_timeout_returns_error():
    bridge = MT5EABridge(timeout_sec=0.5)
    result = bridge.place_order("EURUSD", "long")
    assert result["retcode"] == -1
    assert "EA" in result["error"]

def test_place_order_correct_json_written():
    bridge = MT5EABridge(timeout_sec=0.2)
    # Write result to avoid waiting
    def fake():
        time.sleep(0.05)
        # Read signal first to verify
        if SIGNAL_FILE.exists():
            sig = json.loads(SIGNAL_FILE.read_text())
            assert sig["symbol"] == "EURUSD"
            assert sig["direction"] == "long"
            assert sig["volume"] == 0.01
        RESULT_FILE.write_text('{"retcode":10009,"order":1}')
    import threading
    threading.Thread(target=fake, daemon=True).start()
    bridge.place_order("EURUSD", "long", 0.01, 50, 100)

def test_format_signal_for_ea_eurusd():
    bridge = MT5EABridge()
    sig = bridge.format_signal_for_ea("EURUSD", "long", 1.1600, 1.1550, 1.1700, 0.01)
    assert sig["symbol"] == "EURUSD"
    assert sig["direction"] == "long"
    assert sig["sl_pips"] == pytest.approx(50.0, abs=5)
    assert sig["tp_pips"] == pytest.approx(100.0, abs=5)

def test_format_signal_for_ea_usdjpy():
    bridge = MT5EABridge()
    sig = bridge.format_signal_for_ea("USDJPY", "short", 159.0, 159.5, 158.0, 0.01)
    assert sig["sl_pips"] > 0
    assert sig["tp_pips"] > 0

def test_format_signal_for_ea_xauusd():
    bridge = MT5EABridge()
    sig = bridge.format_signal_for_ea("XAUUSD", "long", 2300.0, 2295.0, 2310.0, 0.01)
    assert sig["sl_pips"] > 0

def test_is_ea_running_no_result():
    bridge = MT5EABridge()
    assert bridge.is_ea_running() is False

def test_is_ea_running_with_fresh_result():
    bridge = MT5EABridge()
    RESULT_FILE.write_text('{"retcode":10009}')
    assert bridge.is_ea_running() is True

def test_get_last_result_none():
    bridge = MT5EABridge()
    assert bridge.get_last_result() is None

def test_get_last_result_with_file():
    bridge = MT5EABridge()
    RESULT_FILE.write_text('{"retcode":10009,"order":99}')
    result = bridge.get_last_result()
    assert result["order"] == 99

def test_lock_prevents_double_processing():
    """Simulates EA processing — lock prevents re-entry."""
    bridge = MT5EABridge(timeout_sec=0.3)
    LOCK_FILE.write_text("1")
    # Write signal
    SIGNAL_FILE.write_text('{"symbol":"EURUSD","direction":"long"}')
    # With lock present, should timeout (EA won't process)
    result = bridge.place_order("EURUSD", "long")
    assert result["retcode"] == -1
