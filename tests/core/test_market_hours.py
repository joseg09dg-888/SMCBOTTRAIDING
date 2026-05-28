"""Tests for market_hours — is_market_open() per symbol."""
from datetime import datetime, timezone
import pytest
from core.market_hours import is_market_open, minutes_until_open


def utc(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# Crypto always open
class TestCrypto:
    def test_btc_open_weekend(self):
        # Saturday
        assert is_market_open("BTCUSDT", utc(2026, 5, 30, 14)) is True

    def test_eth_open_midnight(self):
        assert is_market_open("ETHUSDT", utc(2026, 5, 28, 0)) is True

    def test_sol_open_anytime(self):
        assert is_market_open("SOLUSDT", utc(2026, 5, 28, 3)) is True


# US30 — NYSE only 13:30-20:00 UTC Mon-Fri
class TestUS30:
    def test_open_during_nyse(self):
        # Wednesday 15:00 UTC
        assert is_market_open("US30", utc(2026, 5, 27, 15)) is True

    def test_closed_before_nyse(self):
        # Wednesday 10:00 UTC
        assert is_market_open("US30", utc(2026, 5, 27, 10)) is False

    def test_closed_after_nyse(self):
        # Wednesday 21:00 UTC
        assert is_market_open("US30", utc(2026, 5, 27, 21)) is False

    def test_closed_weekend(self):
        # Saturday
        assert is_market_open("US30", utc(2026, 5, 30, 15)) is False

    def test_closed_friday_after_close(self):
        # Friday 21:00 UTC
        assert is_market_open("US30", utc(2026, 5, 29, 21)) is False

    def test_open_at_exact_open(self):
        assert is_market_open("US30", utc(2026, 5, 27, 13, 30)) is True

    def test_closed_one_minute_before_open(self):
        assert is_market_open("US30", utc(2026, 5, 27, 13, 29)) is False


# NAS100 — same as US30
class TestNAS100:
    def test_open_during_nyse(self):
        assert is_market_open("NAS100", utc(2026, 5, 27, 16)) is True

    def test_closed_at_night(self):
        assert is_market_open("NAS100", utc(2026, 5, 27, 23)) is False


# Forex — Mon-Fri continuous
class TestForex:
    def test_eurusd_open_monday(self):
        assert is_market_open("EURUSD", utc(2026, 5, 25, 8)) is True

    def test_eurusd_open_friday_morning(self):
        assert is_market_open("EURUSD", utc(2026, 5, 29, 10)) is True

    def test_eurusd_closed_friday_evening(self):
        assert is_market_open("EURUSD", utc(2026, 5, 29, 21)) is False

    def test_eurusd_closed_weekend(self):
        assert is_market_open("EURUSD", utc(2026, 5, 30, 12)) is False


# Gold
class TestXAUUSD:
    def test_open_during_day(self):
        assert is_market_open("XAUUSD", utc(2026, 5, 28, 14)) is True

    def test_closed_during_daily_break(self):
        assert is_market_open("XAUUSD", utc(2026, 5, 28, 21)) is False

    def test_open_after_break(self):
        assert is_market_open("XAUUSD", utc(2026, 5, 28, 22)) is True


# minutes_until_open
class TestMinutesUntilOpen:
    def test_zero_when_open(self):
        assert minutes_until_open("BTCUSDT", utc(2026, 5, 28, 10)) == 0

    def test_positive_when_us30_closed(self):
        # 10:00 UTC — US30 opens at 13:30, so 210 min
        mins = minutes_until_open("US30", utc(2026, 5, 28, 10, 0))
        assert mins == 210

    def test_gold_break_returns_nonzero(self):
        mins = minutes_until_open("XAUUSD", utc(2026, 5, 28, 21, 15))
        assert mins > 0
