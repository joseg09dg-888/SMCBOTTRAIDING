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


# US30 -- Axi CFD hours 01:00-22:00 UTC Mon-Fri (break 22:00-23:00)
class TestUS30:
    def test_open_during_day(self):
        # Wednesday 15:00 UTC -- core trading hours
        assert is_market_open("US30", utc(2026, 5, 27, 15)) is True

    def test_open_early_morning(self):
        # Wednesday 10:00 UTC -- was wrongly closed before, now open on Axi
        assert is_market_open("US30", utc(2026, 5, 27, 10)) is True

    def test_open_at_axi_open_hour(self):
        # Wednesday 01:00 UTC -- Axi opens
        assert is_market_open("US30", utc(2026, 5, 27, 1)) is True

    def test_closed_before_axi_open(self):
        # Wednesday 00:30 UTC -- before 01:00 open
        assert is_market_open("US30", utc(2026, 5, 27, 0, 30)) is False

    def test_closed_during_daily_break(self):
        # Wednesday 22:30 UTC -- daily maintenance break
        assert is_market_open("US30", utc(2026, 5, 27, 22)) is False

    def test_closed_after_daily_break(self):
        # Wednesday 23:00 UTC -- still in break before next open at 01:00
        assert is_market_open("US30", utc(2026, 5, 27, 23)) is False

    def test_closed_weekend(self):
        # Saturday
        assert is_market_open("US30", utc(2026, 5, 30, 15)) is False

    def test_closed_friday_after_close(self):
        # Friday 22:00 UTC -- end of week
        assert is_market_open("US30", utc(2026, 5, 29, 22)) is False

    def test_open_friday_afternoon(self):
        # Friday 20:00 UTC -- still open
        assert is_market_open("US30", utc(2026, 5, 29, 20)) is True


# NAS100 -- same as US30
class TestNAS100:
    def test_open_during_day(self):
        assert is_market_open("NAS100", utc(2026, 5, 27, 16)) is True

    def test_closed_at_night(self):
        assert is_market_open("NAS100", utc(2026, 5, 27, 23)) is False

    def test_open_early_morning(self):
        assert is_market_open("NAS100", utc(2026, 5, 27, 5)) is True


# Forex -- Mon-Fri continuous
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

    def test_us30_zero_when_open(self):
        # 10:00 UTC -- US30 is open on Axi (01:00-22:00)
        assert minutes_until_open("US30", utc(2026, 5, 28, 10)) == 0

    def test_us30_before_open_returns_correct_minutes(self):
        # 00:30 UTC -- opens at 01:00, so 30 min
        mins = minutes_until_open("US30", utc(2026, 5, 28, 0, 30))
        assert mins == 30

    def test_us30_after_daily_break_returns_correct_minutes(self):
        # 22:30 UTC -- opens at 01:00 next day = 150 min
        mins = minutes_until_open("US30", utc(2026, 5, 28, 22, 30))
        assert mins == 150

    def test_gold_break_returns_nonzero(self):
        mins = minutes_until_open("XAUUSD", utc(2026, 5, 28, 21, 15))
        assert mins > 0
