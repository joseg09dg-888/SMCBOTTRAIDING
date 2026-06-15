# tests/core/test_nightly_reporter_wiring.py
"""Verify TradingSupervisor wires NightlyReporter to Telegram so the
22:00 UTC daily report actually reaches the user, not just stdout."""
import pytest
from core.config import config
from core.supervisor import TradingSupervisor


@pytest.fixture(scope="module")
def supervisor():
    return TradingSupervisor(capital=100_000, demo_mode=True)


class TestNightlyReporterWiring:
    def test_reporter_has_telegram_bot(self, supervisor):
        assert supervisor._reporter._bot is not None

    def test_reporter_chat_id_matches_config(self, supervisor):
        assert supervisor._reporter._chat_id == config.telegram_chat_id

    def test_reporter_bot_is_supervisor_telegram_bot(self, supervisor):
        assert supervisor._reporter._bot is supervisor.telegram.get_bot()
