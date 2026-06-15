# tests/dashboard/test_telegram_bot.py
from dashboard.telegram_bot import TradingTelegramBot


class TestGetBot:
    def test_get_bot_returns_telegram_bot_instance(self):
        bot = TradingTelegramBot()
        result = bot.get_bot()
        assert result is not None
        assert hasattr(result, "send_message")

    def test_get_bot_same_instance_as_internal(self):
        bot = TradingTelegramBot()
        first = bot.get_bot()
        second = bot.get_bot()
        assert first is second
