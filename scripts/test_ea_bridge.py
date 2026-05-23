"""Test the EA bridge once EA is compiled and running in Axi MT5."""
import sys, time, asyncio
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
import os; os.chdir(r'C:\Users\jose-\projects\trading_agent')
from connectors.mt5_ea_bridge import MT5EABridge

bridge = MT5EABridge(timeout_sec=15)

print("=== MT5 EA Bridge Test ===")
print("Make sure SMCBotEA is running on a chart in Axi MT5")
print()

print("Sending EURUSD LONG signal to EA...")
result = bridge.place_order(
    symbol="EURUSD", direction="long",
    volume=0.01, sl_pips=50, tp_pips=100,
    comment="SMC Bot EA Test"
)
print(f"Result: {result}")

if result.get("retcode") == 10009:
    print(f"\nEA TRADE EXECUTED!")
    print(f"  Order:  {result.get('order')}")
    print(f"  Price:  {result.get('price')}")
    # Telegram
    try:
        from dashboard.telegram_bot import TradingTelegramBot
        bot = TradingTelegramBot()
        msg = (
            f"<b>MT5 EA BRIDGE FUNCIONANDO</b>\n"
            f"EURUSD LONG via MQL5 EA\n"
            f"Order: {result.get('order')}\n"
            f"retcode: {result.get('retcode')}"
        )
        asyncio.run(bot.send_glint_alert(msg))
        print("  Telegram: sent!")
    except Exception as e:
        print(f"  Telegram: {e}")
elif result.get("retcode") == -1:
    print(f"\nEA not responding (timeout).")
    print("Check that SMCBotEA is compiled and attached to EURUSD chart.")
else:
    print(f"\nEA returned code {result.get('retcode')}: {result.get('error', result.get('comment', ''))}")
