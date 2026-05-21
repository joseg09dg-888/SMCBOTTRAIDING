"""Test Axi Demo MT5 connection — login 10042896."""
import MetaTrader5 as mt5, time, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN    = int(os.getenv('MT5_LOGIN', 10042896))
PASSWORD = os.getenv('MT5_PASSWORD', 'IMSMCbot*3axi')
SERVER   = os.getenv('MT5_SERVER', 'Demostración de Axi-us50')

print(f"Testing Axi Demo: login={LOGIN}")
print(f"Server: {SERVER}")
print()

# Try multiple server name variants (Spanish accents can cause issues)
SERVERS_TO_TRY = [
    SERVER,
    'Demostracion de Axi-us50',
    'Axi-Demo-us50',
    'Axi-US50 Demo',
    'AxiTrader-Demo',
    'Axi-Demo',
]

for srv in SERVERS_TO_TRY:
    try: mt5.shutdown()
    except: pass
    time.sleep(1)

    ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=srv)
    err = mt5.last_error()
    print(f"  server='{srv}': ok={ok} err={err}")

    if ok:
        info = mt5.account_info()
        print(f"\nCONECTADO!")
        print(f"  Login:    {info.login}")
        print(f"  Balance:  {info.balance} {info.currency}")
        print(f"  Server:   {info.server}")
        print(f"  Type:     {'DEMO' if info.trade_mode == 0 else 'REAL'}")

        # Enable symbols and get data
        mt5.symbol_select('EURUSD', True)
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 5)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD velas: {len(df)}")
            print(f"  Last close:   {df['close'].iloc[-1]:.5f}")

        # Send Telegram notification
        try:
            from dashboard.telegram_bot import TradingTelegramBot
            import asyncio
            bot = TradingTelegramBot()
            msg = (
                f"<b>MT5 AXI DEMO CONECTADO</b>\n"
                f"Login: {info.login}\n"
                f"Balance: ${info.balance:,.2f} {info.currency}\n"
                f"Server: {info.server}\n"
                f"Bot operando forex + crypto 24/7"
            )
            asyncio.run(bot.send_glint_alert(msg))
            print(f"  Telegram: sent!")
        except Exception as e:
            print(f"  Telegram: {e}")

        mt5.shutdown()
        sys.exit(0)

    time.sleep(2)

mt5.shutdown()
print(f"\nAll servers failed. MT5 may need to be open with the account logged in.")
