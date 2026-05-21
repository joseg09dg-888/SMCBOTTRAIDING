"""Test Axi MT5 terminal connection."""
import MetaTrader5 as mt5, time, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

AXI_PATH = r'C:\Program Files\Axi MetaTrader 5 Terminal\terminal64.exe'
LOGIN    = int(os.getenv('MT5_LOGIN', 10042896))
PASSWORD = os.getenv('MT5_PASSWORD', 'IMSMCbot*3axi')
SERVER   = os.getenv('MT5_SERVER', 'Demostración de Axi-us50')

print(f"Axi terminal: {AXI_PATH}")
print(f"Login: {LOGIN} | Server: {SERVER}")
print()

servers = [
    SERVER,
    'Demostracion de Axi-us50',
    'Axi-Demo-us50',
    'Axi-us50 Demo',
    'AxiTrader-Demo',
    'Axi-Demo',
]

for srv in servers:
    try: mt5.shutdown()
    except: pass
    time.sleep(1)

    ok = mt5.initialize(path=AXI_PATH, login=LOGIN, password=PASSWORD, server=srv)
    err = mt5.last_error()
    print(f"  server='{srv}': ok={ok} err={err}")

    if ok:
        info = mt5.account_info()
        print(f"\nCONECTADO!")
        print(f"  Login:   {info.login}")
        print(f"  Balance: {info.balance} {info.currency}")
        print(f"  Server:  {info.server}")
        print(f"  Equity:  {info.equity}")
        print(f"  Type:    {'DEMO' if info.trade_mode == 0 else 'REAL'}")

        for sym in ['EURUSD','GBPUSD','XAUUSD','USDJPY','GBPJPY']:
            mt5.symbol_select(sym, True)

        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 5)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            last = df['close'].iloc[-1]
            print(f"  EURUSD velas: {len(df)} | last close: {last:.5f}")
        else:
            print(f"  EURUSD: {mt5.last_error()}")

        # Telegram notification
        try:
            import asyncio
            from dashboard.telegram_bot import TradingTelegramBot
            bot = TradingTelegramBot()
            msg = (
                f"<b>MT5 AXI DEMO CONECTADO</b>\n"
                f"Login: {info.login}\n"
                f"Balance: ${info.balance:,.2f} {info.currency}\n"
                f"Server: {info.server}\n"
                f"Bot operando forex + crypto 24/7"
            )
            asyncio.run(bot.send_glint_alert(msg))
            print("  Telegram: sent!")
        except Exception as e:
            print(f"  Telegram error: {e}")

        mt5.shutdown()
        sys.exit(0)

    time.sleep(2)

mt5.shutdown()
print("\nAll servers failed")
print(f"Last error: {mt5.last_error()}")
