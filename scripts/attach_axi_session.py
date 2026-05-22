"""Attach to running Axi terminal — no path, no crash."""
import MetaTrader5 as mt5, time, sys, os, psutil
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN    = int(os.getenv('MT5_LOGIN', 10042896))
PASSWORD = os.getenv('MT5_PASSWORD', 'IMSMCbot*3axi')
SERVER   = os.getenv('MT5_SERVER', 'Demostración de Axi-us50')

# Check what terminals are running
print("=== Running MT5 terminals ===")
for p in psutil.process_iter(['name','pid','memory_info','exe']):
    if 'terminal64' in p.info['name'].lower():
        ram = p.info['memory_info'].rss // (1024*1024)
        print(f"  PID={p.info['pid']} RAM={ram}MB exe={p.info['exe']}")

print()
print(f"Testing: login={LOGIN}")

# Method 1: Attach to active session (no path, no server — just attach)
print("\nMethod 1: mt5.initialize() — attach to active session")
try: mt5.shutdown()
except: pass
time.sleep(1)
ok = mt5.initialize(timeout=10000)
err = mt5.last_error()
print(f"  Result: ok={ok} err={err}")
if ok:
    info = mt5.account_info()
    if info:
        print(f"  CONNECTED! Login:{info.login} Balance:{info.balance} {info.currency}")
        print(f"  Server: {info.server}")
        mt5.symbol_select('EURUSD', True)
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD: {df['close'].iloc[-1]:.5f}")
        mt5.shutdown()
        sys.exit(0)
    mt5.shutdown()

# Method 2: Login to existing session (no path — don't crash terminal)
print("\nMethod 2: mt5.initialize(login, password, server) — no path")
for srv in [SERVER, 'Demostracion de Axi-us50', 'Axi-Demo']:
    try: mt5.shutdown()
    except: pass
    time.sleep(2)
    ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=srv, timeout=30000)
    err = mt5.last_error()
    print(f"  server='{srv}': ok={ok} err={err}")
    if ok:
        info = mt5.account_info()
        print(f"  CONNECTED! Login:{info.login} Balance:{info.balance} {info.currency} Server:{info.server}")
        mt5.symbol_select('EURUSD', True)
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD: {df['close'].iloc[-1]:.5f}")
        # Send Telegram notification
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
            print(f"  Telegram: {e}")
        mt5.shutdown()
        sys.exit(0)
    time.sleep(2)

mt5.shutdown()
print("\nNot connected.")
print(f"Error: {mt5.last_error()}")
print()
print("If MT5 shows LOGIN DIALOG on screen:")
print("  -> Enter password 'IMSMCbot*3axi' in the dialog")
print("  -> Wait for charts to show prices")
print("  -> Then run this script again")
