"""Final Axi connection test with correct UTF-8 server name."""
import MetaTrader5 as mt5, time, sys, os, psutil
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')

LOGIN = 10042896
PWD   = 'IMSMCbot*3axi'

# Check terminal
for p in psutil.process_iter(['name','memory_info']):
    if 'terminal64' in p.info['name'].lower():
        ram = p.info['memory_info'].rss // (1024*1024)
        print(f"Terminal RAM: {ram}MB")

# Read the server name directly from the config file we wrote
cfg = os.path.expandvars(
    r'%APPDATA%\MetaQuotes\Terminal\6FBEE76C719DC78AB2AE839B5A0C7442\config\common.ini'
)
srv_from_config = 'Demostración de Axi-us50'
if os.path.exists(cfg):
    for line in open(cfg, encoding='utf-8').readlines():
        if line.startswith('Server='):
            srv_from_config = line.strip().replace('Server=', '')
            break
print(f"Server from config: '{srv_from_config}'")
print(f"Server bytes: {srv_from_config.encode('utf-8').hex()}")
print()

# Try all combinations
servers = [
    srv_from_config,                    # Exact from config (UTF-8)
    'Demostración de Axi-us50',         # With Spanish accent (unicode)
    'Demostracion de Axi-us50',         # Without accent
    'Axi-Demo',
    'Axi-US50-Demo',
    'Axi-US51-Demo',
]

for srv in servers:
    try: mt5.shutdown()
    except: pass
    time.sleep(1)
    print(f"Trying: '{srv}'")
    ok = mt5.initialize(login=LOGIN, password=PWD, server=srv, timeout=30000)
    err = mt5.last_error()
    print(f"  ok={ok} err={err}")
    if ok:
        info = mt5.account_info()
        print(f"\nCONNECTED!")
        print(f"  Login:   {info.login}")
        print(f"  Balance: {info.balance} {info.currency}")
        print(f"  Server:  {info.server}")
        print(f"  Type:    {'DEMO' if info.trade_mode == 0 else 'REAL'}")
        mt5.symbol_select('EURUSD', True)
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD:  {df['close'].iloc[-1]:.5f}")
        # Send Telegram
        try:
            import asyncio
            from dashboard.telegram_bot import TradingTelegramBot
            bot = TradingTelegramBot()
            msg = (
                f"<b>MT5 AXI DEMO CONECTADO</b>\n"
                f"Login: {info.login}\n"
                f"Balance: ${info.balance:,.2f} {info.currency}\n"
                f"Server: {info.server}"
            )
            asyncio.run(bot.send_glint_alert(msg))
            print("  Telegram: sent!")
        except Exception as e:
            print(f"  Telegram: {e}")
        mt5.shutdown()
        sys.exit(0)
    time.sleep(2)

mt5.shutdown()
print(f"\nAll failed. Last error: {mt5.last_error()}")
print()
print("ACCION REQUERIDA:")
print("En el terminal Axi que está abierto:")
print("  1. Si ve diálogo de login: ingresa password 'IMSMCbot*3axi' y click OK")
print("  2. Si ve charts con precios: el problema es otro (credenciales incorrectas)")
print("  3. Si charts están negros: el servidor no pudo autenticar")
