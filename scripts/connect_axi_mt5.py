"""Connect to Axi MT5 demo account."""
import MetaTrader5 as mt5, time, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN    = int(os.getenv('MT5_LOGIN', 0))
PASSWORD = os.getenv('MT5_PASSWORD', 'IMSMCbot*2axi')
SERVER   = os.getenv('MT5_SERVER', 'Axi-MT5 Demo')

# Possible Axi server names to try
AXI_SERVERS = [
    SERVER,
    'Axi-MT5 Demo',
    'AxiTrader-Demo',
    'AxiFinancial-Demo',
    'Axi-Demo',
    'AxiTrader-MT5 Demo',
    'Axi-Live',
]

print(f"=== CONNECTING AXI MT5 ===")
print(f"Login: {LOGIN} | Server: {SERVER}")
print(f"Password: {'*' * len(PASSWORD)}")
print()

if LOGIN == 0:
    print("ERROR: MT5_LOGIN not set in .env")
    print("Check email joseg09.dg@gmail.com for Axi account credentials")
    sys.exit(1)

# Find Axi MT5 terminal
import glob
terminals = glob.glob(r"C:\Program Files*\**\terminal64.exe", recursive=True)
print(f"MT5 terminals found: {terminals}")

# Try each server
for srv in AXI_SERVERS:
    for path in [None] + terminals:
        try: mt5.shutdown()
        except: pass
        time.sleep(1)

        if path:
            ok = mt5.initialize(path=path, login=LOGIN, password=PASSWORD, server=srv)
        else:
            ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=srv)

        err = mt5.last_error()
        print(f"  srv={srv} path={path is not None}: ok={ok} err={err}")

        if ok:
            info = mt5.account_info()
            print(f"\nCONNECTED!")
            print(f"  Login:   {info.login}")
            print(f"  Balance: {info.balance} {info.currency}")
            print(f"  Server:  {info.server}")
            print(f"  Type:    {'DEMO' if info.trade_mode == 0 else 'REAL'}")

            # Enable forex symbols
            for sym in ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY', 'GBPJPY']:
                result = mt5.symbol_select(sym, True)
                print(f"  {sym}: {'enabled' if result else 'failed'}")

            # Test EURUSD
            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 3)
            if rates is not None:
                import pandas as pd
                df = pd.DataFrame(rates)
                print(f"  EURUSD last close: {df['close'].iloc[-1]:.5f}")

            mt5.shutdown()
            sys.exit(0)
        time.sleep(2)

mt5.shutdown()
print("\nAll connection attempts failed.")
print("Possible causes:")
print("  1. MT5 Axi not installed or not open")
print("  2. Login number incorrect (check email)")
print("  3. Server name incorrect")
print("  4. Algo Trading not enabled in MT5")
