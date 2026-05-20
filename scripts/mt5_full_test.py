"""Test MT5 with full diagnostic."""
import sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

import MetaTrader5 as mt5

print("=== MT5 Full Diagnostic ===")
print(f"Package version: {mt5.__version__}")
print()

# Try every combination
attempts = [
    {"desc": "With credentials",
     "args": {"login": 8889, "password": "IMSMCbot", "server": "MetaQuotes-Demo"}},
    {"desc": "No credentials (active session)",
     "args": {}},
    {"desc": "With terminal path",
     "args": {"path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
              "login": 8889, "password": "IMSMCbot", "server": "MetaQuotes-Demo"}},
    {"desc": "With portable path",
     "args": {"path": r"C:\Program Files\MetaTrader 5\terminal64.exe"}},
]

for a in attempts:
    try:
        mt5.shutdown()
    except Exception:
        pass
    import time; time.sleep(0.5)

    try:
        ok = mt5.initialize(**a["args"])
        err = mt5.last_error()
        print(f"{a['desc']}: {'OK' if ok else 'FAIL'} | error={err}")
        if ok:
            info = mt5.account_info()
            if info:
                print(f"  Login: {info.login} | Balance: {info.balance} {info.currency} | Server: {info.server}")
                # Try to get EURUSD data
                import pandas as pd
                rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 3)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    print(f"  EURUSD data: {len(df)} candles | last close: {df['close'].iloc[-1]:.5f}")
                else:
                    print(f"  EURUSD data: FAIL {mt5.last_error()}")
            mt5.shutdown()
            print()
            print("SUCCESS! Use this connection method.")
            break
    except Exception as e:
        print(f"{a['desc']}: EXCEPTION {e}")
    print()

print()
print("If ALL failed with -6 'Authorization failed':")
print("  1. In MT5: Tools -> Options -> Expert Advisors")
print("  2. Check: 'Allow automated trading'")
print("  3. Check: 'Allow DLL imports'")
print("  4. Click OK and RESTART MT5 completely")
print("  5. Then run this script again")
