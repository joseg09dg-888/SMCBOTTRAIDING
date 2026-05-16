"""Test MT5 connection and OHLCV fetch."""
import sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\jose-\projects\trading_agent\.env')

import MetaTrader5 as mt5

print("=== MT5 Connection Tests ===")
print(f"MT5 version: {mt5.__version__}")

# Test 1: with credentials
print("\nTest 1: initialize(login=8889, password='IMSMCbot', server='MetaQuotes-Demo')")
ok = mt5.initialize(login=8889, password='IMSMCbot', server='MetaQuotes-Demo')
print(f"  Result: {ok} | Error: {mt5.last_error()}")
if ok:
    info = mt5.account_info()
    print(f"  Login: {info.login} | Balance: {info.balance} {info.currency} | Server: {info.server}")
    print("\nTest OHLCV EURUSD H1:")
    import pandas as pd
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 5)
    if rates is not None:
        df = pd.DataFrame(rates)
        print(f"  Got {len(df)} candles | Last close: {df['close'].iloc[-1]}")
    else:
        print(f"  No data: {mt5.last_error()}")
    mt5.shutdown()
else:
    mt5.shutdown()

# Test 2: no credentials
print("\nTest 2: initialize() (active session)")
ok2 = mt5.initialize()
print(f"  Result: {ok2} | Error: {mt5.last_error()}")
if ok2:
    info = mt5.account_info()
    if info:
        print(f"  Login: {info.login} | Balance: {info.balance} | Server: {info.server}")
    mt5.shutdown()
else:
    mt5.shutdown()

# Test 3: with path
print("\nTest 3: with terminal path")
path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ok3 = mt5.initialize(path=path, login=8889, password='IMSMCbot', server='MetaQuotes-Demo')
print(f"  Result: {ok3} | Error: {mt5.last_error()}")
if ok3:
    info = mt5.account_info()
    print(f"  Login: {info.login} | Balance: {info.balance}")
    mt5.shutdown()
