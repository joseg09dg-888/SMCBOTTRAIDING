"""Test Axi MT5 connection with real credentials."""
import MetaTrader5 as mt5, time, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN    = int(os.getenv('MT5_LOGIN', 60290663))
PASSWORD = os.getenv('MT5_PASSWORD', 'IMSMCbot*2axi')
SERVER   = os.getenv('MT5_SERVER', 'Axi-US51-En vivo')

print(f"Testing: login={LOGIN} server={SERVER}")
print()

for attempt in range(4):
    try: mt5.shutdown()
    except: pass
    time.sleep(2)

    ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER)
    err = mt5.last_error()
    print(f"Attempt {attempt+1}: ok={ok} error={err}")

    if ok:
        info = mt5.account_info()
        print(f"\nCONECTADO!")
        print(f"  Login:    {info.login}")
        print(f"  Balance:  {info.balance} {info.currency}")
        print(f"  Server:   {info.server}")
        print(f"  Equity:   {info.equity}")
        print(f"  Type:     {'DEMO' if info.trade_mode == 0 else 'REAL'}")

        # Enable forex symbols
        print("\nActivando símbolos forex:")
        for sym in ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY', 'GBPJPY']:
            r = mt5.symbol_select(sym, True)
            info2 = mt5.symbol_info(sym)
            price = info2.bid if info2 else 0
            print(f"  {sym}: {'OK' if r else 'FAIL'} bid={price:.5f}")

        # Test EURUSD data
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 3)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"\nEURUSD 1H data: {len(df)} candles")
            print(f"  Last close: {df['close'].iloc[-1]:.5f}")
        else:
            print(f"EURUSD data: {mt5.last_error()}")

        mt5.shutdown()
        sys.exit(0)

    time.sleep(3)

mt5.shutdown()
print(f"\nFailed. Last error: {mt5.last_error()}")
print("Make sure Axi MT5 terminal is open and Algo Trading is enabled")
