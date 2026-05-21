"""Test XM MT5 connection with new credentials from email."""
import MetaTrader5 as mt5, time, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN    = int(os.getenv('MT5_LOGIN', 345308080))
PASSWORD = os.getenv('MT5_PASSWORD', 'IMSMCbot*2')
SERVER   = os.getenv('MT5_SERVER', 'XMGlobal-MT5 10')

print(f"Testing: login={LOGIN} server={SERVER}")
print()

# Try connecting
for attempt in range(3):
    try: mt5.shutdown()
    except: pass
    time.sleep(1)

    ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER)
    err = mt5.last_error()
    print(f"Intento {attempt+1}: ok={ok} error={err}")

    if ok:
        info = mt5.account_info()
        print(f"\nCONECTADO!")
        print(f"  Login:    {info.login}")
        print(f"  Server:   {info.server}")
        print(f"  Balance:  {info.balance} {info.currency}")
        print(f"  Equity:   {info.equity}")
        print(f"  Type:     {'DEMO' if info.trade_mode == 0 else 'REAL'}")

        # Test EURUSD data
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 3)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD:   last close = {df['close'].iloc[-1]:.5f}")
        else:
            print(f"  EURUSD:   {mt5.last_error()}")

        mt5.shutdown()
        sys.exit(0)

    time.sleep(3)

print("\nNo conectó. Verificar que MT5 de XM esté abierto.")
