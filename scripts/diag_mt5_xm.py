"""Diagnose MT5 XM connection — all methods."""
import MetaTrader5 as mt5, os, sys, time
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

print('=== DIAGNÓSTICO MT5 XM ===')

LOGIN = 345308080
PWD   = 'IMSMCbot*2'
SRV   = 'XMGlobal-MT5 10'

# Check running process
import psutil
for p in psutil.process_iter(['name','memory_info','pid']):
    if 'terminal64' in p.info['name'].lower():
        ram = p.info['memory_info'].rss // (1024*1024)
        print(f'MT5 process: PID={p.info["pid"]} RAM={ram}MB')

# Test 1: direct attach
try: mt5.shutdown()
except: pass
time.sleep(1)
r1 = mt5.initialize()
e1 = mt5.last_error()
print(f'\nTest1 init(): ok={r1} err={e1}')
if r1:
    i = mt5.account_info()
    print(f'  Account: {i}')
    mt5.shutdown()

# Test 2: with credentials
try: mt5.shutdown()
except: pass
time.sleep(1)
r2 = mt5.initialize(login=LOGIN, password=PWD, server=SRV)
e2 = mt5.last_error()
print(f'Test2 with creds: ok={r2} err={e2}')
if r2:
    i = mt5.account_info()
    print(f'  Login={i.login} Balance={i.balance} Server={i.server}')
    rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 0, 3)
    print(f'  EURUSD rates: {rates is not None}')
    if rates is not None:
        import pandas as pd
        df = pd.DataFrame(rates)
        print(f'  Last close: {df["close"].iloc[-1]:.5f}')
    mt5.shutdown()
    sys.exit(0)

# Test 3: path explicit
try: mt5.shutdown()
except: pass
time.sleep(1)
r3 = mt5.initialize(
    path=r'C:\Program Files\MetaTrader 5\terminal64.exe',
    login=LOGIN, password=PWD, server=SRV
)
e3 = mt5.last_error()
print(f'Test3 explicit path: ok={r3} err={e3}')
if r3:
    i = mt5.account_info()
    print(f'  Login={i.login} Balance={i.balance}')
    mt5.shutdown()
    sys.exit(0)

mt5.shutdown()
print(f'\nSUMMARY: All failed. Last error code: {e2[0]}')
print('ERROR CODES:')
print('  -6  = Authorization failed (Algo Trading disabled or not logged in)')
print(' -10002 = IPC recv failed (terminal crashed/restarting)')
print(' -10005 = IPC timeout (server unreachable)')
