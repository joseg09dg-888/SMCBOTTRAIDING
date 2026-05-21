"""Force XM MT5 login — kills old MT5, fresh start, long timeout."""
import subprocess, time, os, sys, psutil
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')

LOGIN = 345308080
PWD   = 'IMSMCbot*2'
SRV   = 'XMGlobal-MT5 10'
MT5EXE = r'C:\Program Files\MetaTrader 5\terminal64.exe'

# 1. Kill ALL MT5 processes
for p in psutil.process_iter(['name','pid']):
    if 'terminal64' in p.info['name'].lower():
        print(f"Killing MT5 PID {p.info['pid']}")
        try: p.kill()
        except: pass
time.sleep(3)

# 2. Fix config
cfg = os.path.expandvars(r'%APPDATA%\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\config\common.ini')
if os.path.exists(cfg):
    import re
    content = open(cfg).read()
    content = re.sub(r'Login=\d+', f'Login={LOGIN}', content)
    content = re.sub(r'Server=[^\r\n]+', f'Server={SRV}', content)
    content = re.sub(r'Enabled=\d', 'Enabled=1', content)
    content = re.sub(r'Api=\d', 'Api=1', content)
    open(cfg, 'w').write(content)
    print(f"Config: Login={LOGIN} Server={SRV} Enabled=1 Api=1")

# 3. Start MT5 fresh
print("Starting MT5...")
subprocess.Popen([MT5EXE])

# 4. Wait for FULL load (stable RAM > 100MB for 10s)
print("Waiting for full load...")
stable_count = 0
for i in range(40):
    time.sleep(3)
    ram = 0
    for p in psutil.process_iter(['name','memory_info']):
        if 'terminal64' in p.info['name'].lower():
            ram = p.info['memory_info'].rss // (1024*1024)
    print(f"  t={i*3}s RAM={ram}MB")
    if ram > 100:
        stable_count += 1
        if stable_count >= 3:
            print("  MT5 stable!")
            break
    else:
        stable_count = 0

# 5. Try Python connection with LONG timeout
import MetaTrader5 as mt5
print("\nConnecting Python (timeout=120s)...")
for attempt in range(5):
    try: mt5.shutdown()
    except: pass
    time.sleep(3)

    ok = mt5.initialize(login=LOGIN, password=PWD, server=SRV, timeout=120000)
    err = mt5.last_error()
    print(f"Attempt {attempt+1}: ok={ok} err={err}")

    if ok:
        info = mt5.account_info()
        print(f"\nCONNECTED!")
        print(f"  Login:   {info.login}")
        print(f"  Balance: {info.balance} {info.currency}")
        print(f"  Server:  {info.server}")
        # Enable symbols
        for sym in ['EURUSD','GBPUSD','XAUUSD','USDJPY','GBPJPY']:
            mt5.symbol_select(sym, True)
        # Get EURUSD
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD: {df['close'].iloc[-1]:.5f}")
        mt5.shutdown()
        sys.exit(0)
    time.sleep(5)

mt5.shutdown()
print("\nMT5 could not connect after 5 attempts")
print("Bot will use yfinance for forex (already active)")
print("Run: .venv\\Scripts\\python scripts/force_demo_trade.py EURUSD 1h")
