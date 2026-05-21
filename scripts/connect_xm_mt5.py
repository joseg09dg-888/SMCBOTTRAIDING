"""Connect existing MT5 terminal to XM account."""
import subprocess, time, os, sys
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')

import MetaTrader5 as mt5

LOGIN    = 345308080
PASSWORD = 'IMSMCbot*2'
SERVER   = 'XMGlobal-MT5 10'
MT5_EXE  = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Configure MT5 data directory for XM
import json
cfg_dir = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\config")
cfg_path = os.path.join(cfg_dir, "common.ini")

if os.path.exists(cfg_path):
    content = open(cfg_path).read()
    # Update credentials
    import re
    content = re.sub(r'Login=\d+', f'Login={LOGIN}', content)
    content = re.sub(r'Server=[^\r\n]+', f'Server={SERVER}', content)
    content = re.sub(r'Enabled=\d', 'Enabled=1', content)
    content = re.sub(r'Api=\d', 'Api=1', content)
    open(cfg_path, 'w').write(content)
    print(f"Config updated: Login={LOGIN} Server={SERVER} Enabled=1 Api=1")

# Kill old MT5
subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
time.sleep(2)

# Open MT5
print(f"Opening MT5...")
subprocess.Popen([MT5_EXE])

# Wait for MT5 to load and connect to XM server
print("Waiting for MT5 to load and connect to XM server...")
for i in range(20):
    time.sleep(3)
    # Check RAM
    import psutil
    for proc in psutil.process_iter(['name', 'memory_info']):
        if 'terminal64' in proc.info['name'].lower():
            ram = proc.info['memory_info'].rss // (1024*1024)
            print(f"  t={i*3}s RAM={ram}MB", end="")
            if ram > 100:
                print(" -> loaded!")
                break
            else:
                print()

# Try connecting
print("\nTrying Python connection...")
for attempt in range(5):
    try: mt5.shutdown()
    except: pass
    time.sleep(2)

    ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER)
    err = mt5.last_error()
    print(f"Attempt {attempt+1}: ok={ok} err={err}")

    if ok:
        i = mt5.account_info()
        print(f"\nCONNECTED!")
        print(f"  Login:   {i.login}")
        print(f"  Server:  {i.server}")
        print(f"  Balance: {i.balance} {i.currency}")
        print(f"  Equity:  {i.equity}")
        # Test EURUSD
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD:  {df['close'].iloc[-1]:.5f}")
        mt5.shutdown()
        sys.exit(0)

    time.sleep(3)

print("\nStill not connected. MT5 may need manual login.")
print("If MT5 is open: enter Login=345308080 Password=IMSMCbot*2 Server=XMGlobal-MT5 10")
