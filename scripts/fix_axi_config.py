"""Fix Axi MT5 config — write with proper UTF-8 encoding."""
import os, re, subprocess, time

os.chdir(r'C:\Users\jose-\projects\trading_agent')

# Find Axi data directory
axi_dir = os.path.expandvars(r'%APPDATA%\MetaQuotes\Terminal\6FBEE76C719DC78AB2AE839B5A0C7442')
cfg_dir  = os.path.join(axi_dir, 'config')
cfg_path = os.path.join(cfg_dir, 'common.ini')

os.makedirs(cfg_dir, exist_ok=True)

# Write config with CORRECT encoding (UTF-8, no BOM)
# Server name must match EXACTLY what Axi uses
SERVER_NAMES = [
    'Demostración de Axi-us50',   # Spanish with tilde
    'Demostracion de Axi-us50',   # Without accent (ASCII safe)
    'Axi-Demo',
    'Axi-us50 Demo',
]

for srv in SERVER_NAMES:
    cfg = (
        '[Common]\n'
        'Login=10042896\n'
        f'Server={srv}\n'
        'ProxyEnable=0\n'
        'ProxyType=0\n'
        'ProxyAddress=\n'
        '[Experts]\n'
        'AllowDllImport=1\n'
        'Enabled=1\n'
        'Account=1\n'
        'Profile=1\n'
        'Chart=0\n'
        'Api=1\n'
        'WebRequest=1\n'
    )
    # Write UTF-8 without BOM
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(cfg)

    print(f"Config written with server: '{srv}'")
    print(f"File: {cfg_path}")

    # Verify
    content = open(cfg_path, encoding='utf-8').read()
    print(f"Login: {[l for l in content.split() if l.startswith('Login=')]}")
    print(f"Server: {[l for l in content.split() if l.startswith('Server=')]}")

    # Open terminal fresh
    subprocess.run(['taskkill', '/F', '/IM', 'terminal64.exe'], capture_output=True)
    time.sleep(2)
    print(f"Opening Axi terminal...")
    subprocess.Popen([r'C:\Program Files\Axi MetaTrader 5 Terminal\terminal64.exe'])

    # Watch RAM for 20s — if it stabilizes >100MB, the server name is correct
    import psutil
    peak_ram = 0
    stable = False
    for i in range(20):
        time.sleep(2)
        for p in psutil.process_iter(['name','memory_info']):
            if 'terminal64' in p.info['name'].lower():
                ram = p.info['memory_info'].rss // (1024*1024)
                if ram > peak_ram: peak_ram = ram
                print(f"  t={i*2}s RAM={ram}MB", end='\r')
                if ram > 150:
                    stable = True

    print(f"\nPeak RAM: {peak_ram}MB | Stable: {stable}")

    if stable:
        print(f"SUCCESS! Server '{srv}' works — terminal stabilized!")
        break
    else:
        print(f"Terminal crashed with server '{srv}', trying next...")
        print()

# Final test
print("\nFinal: testing Python connection...")
import MetaTrader5 as mt5, sys
for srv in SERVER_NAMES:
    mt5.shutdown(); time.sleep(1)
    ok = mt5.initialize(login=10042896, password='IMSMCbot*3axi', server=srv, timeout=15000)
    print(f"  {srv}: ok={ok} err={mt5.last_error()}")
    if ok:
        i = mt5.account_info()
        print(f"  CONNECTED! Login:{i.login} Balance:{i.balance} {i.currency} Server:{i.server}")
        mt5.shutdown()
        sys.exit(0)
mt5.shutdown()
