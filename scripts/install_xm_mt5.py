"""Download and install XM MT5."""
import urllib.request, subprocess, os, time, glob, sys

# XM MT5 installer URL
urls = [
    "https://download.mql5.com/cdn/web/xm.global.limited/mt5/xmglobal5setup.exe",
    "https://download.mql5.com/cdn/web/trading.point.of.financial.instruments/mt5/xmtrading5setup.exe",
]

installer = r"C:\Windows\Temp\xm_mt5_setup.exe"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "Accept": "*/*",
}

downloaded = False
for url in urls:
    try:
        print(f"Downloading: {url}")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r, open(installer, "wb") as f:
            total = 0
            while True:
                chunk = r.read(65536)
                if not chunk: break
                f.write(chunk)
                total += len(chunk)
            print(f"Downloaded: {total//1024//1024} MB -> {installer}")
        downloaded = True
        break
    except Exception as e:
        print(f"  Failed: {e}")

if not downloaded:
    print("All download URLs failed")
    sys.exit(1)

# Kill existing MT5
subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
time.sleep(2)

# Install silently
print("Installing XM MT5...")
result = subprocess.run([installer, "/auto"], capture_output=False, timeout=120)
print(f"Install exit code: {result.returncode}")
time.sleep(10)

# Find installation
found = glob.glob(r"C:\Program Files*\**\terminal64.exe", recursive=True)
print(f"MT5 installations: {found}")

# Open it
if found:
    newest = sorted(found, key=os.path.getmtime, reverse=True)[0]
    print(f"Opening: {newest}")
    subprocess.Popen([newest])
    time.sleep(20)

    # Test connection
    import MetaTrader5 as mt5
    sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
    ok = mt5.initialize(login=345308080, password='IMSMCbot*2', server='XMGlobal-MT5 10')
    print(f"\nmt5.initialize(): ok={ok} err={mt5.last_error()}")
    if ok:
        i = mt5.account_info()
        print(f"CONNECTED! Login:{i.login} Balance:{i.balance} {i.currency} Server:{i.server}")
        mt5.shutdown()
