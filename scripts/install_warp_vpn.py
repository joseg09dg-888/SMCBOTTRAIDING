"""Install Cloudflare WARP (free VPN) and connect."""
import urllib.request, subprocess, os, sys, time, json

print("=== INSTALLING CLOUDFLARE WARP (Free VPN) ===")
print("No registration required. Encrypts all traffic.")
print()

WARP_URL = "https://1.1.1.1/Cloudflare_WARP_Release-x64.msi"
WARP_PATH = r"C:\Windows\Temp\warp_install.msi"
WARP_CLI  = r"C:\Program Files\Cloudflare\Cloudflare WARP\warp-cli.exe"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}

# Check if already installed
if os.path.exists(WARP_CLI):
    print(f"WARP already installed: {WARP_CLI}")
else:
    print(f"Downloading WARP from {WARP_URL}...")
    try:
        req = urllib.request.Request(WARP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r, open(WARP_PATH, "wb") as f:
            total = 0
            while True:
                chunk = r.read(65536)
                if not chunk: break
                f.write(chunk)
                total += len(chunk)
        print(f"Downloaded: {total//1024//1024} MB -> {WARP_PATH}")

        print("Installing WARP silently...")
        result = subprocess.run(
            ["msiexec", "/i", WARP_PATH, "/quiet", "/norestart"],
            capture_output=True, timeout=120
        )
        print(f"Install exit code: {result.returncode}")
        time.sleep(15)  # Let service start

    except Exception as e:
        print(f"Download/install error: {e}")
        print("Trying alternate download...")

# Try warp-cli connect
warp_paths = [
    WARP_CLI,
    r"C:\Program Files\Cloudflare\Cloudflare WARP\warp-cli.exe",
    r"C:\Program Files (x86)\Cloudflare\Cloudflare WARP\warp-cli.exe",
]

warp_exe = None
for p in warp_paths:
    if os.path.exists(p):
        warp_exe = p
        print(f"Found WARP CLI: {warp_exe}")
        break

if warp_exe:
    print("\nConnecting WARP...")
    # Register without email (zero trust not needed for basic VPN)
    subprocess.run([warp_exe, "registration", "new"], capture_output=True, timeout=30)
    result = subprocess.run([warp_exe, "connect"], capture_output=True, text=True, timeout=30)
    print(f"Connect result: {result.stdout} {result.stderr}")

    time.sleep(5)
    status = subprocess.run([warp_exe, "status"], capture_output=True, text=True, timeout=10)
    print(f"Status: {status.stdout.strip()}")

    # Check connectivity
    print("\nTesting connectivity after WARP...")
    import socket
    for host in ["mt5.metaquotes.net", "testnet.binance.vision"]:
        try:
            socket.create_connection((host, 443), timeout=5)
            print(f"  {host}:443 -> REACHABLE via WARP!")
        except:
            print(f"  {host}:443 -> still blocked")
else:
    print("WARP not installed successfully")
    sys.exit(1)
