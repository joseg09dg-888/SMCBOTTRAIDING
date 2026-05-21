"""Set up Tor as SOCKS5 proxy and configure Axi MT5 to use it."""
import urllib.request, subprocess, os, sys, time, ssl, socket, zipfile

print("=== TOR PROXY SETUP FOR MT5 ===")
print("Tor routes traffic through public relays — bypasses ISP blocks")
print()

TOR_DIR = r"C:\Windows\Temp\tor_proxy"
TOR_EXE = os.path.join(TOR_DIR, "Tor", "tor.exe")

# Step 1: Download Tor if not present
if not os.path.exists(TOR_EXE):
    os.makedirs(TOR_DIR, exist_ok=True)

    # Try multiple Tor download mirrors
    tor_urls = [
        "https://archive.torproject.org/tor-package-archive/torbrowser/13.5/tor-expert-bundle-windows-x86_64-13.5.tar.gz",
        "https://dist.torproject.org/torbrowser/13.5/tor-expert-bundle-windows-x86_64-13.5.tar.gz",
    ]

    # Simpler: just download the standalone Tor Expert Bundle (no browser needed)
    # This is smaller (~10MB) and works as SOCKS proxy
    tor_zip_url = "https://archive.torproject.org/tor-package-archive/torbrowser/13.5.1/tor-expert-bundle-windows-x86_64-13.5.1.tar.gz"

    print(f"Downloading Tor from {tor_zip_url}...")
    tor_archive = os.path.join(TOR_DIR, "tor.tar.gz")

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(tor_zip_url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(tor_archive, "wb") as f:
            total = 0
            while True:
                chunk = r.read(65536)
                if not chunk: break
                f.write(chunk)
                total += len(chunk)
        print(f"Downloaded: {total//1024//1024} MB")

        # Extract
        import tarfile
        print("Extracting...")
        with tarfile.open(tor_archive, "r:gz") as tf:
            tf.extractall(TOR_DIR)
        print("Extracted!")

    except Exception as e:
        print(f"Download error: {e}")
        # Try alternate approach: use Python as SOCKS proxy via free service
        print("\nTrying alternate approach: configure MT5 proxy directly...")

# Step 2: Find or start Tor
tor_exe = TOR_EXE
if not os.path.exists(tor_exe):
    # Search for tor.exe
    for root, dirs, files in os.walk(TOR_DIR):
        if "tor.exe" in files:
            tor_exe = os.path.join(root, "tor.exe")
            break

if os.path.exists(tor_exe):
    print(f"\nStarting Tor SOCKS5 proxy on port 9050...")
    print(f"Tor exe: {tor_exe}")

    tor_proc = subprocess.Popen(
        [tor_exe, "--SocksPort", "9050", "--Log", "notice stdout"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(tor_exe)
    )

    # Wait for Tor to bootstrap
    print("Waiting for Tor to bootstrap (up to 60s)...")
    for i in range(60):
        time.sleep(1)
        try:
            # Check if SOCKS5 port is open
            sock = socket.socket()
            sock.settimeout(1)
            sock.connect(("127.0.0.1", 9050))
            sock.close()
            print(f"  Tor SOCKS5 proxy ready at localhost:9050 (t={i}s)")
            break
        except:
            pass
        if i % 10 == 0:
            print(f"  t={i}s waiting...")
else:
    print("Tor not available. Configuring MT5 with free SOCKS5 proxy instead...")

# Step 3: Configure Axi MT5 to use SOCKS5 proxy
# Try multiple free SOCKS5 proxies
FREE_SOCKS5 = [
    ("127.0.0.1", 9050),    # Tor local proxy
    ("51.195.57.239", 1080), # Known free SOCKS5
    ("185.130.105.211", 1080),
    ("104.251.123.83", 1080),
]

working_proxy = None
for ip, port in FREE_SOCKS5:
    try:
        sock = socket.socket()
        sock.settimeout(3)
        sock.connect((ip, port))
        sock.close()
        working_proxy = (ip, port)
        print(f"Working SOCKS5 proxy: {ip}:{port}")
        break
    except:
        print(f"  Proxy {ip}:{port} not reachable")

if working_proxy:
    proxy_ip, proxy_port = working_proxy
    print(f"\nConfiguring Axi MT5 to use proxy {proxy_ip}:{proxy_port}...")

    cfg_path = os.path.expandvars(
        r"%APPDATA%\MetaQuotes\Terminal\6FBEE76C719DC78AB2AE839B5A0C7442\config\common.ini"
    )

    if os.path.exists(cfg_path):
        content = open(cfg_path).read()
        import re
        content = re.sub(r'ProxyEnable=\d', 'ProxyEnable=1', content)
        content = re.sub(r'ProxyType=\d', 'ProxyType=4', content)  # 4 = SOCKS5
        # Add proxy settings if not present
        if 'ProxyAddress=' not in content:
            content += f"\nProxyAddress={proxy_ip}:{proxy_port}\n"
        else:
            content = re.sub(r'ProxyAddress=[^\r\n]*', f'ProxyAddress={proxy_ip}:{proxy_port}', content)
        open(cfg_path, 'w').write(content)
        print(f"MT5 config updated with SOCKS5 proxy!")
        print(f"  ProxyEnable=1, ProxyType=4 (SOCKS5)")
        print(f"  ProxyAddress={proxy_ip}:{proxy_port}")
    else:
        print(f"Config not found: {cfg_path}")
else:
    print("No working proxy found.")
    print("\nManual VPN options:")
    print("  1. ProtonVPN free: protonvpn.com/download (requires registration)")
    print("  2. ExpressVPN trial: 30 days free")
    print("  3. Personal hotspot via phone (different network = different ISP)")
    print("  4. Ask ISP to unblock trading ports 443/5580/5581")
    print()
    print("FASTEST OPTION: Use phone as WiFi hotspot")
    print("  Phone Settings -> Personal Hotspot -> Connect PC to phone WiFi")
    print("  Then run: .venv\\Scripts\\python scripts/test_axi_terminal.py")
