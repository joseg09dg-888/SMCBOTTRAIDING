"""Configure Windows built-in VPN with free credentials (no software install)."""
import subprocess, urllib.request, re, time, sys, ssl

print("=== WINDOWS BUILT-IN VPN SETUP ===")
print("Using free VPN credentials (no software install needed)")
print()

# VPNBook provides free PPTP credentials (public)
# Note: PPTP is less secure but works for port unblocking
VPNBOOK_CREDENTIALS_URL = "https://www.vpnbook.com/"

# Free credentials (these rotate, fetching current ones)
# Fallback hardcoded known-good entries
FREE_VPN_SERVERS = [
    # VPNBook PPTP free servers (publicly available)
    {"name": "VPNBook-US1",  "server": "us1.vpnbook.com",  "user": "vpnbook", "pass": ""},
    {"name": "VPNBook-US2",  "server": "us2.vpnbook.com",  "user": "vpnbook", "pass": ""},
    {"name": "VPNBook-EU1",  "server": "euro1.vpnbook.com","user": "vpnbook", "pass": ""},
    {"name": "VPNBook-EU2",  "server": "euro2.vpnbook.com","user": "vpnbook", "pass": ""},
    {"name": "VPNBook-CA",   "server": "ca1.vpnbook.com",  "user": "vpnbook", "pass": ""},
]

# Try to get current password from VPNBook website
vpnbook_password = "vcamtg7"  # Default/fallback
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        VPNBOOK_CREDENTIALS_URL,
        headers={"User-Agent": "Mozilla/5.0 Chrome/120.0"}
    )
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        html = r.read().decode('utf-8', errors='replace')
    # Find password pattern
    patterns = [
        r'Password[:\s]*<[^>]*>([a-zA-Z0-9]+)',
        r'password[:\s]+([a-zA-Z0-9]{6,12})',
        r'<strong>([a-zA-Z0-9]{6,12})</strong>.*?password',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.I)
        if m:
            vpnbook_password = m.group(1)
            print(f"Got current VPNBook password: {vpnbook_password}")
            break
    else:
        print(f"Using default password: {vpnbook_password}")
except Exception as e:
    print(f"Could not fetch credentials: {e}. Using default.")

# Update password in servers list
for s in FREE_VPN_SERVERS:
    s["pass"] = vpnbook_password

print(f"\nVPN servers to try: {len(FREE_VPN_SERVERS)}")
print()

# Remove any existing VPN connections we created
for s in FREE_VPN_SERVERS:
    subprocess.run(
        ['powershell', '-Command', f'Remove-VpnConnection -Name "{s["name"]}" -Force -ErrorAction SilentlyContinue'],
        capture_output=True
    )

# Add and connect VPN connections
for s in FREE_VPN_SERVERS:
    name   = s["name"]
    server = s["server"]
    user   = s["user"]
    pwd    = s["pass"]

    print(f"Adding {name} -> {server}...")
    # Add PPTP VPN connection
    add_cmd = (
        f'Add-VpnConnection -Name "{name}" -ServerAddress "{server}" '
        f'-TunnelType Pptp -AuthenticationMethod MSChapv2 '
        f'-Force -PassThru'
    )
    r = subprocess.run(['powershell', '-Command', add_cmd], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        print(f"  Failed to add: {r.stderr[:100]}")
        continue

    print(f"  Connecting to {name}...")
    connect_cmd = (
        f'$cred = New-Object PSCredential("{user}", (ConvertTo-SecureString "{pwd}" -AsPlainText -Force)); '
        f'rasdial "{name}" {user} {pwd}'
    )
    r2 = subprocess.run(
        ['powershell', '-Command', f'rasdial "{name}" {user} {pwd}'],
        capture_output=True, text=True, timeout=30
    )
    print(f"  Connect result: {r2.returncode} | {r2.stdout.strip()[:100]}")

    if r2.returncode == 0 or "connected" in r2.stdout.lower():
        print(f"\nCONNECTED via {name}!")
        time.sleep(3)

        # Test connectivity
        import socket
        connected_server = ""
        for host, port in [("mt5.metaquotes.net", 443), ("testnet.binance.vision", 443)]:
            try:
                socket.create_connection((host, port), timeout=5)
                print(f"  {host}:{port} -> NOW REACHABLE!")
                connected_server = name
            except:
                print(f"  {host}:{port} -> still blocked")

        if connected_server:
            print(f"\nVPN {name} is working!")
            break
        else:
            print("  VPN connected but servers still blocked, trying next...")
            subprocess.run(['rasdial', name, '/disconnect'], capture_output=True)
    else:
        print(f"  Connection failed")

print("\nVPN setup complete. Run scripts/test_axi_terminal.py to test MT5.")
