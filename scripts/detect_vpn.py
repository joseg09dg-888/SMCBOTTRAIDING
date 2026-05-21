"""Detect installed VPNs and try to connect."""
import subprocess, os, glob, sys

print("=== VPN DETECTION ===")

# 1. Check common VPN paths
vpn_paths = [
    r"C:\Program Files\NordVPN",
    r"C:\Program Files\ExpressVPN",
    r"C:\Program Files\Proton\VPN",
    r"C:\Program Files\ProtonVPN",
    r"C:\Program Files\Windscribe",
    r"C:\Program Files\Mullvad VPN",
    r"C:\Program Files\OpenVPN",
    r"C:\Program Files\Surfshark",
    r"C:\Program Files\CyberGhost VPN",
    r"C:\Program Files\PureVPN",
    r"C:\Program Files (x86)\NordVPN",
    r"C:\Program Files (x86)\ExpressVPN",
]
found_vpns = []
for path in vpn_paths:
    if os.path.exists(path):
        exes = glob.glob(os.path.join(path, "**", "*.exe"), recursive=True)
        print(f"  FOUND: {path}")
        for e in exes[:3]: print(f"    {e}")
        found_vpns.append(path)

if not found_vpns:
    print("  No VPN in standard paths")

# 2. Check Windows registry for any VPN software
print("\n=== Registry search (VPN-related) ===")
try:
    result = subprocess.run(
        ['reg', 'query',
         r'HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
         '/s', '/f', 'vpn'],
        capture_output=True, text=True, timeout=10
    )
    lines = [l for l in result.stdout.split('\n') if 'DisplayName' in l or 'InstallLocation' in l]
    for l in lines[:20]:
        print(f"  {l.strip()}")
except Exception as e:
    print(f"  Registry search error: {e}")

# 3. Check running VPN processes
print("\n=== VPN processes running ===")
try:
    result = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=5)
    vpn_procs = [l for l in result.stdout.split('\n')
                 if any(v in l.lower() for v in ['vpn', 'nord', 'express', 'proton', 'windscribe', 'mullvad', 'wireguard', 'openvpn'])]
    if vpn_procs:
        for p in vpn_procs: print(f"  {p.strip()}")
    else:
        print("  No VPN processes found")
except Exception as e:
    print(f"  Process check error: {e}")

# 4. Check Windows VPN adapters
print("\n=== VPN network adapters ===")
try:
    result = subprocess.run(
        ['powershell', '-Command',
         'Get-NetAdapter | Where-Object { $_.InterfaceDescription -match "VPN|TAP|TUN|WireGuard|OpenVPN|NordVPN|ExpressVPN" } | Select-Object Name,InterfaceDescription,Status'],
        capture_output=True, text=True, timeout=10
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    else:
        print("  No VPN adapters found")
except Exception as e:
    print(f"  Adapter check error: {e}")

# 5. Check Windows built-in VPN connections
print("\n=== Windows VPN connections ===")
try:
    result = subprocess.run(
        ['powershell', '-Command',
         'Get-VpnConnection 2>$null | Select-Object Name,ServerAddress,ConnectionStatus'],
        capture_output=True, text=True, timeout=10
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    else:
        print("  No Windows VPN connections configured")
except Exception as e:
    print(f"  VPN connection check: {e}")

print(f"\n=== SUMMARY ===")
print(f"VPN installations found: {len(found_vpns)}")
for v in found_vpns:
    print(f"  {v}")
