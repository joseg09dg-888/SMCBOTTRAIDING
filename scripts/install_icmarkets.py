import subprocess, urllib.request, os, time

path = r"C:\Users\jose-\Downloads\icmarkets_mt5.exe"

# Try multiple URLs with browser User-Agent
urls = [
    "https://download.mql5.com/cdn/web/ic.markets.sc.ltd/mt5/icmarketssc5setup.exe",
    "https://download.mql5.com/cdn/web/ic.markets.eu.ltd/mt5/icmarketseu5setup.exe",
    "https://download.mql5.com/cdn/web/ic.markets.au.pty.ltd/mt5/icmarketsau5setup.exe",
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.mql5.com/",
}

downloaded = False
for url in urls:
    try:
        print(f"Descargando: {url}")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            downloaded_bytes = 0
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded_bytes += len(chunk)
            print(f"Descargado: {downloaded_bytes//1024//1024} MB -> {path}")
        downloaded = True
        break
    except Exception as e:
        print(f"  Fallo: {e}")

if not downloaded:
    print("Intentando con requests...")
    import subprocess as sp
    sp.run(["pip", "install", "requests", "-q"], check=False)
    import requests
    for url in urls:
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                print(f"Descargado via requests: {os.path.getsize(path)//1024//1024} MB")
                downloaded = True
                break
        except Exception as e:
            print(f"  requests fallo: {e}")

if not downloaded:
    print("ERROR: No se pudo descargar")
    exit(1)

sz = os.path.getsize(path)
print(f"Archivo: {path} ({sz//1024//1024} MB)")

print("Instalando con /auto...")
result = subprocess.run([path, "/auto"], capture_output=False)
print(f"Exit code: {result.returncode}")

time.sleep(8)
import glob
found = glob.glob(r"C:\Program Files*\**\terminal64.exe", recursive=True)
print(f"MT5 instalaciones encontradas: {found}")
