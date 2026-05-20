"""Try to connect to MT5 immediately after startup, before broker auth."""
import MetaTrader5 as mt5
import subprocess, time, os, sys

print("Cerrando MT5 previo...")
subprocess.run(['taskkill', '/F', '/IM', 'terminal64.exe'], capture_output=True)
time.sleep(2)

print("Arrancando MT5...")
subprocess.Popen(r'C:\Program Files\MetaTrader 5\terminal64.exe')

print("Intentando conectar en los primeros 15 segundos (antes de auth servidor)...")
for i in range(15):
    time.sleep(1)
    mt5.shutdown()
    ok = mt5.initialize()
    err = mt5.last_error()

    # Check RAM
    try:
        import psutil
        for p in psutil.process_iter(['name','memory_info']):
            if 'terminal64' in p.info['name'].lower():
                ram = p.info['memory_info'].rss // (1024*1024)
                print(f"  t={i+1}s ok={ok} err={err} RAM={ram}MB")
                break
    except:
        print(f"  t={i+1}s ok={ok} err={err}")

    if ok:
        i = mt5.account_info()
        print(f"\nCONECTADO! Login:{i.login} Balance:{i.balance} {i.currency}")
        r = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
        print(f"EURUSD data: {r is not None}")
        mt5.shutdown()
        sys.exit(0)

print("\nNo conectó en 15 segundos")
print("Error final:", mt5.last_error())
