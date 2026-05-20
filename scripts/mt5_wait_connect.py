"""Wait for MT5 to fully load and connect."""
import MetaTrader5 as mt5
import time, sys

print("Esperando que MT5 se conecte al servidor...")
for attempt in range(20):  # try for up to 100 seconds
    time.sleep(5)
    try:
        mt5.shutdown()
    except:
        pass

    ok = mt5.initialize()
    err = mt5.last_error()
    ram_msg = ""

    try:
        import psutil
        for proc in psutil.process_iter(['name', 'memory_info']):
            if 'terminal64' in proc.info['name'].lower():
                ram_mb = proc.info['memory_info'].rss // (1024*1024)
                ram_msg = f" | MT5 RAM:{ram_mb}MB"
    except:
        pass

    print(f"[{attempt+1}/20] ok={ok} err={err}{ram_msg}")

    if ok:
        i = mt5.account_info()
        print(f"\n✅ CONECTADO!")
        print(f"  Login:   {i.login}")
        print(f"  Balance: {i.balance} {i.currency}")
        print(f"  Server:  {i.server}")

        # Test EURUSD data
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 3)
        if rates is not None:
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"  EURUSD ultimo close: {df['close'].iloc[-1]:.5f}")

        # Test XAUUSD
        rates2 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 1)
        if rates2 is not None:
            df2 = pd.DataFrame(rates2)
            print(f"  XAUUSD ultimo close: {df2['close'].iloc[-1]:.2f}")

        mt5.shutdown()
        sys.exit(0)

print("\n❌ MT5 no conectó en 100 segundos")
print("Acciones posibles:")
print("  1. Abrir MT5 manualmente y loguearte")
print("  2. Verificar que MT5 muestre cotizaciones (no pantalla de login)")
