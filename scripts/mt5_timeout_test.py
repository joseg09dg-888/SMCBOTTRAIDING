"""Test MT5 with extended timeout and all combinations."""
import MetaTrader5 as mt5
import time

print("Testing with timeout=60000ms (60 seconds)...")

# Method 1: with server + long timeout
mt5.shutdown()
time.sleep(1)
ok = mt5.initialize(login=8889, password='IMSMCbot',
                    server='BrokerGroup-Live24', timeout=60000)
print(f"BrokerGroup-Live24 (60s timeout): ok={ok} err={mt5.last_error()}")
if ok:
    i = mt5.account_info()
    print(f"CONECTADO! Login:{i.login} Balance:{i.balance} {i.currency} Server:{i.server}")
    r = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
    print(f"EURUSD data: {r is not None}")
    mt5.shutdown()
    exit(0)

# Method 2: no server, long timeout
mt5.shutdown()
time.sleep(1)
ok2 = mt5.initialize(timeout=60000)
print(f"No server (60s timeout): ok={ok2} err={mt5.last_error()}")
if ok2:
    i = mt5.account_info()
    print(f"CONECTADO! Login:{i.login} Balance:{i.balance}")
    mt5.shutdown()
    exit(0)

print("\nAun sin conectar — verifica en MT5 que hay cotizaciones en vivo")
print("Si MT5 muestra SOLO una barra 'Connecting...' — el servidor esta caido")
