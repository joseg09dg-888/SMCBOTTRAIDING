import sys
print("1. importando MetaTrader5...", flush=True)
import MetaTrader5 as mt5
print(f"2. version paquete: {mt5.__version__}", flush=True)
print("3. llamando mt5.initialize() sin argumentos (attach a sesion activa)...", flush=True)
ok = mt5.initialize(timeout=10000)
print(f"4. resultado initialize(): {ok}", flush=True)
print(f"5. last_error: {mt5.last_error()}", flush=True)
if ok:
    acc = mt5.account_info()
    print(f"6. account_info: {acc}", flush=True)
    ti = mt5.terminal_info()
    print(f"7. terminal_info: {ti}", flush=True)
mt5.shutdown()
print("8. shutdown ok", flush=True)
