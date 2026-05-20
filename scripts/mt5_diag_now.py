import MetaTrader5 as mt5, time
print("Intentando conectar a MT5...")
for i in range(3):
    result = mt5.initialize()
    error = mt5.last_error()
    print(f"Intento {i+1}: {result} | Error: {error}")
    if result:
        info = mt5.account_info()
        print(f"Login: {info.login}")
        print(f"Server: {info.server}")
        print(f"Balance: {info.balance}")
        mt5.shutdown()
        break
    time.sleep(3)
