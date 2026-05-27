"""Close any open XAUUSD SELL position immediately."""
import sys; sys.path.insert(0, ".")
import MetaTrader5 as mt5

mt5.initialize()
acc_before = mt5.account_info()
print(f"Balance antes: ${acc_before.balance:,.2f} | Equity: ${acc_before.equity:,.2f}")

positions = mt5.positions_get() or []
xau_sells = [p for p in positions if p.symbol == "XAUUSD" and p.type == 1]

if not xau_sells:
    print("No hay XAUUSD SELL abierta.")
else:
    for p in xau_sells:
        print(f"Cerrando #{p.ticket} XAUUSD SELL {p.volume}L @{p.price_open:.2f} | profit={p.profit:+.2f}")
        tick = mt5.symbol_info_tick("XAUUSD")
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       "XAUUSD",
            "volume":       p.volume,
            "type":         mt5.ORDER_TYPE_BUY,
            "position":     p.ticket,
            "price":        tick.ask,
            "deviation":    100,
            "comment":      "Close bad signal",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  CERRADA. Perdida realizada: ${p.profit:+.2f}")
        else:
            code = result.retcode if result else "?"
            print(f"  FALLO retcode={code} {result.comment if result else mt5.last_error()}")

acc_after = mt5.account_info()
print(f"\nBalance ahora: ${acc_after.balance:,.2f} | Equity: ${acc_after.equity:,.2f}")

remaining = mt5.positions_get() or []
print(f"Posiciones restantes: {len(remaining)}")
for p in remaining:
    side = "BUY" if p.type == 0 else "SELL"
    print(f"  #{p.ticket} {p.symbol} {side} {p.volume}L profit={p.profit:+.2f}")

mt5.shutdown()
