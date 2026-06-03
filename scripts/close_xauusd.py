"""Close broken XAUUSD SELL #60015449 — TP only 0.48 pts, SL 233 pts."""
import sys; sys.path.insert(0, ".")
import MetaTrader5 as mt5

mt5.initialize()

TARGET_TICKET = 60015449
positions = mt5.positions_get(ticket=TARGET_TICKET) or []
if not positions:
    print(f"Posicion #{TARGET_TICKET} no encontrada — ya cerrada o ticket incorrecto")
    # Try finding any XAUUSD SELL with broken TP
    all_pos = mt5.positions_get() or []
    for p in all_pos:
        if p.symbol == "XAUUSD" and p.type == 1:
            tp_dist = abs(p.tp - p.price_open) if p.tp > 0 else 999
            print(f"  #{p.ticket} XAUUSD SELL profit={p.profit:+.2f} tp_dist={tp_dist:.3f}")
    mt5.shutdown()
    raise SystemExit(0)

p = positions[0]
tick = mt5.symbol_info_tick(p.symbol)
close_price = tick.ask  # closing a SELL = BUY at ask

req = {
    "action":       mt5.TRADE_ACTION_DEAL,
    "symbol":       p.symbol,
    "volume":       p.volume,
    "type":         mt5.ORDER_TYPE_BUY,
    "position":     p.ticket,
    "price":        close_price,
    "deviation":    50,
    "comment":      "Close broken TP",
    "type_time":    mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
result = mt5.order_send(req)
if result and result.retcode == mt5.TRADE_RETCODE_DONE:
    print(f"CERRADA #{p.ticket} XAUUSD SELL | profit realizado: {p.profit:+.2f} USD")
else:
    code = result.retcode if result else "None"
    comment = result.comment if result else mt5.last_error()
    print(f"FALLO: retcode={code} | {comment}")

acc = mt5.account_info()
print(f"\nBalance actual: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}")
mt5.shutdown()
