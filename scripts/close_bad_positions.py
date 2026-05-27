"""Close duplicate USDJPY + broken XAUUSD positions."""
import sys, os
sys.path.insert(0, ".")
import MetaTrader5 as mt5

mt5.initialize()
positions = mt5.positions_get() or []
print(f"Posiciones abiertas: {len(positions)}")

# Group by symbol
by_symbol = {}
for p in positions:
    by_symbol.setdefault(p.symbol, []).append(p)

# Close logic
to_close = []

# USDJPY: keep best (highest profit), close rest
if "USDJPY" in by_symbol:
    usdjpy = sorted(by_symbol["USDJPY"], key=lambda p: p.profit, reverse=True)
    keep = usdjpy[0]
    print(f"USDJPY: keep #{keep.ticket} profit={keep.profit:+.2f}")
    for p in usdjpy[1:]:
        to_close.append(p)
        print(f"USDJPY: close #{p.ticket} profit={p.profit:+.2f}")

# XAUUSD: TP broken (4483.060 vs entry 4483.540 = 0.48pts) — close it
if "XAUUSD" in by_symbol:
    for p in by_symbol["XAUUSD"]:
        tp_dist = abs(p.tp - p.price_open) if p.tp > 0 else 999
        print(f"XAUUSD: #{p.ticket} profit={p.profit:+.2f} tp_dist={tp_dist:.3f} sl_dist={abs(p.sl - p.price_open):.3f}")
        if tp_dist < 1.0:  # TP within 1 point — broken
            to_close.append(p)
            print(f"  -> CLOSING (TP broken: only {tp_dist:.3f} pts away)")

print(f"\nCerrando {len(to_close)} posiciones...")
for p in to_close:
    close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(p.symbol)
    close_price = tick.bid if p.type == 0 else tick.ask

    req = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    p.symbol,
        "volume":    p.volume,
        "type":      close_type,
        "position":  p.ticket,
        "price":     close_price,
        "deviation": 50,
        "comment":   "Audit close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  CLOSED #{p.ticket} {p.symbol} profit was {p.profit:+.2f}")
    else:
        code = result.retcode if result else "None"
        comment = result.comment if result else mt5.last_error()
        print(f"  FAIL #{p.ticket}: retcode={code} {comment}")

# Final state
positions2 = mt5.positions_get() or []
acc = mt5.account_info()
print(f"\nEstado final:")
print(f"  Posiciones: {len(positions2)}")
print(f"  Balance: ${acc.balance:,.2f}")
print(f"  Equity:  ${acc.equity:,.2f}")
for p in positions2:
    side = "BUY" if p.type == 0 else "SELL"
    print(f"  #{p.ticket} {p.symbol} {side} profit={p.profit:+.2f}")

mt5.shutdown()
