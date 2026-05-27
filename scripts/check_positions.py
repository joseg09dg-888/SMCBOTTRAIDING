import sys; sys.path.insert(0, ".")
import MetaTrader5 as mt5

mt5.initialize()
acc = mt5.account_info()
print(f"Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f} | Profit abierto: ${acc.profit:+.2f}")
positions = mt5.positions_get() or []
print(f"Posiciones abiertas: {len(positions)}")
for p in positions:
    side = "BUY" if p.type == 0 else "SELL"
    tp_dist = abs(p.tp - p.price_open) if p.tp > 0 else 999
    sl_dist = abs(p.sl - p.price_open) if p.sl > 0 else 999
    print(f"  #{p.ticket} {p.symbol} {side} {p.volume}L @{p.price_open:.3f} profit={p.profit:+.2f} TP_dist={tp_dist:.3f} SL_dist={sl_dist:.3f}")
mt5.shutdown()
