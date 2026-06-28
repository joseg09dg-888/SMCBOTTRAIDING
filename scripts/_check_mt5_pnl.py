import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("MT5 no disponible:", mt5.last_error())
    exit()

info = mt5.account_info()
print("Balance:", info.balance)
print("Equity:", info.equity)
print("Profit flotante:", info.profit)
print("Initial balance era ~99470")
print("P&L total estimado:", round(info.balance - 99470, 2))

from_date = datetime.now() - timedelta(days=7)
deals = mt5.history_deals_get(from_date, datetime.now())
if deals:
    print("\nDEALS ULTIMOS 7 DIAS:", len(deals))
    pnl_total = 0
    for d in deals:
        if d.profit != 0:
            pnl_total += d.profit
            dt = datetime.fromtimestamp(d.time)
            tipo = "BUY" if d.type == 0 else "SELL"
            print(f"  {dt.strftime('%m-%d %H:%M')} {d.symbol:10s} {tipo:4s} vol={d.volume} price={d.price} PNL={round(d.profit,2):+.2f} comment={d.comment}")
    print("TOTAL PNL 7d:", round(pnl_total, 2))
else:
    print("Sin deals en 7 dias")

positions = mt5.positions_get()
if positions:
    print("\nPOSICIONES ABIERTAS:", len(positions))
    for p in positions:
        tipo = "LONG" if p.type == 0 else "SHORT"
        print(f"  {p.symbol:10s} {tipo:5s} vol={p.volume} entry={p.price_open} curr={p.price_current} PNL={round(p.profit,2):+.2f} SL={p.sl} TP={p.tp} ticket={p.ticket}")
else:
    print("Sin posiciones abiertas")

mt5.shutdown()
