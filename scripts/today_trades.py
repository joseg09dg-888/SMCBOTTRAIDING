"""Show today's MT5 trades and current positions."""
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timezone

mt5.initialize()

now       = datetime.now(timezone.utc)
from_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
to_date   = now

deals = mt5.history_deals_get(from_date, to_date)
entries = [d for d in (deals or []) if d.entry == 0 and d.type in (0, 1)]
closes  = [d for d in (deals or []) if d.entry == 1 and d.profit != 0]

print("=" * 50)
print("TRADES MT5 HOY (entradas)")
print("=" * 50)
for d in sorted(entries, key=lambda x: x.time):
    t  = datetime.fromtimestamp(d.time, tz=timezone.utc)
    dr = "BUY" if d.type == 0 else "SELL"
    print(f"  {t.strftime('%H:%M UTC')}  {d.symbol}  {dr}  {d.volume}L @ {d.price:.4f}")

print()
print("TRADES MT5 HOY (cierres)")
for d in sorted(closes, key=lambda x: x.time):
    t   = datetime.fromtimestamp(d.time, tz=timezone.utc)
    res = "WIN" if d.profit > 0 else "LOSS"
    print(f"  {t.strftime('%H:%M UTC')}  {d.symbol}  {res}  P&L=${d.profit:+.2f}")

wins  = sum(1 for d in closes if d.profit > 0)
loss  = sum(1 for d in closes if d.profit <= 0)
net   = sum(d.profit for d in closes)
print(f"\n  RESUMEN: {wins}W / {loss}L  |  Neto hoy: ${net:+.2f}")

print()
print("POSICIONES ABIERTAS AHORA")
pos = mt5.positions_get()
if pos:
    for p in pos:
        dr = "BUY" if p.type == 0 else "SELL"
        estado = "GANANDO" if p.profit >= 0 else "PERDIENDO"
        print(f"  {p.symbol} {dr} {p.volume}L  PnL=${p.profit:+.2f} ({estado})")
        print(f"    Entry={p.price_open:.4f}  SL={p.sl:.4f}  TP={p.tp:.4f}")
else:
    print("  Sin posiciones abiertas")

info = mt5.account_info()
if info:
    net_total = info.balance - 100_000
    print(f"\nBalance: ${info.balance:,.2f}  |  Neto total: ${net_total:+.2f} ({net_total/100_000*100:+.3f}%)")

mt5.shutdown()
