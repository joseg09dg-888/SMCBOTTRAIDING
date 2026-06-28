import MetaTrader5 as mt5, os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

load_dotenv()
mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

acc = mt5.account_info()
print(f"Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f} | Float: ${acc.profit:.2f}")
print(f"Perdida total desde $100K: ${acc.balance - 100000:.2f}")
print()

# Last 48 hours
from_dt = datetime.now(timezone.utc) - timedelta(hours=48)
history = mt5.history_deals_get(from_dt, datetime.now(timezone.utc))

print("=== ULTIMAS 48 HORAS (todos los deals con P&L) ===")
total = 0.0
for d in history:
    if d.profit == 0 and d.commission == 0 and d.swap == 0:
        continue
    dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
    pnl = d.profit + d.commission + d.swap
    total += pnl
    comment = d.comment or "(manual)"
    result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")
    print(f"{dt.strftime('%m-%d %H:%M')} | {d.symbol:8s} | {result} | ${pnl:+.2f} | {comment}")

print(f"\nTotal 48h: ${total:.2f}")
print()

# Open positions
positions = mt5.positions_get()
print("=== POSICIONES ABIERTAS ===")
if positions:
    for p in positions:
        side = 'BUY' if p.type == 0 else 'SELL'
        print(f"#{p.ticket} {p.symbol} {side} {p.volume}L | entry={p.price_open:.5f} | now={p.price_current:.5f} | float=${p.profit:.2f} | SL={p.sl:.5f} | TP={p.tp:.5f}")
else:
    print("Sin posiciones abiertas")

mt5.shutdown()
