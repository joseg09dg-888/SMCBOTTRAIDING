import MetaTrader5 as mt5, os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

load_dotenv()
mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

acc = mt5.account_info()
print(f"Balance: ${acc.balance:.2f} | Perdida total cerrada: ${acc.balance - 100000:.2f}")
print(f"Equity ahora: ${acc.equity:.2f} | Float: ${acc.profit:.2f}")
print()

from_dt = datetime.now(timezone.utc) - timedelta(days=7)
history = mt5.history_deals_get(from_dt, datetime.now(timezone.utc))

wins, losses, total_pnl = 0, 0, 0.0
today_pnl = 0.0
today = datetime.now(timezone.utc).date()

for d in history:
    if d.profit == 0:
        continue
    total_pnl += d.profit
    dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
    if dt.date() == today:
        today_pnl += d.profit
    if d.profit > 0:
        wins += 1
    else:
        losses += 1

print(f"7 dias: {wins}W / {losses}L | P&L cerrado: ${total_pnl:.2f}")
print(f"Hoy cerrado: ${today_pnl:.2f}")
print()

# posiciones abiertas
positions = mt5.positions_get()
if positions:
    for p in positions:
        side = 'BUY' if p.type == 0 else 'SELL'
        print(f"ABIERTA: {p.symbol} {side} | P&L float: ${p.profit:.2f} | SL={p.sl:.1f} | TP={p.tp:.1f}")
else:
    print("Sin posiciones abiertas")

mt5.shutdown()
