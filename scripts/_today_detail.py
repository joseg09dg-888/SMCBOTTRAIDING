import MetaTrader5 as mt5, os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

load_dotenv()
mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
history = mt5.history_deals_get(today_start, datetime.now(timezone.utc))

print("=== HOY (trades cerrados) ===")
total = 0.0
for d in history:
    if d.profit == 0 and d.commission == 0:
        continue
    dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
    pnl = d.profit + d.commission
    total += pnl
    comment = d.comment or "(sin comentario = manual)"
    result = "WIN" if pnl > 0 else "LOSS"
    print(f"{dt.strftime('%H:%M')} | {d.symbol:8s} | {result} | ${pnl:+.2f} | {comment}")

print(f"\nTotal hoy cerrado: ${total:.2f}")
mt5.shutdown()
