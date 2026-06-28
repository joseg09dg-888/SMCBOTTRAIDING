import MetaTrader5 as mt5, os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from collections import defaultdict
load_dotenv()
mt5.initialize()
mt5.login(int(os.getenv("MT5_LOGIN")), os.getenv("MT5_PASSWORD"), os.getenv("MT5_SERVER"))
account = mt5.account_info()
print(f"Balance: {account.balance:.2f} | Drawdown: {((100000-account.balance)/100000)*100:.2f}%")
from_dt = datetime.now(timezone.utc) - timedelta(days=10)
deals = mt5.history_deals_get(from_dt, datetime.now(timezone.utc))
by_day = defaultdict(lambda: {"w":0,"l":0,"pnl":0.0,"syms":set()})
if deals:
    for d in deals:
        if d.profit == 0 or d.type not in (0,1): continue
        day = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime("%Y-%m-%d")
        if d.profit > 0: by_day[day]["w"] += 1
        else: by_day[day]["l"] += 1
        by_day[day]["pnl"] += d.profit
        by_day[day]["syms"].add(d.symbol)
print("DIA        | W  |  L  |  WR% | NET PNL  | PARES")
print("-"*65)
for day in sorted(by_day.keys(), reverse=True):
    d = by_day[day]
    total = d["w"]+d["l"]
    wr = d["w"]/total*100 if total else 0
    syms = ",".join(sorted(d["syms"]))
    print(f"{day} | {d['w']:2d} | {d['l']:3d} | {wr:4.0f}% | {d['pnl']:+8.2f} | {syms}")
mt5.shutdown()
