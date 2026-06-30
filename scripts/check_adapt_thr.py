import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

mt5.initialize(login=10042896, password='IMSMCbot*3axi', server='Axi-US50-Demo')
desde = datetime.now(timezone.utc) - timedelta(days=14)
hasta = datetime.now(timezone.utc)
deals = mt5.history_deals_get(desde, hasta)
closed = [d for d in (deals or []) if d.type in (0,1) and d.entry==1 and d.volume > 0.10]
recent = sorted(closed, key=lambda d: d.time)[-10:]
print(f'Total deals cerrados (vol>0.1L, 14 dias): {len(closed)}')
print('Ultimos 10:')
for d in recent:
    t = datetime.fromtimestamp(d.time, tz=timezone.utc)
    status = "WIN" if d.profit > 0 else "LOSS"
    print(f'  {t.strftime("%m-%d %H:%M")} {d.symbol} vol={d.volume} profit={d.profit:.2f} [{status}]')
wins = sum(1 for d in recent if d.profit > 0)
print(f'Wins={wins}/{len(recent)} WR={wins/max(len(recent),1)*100:.0f}%')
mt5.shutdown()
