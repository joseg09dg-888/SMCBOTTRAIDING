from datetime import datetime, timezone
from core.market_hours import is_market_open, minutes_until_open

now = datetime.now(timezone.utc)
print(f"UTC: {now.strftime('%Y-%m-%d %H:%M')} weekday={now.weekday()}")
for s in ['EURUSD','GBPUSD','AUDUSD','USDCAD','NAS100']:
    abierto = is_market_open(s, now)
    mins = minutes_until_open(s, now)
    estado = "ABIERTO" if abierto else f"CERRADO (abre en {mins}min)"
    print(f"  {s}: {estado}")
