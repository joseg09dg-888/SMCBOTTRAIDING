"""Explain exactly why bot is or isn't trading right now."""
import sys, json
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
dead_hours = {22, 23, 0, 1, 2, 3, 4, 5, 6}

print("=" * 50)
print("POR QUE EL BOT NO OPERA AHORA")
print("=" * 50)
print()
print(f"Hora UTC actual: {now.strftime('%H:%M')} UTC")
print()

reasons = []

# 1. Dead hours check
if now.hour in dead_hours:
    minutes_to_7 = ((7 - now.hour) % 24) * 60 - now.minute
    reasons.append(f"HORA MUERTA ({now.hour}:00 UTC) — bloqueo 22:00-06:59 UTC")
    reasons.append(f"  -> Reanuda en {minutes_to_7} minutos a las 07:00 UTC")
else:
    print(f"[OK] Hora activa ({now.hour}:00 UTC — sesion London/NY)")

# 2. Daily limit
with open('memory/daily_trades.json') as f:
    dt = json.load(f)
today = now.strftime('%Y-%m-%d')
trades_today = dt.get(today, 0)

if trades_today >= 5:
    reasons.append(f"LIMITE DIARIO ({trades_today}/5) — reset en 00:00 UTC")
    tomorrow_reset = (24 - now.hour) * 60 - now.minute
    reasons.append(f"  -> Reset en {tomorrow_reset} minutos")
else:
    print(f"[OK] Trades hoy: {trades_today}/5 (quedan {5-trades_today} slots)")

# 3. Market hours
if now.weekday() == 5:  # Saturday
    reasons.append("MERCADO CERRADO (sabado)")
elif now.weekday() == 6:  # Sunday
    reasons.append("MERCADO CERRADO (domingo)")
else:
    print(f"[OK] Mercado abierto ({['Lun','Mar','Mie','Jue','Vie'][now.weekday()]})")

if reasons:
    print()
    print("RAZONES POR LAS QUE NO OPERA:")
    for r in reasons:
        print(f"  {r}")
    print()
    print("El bot SI esta buscando setups — solo esperando la hora correcta.")
    print("Es CORRECTO no operar de madrugada — spreads altos, liquidez baja.")
else:
    print()
    print("El bot PUEDE operar ahora — buscando setup con score >= 90")

print()
print("PROXIMAS HORAS ACTIVAS:")
print("  07:00 - 21:59 UTC = operaciones permitidas")
print("  22:00 - 06:59 UTC = bloqueado (madrugada, spreads malos)")
print()
print(f"Dias restantes del sprint 4-dias:")
print(f"  Mañana: hasta 5 trades")
print(f"  Pasado: hasta 5 trades")
print(f"  Total disponible: ~{(5-trades_today) + 5 + 5 + 5} trades mas")
