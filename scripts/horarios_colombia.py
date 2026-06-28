"""Show trading schedule in Colombia time (UTC-5)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

COLOMBIA_OFFSET = -5  # UTC-5

print("=" * 55)
print("HORARIOS DEL BOT EN HORA COLOMBIA (UTC-5)")
print("=" * 55)
print()

dead_utc = {22, 23, 0, 1, 2, 3, 4, 5, 6}

print("HORA COLOMBIA  |  HORA UTC  |  ESTADO")
print("-" * 55)
for h_col in range(0, 24):
    h_utc = (h_col + 5) % 24  # Colombia = UTC - 5, so UTC = Colombia + 5
    estado = "BLOQUEADO (hora muerta)" if h_utc in dead_utc else "ACTIVO - puede operar"
    marker = " <-- TU AHORA (7 PM)" if h_col == 19 else ""
    print(f"  {h_col:02d}:00       |   {h_utc:02d}:00 UTC  |  {estado}{marker}")

print()
print("SESIONES DE MERCADO (hora Colombia):")
print("  Tokyo/Asia: 20:00 - 05:00 (USDJPY, AUDUSD, GBPJPY)")
print("  Londres:    02:00 - 11:00 (EURUSD, GBPUSD, XAUUSD)")
print("  Nueva York: 08:30 - 17:00 (todo, MEJOR momento)")
print("  Overlap L+NY: 08:30-11:00 (MAXIMA liquidez)")
print()
print("PROBLEMA DETECTADO:")
print("  Bot bloqueado desde 17:00 Colombia hasta 01:00 Colombia")
print("  Eso incluye TU horario de 7 PM (00:00 UTC)")
print()
print("SOLUCION: abrir sesion Asia para JPY/AUD")
print("  USDJPY y AUDUSD son activos a las 7 PM Colombia (22:00 UTC)")
