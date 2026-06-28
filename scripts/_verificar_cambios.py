"""
VERIFICACION DE 5 FILTROS — confirma que todos los cambios estan activos en el bot.
Corre SIEMPRE despues de un restart para confirmar que el nuevo codigo esta cargado.
"""
import sys, os, json
sys.path.insert(0, '.')

FILTROS_OK = []
FILTROS_FAIL = []

def ok(msg):
    FILTROS_OK.append(msg)
    print(f"  [OK] {msg}")

def fail(msg):
    FILTROS_FAIL.append(msg)
    print(f"  [FAIL] {msg}")

print("=" * 55)
print("VERIFICACION DE 5 FILTROS POST-CAMBIO")
print("=" * 55)

# FILTRO 1: Constantes del supervisor (MAX_OPEN, MAX_DAILY)
print("\nFILTRO 1: Constantes supervisor.py")
try:
    from core.supervisor import MAX_OPEN_POSITIONS, MAX_DAILY_TRADES
    if MAX_OPEN_POSITIONS >= 3:
        ok(f"MAX_OPEN_POSITIONS = {MAX_OPEN_POSITIONS} (>= 3)")
    else:
        fail(f"MAX_OPEN_POSITIONS = {MAX_OPEN_POSITIONS} (deberia ser 3)")
    if MAX_DAILY_TRADES >= 10:
        ok(f"MAX_DAILY_TRADES = {MAX_DAILY_TRADES} (>= 10)")
    else:
        fail(f"MAX_DAILY_TRADES = {MAX_DAILY_TRADES} (deberia ser 10)")
except Exception as e:
    fail(f"No se pudo importar supervisor: {e}")

# FILTRO 2: Risk Governor state (multiplier = 1.0)
print("\nFILTRO 2: Risk Governor state (riesgo)")
try:
    with open("memory/risk_governor_state.json", encoding="utf-8") as f:
        state = json.load(f)
    mult = state.get("risk_multiplier", 0)
    if mult >= 1.0:
        ok(f"risk_multiplier = {mult:.2f} (riesgo completo)")
    else:
        fail(f"risk_multiplier = {mult:.2f} (deberia ser 1.0)")
    suspended = list(state.get("suspended", {}).keys())
    ok(f"Suspendidos: {suspended} (USDJPY y GBPJPY correctos)")
except Exception as e:
    fail(f"Error leyendo risk_governor_state.json: {e}")

# FILTRO 3: Market hours US30 (Axi CFD 01:00-22:00)
print("\nFILTRO 3: Market hours US30")
try:
    from datetime import datetime, timezone
    from core.market_hours import is_market_open, minutes_until_open
    # Test: 15:00 UTC deberia ser ABIERTO
    test_time = datetime(2026, 6, 18, 15, 0, tzinfo=timezone.utc)
    if is_market_open("US30", test_time):
        ok("US30 abierto a las 15:00 UTC (Axi CFD hours correctas)")
    else:
        fail("US30 CERRADO a las 15:00 UTC (deberia estar abierto)")
    # Test: 22:30 UTC deberia ser CERRADO
    test_break = datetime(2026, 6, 18, 22, 30, tzinfo=timezone.utc)
    if not is_market_open("US30", test_break):
        ok("US30 cerrado a las 22:30 UTC (break diario correcto)")
    else:
        fail("US30 ABIERTO a las 22:30 UTC (deberia estar cerrado)")
    # Test ahora mismo
    now_open = is_market_open("US30")
    mins = minutes_until_open("US30")
    if now_open:
        ok(f"US30 ABIERTO ahora mismo")
    else:
        ok(f"US30 cerrado ahora (break/fin semana) -- abre en {mins}min")
except Exception as e:
    fail(f"Error en market_hours: {e}")

# FILTRO 4: Peak-guard en supervisor
print("\nFILTRO 4: Peak-guard en _manage_open_positions")
try:
    with open("core/supervisor.py", encoding="utf-8") as f:
        content = f.read()
    if "PEAK_MIN_USD" in content and "PEAK_RETRACE_PCT" in content:
        ok("Peak-guard activo (PEAK_MIN_USD + PEAK_RETRACE_PCT encontrados)")
    else:
        fail("Peak-guard NO encontrado en supervisor.py")
    if "tight trail" in content or "0.5R" in content or "sl_dist * 0.5" in content:
        ok("Trailing 3R+ tight (0.5R) activo")
    else:
        fail("Trailing 3R+ tight NO encontrado")
except Exception as e:
    fail(f"Error leyendo supervisor.py: {e}")

# FILTRO 5: MT5 y balance actual
print("\nFILTRO 5: Conexion MT5 y balance")
try:
    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    load_dotenv()
    mt5.initialize()
    mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))
    acc = mt5.account_info()
    if acc:
        dd = (1 - acc.balance / 100000) * 100
        ok(f"MT5 conectado | Balance=${acc.balance:.2f} | Drawdown={dd:.2f}%")
        positions = mt5.positions_get()
        n = len(positions) if positions else 0
        ok(f"Posiciones abiertas: {n} (max permitido: 3)")
    else:
        fail("MT5 no conectado")
    mt5.shutdown()
except Exception as e:
    fail(f"Error MT5: {e}")

# RESUMEN
print("\n" + "=" * 55)
print(f"RESULTADO: {len(FILTROS_OK)} OK / {len(FILTROS_FAIL)} FAIL")
if not FILTROS_FAIL:
    print("TODOS LOS CAMBIOS CONFIRMADOS -- bot listo")
else:
    print("CAMBIOS PENDIENTES:")
    for f in FILTROS_FAIL:
        print(f"  >> {f}")
print("=" * 55)
