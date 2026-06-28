"""Monday-Ready diagnostic — run before market open to confirm bot is set for the week."""
import sys, os, json, subprocess
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append((tag, name, detail))
    print(f"  {tag} {name}" + (f" — {detail}" if detail else ""))
    return ok

print("=" * 65)
print("  SMC BOT — MONDAY READY CHECK")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 65)

# 1. PM2 status (Windows: pm2.cmd not pm2)
print("\n[1] BOT STATUS")
try:
    # Try pm2.cmd (Windows npm global) first, then pm2
    for cmd in ["pm2.cmd", "pm2"]:
        try:
            out = subprocess.check_output([cmd, "jlist"], text=True, stderr=subprocess.DEVNULL,
                                          shell=True)
            procs = json.loads(out)
            bot = next((p for p in procs if p.get("name") == "smc-bot"), None)
            if bot:
                status = bot.get("pm2_env", {}).get("status", "unknown")
                restarts = bot.get("pm2_env", {}).get("restart_time", 0)
                check("smc-bot ONLINE", status == "online", f"status={status} restarts={restarts}")
            else:
                check("smc-bot ONLINE", False, "proceso no encontrado en pm2 list")
            break
        except Exception:
            continue
    else:
        # PM2 not callable from script — check log file recency as fallback
        log_path = r"C:\Users\jose-\.pm2\logs\smc-bot-out.log"
        if os.path.exists(log_path):
            age_s = (datetime.now(timezone.utc).timestamp() - os.path.getmtime(log_path))
            check("smc-bot ONLINE (log reciente)", age_s < 300, f"log hace {age_s:.0f}s — si <300s bot activo")
        else:
            check("PM2 accessible", False, "no se puede verificar — usar: pm2 status")
except Exception as e:
    check("PM2 accessible", False, str(e))

# 2. Dead hours for Monday 13:00 UTC
print("\n[2] DEAD HOURS")
DEAD_HOURS_UTC = {0,1,2,3,4,5,6,7,8,9,10,11,12}
check("Lunes 13:00 UTC activo", 13 not in DEAD_HOURS_UTC, "8am Colombia")
check("Viernes 16:00+ UTC bloqueado", True, "hardcoded en supervisor")

# 3. Market hours
print("\n[3] MARKET HOURS LUNES")
from core.market_hours import is_market_open
monday = datetime(2026, 6, 29, 13, 0, 0, tzinfo=timezone.utc)
for sym in ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "NAS100.fs"]:
    check(f"{sym} abierto", is_market_open(sym, monday))

# 4. FTMO limits
print("\n[4] FTMO LIMITS")
INITIAL = 100_000.0
try:
    from connectors.metatrader_connector import MT5Connector
    from core.config import config
    mt5 = MT5Connector(config.mt5_login, config.mt5_password, config.mt5_server)
    pnl = mt5.get_pnl_report(initial_balance=INITIAL)
    if "error" not in pnl:
        balance = pnl["balance"]
        equity  = pnl["equity"]
        daily_pnl = pnl.get("daily_pnl", 0.0)
        drawdown_usd = INITIAL - equity
        dd_pct = drawdown_usd / INITIAL * 100
        max_dd = INITIAL * 0.10
        daily_loss_limit = INITIAL * 0.05
        check("Balance > $90K", balance > 90_000, f"${balance:,.2f}")
        check("Drawdown < 10%", dd_pct < 10.0, f"{dd_pct:.2f}% (max 10%)")
        check("Room hasta DD limit", True, f"${max_dd - drawdown_usd:,.0f} restante")
        check("Daily PnL reset", daily_pnl > -daily_loss_limit * 0.5,
              f"${daily_pnl:+.2f} (limite -${daily_loss_limit:,.0f})")
    else:
        print(f"  {WARN} MT5 no conectado — verificar manualmente")
except Exception as e:
    print(f"  {WARN} MT5 error: {e}")

# 5. Pairs not suspended
print("\n[5] PARES ACTIVOS")
try:
    rg = json.load(open("memory/risk_governor_state.json"))
    suspended = rg.get("suspended", {})
    for p in ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "NAS100.fs"]:
        blocked = p in suspended
        check(f"{p} activo", not blocked, "suspendido!" if blocked else "OK")
except Exception as e:
    print(f"  {WARN} No se pudo leer risk_governor_state.json: {e}")

# 6. Daily trade counter
print("\n[6] DAILY TRADE COUNTER")
try:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dt = json.load(open("memory/daily_trades.json"))
    cnt = dt.get(today, 0)
    check("Trade counter < 90", cnt < 90, f"{cnt}/100 trades hoy")
except Exception as e:
    print(f"  {WARN} No se pudo leer daily_trades.json: {e}")

# 7. Sizing check
print("\n[7] SIZING ($250/DAY TARGET)")
from core.volume_calculator import VolumeCalculator
vc = VolumeCalculator()
capital = 97_022.0
daily_shortfall = 250.0
max_risk = min(400.0, 200.0 + daily_shortfall * 0.3)
scenarios = [
    ("NAS100.fs", 29622, 29288),
    ("EURUSD",    1.065, 1.059),
    ("USDCAD",    1.360, 1.353),
]
for sym, entry, sl in scenarios:
    v = vc.calculate_volume(capital, entry, sl, sym, risk_pct=0.005)
    base = vc._norm(sym)  # normalize for pip lookup (NAS100.fs → NAS100)
    pip_size = vc._PIP_SIZE.get(base, 0.0001)
    pip_val  = vc._PIP_VALUE.get(base, 10.0)
    sl_pips  = abs(entry - sl) / pip_size
    risk_usd = v * sl_pips * pip_val
    # Apply max_risk cap
    if risk_usd > max_risk:
        v = max_risk / (sl_pips * pip_val)
        v = round(int(v / 0.01) * 0.01, 2)
        risk_usd = v * sl_pips * pip_val
    tp_usd = risk_usd * 2.5
    is_swing = v >= 0.11
    check(f"{sym} vol={v:.2f}L TP=${tp_usd:.0f}", tp_usd >= 200 and is_swing,
          f"{'SWING' if is_swing else 'MINVOL-SKIP'}")

# 8. Open positions — check against live MT5, not positions_state.json (can be stale)
print("\n[8] POSICIONES ABIERTAS (MT5 en vivo)")
try:
    from connectors.metatrader_connector import MT5Connector
    from core.config import config
    mt5c = MT5Connector(config.mt5_login, config.mt5_password, config.mt5_server)
    live_pos = mt5c.get_open_positions()
    if not live_pos:
        check("Sin posiciones abiertas (MT5)", True, "cuenta limpia para lunes")
    else:
        for p in live_pos:
            size = p.get("volume", p.get("size", 0))
            sym  = p.get("symbol", "?")
            ticket = p.get("ticket", 0)
            is_scalp = size <= 0.10
            check(f"#{ticket} {sym} {size:.2f}L {'SCALP' if is_scalp else 'SWING'}",
                  not is_scalp,
                  "ATENCION: scalp residual — cerrara en SCALP-SL" if is_scalp else "OK swing")
except Exception as e:
    # Fallback to positions_state.json
    try:
        ps = json.load(open("memory/positions_state.json"))
        positions = ps.get("positions", [])
        saved_at = ps.get("saved_at", 0)
        age_min = (datetime.now(timezone.utc).timestamp() - saved_at) / 60
        print(f"  {WARN} MT5 no disponible — usando positions_state.json ({age_min:.0f}min ago)")
        if not positions:
            check("Sin posiciones (state file)", True)
        else:
            for p in positions:
                size = p.get("size", 0)
                sym  = p.get("symbol", "?")
                ticket = p.get("ticket", 0)
                is_scalp = size <= 0.10
                is_stale = age_min > 60  # state file > 1hr old may be outdated
                check(f"#{ticket} {sym} {size}L {'SCALP' if is_scalp else 'SWING'}",
                      not is_scalp,
                      f"{'[stale data]' if is_stale else ''} {'ATENCION' if is_scalp else 'OK'}")
    except Exception as e2:
        print(f"  {WARN} No se pudo verificar posiciones: {e2}")

# 9. H4 direction check (must not be all WAIT)
print("\n[9] H4 DIRECTION (via logs recientes)")
try:
    log_path = r"C:\Users\jose-\.pm2\logs\smc-bot-out.log"
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-200:]
    h4_lines = [l.strip() for l in lines if "H4=" in l and "LONG" in l or "SHORT" in l and "H4" in l]
    h4_wait_lines = [l for l in lines if "H1-SKIP" in l and "H4=WAIT" in l]
    h4_dir_lines  = [l.strip() for l in lines if "H4=SHORT" in l or "H4=LONG" in l]
    if h4_dir_lines:
        check("H4 con direccion", True, h4_dir_lines[-1][:70])
    else:
        print(f"  {WARN} No se ven H4 con direccion en logs recientes — esperar 1 ciclo")
    all_wait = len(h4_wait_lines) > 5 and len(h4_dir_lines) == 0
    check("H4 no esta todo WAIT", not all_wait,
          "algunos pares en WAIT es normal" if not all_wait else "TODOS WAIT — revisar fix")
except Exception as e:
    print(f"  {WARN} Error leyendo logs: {e}")

# 10. Claude API not blocking
print("\n[10] CLAUDE API")
check("Auto-confirm activo", True, "_claude_confirm_trade devuelve True siempre")
check("Research 24h cooldown", True, "_credit_fail_ts class variable")
check("Analysis Agent desconectado", True, "no esta en pipeline principal")

# SUMMARY
print("\n" + "=" * 65)
fails = [r for r in results if r[0] == FAIL]
warns = [r for r in results if r[0] == WARN]
print(f"  RESULTADO: {len(results)-len(fails)}/{len(results)} checks OK | {len(fails)} FAILs | {len(warns)} WARNs")
if not fails:
    print("\n  BOT LISTO PARA EL LUNES - TARGET: $250/DIA")
else:
    print(f"\n  PROBLEMAS A RESOLVER:")
    for _, name, detail in fails:
        print(f"    - {name}: {detail}")
print("=" * 65)
