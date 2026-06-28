"""
Monitor de sesion en vivo — para usar con /loop durante trading.

Uso: .venv\Scripts\python scripts/_monitor_session.py
Muestra: posiciones, P&L, progreso a $250, alertas criticas.
"""
import os, sys, json, subprocess, datetime as _dt
from dotenv import load_dotenv
load_dotenv()

SESSION_START_HOUR = 13   # UTC
DAILY_TARGET       = 250  # USD
DANGER_LOSS        = -500 # USD — alerta roja
MAX_DD_PCT         = 0.05 # 5% diario FTMO

now  = _dt.datetime.now(_dt.timezone.utc)
hora = now.strftime("%H:%M UTC")

def _bar(pct: float, width: int = 20) -> str:
    filled = int(min(max(pct, 0), 1) * width)
    return "█" * filled + "░" * (width - filled)

def _mt5_data():
    try:
        import MetaTrader5 as mt5
        mt5.initialize()
        mt5.login(int(os.getenv("MT5_LOGIN")), os.getenv("MT5_PASSWORD"), os.getenv("MT5_SERVER"))
        acc  = mt5.account_info()
        pos  = mt5.positions_get() or []
        hist = mt5.history_deals_get(
            _dt.datetime.now(_dt.timezone.utc).replace(hour=0, minute=0, second=0),
            _dt.datetime.now(_dt.timezone.utc)
        ) or []
        realized = sum(d.profit for d in hist if d.entry == 1)
        mt5.shutdown()
        return {
            "balance":  acc.balance,
            "equity":   acc.equity,
            "float":    acc.profit,
            "realized": realized,
            "positions": [
                {
                    "ticket":  p.ticket,
                    "symbol":  p.symbol,
                    "side":    "BUY" if p.type == 0 else "SELL",
                    "vol":     p.volume,
                    "entry":   p.price_open,
                    "current": p.price_current,
                    "profit":  p.profit,
                    "sl":      p.sl,
                    "tp":      p.tp,
                }
                for p in pos
            ],
        }
    except Exception as e:
        return {"error": str(e), "balance": 0, "equity": 0, "float": 0, "realized": 0, "positions": []}

def _recent_logs(n: int = 25) -> list[str]:
    try:
        result = subprocess.run(
            ["pm2", "logs", "smc-bot", "--lines", str(n), "--nostream"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines() + result.stderr.splitlines()
        return [l for l in lines if "smc-bot" in l][-n:]
    except Exception:
        return []

def _scan_stats() -> dict:
    try:
        f = os.path.join("memory", "scan_stats.json")
        if os.path.exists(f):
            return json.load(open(f))
    except Exception:
        pass
    return {}

# ── Collect data ──────────────────────────────────────────────────────────
data  = _mt5_data()
stats = _scan_stats()
logs  = _recent_logs(30)

bal       = data.get("balance", 0)
equity    = data.get("equity", 0)
float_pnl = data.get("float", 0)
realized  = data.get("realized", 0)
positions = data.get("positions", [])
net       = realized + float_pnl
net_pct   = net / 100_000 * 100

target_pct  = net / DAILY_TARGET
danger      = net < DANGER_LOSS
target_done = net >= DAILY_TARGET

# ── Print report ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SESSION MONITOR — {hora}")
print(f"{'='*60}")

# Account
print(f"\nBALANCE:  ${bal:>10,.2f}   EQUITY: ${equity:,.2f}")
print(f"FLOAT:    ${float_pnl:>+10.2f}   REALIZ: ${realized:+.2f}")
print(f"NET HOY:  ${net:>+10.2f}   ({net_pct:+.3f}%)")

# Progress bar
print(f"\nMETA $250: [{_bar(target_pct)}] {target_pct*100:.0f}%")
if target_done:
    print("  ✅ META CUMPLIDA — considera cerrar todo")
elif danger:
    print(f"  🚨 PELIGRO: perdida > ${abs(DANGER_LOSS)} — evaluar PAUSE")
elif net < 0:
    print(f"  🔴 En rojo — esperando recuperacion")
else:
    print(f"  🟡 Progresando — faltan ${DAILY_TARGET - net:.0f}")

# Positions
print(f"\nPOSICIONES ABIERTAS: {len(positions)}")
if positions:
    for p in positions:
        icon  = "🟢" if p["profit"] >= 0 else "🔴"
        sl_d  = abs(p["current"] - p["sl"])
        tp_d  = abs(p["tp"] - p["current"])
        rr_now = tp_d / sl_d if sl_d > 0 else 0
        print(f"  {icon} #{p['ticket']} {p['symbol']} {p['side']} {p['vol']}L")
        print(f"     entry={p['entry']:.4f} now={p['current']:.4f}  P&L=${p['profit']:+.2f}")
        print(f"     SL:{sl_d:.1f}pts | TP:{tp_d:.1f}pts | RR restante:{rr_now:.1f}")
else:
    print("  (sin posiciones)")

# DIM6
consec = stats.get("consecutive_losses", 0)
wr5    = stats.get("wr_last5", None)
mo_pct = stats.get("monthly_profit_pct", 0.0)
print(f"\nDIM6 CIRCUIT:")
print(f"  Perdidas consecutivas: {consec}/3  {'🚨 BLOQUEADO' if consec >= 3 else 'OK'}")
if wr5 is not None:
    print(f"  WR ultimos 5:          {wr5:.0%}  {'⚠️ size=60%' if wr5 < 0.40 else 'OK'}")
print(f"  Profit mensual:        {mo_pct:.1f}%  {'⚠️ size=30%' if mo_pct >= 4.0 else 'OK'}")

# Recent log entries of interest
interesting = [l for l in logs if any(k in l for k in [
    "SWING", "SCALP", "ejecutando", "TRAIL", "META", "CIRCUIT", "BLOCK",
    "SL hit", "TP hit", "cerrado", "ERROR", "skip"
])]
if interesting:
    print(f"\nACTIVIDAD RECIENTE:")
    for l in interesting[-8:]:
        print(f"  {l[l.find('smc-bot'):].replace('0|smc-bot  | ', '')[:100]}")

# Next setups from logs
setups = [l for l in logs if "ejecutando SWING" in l or "ejecutando SCALP" in l]
if setups:
    print(f"\nULTIMO SETUP EJECUTADO:")
    print(f"  {setups[-1][-120:]}")

print(f"\n{'='*60}")
print(f"  Actualizar: .venv\\Scripts\\python scripts\\_monitor_session.py")
print(f"{'='*60}\n")
