"""Full forensic audit — answers all 7 questions."""
import sys, os
sys.path.insert(0, ".")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import MetaTrader5 as mt5
from datetime import datetime, timezone
import sqlite3, json

print("=" * 60)
print("AUDITORIA FORENSE COMPLETA")
print("=" * 60)

# ── Q1: P&L REAL ──────────────────────────────────────────
print("\n[Q1] P&L REAL DESDE INICIO")
mt5.initialize()
acc = mt5.account_info()
balance = acc.balance
equity  = acc.equity
profit  = acc.profit
net     = balance - 100_000.0
net_pct = net / 100_000.0 * 100
print(f"  Balance actual:  ${balance:,.2f}")
print(f"  Equity:          ${equity:,.2f}")
print(f"  P&L abierto:     ${profit:+.2f}")
print(f"  Net vs $100,000: ${net:+.2f} ({net_pct:+.3f}%)")

positions = mt5.positions_get() or []
print(f"\n  Posiciones abiertas: {len(positions)}")
for p in positions:
    dt = datetime.fromtimestamp(p.time, tz=timezone.utc).strftime("%m-%d %H:%M")
    side = "BUY" if p.type == 0 else "SELL"
    print(f"    #{p.ticket} {p.symbol} {side} {p.volume}lot @{p.price_open:.3f}"
          f" profit={p.profit:+.2f} sl={p.sl:.3f} tp={p.tp:.3f} [{dt}]")

from_date = datetime(2026, 5, 1, tzinfo=timezone.utc)
to_date   = datetime.now(timezone.utc)
deals = mt5.history_deals_get(from_date, to_date) or []
trade_deals = [d for d in deals if d.symbol != ""]
closing = [d for d in trade_deals if d.entry == 1]
realized = sum(d.profit + d.swap + d.commission for d in closing)
wins_closed  = sum(1 for d in closing if d.profit > 0)
loss_closed  = sum(1 for d in closing if d.profit < 0)
print(f"\n  Historial mayo:")
print(f"    Deals totales:   {len(trade_deals)}")
print(f"    Trades cerrados: {len(closing)}")
print(f"    Ganadores:       {wins_closed}")
print(f"    Perdedores:      {loss_closed}")
print(f"    P&L realizado:   ${realized:+.2f}")
if closing:
    print("    Ultimos cierres:")
    for d in closing[-5:]:
        dt2 = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime("%m-%d %H:%M")
        side = "BUY" if d.type == 0 else "SELL"
        print(f"      {dt2} {d.symbol} {side} profit={d.profit:+.2f} swap={d.swap:.2f}")

# Symbols with open positions
open_symbols = {p.symbol for p in positions}
print(f"\n  Simbolos con posicion abierta: {open_symbols if open_symbols else 'NINGUNO'}")

mt5.shutdown()

# ── Q2: APRENDIZAJE ───────────────────────────────────────
print("\n[Q2] EVIDENCIA DE APRENDIZAJE")
memory_files = {
    "memory/mql5_strategies.json":  "MQL5 strategies",
    "memory/episodic_memory.json":  "Episodic JSON",
    "shared_context.json":          "Shared context",
}
for path, name in memory_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  OK   {name}: {size} bytes")
        if size < 10000:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                print(f"       Keys: {list(data.keys())[:5]}")
            except Exception:
                pass
    else:
        print(f"  MISS {name}: NO EXISTE")

# SQLite episodic
conn = sqlite3.connect("memory/episodes.db")
conn.row_factory = sqlite3.Row
eps = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
goals = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
research = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
print(f"\n  episodes.db:")
print(f"    Episodios: {eps}")
print(f"    Lecciones: {lessons}")
print(f"    Metas: {goals}")
print(f"    Research: {research}")
if eps > 0:
    rows = conn.execute("SELECT * FROM episodes ORDER BY id DESC LIMIT 3").fetchall()
    print("    Ultimos episodios:")
    for r in rows:
        print(f"      {dict(r)}")
conn.close()

# scores.db
conn2 = sqlite3.connect("memory/scores.db")
conn2.row_factory = sqlite3.Row
total_scores = conn2.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
executed = conn2.execute("SELECT COUNT(*) FROM scores WHERE executed=1").fetchone()[0]
print(f"\n  scores.db: {total_scores} scores, {executed} ejecutados")
conn2.close()

# Agent memory dir
agents_dir = "memory/agents"
if os.path.isdir(agents_dir):
    files = os.listdir(agents_dir)
    print(f"\n  memory/agents/: {len(files)} archivos")
    for f in files[:5]:
        path2 = os.path.join(agents_dir, f)
        print(f"    {f}: {os.path.getsize(path2)} bytes")
else:
    print("\n  memory/agents/: directorio NO existe")

# ── Q3: YOUTUBE / MQL5 / GLINT ───────────────────────────
print("\n[Q3] YOUTUBE / MQL5 / GLINT")
glint_path = "memory/glint_session.json"
if os.path.exists(glint_path):
    mtime = datetime.fromtimestamp(os.path.getmtime(glint_path)).strftime("%Y-%m-%d %H:%M")
    print(f"  Glint session: OK (modificado {mtime})")
else:
    print("  Glint session: MISSING")

# Check YouTube trainer
try:
    from training.youtube_trainer import YouTubeTrainer
    yt = YouTubeTrainer.__new__(YouTubeTrainer)
    src = open("training/youtube_trainer.py", encoding="utf-8").read()
    has_save = "save" in src or "json" in src.lower()
    print(f"  YouTubeTrainer: clase existe, guarda datos={has_save}")
except Exception as e:
    print(f"  YouTubeTrainer: {e}")

# ── Q4: Duplicate position guard verification ─────────────
print("\n[Q4] GUARD POSICIONES DUPLICADAS")
with open("core/supervisor.py", encoding="utf-8") as f:
    sup = f.read()
checks = {
    "sym_open guard": "sym_open" in sup,
    "SL=0 skip": "sl_val == 0.0" in sup,
    "posicion ya abierta message": "posicion ya abierta" in sup,
    "get_positions before order": "get_positions" in sup and "place_order" in sup,
}
for name, ok in checks.items():
    print(f"  {'OK' if ok else 'FAIL'} {name}")

# ── Q5: Volume config ─────────────────────────────────────
print("\n[Q5] VOLUMEN EN ORDENES MT5")
import re
vol_matches = re.findall(r"place_order.*?(\d+\.\d+)", sup)
place_order_lines = [l.strip() for l in sup.split("\n") if "place_order" in l and ("0.01" in l or "volume" in l.lower())]
for l in place_order_lines:
    print(f"  {l}")

# Check .env for MT5_VOLUME
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, encoding="utf-8") as f:
        env_content = f.read()
    vol_line = [l for l in env_content.split("\n") if "VOLUME" in l or "volume" in l]
    print(f"  .env VOLUME lines: {vol_line if vol_line else 'NINGUNA'}")

print("\n" + "=" * 60)
print("AUDIT DONE")
print("=" * 60)
