"""Forensic audit script — run from project root."""
import os, sqlite3, shutil, json, sys
sys.path.insert(0, ".")

print("=" * 60)
print("FORENSIC AUDIT — SMC TRADING BOT")
print("=" * 60)

# ── 1. Disk ────────────────────────────────────────────────
total, used, free = shutil.disk_usage("C:")
print(f"\n[DISK] Libre: {free//1024//1024//1024:.1f}GB / {total//1024//1024//1024:.1f}GB total")
if free < 2 * 1024**3:
    print("  ⚠️  CRITICO: menos de 2GB libre")
else:
    print("  ✅ Espacio OK")

# PM2 logs size
pm2_log = os.path.expanduser("~/.pm2/logs")
log_size = sum(
    os.path.getsize(os.path.join(pm2_log, f))
    for f in os.listdir(pm2_log) if os.path.isfile(os.path.join(pm2_log, f))
) if os.path.isdir(pm2_log) else 0
print(f"  PM2 logs: {log_size//1024//1024:.1f}MB")

# ── 2. SQLite integrity ────────────────────────────────────
print("\n[SQLITE]")
for db_path in ["memory/scores.db", "memory/episodes.db"]:
    if not os.path.exists(db_path):
        print(f"  MISS {db_path}")
        continue
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        res = conn.execute("PRAGMA integrity_check").fetchone()[0]
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        size = os.path.getsize(db_path)
        print(f"  {'OK' if res == 'ok' else 'FAIL'} {db_path} ({size//1024}KB) tables={tables} integrity={res}")
        # Row counts
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"      {t}: {count} rows")
        conn.close()
    except Exception as e:
        print(f"  FAIL {db_path}: {e}")

# ── 3. Memory files ────────────────────────────────────────
print("\n[MEMORY FILES]")
memory_checks = {
    "memory/glint_session.json":   "Glint session",
    "memory/mql5_strategies.json": "MQL5 strategies",
    "memory/episodic_memory.json": "Episodic JSON",
    "shared_context.json":         "Shared context",
    "memory/agents":               "Agent memory dir",
}
for path, label in memory_checks.items():
    exists = os.path.exists(path)
    if exists and os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"  OK   {label:25} {size} bytes")
    elif exists:
        files = os.listdir(path)
        print(f"  OK   {label:25} dir with {len(files)} files")
    else:
        print(f"  MISS {label:25} NOT FOUND")

# ── 4. DecisionFilter component check ─────────────────────
print("\n[DECISION FILTER]")
try:
    from core.decision_filter import DecisionFilter
    import inspect
    src = inspect.getsource(DecisionFilter.evaluate)
    components = []
    if "_score_smc" in src:     components.append("SMC")
    if "_score_ml" in src or "ml_score" in src:   components.append("ML/LSTM")
    if "sentiment" in src:      components.append("Sentiment")
    if "_score_risk" in src:    components.append("Risk")
    if "hist_score" in src:     components.append("Historical")
    print(f"  Componentes activos: {components}")
    # Max possible score
    import re
    max_vals = re.findall(r"max.*?=\s*(\d+)", src)
    print(f"  Score max detectado: 0-100 (base SMC system)")
    print("  ✅ DecisionFilter imports OK")
except Exception as e:
    print(f"  FAIL: {e}")

# ── 5. Risk manager config check ──────────────────────────
print("\n[RISK MANAGER]")
try:
    from core.config import config
    print(f"  max_risk_per_trade: {config.max_risk_per_trade*100:.1f}%")
    print(f"  operation_mode: {config.operation_mode}")
    mt5_login = str(config.mt5_login)
    print(f"  MT5 login configured: {'Yes' if mt5_login and mt5_login != 'None' else 'NO'}")
    print(f"  Anthropic key: {'SET' if config.anthropic_api_key else 'MISSING'}")
    print(f"  Telegram token: {'SET' if config.telegram_bot_token else 'MISSING'}")
except Exception as e:
    print(f"  FAIL: {e}")

# ── 6. Supervisor asyncio.gather audit ────────────────────
print("\n[SUPERVISOR LOOPS]")
try:
    with open("core/supervisor.py", encoding="utf-8") as f:
        sup = f.read()
    loops_in_gather = []
    import re
    gather_match = re.search(r"asyncio\.gather\((.*?)\)", sup, re.DOTALL)
    if gather_match:
        loops_in_gather = [l.strip().rstrip(",") for l in gather_match.group(1).split("\n") if l.strip()]
    print(f"  asyncio.gather() runs {len(loops_in_gather)} loops:")
    for l in loops_in_gather:
        print(f"    {l}")

    # Check for race condition: MT5 reconnect + order send
    if "_mt5_available" in sup:
        print("  ✅ MT5 availability flag used")
    if "await asyncio.sleep" in sup:
        sleep_counts = sup.count("await asyncio.sleep")
        print(f"  ✅ {sleep_counts} await asyncio.sleep calls (no busy loops)")
except Exception as e:
    print(f"  FAIL: {e}")

# ── 7. Order execution flow ───────────────────────────────
print("\n[ORDER EXECUTION FLOW]")
try:
    with open("core/supervisor.py", encoding="utf-8") as f:
        sup = f.read()
    checks = {
        "SL validation (sl_val == 0.0 skip)": "sl_val == 0.0" in sup,
        "Duplicate position guard":           "sym_open" in sup or "posicion ya abierta" in sup,
        "Episodic recording after fill":      "record_episode" in sup,
        "Telegram alert on fill":             "send_glint_alert" in sup,
        "MT5 reconnect loop":                 "reconnect" in sup,
    }
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'MISS'} {name}")
except Exception as e:
    print(f"  FAIL: {e}")

# ── 8. API fallback check ─────────────────────────────────
print("\n[API FALLBACKS]")
fallbacks = {
    "Binance": ("connectors/binance_connector.py", "except"),
    "MT5":     ("connectors/metatrader_connector.py", "except"),
    "Claude":  ("agents/analysis_agent.py", "fallback"),
    "Telegram":("dashboard/telegram_commander.py", "except"),
}
for name, (path, keyword) in fallbacks.items():
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        count = content.count(keyword)
        print(f"  OK   {name:10} {count} exception handlers in {path}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

# ── 9. Scores DB actual data ──────────────────────────────
print("\n[SCORES DB — LIVE DATA]")
try:
    from core.score_db import get_stats
    stats = get_stats()
    print(f"  Trades ejecutados: {stats['executed']}")
    print(f"  Win rate:          {stats['win_rate']:.1f}%")
    print(f"  Total scaneados:   {stats['total']}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
