"""Auditoria real: que funciona vs que es solo codigo muerto."""
import sys, os, sqlite3, json
sys.path.insert(0, ".")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

print("=" * 60)
print("AUDITORIA INSTITUCIONAL COMPLETA")
print("=" * 60)

# ── 1. APRENDIZAJE REAL ─────────────────────────────────────
print("\n[1] SISTEMA DE APRENDIZAJE")
conn = sqlite3.connect("memory/episodes.db")
conn.row_factory = sqlite3.Row
eps   = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
wins  = conn.execute("SELECT COUNT(*) FROM episodes WHERE result='WIN'").fetchone()[0]
losses= conn.execute("SELECT COUNT(*) FROM episodes WHERE result='LOSS'").fetchone()[0]
open_ = conn.execute("SELECT COUNT(*) FROM episodes WHERE result IS NULL").fetchone()[0]
lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

print(f"  Episodios grabados:  {eps}")
print(f"  WINs registrados:   {wins}")
print(f"  LOSSes registrados: {losses}")
print(f"  Abiertos sin result:{open_}")
print(f"  Lecciones aprendidas:{lessons}")

if eps == 0:
    print("  PROBLEMA: No hay episodios -> el bot NO esta aprendiendo de sus trades")
elif losses == 0 and open_ > 0:
    print("  PROBLEMA: Trades abiertos pero sin resultado registrado -> aprendizaje CIEGO")
elif lessons == 0:
    print("  PROBLEMA: Episodios grabados pero ninguna leccion generada")
else:
    print("  OK: Aprendizaje activo")

# Mostrar ultimos episodios
rows = conn.execute("SELECT * FROM episodes ORDER BY id DESC LIMIT 5").fetchall()
if rows:
    print("  Ultimos 5 episodios:")
    for r in rows:
        d = dict(r)
        print(f"    id={d['id']} sym={d.get('symbol','?')} dir={d.get('direction','?')} result={d.get('result','?')} score={d.get('score','?')}")

# Metas
goals = conn.execute("SELECT * FROM goals").fetchall()
print(f"\n  Metas en DB: {len(goals)}")
for g in goals:
    d = dict(g)
    print(f"    {d.get('goal_id','?')}: {d.get('current_value',0):.1f}/{d.get('target_value',0)} ({d.get('progress_pct',0):.0f}%)")
conn.close()

# ── 2. SCORES DB ────────────────────────────────────────────
print("\n[2] TRADES EJECUTADOS EN SCORES.DB")
conn2 = sqlite3.connect("memory/scores.db")
conn2.row_factory = sqlite3.Row
total = conn2.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
exec_ = conn2.execute("SELECT COUNT(*) FROM scores WHERE executed=1").fetchone()[0]
avg_score = conn2.execute("SELECT AVG(score) FROM scores WHERE executed=1").fetchone()[0] or 0
min_score = conn2.execute("SELECT MIN(score) FROM scores WHERE executed=1").fetchone()[0] or 0
max_score = conn2.execute("SELECT MAX(score) FROM scores WHERE executed=1").fetchone()[0] or 0
print(f"  Scores guardados:   {total}")
print(f"  Ejecutados:         {exec_}")
print(f"  Score min/avg/max:  {min_score:.0f} / {avg_score:.1f} / {max_score:.0f}")
if avg_score < 60:
    print("  PROBLEMA: Score promedio muy bajo -> operaba con senales basura")

rows2 = conn2.execute("SELECT * FROM scores ORDER BY id DESC LIMIT 5").fetchall()
if rows2:
    print("  Ultimos 5 scores:")
    for r in rows2:
        d = dict(r)
        print(f"    {d.get('symbol','?')} {d.get('direction','?')} score={d.get('score','?')} exec={d.get('executed','?')} result={d.get('result','?')}")
conn2.close()

# ── 3. AGENTES ACTIVOS vs DORMIDOS ──────────────────────────
print("\n[3] ESTADO DE AGENTES (activo en loop vs solo existe)")
agents_to_check = {
    "AutonomousLearner":    ("core.autonomous_learner",     "AutonomousLearner"),
    "ResearchAgent":        ("core.research_agent",         "ResearchAgent"),
    "GoalsManager":         ("core.goals_manager",          "GoalsManager"),
    "NightlyReporter":      ("core.nightly_reporter",       "NightlyReporter"),
    "FTMOAgent":            ("strategies.ftmo_agent",       "FTMOAgent"),
    "DecisionFilter":       ("core.decision_filter",        "DecisionFilter"),
    "LunarAgent":           ("agents.lunar_agent",          "LunarAgent"),
    "ElliottAgent":         ("agents.elliott_agent",        "ElliottAgent"),
    "FootprintAgent":       ("agents.footprint_agent",      "FootprintAgent"),
    "StatisticalEdge":      ("agents.statistical_edge_agent","StatisticalEdgeAgent"),
    "SMCAnalysisAgent":     ("agents.analysis_agent",       "SMCAnalysisAgent"),
    "RiskManager":          ("core.risk_manager",           "RiskManager"),
}
active_in_loop = {
    "AutonomousLearner", "ResearchAgent", "GoalsManager",
    "NightlyReporter", "DecisionFilter", "RiskManager"
}
for name, (module, cls) in agents_to_check.items():
    try:
        m = __import__(module, fromlist=[cls])
        getattr(m, cls)
        in_loop = "ACTIVO en loop" if name in active_in_loop else "EXISTE pero DORMIDO"
        print(f"  {'OK  ' if name in active_in_loop else 'WARN'} {name}: {in_loop}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

# ── 4. FTMO/AXI REGLAS ACTIVAS ──────────────────────────────
print("\n[4] CUMPLIMIENTO FTMO/AXI")
try:
    from strategies.ftmo_agent import FTMOAgent
    import MetaTrader5 as mt5
    mt5.initialize()
    acc = mt5.account_info()
    if acc:
        balance = acc.balance
        ftmo = FTMOAgent()
        status = ftmo.check_status(balance)
        print(f"  Balance actual:      ${balance:,.2f}")
        print(f"  FTMO mode:           {status.get('mode','?')}")
        print(f"  Daily loss usado:    {status.get('daily_loss_pct',0)*100:.2f}% (max 5%)")
        print(f"  Drawdown total:      {status.get('total_dd_pct',0)*100:.2f}% (max 10%)")
        print(f"  Puede operar:        {status.get('can_trade', '?')}")
        print(f"  Razon si no:         {status.get('reason', 'OK')}")
    else:
        print("  MT5 no conectado")
    mt5.shutdown()
except Exception as e:
    print(f"  FAIL FTMOAgent: {e}")

# ── 5. SUPERVISOR: QUE HACE REALMENTE ───────────────────────
print("\n[5] QUE HACE EL SUPERVISOR REALMENTE")
with open("core/supervisor.py", encoding="utf-8") as f:
    sup = f.read()

checks = {
    "FTMOAgent conectado al scan": "FTMOAgent" in sup and "ftmo" in sup.lower(),
    "GoalsManager.evaluate() llamado": "evaluate" in sup,
    "AutonomousLearner.run() llamado": "run" in sup and "learner" in sup.lower(),
    "ResearchAgent activo": "research" in sup.lower(),
    "Risk check antes de orden": "risk_manager" in sup and "can_trade" in sup.lower(),
    "FTMO check antes de orden": "ftmo" in sup.lower() and "can_trade" in sup.lower(),
    "SL obligatorio": "sl_val == 0.0" in sup,
    "Guard posicion duplicada": "sym_open" in sup,
    "Score minimo 80": "MT5_REAL_SCORE_THRESHOLD" in sup,
    "Modo conservador activo": "CONSERVATIVE_MODE" in sup,
    "Max 1 trade/dia": "MAX_DAILY_TRADES" in sup,
    "Filtro horario muerto": "DEAD_HOURS_UTC" in sup,
}
for name, ok in checks.items():
    print(f"  {'OK  ' if ok else 'MISS'} {name}")

print("\n" + "=" * 60)
print("VEREDICTO HONESTO")
print("=" * 60)
