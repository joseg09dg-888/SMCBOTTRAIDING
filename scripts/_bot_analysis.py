import json, sqlite3
from datetime import datetime, timezone, timedelta

# ── Daily trades ──────────────────────────────────────────────────────────
dt = json.load(open('memory/daily_trades.json'))
print('=== TRADES POR DIA (ultimos 10) ===')
for d, n in sorted(dt.items())[-10:]:
    print(f'  {d}: {n} trades')

# ── Risk governor ─────────────────────────────────────────────────────────
rg = json.load(open('memory/risk_governor_state.json'))
print(f'\n=== RISK GOVERNOR ===')
print(f'Multiplicador riesgo: x{rg["risk_multiplier"]}')
print(f'Suspendidos: {list(rg["suspended"].keys())}')
h = rg["history"][-1]
print(f'Activos: {h["active_symbols"]}')

# ── Episodes DB ───────────────────────────────────────────────────────────
db = sqlite3.connect('memory/episodes.db')
db.row_factory = sqlite3.Row

WIN = "WIN"

row = db.execute(
    "SELECT COUNT(*) total, SUM(CASE WHEN result=? THEN 1 ELSE 0 END) wins,"
    " AVG(pnl) avg_pnl, SUM(pnl) total_pnl FROM episodes WHERE result IS NOT NULL",
    (WIN,)
).fetchone()

total = row["total"] or 0
wins  = row["wins"]  or 0
print(f"\n=== APRENDIZAJE GLOBAL ===")
if total > 0:
    print(f"Total episodios: {total} | Wins: {wins} | WR: {wins/total*100:.1f}%")
    print(f"PnL promedio: {(row['avg_pnl'] or 0):+.2f} por trade")
    print(f"PnL acumulado: {(row['total_pnl'] or 0):+.2f}")
else:
    print("Sin datos suficientes")

# Last 20
rows = db.execute(
    "SELECT symbol, result, pnl, ts FROM episodes"
    " WHERE result IS NOT NULL ORDER BY id DESC LIMIT 20"
).fetchall()
print(f"\n=== ULTIMOS 20 TRADES ===")
w20 = 0
for r in rows:
    icon = "W" if r["result"] == WIN else "L"
    if r["result"] == WIN:
        w20 += 1
    pnl = r["pnl"] or 0
    print(f"  {icon}  {r['symbol']:10} {pnl:+.2f}  {str(r['ts'])[:10]}")
print(f"WR ultimos 20: {w20}/20 = {w20/20*100:.0f}%")

# By symbol
print(f"\n=== RENDIMIENTO POR SIMBOLO ===")
sym_rows = db.execute(
    "SELECT symbol, COUNT(*) n,"
    " SUM(CASE WHEN result=? THEN 1 ELSE 0 END) w,"
    " AVG(pnl) avg_pnl, SUM(pnl) total_pnl"
    " FROM episodes WHERE result IS NOT NULL"
    " GROUP BY symbol ORDER BY n DESC",
    (WIN,)
).fetchall()
for r in sym_rows:
    n   = r["n"]
    w   = r["w"] or 0
    wr  = w / n * 100 if n > 0 else 0
    avg = r["avg_pnl"] or 0
    tot = r["total_pnl"] or 0
    flag = "STOP" if wr < 35 and n >= 8 else ("BIEN" if wr >= 55 else "OK")
    print(f"  [{flag}] {r['symbol']:10} {n:3}tr | WR {wr:.0f}% | avg {avg:+.1f} | total {tot:+.0f}")

# Weekly trend
print(f"\n=== TENDENCIA SEMANAL ===")
weeks = db.execute(
    "SELECT strftime('%Y-W%W', ts) week, COUNT(*) n,"
    " SUM(CASE WHEN result=? THEN 1 ELSE 0 END) w"
    " FROM episodes WHERE result IS NOT NULL"
    " GROUP BY week ORDER BY week DESC LIMIT 5",
    (WIN,)
).fetchall()
for wk in weeks:
    n  = wk["n"]
    w  = wk["w"] or 0
    wr = w / n * 100 if n > 0 else 0
    bar = "#" * int(wr / 10)
    print(f"  {wk['week']}  {n:2}tr  WR {wr:.0f}%  {bar}")

db.close()
