import sqlite3, sys, os
sys.path.insert(0, '.')

# Episodes DB
conn = sqlite3.connect('memory/episodes.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print(f"Tablas en episodes.db: {tables}")

for table in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        print(f"\n[{table}] {n} registros | columnas: {cols}")
        if n > 0:
            cur.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 5")
            for row in cur.fetchall():
                print(f"  {row}")
    except Exception as e:
        print(f"  Error: {e}")
conn.close()

# Scores DB
if os.path.exists('memory/scores.db'):
    conn2 = sqlite3.connect('memory/scores.db')
    cur2 = conn2.cursor()
    cur2.execute("SELECT COUNT(*) FROM scores")
    n = cur2.fetchone()[0]
    cur2.execute("SELECT symbol, setup_type, score, outcome, pnl_pct, ts FROM scores ORDER BY ts DESC LIMIT 8")
    rows = cur2.fetchall()
    print(f"\n[SCORES] {n} registros totales")
    wins = sum(1 for r in rows if r[3] == 1)
    print(f"Ultimos 8: {wins}W/{len(rows)-wins}L")
    for r in rows:
        res = 'WIN' if r[3] == 1 else 'LOSS'
        print(f"  {r[5][:16]} | {r[0]:8s} | score={r[2]} | {res} | pnl={r[4]:.4f}")
    conn2.close()
