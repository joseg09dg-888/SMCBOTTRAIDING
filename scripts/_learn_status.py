import sqlite3

conn = sqlite3.connect('memory/episodes.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM episodes")
total = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM episodes WHERE result='WIN'")
wins = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM episodes WHERE result='LOSS'")
losses = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM episodes WHERE result IS NULL")
nulls = cur.fetchone()[0]

print(f"Total episodes: {total}")
print(f"  WIN:  {wins}")
print(f"  LOSS: {losses}")
print(f"  NULL (sin resultado): {nulls}")

cur.execute("SELECT id, ts, symbol, direction, score, result, pnl FROM episodes WHERE result IS NOT NULL ORDER BY id DESC LIMIT 10")
rows = cur.fetchall()
if rows:
    print("\nEpisodes CON resultados:")
    for r in rows:
        print(f"  id={r['id']} {r['symbol']} {r['direction']} score={r['score']} result={r['result']} pnl={r['pnl']}")
else:
    print("\n[ALERTA] NINGUN episodio tiene resultado registrado -- el aprendizaje no tiene feedback!")

# Check lessons table
cur.execute("SELECT COUNT(*) FROM lessons")
n_lessons = cur.fetchone()[0]
print(f"\nLessons guardadas: {n_lessons}")

# Scores DB win/loss
import os
if os.path.exists('memory/scores.db'):
    conn2 = sqlite3.connect('memory/scores.db')
    cur2 = conn2.cursor()
    cur2.execute("SELECT COUNT(*) FROM scores WHERE outcome='WIN'")
    sw = cur2.fetchone()[0]
    cur2.execute("SELECT COUNT(*) FROM scores WHERE outcome='LOSS'")
    sl = cur2.fetchone()[0]
    cur2.execute("SELECT COUNT(*) FROM scores WHERE outcome IS NULL")
    sn = cur2.fetchone()[0]
    print(f"\nScores DB: {sw}W / {sl}L / {sn} sin outcome")
    cur2.execute("SELECT ts, symbol, score, outcome, pnl_pct FROM scores WHERE outcome IS NOT NULL ORDER BY rowid DESC LIMIT 5")
    for r in cur2.fetchall():
        print(f"  {r[0][:16]} {r[1]:8s} score={r[2]} {r[3]} pnl={r[4]:.4f}")
    conn2.close()

conn.close()
