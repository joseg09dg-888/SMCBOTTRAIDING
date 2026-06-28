"""
Backfill TODOS los episodios NULL con ticket MT5 real.
Busca en historia de 90 dias y registra resultado.
Mas agresivo que _fix_orphaned_episodes.py que solo mira open_episodes.json.
"""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import sqlite3

mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

# Posiciones actualmente abiertas (no tocar)
open_tickets = {p.ticket for p in (mt5.positions_get() or [])}
print(f"Posiciones abiertas (NO tocar): {sorted(open_tickets)}")

# Todos los episodios NULL con ticket real
conn = sqlite3.connect('memory/episodes.db')
c = conn.cursor()
nulls = c.execute(
    'SELECT id, ticket, symbol, setup_type FROM episodes WHERE result IS NULL AND ticket > 0'
).fetchall()
print(f"Episodios NULL con ticket: {len(nulls)}")

# Historia MT5 90 dias
now = datetime.now(timezone.utc)
all_deals = mt5.history_deals_get(now - timedelta(days=90), now) or []
print(f"Deals MT5 ultimos 90 dias: {len(all_deals)}")

# Closing deals por position_id
closing = {}
for d in all_deals:
    if d.entry == 1:  # deal de salida
        closing[d.position_id] = d

fixed = 0
skipped_open = 0
not_found = 0

for ep_id, ticket, symbol, setup in nulls:
    if ticket in open_tickets:
        skipped_open += 1
        continue

    d = closing.get(ticket)
    if not d:
        not_found += 1
        continue

    pnl = round(d.profit + d.swap + d.commission, 2)
    result = 'WIN' if pnl > 0 else 'LOSS'
    lesson = f"Backfill masivo: {result} PnL={pnl:+.2f} precio_cierre={d.price}"

    c.execute(
        '''UPDATE episodes SET result=?, pnl=?, lesson=?, exit_price=?
           WHERE id=?''',
        (result, pnl, lesson, d.price, ep_id)
    )
    print(f"  [OK] ep={ep_id} ticket={ticket} {symbol} -> {result} pnl={pnl:+.2f}")
    fixed += 1

conn.commit()
conn.close()

print(f"\nResumen: {fixed} actualizados | {skipped_open} abiertos (saltados) | {not_found} no encontrados en historial")

# Totales actualizados
conn2 = sqlite3.connect('memory/episodes.db')
c2 = conn2.cursor()
wins   = c2.execute("SELECT COUNT(*) FROM episodes WHERE result='WIN'").fetchone()[0]
losses = c2.execute("SELECT COUNT(*) FROM episodes WHERE result='LOSS'").fetchone()[0]
nulls2 = c2.execute("SELECT COUNT(*) FROM episodes WHERE result IS NULL").fetchone()[0]
conn2.close()

wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
print(f"\nEstado final episodes.db:")
print(f"  WIN:    {wins}")
print(f"  LOSS:   {losses}")
print(f"  NULL:   {nulls2}")
print(f"  WR:     {wr:.1f}%")

# Correr AutonomousLearner con datos frescos
print("\nEjecutando AutonomousLearner con datos actualizados...")
from core.autonomous_learner import AutonomousLearner
conn3 = sqlite3.connect('memory/episodes.db')
learner = AutonomousLearner(conn=conn3)
result = learner.run_analysis()
if result:
    print(f"Learner: {len(result)} grupos con ajustes:")
    for key, data in result.items():
        print(f"  {key}: n={data.get('n','?')} WR={data.get('win_rate',0):.0f}% -> adj={data['weight_adj']:.2f}")
else:
    print("Learner: sin grupos con 20+ muestras aun")
conn3.close()

mt5.shutdown()
print("\nBackfill completo.")
