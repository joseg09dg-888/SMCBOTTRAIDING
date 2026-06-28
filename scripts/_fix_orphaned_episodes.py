"""
Fix: recover outcomes for orphaned episodes (positions that closed during bot restart).
Reads open_episodes.json, finds tickets NOT currently open in MT5,
and backfills their result from MT5 deal history.
"""
import sys, json, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from memory.episodic_db import get_db, update_episode_result

mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

# Currently open tickets
open_pos = mt5.positions_get() or []
open_tickets = {p.ticket for p in open_pos}

# Load open_episodes.json
with open('memory/open_episodes.json', 'r') as f:
    open_episodes = json.load(f)

# Keys are strings in JSON
orphaned = {int(t): eid for t, eid in open_episodes.items()
            if int(t) not in open_tickets}

print(f"Open positions: {sorted(open_tickets)}")
print(f"Tickets in open_episodes.json: {sorted(int(t) for t in open_episodes.keys())}")
print(f"Orphaned (closed but no outcome): {sorted(orphaned.keys())}")

if not orphaned:
    print("Nothing to fix — all episodes match current positions.")
    mt5.shutdown()
    sys.exit(0)

# Fetch deal history (90 days back)
now = datetime.now(timezone.utc)
from_dt = now - timedelta(days=90)
all_deals = mt5.history_deals_get(from_dt, now) or []
print(f"\nTotal MT5 deals in last 90 days: {len(all_deals)}")

# Build lookup: position_id -> closing deal (entry==1 means OUT/close)
closing_deals = {}
for d in all_deals:
    if d.entry == 1:  # exit deal
        closing_deals[d.position_id] = d

conn = get_db()
fixed = 0
missing = 0

for ticket, episode_id in sorted(orphaned.items()):
    d = closing_deals.get(ticket)
    if d:
        pnl = round(d.profit + d.swap + d.commission, 2)
        result = 'WIN' if pnl > 0 else 'LOSS'
        update_episode_result(
            episode_id,
            exit_price=d.price,
            pnl=pnl,
            result=result,
            lesson=f"Backfill: score -> {result} PnL={pnl:+.2f}",
            conn=conn
        )
        print(f"  [FIXED] ticket={ticket} ep={episode_id} -> {result} pnl={pnl:+.2f}")
        fixed += 1
        # Remove from open_episodes.json
        open_episodes.pop(str(ticket), None)
    else:
        print(f"  [MISS]  ticket={ticket} ep={episode_id} -> no closing deal found (manual close? <7 days ago?)")
        missing += 1

# Save cleaned open_episodes.json
with open('memory/open_episodes.json', 'w') as f:
    json.dump(open_episodes, f)

print(f"\nDone: {fixed} fixed, {missing} not found in history.")
print(f"open_episodes.json updated (removed closed orphans).")

# Now run AutonomousLearner to generate lessons from recovered data
from core.autonomous_learner import AutonomousLearner
learner = AutonomousLearner(conn=conn)
result = learner.run_analysis()
if result:
    print(f"\nAutonomousLearner generated {len(result)} weight adjustments:")
    for key, data in result.items():
        print(f"  {key}: WR={data['win_rate']:.1f}% -> weight={data['weight_adj']:.2f}")
else:
    print(f"\nAutonomousLearner: not enough data yet (need 5+ per group)")

conn.close()
mt5.shutdown()
