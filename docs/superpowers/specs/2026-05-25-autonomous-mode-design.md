# SMC Bot — Autonomous Mode Design Spec
**Date:** 2026-05-25  
**Status:** Approved  
**Author:** Claude Code (brainstorming session)

---

## 1. Goal

Transform the SMC trading bot from a rule-based scanner into a self-improving autonomous agent that:
- Learns from every real MT5 trade (win or loss)
- Reasons with Claude API using historical context before each trade
- Researches new strategies continuously (arXiv, MQL5)
- Tracks its own goals and reports progress nightly
- Operates 24/7 without human intervention

All functionality must survive: PM2 restarts, Windows reboots, DNS failures, MT5 disconnects, and network outages.

---

## 2. Architecture Overview

```
supervisor.py  ─── asyncio.gather() ──────────────────────────────
                     │
                     ├── _market_scan_loop()       [existing, 30s]
                     ├── glint.connect()            [existing]
                     ├── _learning_loop()           [NEW, 60min]
                     ├── _research_loop()           [NEW, 2h]
                     ├── _goals_loop()              [NEW, 24h]
                     └── _nightly_report_loop()     [NEW, 22:00 UTC]

Per-trade (inside _send_mt5_real_order):
  BEFORE order → query_similar_episodes() → enrich Claude prompt
  AFTER order  → record_episode()          → write to episodic_db
```

All new modules are **independent** — each catches its own exceptions and retries. A failure in one loop never stops the others.

---

## 3. Foundation: `memory/episodic_db.py`

**SQLite with WAL mode** — chosen for adversarial conditions:
- No external process required
- ACID compliant: partial writes roll back automatically
- WAL mode: concurrent reads + writes without blocking
- Survives kill -9, power loss, PM2 force-restart

### Schema

```sql
-- Every real MT5 trade executed
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    direction   TEXT NOT NULL,          -- BUY | SELL
    entry       REAL NOT NULL,
    sl          REAL,
    tp          REAL,
    ticket      INTEGER,               -- MT5 ticket number
    score       INTEGER,               -- decision score 0-100
    setup_type  TEXT,                  -- CHoCH+OB | BOS | FVG | etc.
    regime      TEXT,                  -- trending | ranging | high_vol
    session     TEXT,                  -- asia | london | ny | overlap
    reasoning   TEXT,                  -- Claude JSON reasoning
    macro_ctx   TEXT,                  -- Glint/news context at trade time
    exit_price  REAL,
    pnl         REAL,
    result      TEXT,                  -- WIN | LOSS | OPEN | BE
    lesson      TEXT                   -- extracted lesson after close
);

-- Learned strategy weights (auto-updated by learning loop)
CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    setup_type  TEXT NOT NULL,
    regime      TEXT,
    session     TEXT,
    win_rate    REAL,
    sample_size INTEGER,
    weight_adj  REAL,                  -- multiplier applied to DecisionFilter
    notes       TEXT
);

-- Bot's autonomous goals
CREATE TABLE IF NOT EXISTS goals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id     TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    metric      TEXT NOT NULL,
    target      REAL NOT NULL,
    current     REAL DEFAULT 0,
    progress_pct REAL DEFAULT 0,
    horizon     TEXT,                  -- short | medium | long | ultimate
    updated_ts  TEXT
);

-- Research items fetched from arXiv, MQL5, Glint
CREATE TABLE IF NOT EXISTS research (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    source      TEXT NOT NULL,         -- arxiv | mql5 | glint
    title       TEXT,
    summary     TEXT,
    url         TEXT,
    applied     INTEGER DEFAULT 0,     -- 1 if incorporated into strategy
    relevance   REAL DEFAULT 0.0       -- 0-1 relevance score
);

-- Nightly reports
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT UNIQUE NOT NULL,
    trades_total INTEGER DEFAULT 0,
    trades_win   INTEGER DEFAULT 0,
    trades_loss  INTEGER DEFAULT 0,
    pnl_day      REAL DEFAULT 0,
    win_rate     REAL DEFAULT 0,
    best_setup   TEXT,
    worst_setup  TEXT,
    lessons_text TEXT,
    plan_tomorrow TEXT,
    goals_snapshot TEXT,               -- JSON of goal progress
    report_text  TEXT                  -- full Telegram message
);
```

### Key functions

```python
# episodic_db.py public API
def get_db() -> sqlite3.Connection          # WAL mode, thread-safe
def record_episode(ep: dict) -> int         # returns episode id
def update_episode_result(id, exit, pnl, result, lesson)
def query_similar_episodes(symbol, setup_type, regime, n=10) -> list
def get_setup_stats() -> dict               # win_rate per setup_type
def get_session_stats() -> dict             # win_rate per session
def save_lesson(lesson: dict)
def save_research(item: dict)
def save_report(report: dict)
def get_goals() -> list
def update_goal(goal_id, current_value)
def seed_goals()                            # insert default goals if empty
```

---

## 4. Subsystem 1: `core/autonomous_learner.py`

**Purpose:** Every 60 minutes, analyze real trade history and auto-adjust DecisionFilter weights.

### `_learning_loop()` in supervisor.py

```python
async def _learning_loop(self):
    while self._running:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self.learner.run_analysis
            )
        except Exception as e:
            print(f"[LEARN] Error: {e}")
        await asyncio.sleep(3600)
```

### `AutonomousLearner.run_analysis()`

1. Query `episodes` where `result IN ('WIN','LOSS')` and `ts > 7 days ago`
2. Compute win_rate grouped by: `setup_type`, `regime`, `session`
3. For each group with >= 5 samples:
   - win_rate > 65% → `weight_adj = 1.20` (boost DecisionFilter threshold down by 5pts)
   - win_rate 50-65% → `weight_adj = 1.00` (no change)
   - win_rate < 50% → `weight_adj = 0.80` (raise threshold by 5pts, harder to trigger)
   - win_rate < 35% with >= 10 samples → `weight_adj = 0.50` (near-disable)
4. Save updated weights to `lessons` table
5. Print summary: `[LEARN] USDJPY+H4+CHoCH: 7/9 WIN (77%) → weight +20%`

### Integration with DecisionFilter

`TradingSupervisor` passes current `weight_adj` to `route_signal()`. DecisionFilter score threshold is multiplied by the inverse of weight_adj (higher weight → lower effective threshold → easier to trigger).

---

## 5. Subsystem 2: Enhanced Claude Reasoning (modify `agents/analysis_agent.py`)

**Purpose:** Before scoring each MT5 signal, Claude reasons with real historical context.

### Enhanced prompt structure

```python
REASONING_PROMPT = """
Eres un trader institucional SMC. Analiza este setup y razona paso a paso.

## DATOS TECNICOS
{smc_analysis}

## EPISODIOS HISTORICOS SIMILARES ({n} trades anteriores del bot)
{episodes_text}
# Format: "USDJPY H4 CHoCH+OB BUY → WIN +45pips (2026-05-20)"

## REGIMEN ACTUAL DE MERCADO
{regime}
# From quant_regime.py HMM: trending_up | trending_down | ranging | high_vol

## CONTEXTO MACRO
{glint_context}

## RAZONAMIENTO REQUERIDO
1. ¿Que hace el Smart Money en este punto?
2. ¿Los episodios historicos respaldan o contradicen este setup?
3. ¿El regimen actual favorece esta estrategia?
4. Leccion aplicable de episodios similares (si hay 3+)
5. Nivel de confianza 0-100 y por que

## RESPONDE SOLO EN JSON VALIDO:
{
  "smart_money_action": "string",
  "historical_support": "supports|contradicts|neutral",
  "regime_fit": "favorable|neutral|unfavorable",
  "lesson_applied": "string or null",
  "decision": "LONG|SHORT|WAIT",
  "confidence": 0-100,
  "justification": "max 2 lines"
}
"""
```

### Score adjustment rules

- `confidence >= 75` AND `historical_support == "supports"` → score = min(100, score + 10)
- `confidence < 40` → WAIT override (don't execute regardless of base score)
- `historical_support == "contradicts"` AND 3+ similar losses → WAIT override
- `regime_fit == "unfavorable"` → score = max(0, score - 15)

### Call chain (order matters)

1. `_scan_mt5_symbol()` runs SMC lite analysis → base score
2. If base score >= threshold → call `SMCAnalysisAgent.reason_with_context(signal, similar_episodes, regime)` 
3. Claude returns JSON → adjust score per rules above → final score
4. If final score >= threshold AND no WAIT override → call `_send_mt5_real_order()`
5. Claude API failure → skip step 2-3, use base score unchanged (never block execution)

### Fallback

If Claude API fails (timeout, quota, DNS) → fall back to existing `_run_smc_lite()` without adjustment. Never block trade execution due to API failure.

---

## 6. Subsystem 3: Episodic Memory Integration (in supervisor.py)

**Purpose:** Record every real MT5 trade as an episode, close it when result is known.

### On trade execution (`_send_mt5_real_order`)

```python
# BEFORE order: query similar episodes to enrich Claude reasoning
similar = query_similar_episodes(symbol, setup_type, regime, n=10)

# AFTER successful order:
episode_id = record_episode({
    "ts": now_utc(),
    "symbol": signal.symbol,
    "timeframe": signal.timeframe,
    "direction": order_type,
    "entry": result["price"],
    "sl": sl_val,
    "tp": tp_val,
    "ticket": result["ticket"],
    "score": signal.decision_score,
    "setup_type": detect_setup_type(signal),  # regex on analysis_text: CHoCH+OB | BOS | FVG | OB | plain
    "regime": current_regime,                 # from quant_regime.py: supervisor holds self._current_regime, updated hourly
    "session": current_session(),             # UTC hour → asia[0-8] | london[8-12] | overlap[12-17] | ny[17-21] | off[21-24]
    "reasoning": claude_json_response,
    "macro_ctx": self._last_glint_text,
})
```

### Episode closing

**Episode closer** runs at the end of every `_market_scan_loop()` cycle (every 30s), queries open episodes, checks MT5 positions via `self.mt5.get_positions()`:
- If position closed → fetch exit price from MT5, compute PnL, classify WIN/LOSS/BE
- Extract lesson via simple rule: "WIN on {setup_type} in {regime} → confirm weight"
- Update episode: `update_episode_result(id, exit, pnl, result, lesson)`

---

## 7. Subsystem 4: `core/research_agent.py`

**Purpose:** Every 2 hours, fetch and store new trading research.

### Sources

| Source | Method | Query |
|--------|--------|-------|
| arXiv | REST API (no auth) | `q-fin.TR` + `cs.AI`, last 7 days |
| MQL5 | httpx GET + regex (no BS4) | /articles category=trading-systems |
| Glint | existing connector | already running |

### `ResearchAgent.run_cycle()`

1. Fetch up to 5 new items per source
2. For each: extract title + abstract/summary
3. Score relevance 0-1: contains SMC/order block/ICT/liquidity → 0.7+
4. Save to `research` table if relevance > 0.4
5. Every 24h: take top 3 relevant unread items, summarize for Claude prompt injection

### Adversarial handling

- All HTTP requests: timeout=10s, retry=1, fail silently
- If source unreachable → skip, log `[RESEARCH] mql5 unavailable`, continue
- No external dependency beyond `httpx` (already installed)

---

## 8. Subsystem 5: `core/goals_manager.py`

**Purpose:** Track autonomous bot goals, evaluate daily progress.

### Default goals (seeded on first run)

```python
INITIAL_GOALS = [
    {"goal_id": "edge_score_50",  "description": "Alcanzar Edge Score 50 en Axi Select",
     "metric": "axi_edge_score",  "target": 50,  "horizon": "short"},
    {"goal_id": "winrate_65",     "description": "Win rate > 65% en 100 trades",
     "metric": "win_rate_pct_100","target": 65,  "horizon": "medium"},
    {"goal_id": "axi_challenge",  "description": "Pasar challenge Axi $5K",
     "metric": "challenge_passed","target": 1,   "horizon": "medium"},
    {"goal_id": "funded_5k",      "description": "Cuenta fondeada $5,000 Axi",
     "metric": "funded_usd",      "target": 5000,"horizon": "long"},
    {"goal_id": "funded_1m",      "description": "Llegar a $1,000,000 fondeado",
     "metric": "funded_usd",      "target": 1000000, "horizon": "ultimate"},
]
```

### `GoalsManager.evaluate()`

Computes current value for each metric from `episodes` table:
- `win_rate_pct_100` = wins/total for last 100 closed episodes
- `axi_edge_score` = (win_rate * 0.6) + (profit_factor * 0.4) * 50 (proxy formula)
- Updates `goals` table with current + progress_pct

---

## 9. Subsystem 6: `core/nightly_reporter.py`

**Purpose:** Every day at 22:00 UTC, generate and send a Telegram report.

### `NightlyReporter.generate_report(date)`

Queries `episodes` for today's trades:
```
trades_today = SELECT * FROM episodes WHERE ts LIKE '{date}%' AND result != 'OPEN'
```

Computes: total, wins, losses, win_rate, total_pnl, best_setup, worst_setup.

Queries `lessons` for recent weight changes.
Queries `goals` for current progress.
Queries `research` for top 1 new article today.

### Report format (Telegram HTML)

```
📊 <b>REPORTE AUTÓNOMO — {date}</b>
━━━━━━━━━━━━━━━━━━━━
Trades hoy: {total} ({wins}W / {losses}L) — {win_rate:.1f}%
P&L: {pnl:+.2f} USD | Drawdown: {dd:.1f}%
━━━━━━━━━━━━━━━━━━━━
🧠 <b>Lecciones:</b>
• {lesson_1}
• {lesson_2}

💡 <b>Plan mañana:</b>
• {plan_1}
• {plan_2}

🎯 <b>Metas:</b>
• Win rate 65%: {winrate_progress:.0f}% completado
• Edge Score Axi: {edge_score}/50

📚 Nuevo: "{research_title}"
━━━━━━━━━━━━━━━━━━━━
```

### Timing

`_nightly_report_loop()` checks current UTC hour every 60s. Fires when `hour == 22 and minute < 2` and hasn't fired today. Falls back to last known data if episodes table is empty.

---

## 10. Testing Strategy

**New test files** (all must coexist with existing 1003 tests):

| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_episodic_db.py` | ~35 | All DB functions, WAL, concurrent write |
| `tests/test_autonomous_learner.py` | ~25 | Weight adjustment logic, edge cases |
| `tests/test_research_agent.py` | ~20 | Fetch mocking, relevance scoring, fallback |
| `tests/test_goals_manager.py` | ~20 | Goal seeding, metric computation, progress |
| `tests/test_nightly_reporter.py` | ~20 | Report generation, empty-data fallback |
| `tests/test_reasoning_prompt.py` | ~15 | Prompt construction, JSON parse, fallback |

**Total new tests: ~135**  
**Target after implementation: 1003 + 135 = ~1138 tests passing**

All tests use in-memory SQLite (`:memory:`) — no disk writes during tests.
All Claude API calls mocked with `unittest.mock.patch`.
All HTTP calls mocked — no external network in tests.

---

## 11. Implementation Order (within single PR)

1. `memory/episodic_db.py` + `tests/test_episodic_db.py`
2. `core/autonomous_learner.py` + `tests/test_autonomous_learner.py`
3. `core/research_agent.py` + `tests/test_research_agent.py`
4. `core/goals_manager.py` + `tests/test_goals_manager.py`
5. `core/nightly_reporter.py` + `tests/test_nightly_reporter.py`
6. Modify `agents/analysis_agent.py` (enhanced reasoning) + `tests/test_reasoning_prompt.py`
7. Modify `core/supervisor.py` (wire all loops + episodic recording)
8. Update `CLAUDE.md`
9. Run full pytest — must be green
10. `git push` to `github.com/joseg09dg-888/SMCBOTTRAIDING`

---

## 12. Non-Goals (explicitly out of scope)

- Real-time position monitoring beyond MT5 connector (already handles this)
- ChromaDB vector search (SQLite sufficient for current trade volume)
- YouTube transcript learning (already in continuous_learning.py, not duplicating)
- UI dashboard (Telegram reports are sufficient)
- Automated FTMO challenge submission (manual for now)

---

## 13. Constraints (from CLAUDE.md)

- NEVER use PowerShell Out-File / Set-Content for .py files → use Write tool only
- NEVER claim "fixed" without verification
- NEVER break 1003 existing tests
- supervisor.py encoding is sensitive — use Write tool for all edits
- MT5 only available via bot process (IPC exclusive) — no separate scripts
