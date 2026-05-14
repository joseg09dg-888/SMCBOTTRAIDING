"""
AgentMemoryManager — 4-layer persistent memory for all bot agents.

Layer 1: Short-term RAM dict (cleared on restart)
Layer 2: Medium-term SQLite (memory/agent_memory.db)
Layer 3: Long-term vector store (ChromaDB if available, JSON fallback)
Layer 4: Per-agent JSON files (memory/agents/<agent>.json)

Plus: shared_context.json for cross-agent communication.
"""
import copy
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


AGENT_NAMES = [
    "smc_agent", "signal_agent", "prediction_agent",
    "lunar_agent", "elliott_agent", "institutional_flow_agent",
    "alternative_data_agent", "microstructure_agent", "fed_sentiment_agent",
    "onchain_agent", "geopolitical_agent", "chaos_agent",
    "retail_psychology_agent", "historical_agent", "glint_connector",
    "risk_manager", "decision_filter", "telegram_commander",
    "market_connector", "youtube_trainer", "supervisor",
]

_AGENT_DEFAULT = {
    "total_signals": 0,
    "correct_signals": 0,
    "accuracy": 0.0,
    "best_conditions": [],
    "worst_conditions": [],
    "last_updated": "",
    "lessons_learned": [],
    "weight_in_score": 1.0,
}

_DDL = """
CREATE TABLE IF NOT EXISTS trade_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    entry        REAL,
    exit_price   REAL,
    pnl          REAL,
    won          INTEGER,          -- 1 or 0
    agents_voted TEXT,             -- JSON list
    timestamp    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent        TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    signal_type  TEXT,             -- "entry", "exit", "alert"
    direction    TEXT,             -- "bullish", "bearish", "neutral"
    score        INTEGER,
    metadata     TEXT,             -- JSON
    timestamp    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    agent               TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    decision            TEXT,
    reason              TEXT,
    score_contribution  INTEGER,
    timestamp           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_success (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern      TEXT    NOT NULL,
    symbol       TEXT,
    timeframe    TEXT,
    wins         INTEGER DEFAULT 0,
    losses       INTEGER DEFAULT 0,
    last_seen    TEXT
);

CREATE TABLE IF NOT EXISTS market_conditions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    condition    TEXT,             -- "trending_up","trending_down","ranging"
    volatility   TEXT,             -- "low","medium","high"
    session      TEXT,
    timestamp    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signal_agent   ON signal_history(agent);
CREATE INDEX IF NOT EXISTS idx_signal_symbol  ON signal_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_symbol   ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_decision_agent ON agent_decisions(agent);
"""


class AgentMemoryManager:
    """Central memory hub for all 21 bot agents."""

    def __init__(
        self,
        db_path: str = "memory/agent_memory.db",
        agents_dir: str = "memory/agents",
        shared_ctx_path: str = "memory/shared_context.json",
        vector_dir: str = "memory/vector_store",
    ):
        # Paths
        self._db_path       = db_path
        self._agents_dir    = Path(agents_dir)
        self._ctx_path      = Path(shared_ctx_path)
        self._vector_dir    = Path(vector_dir)

        # Create dirs
        for p in (self._agents_dir, self._vector_dir,
                  Path(db_path).parent):
            p.mkdir(parents=True, exist_ok=True)

        # Layer 1: Short-term RAM
        self._short_term: Dict[str, Dict[str, Any]] = {
            name: {} for name in AGENT_NAMES
        }

        # Layer 2: SQLite
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

        # Layer 3: Vector store (ChromaDB preferred, JSON fallback)
        self._chroma = None
        self._chroma_col = None
        self._json_kb: List[Dict] = []
        self._init_vector_store()

        # Layer 4: Per-agent JSON
        self._agent_data: Dict[str, Dict] = {}
        self._load_all_agent_data()

        # Shared context
        self._shared_ctx: Dict[str, Any] = self._load_shared_context()

    # ── Layer 1: Short-term ───────────────────────────────────────────────────

    def set_short_term(self, agent: str, key: str, value: Any):
        if agent not in self._short_term:
            self._short_term[agent] = {}
        self._short_term[agent][key] = value

    def get_short_term(self, agent: str, key: str, default: Any = None) -> Any:
        return self._short_term.get(agent, {}).get(key, default)

    def clear_short_term(self, agent: Optional[str] = None):
        if agent:
            self._short_term[agent] = {}
        else:
            for name in self._short_term:
                self._short_term[name] = {}

    # ── Layer 2: SQLite ───────────────────────────────────────────────────────

    def record_signal(
        self,
        agent: str,
        symbol: str,
        signal_type: str,
        direction: str,
        score: int,
        metadata: Optional[Dict] = None,
    ):
        self._conn.execute(
            "INSERT INTO signal_history "
            "(agent, symbol, signal_type, direction, score, metadata, timestamp)"
            " VALUES (?,?,?,?,?,?,?)",
            (agent, symbol, signal_type, direction, score,
             json.dumps(metadata or {}), _now()),
        )
        self._conn.commit()

    def record_trade(
        self,
        symbol: str,
        entry: float,
        exit_price: float,
        pnl: float,
        agents_voted: List[str],
        winning: bool,
    ):
        self._conn.execute(
            "INSERT INTO trade_history "
            "(symbol, entry, exit_price, pnl, won, agents_voted, timestamp)"
            " VALUES (?,?,?,?,?,?,?)",
            (symbol, entry, exit_price, pnl, int(winning),
             json.dumps(agents_voted), _now()),
        )
        self._conn.commit()

    def record_decision(
        self,
        agent: str,
        symbol: str,
        decision: str,
        reason: str,
        score_contribution: int,
    ):
        self._conn.execute(
            "INSERT INTO agent_decisions "
            "(agent, symbol, decision, reason, score_contribution, timestamp)"
            " VALUES (?,?,?,?,?,?)",
            (agent, symbol, decision, reason, score_contribution, _now()),
        )
        self._conn.commit()

    def record_pattern(self, pattern: str, symbol: str, timeframe: str, won: bool):
        row = self._conn.execute(
            "SELECT id FROM pattern_success WHERE pattern=? AND symbol=? AND timeframe=?",
            (pattern, symbol, timeframe),
        ).fetchone()
        if row:
            col = "wins" if won else "losses"
            self._conn.execute(
                f"UPDATE pattern_success SET {col}={col}+1, last_seen=? WHERE id=?",
                (_now(), row["id"]),
            )
        else:
            self._conn.execute(
                "INSERT INTO pattern_success (pattern, symbol, timeframe, wins, losses, last_seen)"
                " VALUES (?,?,?,?,?,?)",
                (pattern, symbol, timeframe, int(won), int(not won), _now()),
            )
        self._conn.commit()

    def get_agent_stats(self, agent: str) -> Dict:
        signals = self._conn.execute(
            "SELECT COUNT(*) as n FROM signal_history WHERE agent=?", (agent,)
        ).fetchone()["n"]
        decisions = self._conn.execute(
            "SELECT AVG(score_contribution) as avg_score FROM agent_decisions WHERE agent=?",
            (agent,),
        ).fetchone()["avg_score"]
        return {
            "agent":       agent,
            "total_signals_db": signals,
            "avg_score_contribution": round(decisions or 0, 2),
        }

    def get_recent_trades(self, days: int = 7) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM trade_history WHERE timestamp > ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pattern_stats(self, pattern: str = "") -> List[Dict]:
        if pattern:
            rows = self._conn.execute(
                "SELECT * FROM pattern_success WHERE pattern LIKE ?", (f"%{pattern}%",)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pattern_success ORDER BY wins DESC LIMIT 20"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Layer 3: Vector store ─────────────────────────────────────────────────

    def _init_vector_store(self):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._vector_dir))
            self._chroma_col = client.get_or_create_collection("bot_knowledge")
            self._chroma = client
        except Exception:
            # JSON fallback — load existing KB
            kb_file = self._vector_dir / "knowledge_base.json"
            if kb_file.exists():
                try:
                    self._json_kb = json.loads(kb_file.read_text(encoding="utf-8"))
                except Exception:
                    self._json_kb = []

    def store_knowledge(self, agent: str, text: str, metadata: Optional[Dict] = None):
        meta = {"agent": agent, "timestamp": _now(), **(metadata or {})}
        if self._chroma_col is not None:
            doc_id = f"{agent}_{int(time.time()*1000)}"
            try:
                self._chroma_col.add(
                    documents=[text], metadatas=[meta], ids=[doc_id]
                )
                return
            except Exception:
                pass
        # JSON fallback
        self._json_kb.append({"text": text, "meta": meta})
        self._save_json_kb()

    def search_knowledge(self, query: str, n_results: int = 5) -> List[Dict]:
        if self._chroma_col is not None:
            try:
                res = self._chroma_col.query(
                    query_texts=[query], n_results=min(n_results, 10)
                )
                return [
                    {"text": doc, "meta": meta}
                    for doc, meta in zip(
                        res["documents"][0], res["metadatas"][0]
                    )
                ]
            except Exception:
                pass
        # JSON fallback: simple keyword search
        query_lower = query.lower()
        matches = [
            e for e in self._json_kb
            if any(w in e["text"].lower() for w in query_lower.split())
        ]
        return matches[:n_results]

    def _save_json_kb(self):
        kb_file = self._vector_dir / "knowledge_base.json"
        try:
            kb_file.write_text(
                json.dumps(self._json_kb[-2000:], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_knowledge_count(self) -> int:
        if self._chroma_col is not None:
            try:
                return self._chroma_col.count()
            except Exception:
                pass
        return len(self._json_kb)

    # ── Layer 4: Per-agent JSON ───────────────────────────────────────────────

    def _agent_file(self, agent: str) -> Path:
        return self._agents_dir / f"{agent}.json"

    def _load_all_agent_data(self):
        for name in AGENT_NAMES:
            f = self._agent_file(name)
            if f.exists():
                try:
                    self._agent_data[name] = json.loads(f.read_text(encoding="utf-8"))
                    continue
                except Exception:
                    pass
            # Create default
            self._agent_data[name] = copy.deepcopy(_AGENT_DEFAULT)
            self._save_agent_data(name)

    def _save_agent_data(self, agent: str):
        try:
            self._agent_file(agent).write_text(
                json.dumps(self._agent_data[agent], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def update_agent_data(self, agent: str, signal_correct: bool,
                           condition: Optional[str] = None):
        if agent not in self._agent_data:
            self._agent_data[agent] = copy.deepcopy(_AGENT_DEFAULT)
        d = self._agent_data[agent]
        d["total_signals"]   += 1
        d["correct_signals"] += int(signal_correct)
        total = d["total_signals"]
        d["accuracy"]     = round(d["correct_signals"] / total, 4) if total else 0.0
        d["last_updated"] = _now()

        if condition:
            bucket = "best_conditions" if signal_correct else "worst_conditions"
            lst = d[bucket]
            if condition not in lst:
                lst.append(condition)
            d[bucket] = lst[-10:]   # keep last 10

        self._save_agent_data(agent)

    def get_agent_accuracy(self, agent: str) -> float:
        return self._agent_data.get(agent, _AGENT_DEFAULT)["accuracy"]

    def update_agent_weight(self, agent: str, weight: float):
        if agent not in self._agent_data:
            self._agent_data[agent] = copy.deepcopy(_AGENT_DEFAULT)
        self._agent_data[agent]["weight_in_score"] = round(max(0.0, min(weight, 3.0)), 4)
        self._save_agent_data(agent)

    def get_agent_weight(self, agent: str) -> float:
        return self._agent_data.get(agent, _AGENT_DEFAULT)["weight_in_score"]

    def add_lesson(self, agent: str, lesson: str):
        if agent not in self._agent_data:
            self._agent_data[agent] = copy.deepcopy(_AGENT_DEFAULT)
        lessons = self._agent_data[agent]["lessons_learned"]
        lessons.append({"lesson": lesson, "timestamp": _now()})
        self._agent_data[agent]["lessons_learned"] = lessons[-20:]  # keep last 20
        self._save_agent_data(agent)
        # Also store in vector KB
        self.store_knowledge(agent, lesson, {"type": "lesson"})

    def get_all_agent_stats(self) -> Dict[str, Dict]:
        return {
            name: {
                "accuracy":       self.get_agent_accuracy(name),
                "weight":         self.get_agent_weight(name),
                "total_signals":  self._agent_data.get(name, {}).get("total_signals", 0),
                "correct_signals": self._agent_data.get(name, {}).get("correct_signals", 0),
            }
            for name in AGENT_NAMES
        }

    # ── Shared context ────────────────────────────────────────────────────────

    def _load_shared_context(self) -> Dict:
        if self._ctx_path.exists():
            try:
                return json.loads(self._ctx_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "market_condition":  "unknown",
            "last_event":        "",
            "recent_avg_score":  0,
            "best_agents_week":  [],
            "active_alerts":     [],
            "last_updated":      "",
        }

    def _save_shared_context(self):
        try:
            self._shared_ctx["last_updated"] = _now()
            self._ctx_path.write_text(
                json.dumps(self._shared_ctx, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def set_shared_context(self, key: str, value: Any):
        self._shared_ctx[key] = value
        self._save_shared_context()

    def get_shared_context(self, key: str, default: Any = None) -> Any:
        return self._shared_ctx.get(key, default)

    def get_full_shared_context(self) -> Dict:
        return dict(self._shared_ctx)

    def broadcast_alert(self, agent: str, message: str):
        """Any agent can post an alert visible to all others."""
        alerts = self._shared_ctx.get("active_alerts", [])
        alerts.append({"from": agent, "message": message, "timestamp": _now()})
        self._shared_ctx["active_alerts"] = alerts[-10:]
        self._save_shared_context()

    # ── Summary for /memory command ───────────────────────────────────────────

    def memory_summary(self) -> str:
        trades  = self._conn.execute("SELECT COUNT(*) as n FROM trade_history").fetchone()["n"]
        signals = self._conn.execute("SELECT COUNT(*) as n FROM signal_history").fetchone()["n"]
        patterns = self._conn.execute("SELECT COUNT(*) as n FROM pattern_success").fetchone()["n"]
        kb_count = self.get_knowledge_count()

        all_stats = self.get_all_agent_stats()
        ranked = sorted(
            [(n, s) for n, s in all_stats.items() if s["total_signals"] > 0],
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        underperforming = [
            n for n, s in all_stats.items()
            if s["total_signals"] > 10 and s["accuracy"] < 0.35
        ]

        top3 = "\n".join(
            f"{'🥇🥈🥉'[i]} {n}: {s['accuracy']*100:.0f}% ({s['total_signals']} señales)"
            for i, (n, s) in enumerate(ranked[:3])
        ) or "Sin señales registradas aún"

        under_str = (
            "\n".join(f"⚠️ {n}: {all_stats[n]['accuracy']*100:.0f}%" for n in underperforming)
            if underperforming else "✅ Todos en rango aceptable"
        )

        return (
            "🧠 *MEMORIA DEL BOT*\n"
            "─────────────────────\n"
            f"Trades recordados: {trades:,}\n"
            f"Señales registradas: {signals:,}\n"
            f"Patrones aprendidos: {patterns:,}\n"
            f"Conocimiento en KB: {kb_count:,} entradas\n"
            "─────────────────────\n"
            f"Top agentes por accuracy:\n{top3}\n"
            "─────────────────────\n"
            f"Agentes en revisión:\n{under_str}"
        )

    def close(self):
        self._conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
