# core/risk_governor.py
"""
RiskGovernor: autonomous circuit breaker for MT5-real trading.

Runs on a schedule (wired into TradingSupervisor as _risk_governor_loop).
Each cycle it:

  1. Looks at the last `min_trades` CLOSED real trades per symbol (from MT5
     deal history). If a symbol's rolling win rate drops below `suspend_wr`,
     the symbol is removed from the active scan list automatically.
  2. Re-activates a suspended symbol after `cooldown_hours` so it gets a
     fresh evaluation window with whatever fixes were made in the meantime.
  3. Looks at current account drawdown vs the $100K starting balance and
     scales a global risk multiplier down in tiers as drawdown grows,
     stepping it back up one tier at a time as drawdown recovers.

State persists to memory/risk_governor_state.json so decisions survive PM2
restarts. The pure decision logic (evaluate/format_report) takes plain dicts
so it can be unit-tested without a live MT5 connection.
"""
import json
import os
from datetime import datetime, timedelta, timezone


class RiskGovernor:
    def __init__(
        self,
        all_symbols,
        state_path="memory/risk_governor_state.json",
        min_trades=8,
        suspend_wr=0.25,
        cooldown_hours=168,  # 7 days
        dd_tiers=(),  # drawdown tiers desactivados — volumen fijo para cumplir objetivo $250/dia
        initial_suspended=None,
    ):
        self.all_symbols = list(all_symbols)
        self.state_path = state_path
        self.min_trades = min_trades
        self.suspend_wr = suspend_wr
        self.cooldown_hours = cooldown_hours
        self.dd_tiers = sorted(dd_tiers, key=lambda x: -x[0])
        self._state = self._load_state(initial_suspended or {})

    # ── persistence ─────────────────────────────────────────────────
    def _load_state(self, initial_suspended: dict) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    state = json.load(f)
                state.setdefault("suspended", {})
                state.setdefault("risk_multiplier", 1.0)
                state.setdefault("history", [])
                return state
            except Exception:
                pass
        now = datetime.now(timezone.utc).isoformat()
        state = {
            "suspended": {
                sym: {"reason": reason, "since": now}
                for sym, reason in initial_suspended.items()
            },
            "risk_multiplier": 1.0,
            "history": [],
        }
        self._write_state(state)
        return state

    def _write_state(self, state: dict) -> None:
        dirpath = os.path.dirname(self.state_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def save_state(self) -> None:
        self._write_state(self._state)

    # ── queries ─────────────────────────────────────────────────────
    def active_symbols(self) -> list:
        return [s for s in self.all_symbols if s not in self._state["suspended"]]

    def suspended_symbols(self) -> dict:
        return dict(self._state["suspended"])

    def risk_multiplier(self) -> float:
        return self._state.get("risk_multiplier", 1.0)

    # ── decision logic ──────────────────────────────────────────────
    @staticmethod
    def stats_from_deals(deals: list) -> dict:
        """deals: list of {'profit': float} dicts (oldest..newest)."""
        n = len(deals)
        if n == 0:
            return {"n": 0, "wins": 0, "wr": None, "pnl": 0.0}
        wins = sum(1 for d in deals if d["profit"] > 0)
        pnl = sum(d["profit"] for d in deals)
        return {"n": n, "wins": wins, "wr": wins / n, "pnl": pnl}

    def evaluate(self, symbol_deals: dict, drawdown_pct: float) -> dict:
        """
        symbol_deals: {symbol: [ {'profit': float, ...}, ... ]} oldest..newest,
                       only the most recent `min_trades` are considered.
        drawdown_pct: (initial_balance - balance) / initial_balance, >= 0 means loss.
        Returns a dict describing what changed this cycle.
        """
        now = datetime.now(timezone.utc)
        changes = {"suspended": [], "reactivated": [], "risk_multiplier": None}

        for sym in self.all_symbols:
            deals = (symbol_deals.get(sym) or [])[-self.min_trades:]
            stats = self.stats_from_deals(deals)
            is_suspended = sym in self._state["suspended"]

            if is_suspended:
                since_raw = self._state["suspended"][sym].get("since")
                try:
                    since = datetime.fromisoformat(since_raw)
                except Exception:
                    since = now
                # Per-entry cooldown_hours overrides class default (allows 30-day bans)
                entry_cooldown = self._state["suspended"][sym].get("cooldown_hours", self.cooldown_hours)
                if now - since >= timedelta(hours=entry_cooldown):
                    del self._state["suspended"][sym]
                    changes["reactivated"].append({"symbol": sym, "reason": "cooldown expirado, reevaluando"})
            else:
                if stats["n"] >= self.min_trades and stats["wr"] is not None and stats["wr"] < self.suspend_wr:
                    self._state["suspended"][sym] = {
                        "reason": (
                            f"WR {stats['wr']*100:.1f}% en ultimos {stats['n']} trades "
                            f"(pnl ${stats['pnl']:+.2f})"
                        ),
                        "since": now.isoformat(),
                    }
                    changes["suspended"].append({"symbol": sym, **stats})

        # Drawdown-based risk multiplier
        new_mult = 1.0
        for dd_thresh, mult in self.dd_tiers:
            if drawdown_pct >= dd_thresh:
                new_mult = mult
                break
        old_mult = self._state.get("risk_multiplier", 1.0)
        if new_mult != old_mult:
            if new_mult < old_mult:
                # drawdown worsened beyond a tier -> cut immediately
                self._state["risk_multiplier"] = new_mult
            else:
                # drawdown recovered -> step up only one tier at a time
                tiers_sorted = sorted({1.0, *(m for _, m in self.dd_tiers)})
                idx = tiers_sorted.index(old_mult) if old_mult in tiers_sorted else 0
                self._state["risk_multiplier"] = tiers_sorted[min(idx + 1, len(tiers_sorted) - 1)]
            changes["risk_multiplier"] = {
                "from": old_mult,
                "to": self._state["risk_multiplier"],
                "drawdown_pct": drawdown_pct,
            }

        self._state["last_run"] = now.isoformat()
        self._state["last_drawdown_pct"] = drawdown_pct
        self._state.setdefault("history", []).append({
            "ts": now.isoformat(),
            "active_symbols": self.active_symbols(),
            "risk_multiplier": self._state["risk_multiplier"],
            "changes": changes,
        })
        self._state["history"] = self._state["history"][-50:]
        self.save_state()
        return changes

    @staticmethod
    def has_changes(changes: dict) -> bool:
        return bool(changes["suspended"] or changes["reactivated"] or changes["risk_multiplier"])

    # ── reporting ───────────────────────────────────────────────────
    def status_line(self) -> str:
        active = self.active_symbols()
        susp = self._state["suspended"]
        line = f"RiskGov: {len(active)} activos ({', '.join(active)}) | riesgo x{self.risk_multiplier():.2f}"
        if susp:
            line += f" | suspendidos: {', '.join(susp.keys())}"
        return line

    def format_report(self, changes: dict, balance: float, drawdown_pct: float) -> str:
        lines = ["<b>RISK GOVERNOR -- ajuste automatico</b>"]
        lines.append(f"Balance: ${balance:,.2f} | Drawdown: {drawdown_pct*100:.2f}%")
        lines.append(f"Riesgo activo: x{self.risk_multiplier():.2f}")
        active = self.active_symbols()
        lines.append(f"Pares activos ({len(active)}): {', '.join(active) if active else '(ninguno)'}")
        if self._state["suspended"]:
            lines.append("Suspendidos:")
            for sym, info in self._state["suspended"].items():
                lines.append(f"  - {sym}: {info['reason']}")
        if changes["suspended"]:
            lines.append("\n<b>NUEVO -- suspendido este ciclo:</b>")
            for c in changes["suspended"]:
                lines.append(f"  - {c['symbol']}: WR {c['wr']*100:.1f}% / {c['n']} trades, pnl ${c['pnl']:+.2f}")
        if changes["reactivated"]:
            lines.append("\n<b>NUEVO -- reactivado este ciclo:</b>")
            for c in changes["reactivated"]:
                lines.append(f"  - {c['symbol']}: {c['reason']}")
        if changes["risk_multiplier"]:
            cm = changes["risk_multiplier"]
            lines.append(
                f"\n<b>NUEVO -- riesgo ajustado:</b> x{cm['from']:.2f} -> x{cm['to']:.2f} "
                f"(drawdown {cm['drawdown_pct']*100:.2f}%)"
            )
        return "\n".join(lines)


def fetch_recent_deals_by_symbol(symbols, lookback_days=45, max_per_symbol=20) -> dict:
    """
    Live MT5 helper: returns {symbol: [{'profit': float, 'time': int}, ...]}
    (oldest..newest, most recent `max_per_symbol` closed positions per symbol).
    Profit includes swap + commission. Requires an initialized MT5 session.
    """
    import MetaTrader5 as mt5

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    until = datetime.now(timezone.utc) + timedelta(days=1)
    deals = mt5.history_deals_get(since, until) or []

    positions: dict = {}
    for d in deals:
        if d.entry == 1 and d.volume > 0.10:  # solo swings — ignorar micro-scalps
            positions.setdefault((d.symbol, d.position_id), []).append(d)

    by_symbol = {s: [] for s in symbols}
    for (sym, _pid), ds in positions.items():
        if sym not in by_symbol:
            continue
        total = sum(d.profit + d.swap + d.commission for d in ds)
        ts = max(d.time for d in ds)
        by_symbol[sym].append({"profit": total, "time": ts})

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x["time"])
        by_symbol[sym] = by_symbol[sym][-max_per_symbol:]

    return by_symbol
