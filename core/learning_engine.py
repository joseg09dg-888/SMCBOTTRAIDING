"""
LearningEngine — daily review, weight adjustment, and Telegram report.

Each day at 23:00 UTC:
1. Reviews trades and signals from the day
2. Updates each agent's accuracy and weight
3. Generates and sends a Telegram report
4. Stores lessons learned in AgentMemoryManager
"""
import asyncio
from datetime import datetime, timezone, date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from core.agent_memory import AgentMemoryManager, AGENT_NAMES


# Weight adjustment rules
_ACCURACY_THRESHOLDS = {
    "increase":  0.70,   # accuracy > 70% → weight +5%
    "maintain":  0.50,   # 50-70% → no change
    "reduce":    0.30,   # < 50% → weight -10%
    "disable":   0.30,   # < 30% for 7 consecutive days → weight → 0.1 (not 0, soft disable)
}
_WEIGHT_INCREASE = 0.05   # +5%
_WEIGHT_REDUCE   = 0.10   # -10%


class LearningEngine:
    """
    Analyses agent performance daily, adjusts weights, stores lessons,
    and sends a Telegram performance report.
    """

    def __init__(self, memory: AgentMemoryManager):
        self.memory = memory
        self._last_review_date: Optional[date] = None

    # ── Daily review ──────────────────────────────────────────────────────────

    def daily_review(self) -> Dict:
        """
        Reviews today's trades and signals.
        Returns a dict with performance data per agent.
        """
        trades  = self.memory.get_recent_trades(days=1)
        all_stats = self.memory.get_all_agent_stats()

        performance: Dict[str, Dict] = {}
        for agent in AGENT_NAMES:
            stats = all_stats[agent]
            acc   = stats["accuracy"]
            weight = stats["weight"]
            action = self._weight_action(acc, stats["total_signals"])
            performance[agent] = {
                "accuracy":       acc,
                "weight":         weight,
                "total_signals":  stats["total_signals"],
                "action":         action,  # "increase", "maintain", "reduce", "new"
            }

        # Adjust weights based on accuracy
        self.adjust_agent_weights(performance)

        # Update shared context
        winning = sum(1 for t in trades if t.get("won"))
        total   = len(trades)
        daily_pnl = sum(t.get("pnl", 0) for t in trades)
        self.memory.set_shared_context("recent_avg_score",
            round(daily_pnl, 2) if trades else 0)

        top_agents = sorted(
            [(n, p["accuracy"]) for n, p in performance.items()
             if p["total_signals"] > 0],
            key=lambda x: x[1], reverse=True,
        )[:3]
        self.memory.set_shared_context("best_agents_week",
            [n for n, _ in top_agents])

        self._last_review_date = date.today()

        return {
            "trades_today":   total,
            "wins_today":     winning,
            "losses_today":   total - winning,
            "daily_pnl":      daily_pnl,
            "win_rate":       winning / total if total else 0.0,
            "agent_performance": performance,
        }

    def _weight_action(self, accuracy: float, total_signals: int) -> str:
        if total_signals < 5:
            return "new"
        if accuracy >= _ACCURACY_THRESHOLDS["increase"]:
            return "increase"
        if accuracy >= _ACCURACY_THRESHOLDS["maintain"]:
            return "maintain"
        return "reduce"

    def adjust_agent_weights(self, performance: Dict[str, Dict]):
        """Adjusts each agent's weight in AgentMemoryManager based on accuracy."""
        for agent, data in performance.items():
            action = data["action"]
            current_weight = data["weight"]
            if action == "increase":
                new_weight = current_weight * (1 + _WEIGHT_INCREASE)
            elif action == "reduce":
                new_weight = current_weight * (1 - _WEIGHT_REDUCE)
                # Generate lesson if badly underperforming
                if data["accuracy"] < _ACCURACY_THRESHOLDS["disable"]:
                    self.memory.add_lesson(
                        agent,
                        f"Accuracy below 30% ({data['accuracy']*100:.0f}%) — "
                        f"weight reduced to minimum. Review conditions.",
                    )
            else:
                continue   # "maintain" or "new" — no change
            self.memory.update_agent_weight(agent, new_weight)

    # ── Evaluate individual signal ────────────────────────────────────────────

    def evaluate_agent_signal(
        self,
        agent: str,
        signal_was_correct: bool,
        condition: Optional[str] = None,
    ):
        """Called after a trade closes to record agent accuracy."""
        self.memory.update_agent_data(agent, signal_was_correct, condition)

    # ── Leaderboard helpers ───────────────────────────────────────────────────

    def get_top_agents(self, n: int = 3) -> List[Tuple[str, float]]:
        """Returns top N agents by accuracy (min 5 signals)."""
        stats = self.memory.get_all_agent_stats()
        ranked = sorted(
            [(name, s["accuracy"]) for name, s in stats.items()
             if s["total_signals"] >= 5],
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:n]

    def get_underperforming_agents(self) -> List[Tuple[str, float]]:
        """Returns agents with accuracy < 35% and >= 10 signals."""
        stats = self.memory.get_all_agent_stats()
        return [
            (name, s["accuracy"])
            for name, s in stats.items()
            if s["total_signals"] >= 10 and s["accuracy"] < 0.35
        ]

    # ── Daily Telegram report ─────────────────────────────────────────────────

    def generate_daily_report(self, review_data: Optional[Dict] = None) -> str:
        if review_data is None:
            review_data = self.daily_review()

        total   = review_data["trades_today"]
        wins    = review_data["wins_today"]
        losses  = review_data["losses_today"]
        pnl     = review_data["daily_pnl"]
        wr      = review_data["win_rate"]
        perf    = review_data["agent_performance"]

        pnl_sign = "+" if pnl >= 0 else ""
        wr_pct   = f"{wr*100:.0f}%"

        # Agent learning lines
        learning_lines = []
        for agent, data in sorted(
            perf.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        ):
            if data["total_signals"] < 2:
                continue
            acc = data["accuracy"] * 100
            action = data["action"]
            if action == "increase":
                icon = "OK"
                note = f"peso +{_WEIGHT_INCREASE*100:.0f}%"
            elif action == "reduce":
                icon = "WARN" if acc >= 30 else "BAD"
                note = f"peso -{_WEIGHT_REDUCE*100:.0f}%"
            else:
                icon = "-"
                note = "sin cambio"
            label = agent.replace("_", " ")
            learning_lines.append(f"{icon} {label}: {acc:.0f}% accuracy -> {note}")

        learning_str = "\n".join(learning_lines[:8]) or "Sin senales suficientes hoy"

        # Lessons
        top_agents = self.get_top_agents(1)
        lesson_str = ""
        if top_agents:
            best_name, _ = top_agents[0]
            lessons = self.memory._agent_data.get(best_name, {}).get("lessons_learned", [])
            if lessons:
                lesson_str = f"\nLeccion del dia ({best_name}):\n{lessons[-1]['lesson']}"

        return (
            f"REPORTE DIARIO DEL BOT\n"
            f"Trades: {total} | Ganados: {wins} | Perdidos: {losses}\n"
            f"Win Rate: {wr_pct} | P&L: {pnl_sign}${abs(pnl):.2f}\n"
            f"─────────────────────\n"
            f"Aprendizaje del dia:\n"
            f"{learning_str}"
            f"{lesson_str}"
        )

    async def schedule_daily_report(
        self,
        send_fn: Callable,
        report_hour_utc: int = 23,
    ):
        """
        Coroutine that fires daily at report_hour_utc UTC.
        Pass send_fn = async callable that accepts a string message.
        """
        while True:
            now = datetime.now(timezone.utc)
            # Seconds until next report_hour
            next_report = now.replace(
                hour=report_hour_utc, minute=0, second=0, microsecond=0
            )
            if next_report <= now:
                next_report += timedelta(days=1)
            wait_seconds = (next_report - now).total_seconds()
            await asyncio.sleep(wait_seconds)

            try:
                review = self.daily_review()
                report = self.generate_daily_report(review)
                await send_fn(report)
            except Exception as e:
                print(f"[LearningEngine] Daily report error: {e}")
