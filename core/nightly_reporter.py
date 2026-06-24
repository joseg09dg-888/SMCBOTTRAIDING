# core/nightly_reporter.py
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from memory.episodic_db import get_db, save_report, get_goals
from core.goals_manager import GoalsManager


class NightlyReporter:
    """
    Generates and optionally sends nightly Telegram reports at 22:00 UTC.
    Reads from episodic DB: today's trades, lessons, goals, top research.
    """

    def __init__(self, conn: sqlite3.Connection = None,
                 telegram_bot=None, chat_id: str = None):
        self._conn      = conn
        self._bot       = telegram_bot
        self._chat_id   = chat_id
        self._fired_dates: set = set()

    def _get_conn(self) -> sqlite3.Connection:
        conn = self._conn or get_db()
        conn.row_factory = sqlite3.Row
        return conn

    def should_fire(self, now: datetime = None) -> bool:
        dt  = now or datetime.now(timezone.utc)
        key = dt.strftime("%Y-%m-%d")
        # Reporte nightly 22:00 UTC (5pm Colombia) Y cierre de jornada 05:00 UTC (midnight Colombia)
        fire_nightly  = dt.hour == 22 and dt.minute < 5
        fire_endofday = dt.hour == 5  and dt.minute < 5
        return (fire_nightly or fire_endofday) and key not in self._fired_dates

    def mark_fired(self, date_str: str):
        self._fired_dates.add(date_str)

    def generate_eod_report(self, balance: float, net_today: float, target: float = 245.0) -> str:
        """Reporte de cierre de jornada — balance y neto del dia vs meta $250."""
        met = net_today >= target
        emoji = "✅" if met else "❌"
        pct_month = (net_today / 98000) * 100 if net_today > 0 else 0
        return (
            f"<b>📊 CIERRE DE JORNADA</b>\n"
            f"{'─' * 25}\n"
            f"Balance: <b>${balance:,.2f}</b>\n"
            f"Neto del dia: <b>${net_today:+.2f}</b>\n"
            f"Meta $250/dia: {emoji} {'CUMPLIDA' if met else 'NO cumplida'}\n"
            f"{'─' * 25}\n"
            f"Aporte al 5% mensual: <b>{pct_month:.3f}%</b>\n"
            f"Acumulado desde $100K: <b>${balance - 100000:+,.2f}</b>"
        )

    def generate_report(self, date: str) -> str:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT symbol, setup_type, pnl, result, lesson
               FROM episodes WHERE ts LIKE ? AND result != 'OPEN'""",
            (f"{date}%",),
        ).fetchall()

        if not rows:
            return (
                f"<b>REPORTE AUTONOMO - {date}</b>\n"
                f"Sin trades hoy. Bot en espera de setup."
            )

        wins   = sum(1 for r in rows if r["result"] == "WIN")
        losses = sum(1 for r in rows if r["result"] == "LOSS")
        total  = wins + losses
        pnl    = sum(r["pnl"] or 0 for r in rows)
        wr     = (wins / total * 100) if total > 0 else 0.0

        setup_pnl: dict = {}
        for r in rows:
            st = r["setup_type"] or "unknown"
            setup_pnl.setdefault(st, 0)
            setup_pnl[st] += r["pnl"] or 0
        best_setup  = max(setup_pnl, key=setup_pnl.get) if setup_pnl else "-"
        worst_setup = min(setup_pnl, key=setup_pnl.get) if setup_pnl else "-"

        lessons = [r["lesson"] for r in rows if r["lesson"]]
        lesson1 = lessons[0] if lessons else "Continuar en demo"
        lesson2 = lessons[1] if len(lessons) > 1 else "Monitorear regimen"

        try:
            gm      = GoalsManager(conn=conn)
            metrics = gm.evaluate()
            wr_prog = min(metrics.get("win_rate_pct_100", 0), 100)
            edge    = metrics.get("axi_edge_score", 0)
        except Exception as e:
            print(f"[NIGHTLY] GoalsManager error: {e}", flush=True)
            gm      = None
            wr_prog = 0
            edge    = 0.0

        research_row = conn.execute(
            "SELECT title FROM research ORDER BY id DESC LIMIT 1"
        ).fetchone()
        research_title = research_row["title"] if research_row else "Sin novedades"

        text = (
            f"<b>REPORTE AUTONOMO - {date}</b>\n"
            f"- - - - - - - - - - - -\n"
            f"Trades: {total} ({wins}W / {losses}L) - {wr:.1f}%\n"
            f"P&L: {pnl:+.2f} USD\n"
            f"- - - - - - - - - - - -\n"
            f"<b>Lecciones:</b>\n"
            f"- {lesson1}\n"
            f"- {lesson2}\n"
            f"<b>Plan manana:</b>\n"
            f"- Mejor setup: {best_setup}\n"
            f"- Evitar: {worst_setup}\n"
            f"<b>Metas:</b>\n"
            f"- Win rate 65%: {wr_prog:.0f}% completado\n"
            f"- Edge Score Axi: {edge:.1f}/50\n"
            f"<b>Nuevo:</b> {research_title[:80]}\n"
            f"- - - - - - - - - - - -"
        )

        save_report({
            "date": date, "trades_total": total, "trades_win": wins,
            "trades_loss": losses, "pnl_day": pnl, "win_rate": wr,
            "best_setup": best_setup, "worst_setup": worst_setup,
            "lessons_text": " | ".join(lessons[:3]),
            "plan_tomorrow": f"Focus {best_setup}",
            "goals_snapshot": gm.get_goals_snapshot() if gm else "",
            "report_text": text,
        }, conn=conn)
        return text

    def generate_and_save(self, date: str) -> str:
        return self.generate_report(date)

    async def send(self, date: str):
        report = self.generate_report(date)
        if self._bot and self._chat_id:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=report,
                parse_mode="HTML",
            )
        else:
            print(
                f"[NIGHTLY] Report ready (no Telegram):\n{report[:200]}",
                flush=True,
            )
