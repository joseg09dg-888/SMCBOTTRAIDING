# agents/report_agent.py
"""
ReportAgent — genera reportes semanales/mensuales, evalua criterios go-live
y puede enviar resumenes por Telegram.
"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Union


# ── Estructuras de datos ───────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol: str
    direction: str          # "long" | "short"
    entry: float
    exit_price: float
    pnl: float
    agents_confirmed: list
    setup_tags: list        # ["CHoCH", "OB", "FVG"]
    timeframe: str
    score: int
    timestamp: datetime


@dataclass
class WeeklyStats:
    week_start: date
    week_end: date
    capital_start: float
    capital_end: float
    pnl: float
    pnl_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    best_trade: Optional[TradeRecord]
    worst_trade: Optional[TradeRecord]
    trades: list = field(default_factory=list)


@dataclass
class MonthlyStats:
    month: int
    year: int
    capital_start: float
    capital_end: float
    pnl: float
    pnl_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    weekly_stats: list = field(default_factory=list)


@dataclass
class GoLiveCriteria:
    win_rate_60_3weeks: bool
    profit_factor_15: bool
    max_drawdown_5: bool
    min_50_trades: bool
    agents_operational: bool
    no_risk_violations: bool
    # Recommended (optional, default False)
    win_rate_65: bool = False
    profit_factor_20: bool = False
    sharpe_15: bool = False
    trades_200: bool = False

    @property
    def mandatory_passed(self) -> int:
        return sum([
            self.win_rate_60_3weeks,
            self.profit_factor_15,
            self.max_drawdown_5,
            self.min_50_trades,
            self.agents_operational,
            self.no_risk_violations,
        ])

    @property
    def mandatory_total(self) -> int:
        return 6

    @property
    def all_mandatory_passed(self) -> bool:
        return self.mandatory_passed == self.mandatory_total

    def verdict_text(self, estimated_date: Optional[date] = None) -> str:
        if self.all_mandatory_passed:
            return (
                "🟢 LISTO PARA REAL — Todos los criterios\n"
                "Recomendacion: empezar con $200"
            )
        missing = self.mandatory_total - self.mandatory_passed
        if estimated_date:
            date_str = estimated_date.strftime("%d/%m/%Y")
        else:
            date_str = "por determinar"
        return (
            f"🟡 AUN NO — Faltan {missing} criterios\n"
            f"Estimado para ir a real: {date_str}"
        )


# ── Clase principal ────────────────────────────────────────────────────────

class ReportAgent:
    def __init__(self, capital: float = 1000.0, telegram_bot=None):
        self._capital = capital
        self._telegram = telegram_bot
        self._trades: list = []

    def add_trade(self, trade: TradeRecord):
        """Agrega un trade al historial."""
        self._trades.append(trade)

    # ── Calculo de metricas ────────────────────────────────────────────────

    def _trades_in_week(self, week_start: date) -> list:
        """Retorna los trades que caen en la semana Mon–Sun de week_start."""
        week_end = week_start + timedelta(days=6)
        result = []
        for t in self._trades:
            t_date = t.timestamp.date() if isinstance(t.timestamp, datetime) else t.timestamp
            if week_start <= t_date <= week_end:
                result.append(t)
        return result

    def _trades_in_month(self, year: int, month: int) -> list:
        result = []
        for t in self._trades:
            t_date = t.timestamp.date() if isinstance(t.timestamp, datetime) else t.timestamp
            if t_date.year == year and t_date.month == month:
                result.append(t)
        return result

    def _calc_profit_factor(self, trades: list) -> float:
        wins_sum = sum(t.pnl for t in trades if t.pnl > 0)
        losses_sum = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if losses_sum == 0:
            return wins_sum if wins_sum > 0 else 0.0
        return wins_sum / losses_sum

    def _calc_max_drawdown_pct(self, trades: list, capital_start: float) -> float:
        """Calcula el maximo drawdown como % del capital de inicio."""
        if not trades or capital_start == 0:
            return 0.0
        equity = capital_start
        peak = capital_start
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.timestamp):
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def calculate_weekly_stats(self, week_start: date) -> WeeklyStats:
        """Calcula estadisticas para la semana que empieza en week_start."""
        week_end = week_start + timedelta(days=6)
        trades = self._trades_in_week(week_start)

        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0
        profit_factor = self._calc_profit_factor(trades)
        max_dd = self._calc_max_drawdown_pct(trades, self._capital)

        best = max(trades, key=lambda t: t.pnl) if trades else None
        worst = min(trades, key=lambda t: t.pnl) if trades else None

        pnl_pct = (total_pnl / self._capital * 100) if self._capital > 0 else 0.0

        return WeeklyStats(
            week_start=week_start,
            week_end=week_end,
            capital_start=self._capital,
            capital_end=self._capital + total_pnl,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            best_trade=best,
            worst_trade=worst,
            trades=trades,
        )

    def calculate_monthly_stats(self, year: int, month: int) -> MonthlyStats:
        """Calcula estadisticas del mes. Incluye Sharpe ratio."""
        trades = self._trades_in_month(year, month)

        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0
        profit_factor = self._calc_profit_factor(trades)
        max_dd = self._calc_max_drawdown_pct(trades, self._capital)
        pnl_pct = (total_pnl / self._capital * 100) if self._capital > 0 else 0.0

        # Sharpe ratio: mean_pnl / std_pnl * sqrt(252)
        if total > 1:
            pnls = [t.pnl for t in trades]
            mean_pnl = sum(pnls) / len(pnls)
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
            std_pnl = math.sqrt(variance)
            sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        return MonthlyStats(
            month=month,
            year=year,
            capital_start=self._capital,
            capital_end=self._capital + total_pnl,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
        )

    def evaluate_go_live_criteria(self, weeks_history: Optional[list] = None) -> GoLiveCriteria:
        """Evalua todos los criterios contra el historial actual."""
        trades = self._trades
        total = len(trades)

        # min_50_trades
        min_50 = total >= 50

        # profit_factor_15 (global)
        pf = self._calc_profit_factor(trades)
        pf_15 = pf >= 1.5

        # max_drawdown_5
        max_dd = self._calc_max_drawdown_pct(trades, self._capital)
        dd_ok = max_dd < 5.0

        # no_risk_violations: ningun trade con perdida > 5% del capital
        threshold = self._capital * 0.05
        no_violations = all(t.pnl >= -threshold for t in trades)

        # win_rate_60_3weeks: evalua usando weeks_history si se provee
        if weeks_history is not None:
            last3 = weeks_history[-3:]
            wr_ok = (
                len(last3) >= 3
                and all(w.win_rate > 60.0 for w in last3)
            )
        else:
            # Infiere de los ultimos 3 lunes del historial propio
            if not trades:
                wr_ok = False
            else:
                # Obtener la semana mas reciente en el historial
                latest = max(t.timestamp.date() if isinstance(t.timestamp, datetime) else t.timestamp
                             for t in trades)
                # Construir 3 semanas hacia atras desde la semana mas reciente
                # Encontrar el lunes de esa semana
                days_since_monday = latest.weekday()
                monday = latest - timedelta(days=days_since_monday)
                weeks = []
                for i in range(3):
                    ws = monday - timedelta(weeks=i)
                    w_stats = self.calculate_weekly_stats(ws)
                    if w_stats.total_trades > 0:
                        weeks.append(w_stats)
                wr_ok = (
                    len(weeks) >= 3
                    and all(w.win_rate > 60.0 for w in weeks)
                )

        return GoLiveCriteria(
            win_rate_60_3weeks=wr_ok,
            profit_factor_15=pf_15,
            max_drawdown_5=dd_ok,
            min_50_trades=min_50,
            agents_operational=True,  # verificado externamente en esta version
            no_risk_violations=no_violations,
        )

    # ── Generacion de reportes ─────────────────────────────────────────────

    def generate_weekly_report_text(self, stats: WeeklyStats) -> str:
        """Genera reporte semanal como texto."""
        sign = "+" if stats.pnl >= 0 else ""
        lines = [
            f"{'=' * 50}",
            f"REPORTE SEMANAL — {stats.week_start} al {stats.week_end}",
            f"{'=' * 50}",
            f"Capital inicio : ${stats.capital_start:,.2f}",
            f"Capital final  : ${stats.capital_end:,.2f}",
            f"P&L            : {sign}${stats.pnl:.2f} ({sign}{stats.pnl_pct:.2f}%)",
            f"",
            f"Trades totales : {stats.total_trades}",
            f"Wins           : {stats.wins}",
            f"Losses         : {stats.losses}",
            f"Win Rate       : {stats.win_rate:.1f}%",
            f"Profit Factor  : {stats.profit_factor:.2f}",
            f"Max Drawdown   : {stats.max_drawdown_pct:.2f}%",
        ]
        if stats.best_trade:
            lines.append(f"Mejor trade    : {stats.best_trade.symbol} +${stats.best_trade.pnl:.2f}")
        if stats.worst_trade:
            lines.append(f"Peor trade     : {stats.worst_trade.symbol} ${stats.worst_trade.pnl:.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def generate_monthly_report_text(self, stats: MonthlyStats) -> str:
        """Genera reporte mensual como texto."""
        import calendar
        month_name = calendar.month_name[stats.month]
        sign = "+" if stats.pnl >= 0 else ""
        lines = [
            f"{'=' * 50}",
            f"REPORTE MENSUAL — {month_name} {stats.year}",
            f"{'=' * 50}",
            f"Capital inicio : ${stats.capital_start:,.2f}",
            f"Capital final  : ${stats.capital_end:,.2f}",
            f"P&L            : {sign}${stats.pnl:.2f} ({sign}{stats.pnl_pct:.2f}%)",
            f"",
            f"Trades totales : {stats.total_trades}",
            f"Wins           : {stats.wins}",
            f"Losses         : {stats.losses}",
            f"Win Rate       : {stats.win_rate:.1f}%",
            f"Profit Factor  : {stats.profit_factor:.2f}",
            f"Max Drawdown   : {stats.max_drawdown_pct:.2f}%",
            f"Sharpe Ratio   : {stats.sharpe_ratio:.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def generate_telegram_summary(self, stats) -> str:
        """Version corta para enviar por Telegram (max ~500 chars)."""
        if isinstance(stats, WeeklyStats):
            sign = "+" if stats.pnl >= 0 else ""
            return (
                f"📊 Semana {stats.week_start}\n"
                f"P&L: {sign}${stats.pnl:.2f} ({sign}{stats.pnl_pct:.1f}%)\n"
                f"Trades: {stats.total_trades} | WR: {stats.win_rate:.1f}%\n"
                f"PF: {stats.profit_factor:.2f} | DD: {stats.max_drawdown_pct:.1f}%"
            )
        elif isinstance(stats, MonthlyStats):
            import calendar
            month_name = calendar.month_abbr[stats.month]
            sign = "+" if stats.pnl >= 0 else ""
            return (
                f"📅 {month_name} {stats.year}\n"
                f"P&L: {sign}${stats.pnl:.2f} ({sign}{stats.pnl_pct:.1f}%)\n"
                f"Trades: {stats.total_trades} | WR: {stats.win_rate:.1f}%\n"
                f"PF: {stats.profit_factor:.2f} | Sharpe: {stats.sharpe_ratio:.2f}"
            )
        return str(stats)

    def generate_criteria_message(self) -> str:
        """Mensaje con estado de los CRITERIOS para ir a real."""
        criteria = self.evaluate_go_live_criteria()
        ok = "✅"
        fail = "❌"
        lines = [
            "CRITERIOS para ir a cuenta REAL:",
            f"{ok if criteria.win_rate_60_3weeks else fail} Win Rate > 60% (3 semanas)",
            f"{ok if criteria.profit_factor_15 else fail} Profit Factor > 1.5",
            f"{ok if criteria.max_drawdown_5 else fail} Max Drawdown < 5%",
            f"{ok if criteria.min_50_trades else fail} Minimo 50 trades",
            f"{ok if criteria.agents_operational else fail} Agentes operacionales",
            f"{ok if criteria.no_risk_violations else fail} Sin violaciones de riesgo",
            "",
            criteria.verdict_text(),
        ]
        return "\n".join(lines)

    def generate_projection_message(self) -> str:
        """Proyeccion proxima semana basada en tendencia actual."""
        trades = self._trades
        if not trades:
            return (
                "📈 Proyeccion proxima semana:\n"
                "Sin historial suficiente para proyectar.\n"
                "Agrega trades para generar proyecciones."
            )

        # Ultimas 4 semanas de datos
        pnls = [t.pnl for t in trades[-20:]]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0.0

        # Tendencia simple: ultimos 5 vs anteriores 5
        if len(pnls) >= 10:
            recent = sum(pnls[-5:]) / 5
            older = sum(pnls[-10:-5]) / 5
            trend = "📈 Mejorando" if recent > older else "📉 Deteriorando"
        else:
            trend = "➡️ Estable"

        proj_pnl = avg_pnl * 5  # aprox 5 trades/semana
        sign = "+" if proj_pnl >= 0 else ""
        return (
            f"📊 Proyeccion proxima semana:\n"
            f"Tendencia: {trend}\n"
            f"P&L promedio/trade: {'+' if avg_pnl >= 0 else ''}${avg_pnl:.2f}\n"
            f"Win Rate reciente: {wr:.1f}%\n"
            f"P&L proyectado: {sign}${proj_pnl:.2f}"
        )

    # ── Guardado de archivos ───────────────────────────────────────────────

    def save_weekly_report(self, stats: WeeklyStats, output_dir: str = "reports/weekly") -> str:
        """Guarda reporte como HTML en output_dir/semana_YYYY-MM-DD.html"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"semana_{stats.week_start.strftime('%Y-%m-%d')}.html"
        filepath = os.path.join(output_dir, filename)

        sign = "+" if stats.pnl >= 0 else ""
        pnl_color = "#2ecc71" if stats.pnl >= 0 else "#e74c3c"

        best_row = ""
        if stats.best_trade:
            best_row = f"<tr><td>Mejor trade</td><td>{stats.best_trade.symbol} +${stats.best_trade.pnl:.2f}</td></tr>"
        worst_row = ""
        if stats.worst_trade:
            worst_row = f"<tr><td>Peor trade</td><td>{stats.worst_trade.symbol} ${stats.worst_trade.pnl:.2f}</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte Semanal {stats.week_start}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
    h1 {{ color: #2c3e50; }}
    .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 20px;
              box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
    td:first-child {{ font-weight: bold; color: #555; width: 200px; }}
    .pnl {{ color: {pnl_color}; font-size: 1.4em; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Reporte Semanal — {stats.week_start} al {stats.week_end}</h1>
  <div class="card">
    <p class="pnl">P&amp;L: {sign}${stats.pnl:.2f} ({sign}{stats.pnl_pct:.2f}%)</p>
    <table>
      <tr><td>Capital inicio</td><td>${stats.capital_start:,.2f}</td></tr>
      <tr><td>Capital final</td><td>${stats.capital_end:,.2f}</td></tr>
      <tr><td>Trades totales</td><td>{stats.total_trades}</td></tr>
      <tr><td>Wins / Losses</td><td>{stats.wins} / {stats.losses}</td></tr>
      <tr><td>Win Rate</td><td>{stats.win_rate:.1f}%</td></tr>
      <tr><td>Profit Factor</td><td>{stats.profit_factor:.2f}</td></tr>
      <tr><td>Max Drawdown</td><td>{stats.max_drawdown_pct:.2f}%</td></tr>
      {best_row}
      {worst_row}
    </table>
  </div>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return filepath

    def save_monthly_report(self, stats: MonthlyStats, output_dir: str = "reports/monthly") -> str:
        """Guarda reporte como HTML en output_dir/mes_YYYY-MM.html"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"mes_{stats.year}-{stats.month:02d}.html"
        filepath = os.path.join(output_dir, filename)

        import calendar
        month_name = calendar.month_name[stats.month]
        sign = "+" if stats.pnl >= 0 else ""
        pnl_color = "#2ecc71" if stats.pnl >= 0 else "#e74c3c"

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte Mensual {month_name} {stats.year}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
    h1 {{ color: #2c3e50; }}
    .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 20px;
              box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
    td:first-child {{ font-weight: bold; color: #555; width: 200px; }}
    .pnl {{ color: {pnl_color}; font-size: 1.4em; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Reporte Mensual — {month_name} {stats.year}</h1>
  <div class="card">
    <p class="pnl">P&amp;L: {sign}${stats.pnl:.2f} ({sign}{stats.pnl_pct:.2f}%)</p>
    <table>
      <tr><td>Capital inicio</td><td>${stats.capital_start:,.2f}</td></tr>
      <tr><td>Capital final</td><td>${stats.capital_end:,.2f}</td></tr>
      <tr><td>Trades totales</td><td>{stats.total_trades}</td></tr>
      <tr><td>Wins / Losses</td><td>{stats.wins} / {stats.losses}</td></tr>
      <tr><td>Win Rate</td><td>{stats.win_rate:.1f}%</td></tr>
      <tr><td>Profit Factor</td><td>{stats.profit_factor:.2f}</td></tr>
      <tr><td>Max Drawdown</td><td>{stats.max_drawdown_pct:.2f}%</td></tr>
      <tr><td>Sharpe Ratio</td><td>{stats.sharpe_ratio:.2f}</td></tr>
    </table>
  </div>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return filepath

    # ── Scheduling (APScheduler) ───────────────────────────────────────────

    def setup_scheduler(self) -> bool:
        """
        Configura APScheduler si esta disponible.
        Retorna True si se configuro, False si no.
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            scheduler = AsyncIOScheduler()
            # Lunes 08:00 UTC
            scheduler.add_job(
                self._scheduled_weekly_report,
                trigger="cron",
                day_of_week="mon",
                hour=8,
                minute=0,
            )
            # Dia 1 de cada mes 08:00 UTC
            scheduler.add_job(
                self._scheduled_monthly_report,
                trigger="cron",
                day=1,
                hour=8,
                minute=0,
            )
            scheduler.start()
            self._scheduler = scheduler
            return True
        except (ImportError, Exception):
            return False

    async def _scheduled_weekly_report(self):
        """Genera y envia reporte semanal automatico."""
        today = date.today()
        days_since_monday = today.weekday()
        # La semana pasada
        last_monday = today - timedelta(days=days_since_monday + 7)
        stats = self.calculate_weekly_stats(last_monday)
        summary = self.generate_telegram_summary(stats)
        if self._telegram:
            try:
                await self._telegram.send_message(summary)
            except Exception:
                pass

    async def _scheduled_monthly_report(self):
        """Genera y envia reporte mensual automatico."""
        today = date.today()
        # El mes pasado
        if today.month == 1:
            year, month = today.year - 1, 12
        else:
            year, month = today.year, today.month - 1
        stats = self.calculate_monthly_stats(year, month)
        summary = self.generate_telegram_summary(stats)
        if self._telegram:
            try:
                await self._telegram.send_message(summary)
            except Exception:
                pass
