"""
FTMOAgent - FTMO Challenge rules enforcement and simulation.

Hardcoded FTMO 2026 rules. 100% deterministic - no external APIs.
Only stdlib dependencies: datetime, dataclasses, enum.
"""
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Optional


class ChallengeType(Enum):
    TWO_STEP = "2step"
    ONE_STEP  = "1step"


class ChallengeStatus(Enum):
    ACTIVE   = "active"
    PASSED   = "passed"
    FAILED   = "failed"
    PAUSED   = "paused"   # daily limit hit, resume next day


@dataclass
class FTMORules:
    """Hardcoded FTMO 2026 rules."""
    challenge_type: ChallengeType
    initial_balance: float
    profit_target_pct: float      # 0.10 = 10%
    max_daily_loss_pct: float     # 0.05 = 5%
    max_total_drawdown_pct: float  # 0.10 = 10%
    trailing_drawdown: bool        # True for 1-step
    min_trading_days: int          # 4
    profit_split_pct: float        # 0.80 or 0.90
    max_risk_per_trade_pct: float = 0.005  # 0.5%
    max_trades_per_day: int = 2
    consistency_limit_pct: float = 0.30   # no day > 30% of total profit
    news_blackout_minutes: int = 2        # 2 min before/after high impact

    @property
    def profit_target_usd(self) -> float:
        return self.initial_balance * self.profit_target_pct

    @property
    def max_daily_loss_usd(self) -> float:
        return self.initial_balance * self.max_daily_loss_pct

    @property
    def max_total_drawdown_usd(self) -> float:
        return self.initial_balance * self.max_total_drawdown_pct

    @property
    def safety_daily_stop_usd(self) -> float:
        """Stop at 60% of limit for safety margin."""
        return self.max_daily_loss_usd * 0.60

    @property
    def safety_drawdown_stop_usd(self) -> float:
        return self.max_total_drawdown_usd * 0.70


@dataclass
class DailyRecord:
    date: date
    pnl: float
    trades: int
    best_trade_pnl: float
    worst_trade_pnl: float


@dataclass
class ChallengeState:
    rules: FTMORules
    status: ChallengeStatus
    current_balance: float
    start_date: date
    trading_days: int            # days with at least 1 trade
    daily_pnl_today: float
    total_pnl: float
    max_drawdown_reached_pct: float
    consecutive_losses: int
    daily_records: list = field(default_factory=list)  # list[DailyRecord]

    @property
    def progress_pct(self) -> float:
        if self.rules.profit_target_usd == 0:
            return 0.0
        return self.total_pnl / self.rules.profit_target_usd

    @property
    def days_elapsed(self) -> int:
        return (date.today() - self.start_date).days + 1

    def estimated_days_remaining(self) -> int:
        """Based on current daily pace."""
        if self.trading_days == 0 or self.total_pnl <= 0:
            return 30  # default estimate
        daily_avg = self.total_pnl / max(self.trading_days, 1)
        remaining_profit = self.rules.profit_target_usd - self.total_pnl
        if daily_avg <= 0:
            return 30
        return max(0, int(remaining_profit / daily_avg))


class FTMOAgent:
    """
    FTMO Challenge rules enforcement and simulation.
    100% deterministic - no external APIs.
    """

    @staticmethod
    def create_rules(challenge_type: ChallengeType,
                     initial_balance: float) -> FTMORules:
        """Factory: create rules for given challenge type."""
        if challenge_type == ChallengeType.TWO_STEP:
            return FTMORules(
                challenge_type=challenge_type,
                initial_balance=initial_balance,
                profit_target_pct=0.10,
                max_daily_loss_pct=0.05,
                max_total_drawdown_pct=0.10,
                trailing_drawdown=False,
                min_trading_days=4,
                profit_split_pct=0.80,
            )
        else:  # ONE_STEP
            return FTMORules(
                challenge_type=challenge_type,
                initial_balance=initial_balance,
                profit_target_pct=0.10,
                max_daily_loss_pct=0.03,
                max_total_drawdown_pct=0.10,
                trailing_drawdown=True,
                min_trading_days=4,
                profit_split_pct=0.90,
            )

    @staticmethod
    def new_challenge(initial_balance: float = 10000.0,
                      challenge_type: ChallengeType = ChallengeType.TWO_STEP,
                      start_date: date = None) -> ChallengeState:
        """Create a new challenge state."""
        rules = FTMOAgent.create_rules(challenge_type, initial_balance)
        return ChallengeState(
            rules=rules,
            status=ChallengeStatus.ACTIVE,
            current_balance=initial_balance,
            start_date=start_date or date.today(),
            trading_days=0,
            daily_pnl_today=0.0,
            total_pnl=0.0,
            max_drawdown_reached_pct=0.0,
            consecutive_losses=0,
        )

    # ── Risk checks ────────────────────────────────────────────────────────

    def check_daily_loss_limit(self, state: ChallengeState) -> tuple:
        """
        Returns (can_trade, reason).
        can_trade=False if daily_pnl_today <= -max_daily_loss_usd
        """
        limit = state.rules.max_daily_loss_usd
        if state.daily_pnl_today <= -limit:
            return False, f"Daily loss limit hit: ${abs(state.daily_pnl_today):.2f} / ${limit:.2f}"
        # Safety stop at 60% of limit
        safety = state.rules.safety_daily_stop_usd
        if state.daily_pnl_today <= -safety:
            return False, f"Safety stop: ${abs(state.daily_pnl_today):.2f} >= 60% of daily limit"
        return True, "Daily loss OK"

    def check_drawdown_limit(self, state: ChallengeState, equity: float = None) -> tuple:
        """
        Returns (can_trade, reason).
        Uses equity (balance + unrealized P&L) when provided; else uses closed balance.
        FTMO counts drawdown from worst of balance or equity.
        """
        effective = min(state.current_balance, equity) if equity is not None and equity > 0 else state.current_balance
        drawdown_pct = (state.rules.initial_balance - effective) / state.rules.initial_balance
        state.max_drawdown_reached_pct = max(state.max_drawdown_reached_pct, drawdown_pct)
        limit = state.rules.max_total_drawdown_pct
        if drawdown_pct >= limit:
            return False, f"Max drawdown hit: {drawdown_pct*100:.2f}% / {limit*100:.2f}%"
        safety = state.rules.max_total_drawdown_pct * 0.70
        if drawdown_pct >= safety:
            return False, f"Approaching drawdown limit: {drawdown_pct*100:.2f}% (safety stop at {safety*100:.2f}%)"
        return True, "Drawdown OK"

    def check_consistency_rule(self, state: ChallengeState) -> tuple:
        """
        Best day rule: no single day > 30% of total profit.
        Returns (passes, reason).
        """
        if state.total_pnl <= 0:
            return True, "No profit yet"
        best_day = max((r.pnl for r in state.daily_records if r.pnl > 0), default=0.0)
        if best_day == 0:
            return True, "No winning days yet"
        best_day_pct = best_day / state.total_pnl
        limit = state.rules.consistency_limit_pct
        if best_day_pct > limit:
            return False, f"Consistency violation: best day is {best_day_pct*100:.1f}% of profit (limit {limit*100:.0f}%)"
        return True, f"Consistency OK: best day {best_day_pct*100:.1f}%"

    def can_trade(self, state: ChallengeState,
                  as_of: datetime = None,
                  equity: float = None) -> tuple:
        """
        Full pre-trade check. Returns (allowed, reason).
        Checks: status, daily loss, drawdown (equity-aware), consecutive losses, time filters.
        """
        if state.status == ChallengeStatus.FAILED:
            return False, "Challenge failed"
        if state.status == ChallengeStatus.PASSED:
            return False, "Challenge already passed"
        if state.status == ChallengeStatus.PAUSED:
            return False, "Paused - daily limit hit, resume next trading day"

        daily_ok, daily_msg = self.check_daily_loss_limit(state)
        if not daily_ok:
            return False, daily_msg

        dd_ok, dd_msg = self.check_drawdown_limit(state, equity=equity)
        if not dd_ok:
            return False, dd_msg

        if state.consecutive_losses >= 3:
            return False, "3 consecutive losses - 24h pause activated"

        # Time filter: don't trade Monday first 2 hours (00:00-02:00 UTC)
        dt = as_of or datetime.now(timezone.utc)
        if dt.weekday() == 0 and dt.hour < 2:
            return False, "Monday first 2 hours - no trading"

        # Friday after 16:00 UTC (NY session close - per FTMO/Axi rules)
        if dt.weekday() == 4 and dt.hour >= 16:
            return False, "Friday after 16:00 UTC - no trading"

        return True, "OK to trade"

    # ── Trade recording ────────────────────────────────────────────────────

    def record_trade(self, state: ChallengeState, pnl: float,
                     trade_date: date = None) -> ChallengeState:
        """
        Record a completed trade. Updates:
        - current_balance
        - daily_pnl_today
        - total_pnl
        - consecutive_losses
        - status (FAILED if limits breached)
        """
        d = trade_date or date.today()

        # Update balance and PnL
        state.current_balance += pnl
        state.daily_pnl_today += pnl
        state.total_pnl       += pnl

        # Track consecutive losses
        if pnl < 0:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0

        # Update daily record
        existing = next((r for r in state.daily_records if r.date == d), None)
        if existing:
            existing.pnl += pnl
            existing.trades += 1
            existing.best_trade_pnl  = max(existing.best_trade_pnl, pnl)
            existing.worst_trade_pnl = min(existing.worst_trade_pnl, pnl)
        else:
            state.daily_records.append(DailyRecord(d, pnl, 1, pnl, pnl))
            state.trading_days += 1

        # Check failure conditions (drawdown takes priority)
        dd_ok, dd_msg = self.check_drawdown_limit(state)
        if not dd_ok:
            state.status = ChallengeStatus.FAILED
            return state

        daily_ok, _ = self.check_daily_loss_limit(state)
        if not daily_ok:
            state.status = ChallengeStatus.PAUSED

        # Check pass conditions only when not already failed/paused by daily limit
        if state.status not in (ChallengeStatus.FAILED,):
            if (state.total_pnl >= state.rules.profit_target_usd and
                    state.trading_days >= state.rules.min_trading_days):
                cons_ok, _ = self.check_consistency_rule(state)
                if cons_ok:
                    state.status = ChallengeStatus.PASSED

        return state

    def new_trading_day(self, state: ChallengeState) -> ChallengeState:
        """Reset daily PnL and reactivate if paused. Streak preserved across midnight."""
        state.daily_pnl_today = 0.0
        # consecutive_losses intentionally NOT reset here — streak persists across days
        if state.status == ChallengeStatus.PAUSED:
            state.status = ChallengeStatus.ACTIVE
        return state

    # ── News filter ────────────────────────────────────────────────────────

    def is_news_blackout(self, event_time: datetime,
                         as_of: datetime = None,
                         blackout_minutes: int = 2) -> bool:
        """True if within blackout window of a news event."""
        dt = as_of or datetime.now(timezone.utc)
        blackout = timedelta(minutes=blackout_minutes)
        return abs(dt - event_time) <= blackout

    @staticmethod
    def rules_blackout_minutes(default: int = 2) -> int:
        return default

    def get_upcoming_news_blackouts(
        self,
        ftmo_rules: FTMORules,
        as_of: datetime = None,
    ) -> list:
        """
        Returns list of (event_name, event_time) for events in next 24h.
        Uses EventDrivenStrategy if available, else returns static list.
        """
        try:
            from strategies.event_driven import EventDrivenStrategy
            strat = EventDrivenStrategy()
            dt = as_of or datetime.now(timezone.utc)
            upcoming = strat.get_upcoming_events(dt, days_ahead=1)
            result = []
            for ev in upcoming:
                if ev.impact.value in ("critical", "high"):
                    result.append((ev.name, ev.scheduled_at))
            return result
        except Exception:
            return []

    # ── Reporting ─────────────────────────────────────────────────────────

    def format_daily_report(self, state: ChallengeState) -> str:
        """HTML Telegram report for /ftmo command."""
        status_emoji = {
            ChallengeStatus.ACTIVE:  "✅",
            ChallengeStatus.PASSED:  "🏆",
            ChallengeStatus.FAILED:  "❌",
            ChallengeStatus.PAUSED:  "⏸️",
        }.get(state.status, "❓")

        progress_pct = state.progress_pct * 100
        drawdown_pct = state.max_drawdown_reached_pct * 100
        daily_loss_pct = abs(min(state.daily_pnl_today, 0)) / state.rules.initial_balance * 100

        est_days = state.estimated_days_remaining()
        daily_limit_ok  = state.daily_pnl_today > -state.rules.safety_daily_stop_usd
        drawdown_ok     = state.max_drawdown_reached_pct < state.rules.max_total_drawdown_pct * 0.70
        cons_ok, _      = self.check_consistency_rule(state)

        return (
            f"<b>📊 FTMO CHALLENGE - {state.rules.challenge_type.value.upper()}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Balance: <b>${state.current_balance:,.2f}</b> "
            f"({'+' if state.total_pnl>=0 else ''}{state.total_pnl/state.rules.initial_balance*100:.2f}%)\n"
            f"Target: ${state.rules.profit_target_usd:,.0f} | "
            f"Logrado: ${state.total_pnl:,.2f} ({progress_pct:.1f}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"P&L hoy: {'+' if state.daily_pnl_today>=0 else ''}${state.daily_pnl_today:,.2f} "
            f"{'✅' if daily_limit_ok else '⚠️'} (límite: ${state.rules.max_daily_loss_usd:,.0f})\n"
            f"Drawdown: {drawdown_pct:.2f}% {'✅' if drawdown_ok else '⚠️'} "
            f"(límite: {state.rules.max_total_drawdown_pct*100:.0f}%)\n"
            f"Días operados: {state.trading_days}/{state.rules.min_trading_days} mínimo\n"
            f"Consistencia: {'✅' if cons_ok else '⚠️'}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Estado: {status_emoji} <b>{state.status.value.upper()}</b>\n"
            f"Estimado para pasar: {est_days} días más\n"
            f"Profit split: {state.rules.profit_split_pct*100:.0f}%"
        )

    def format_risk_alert(self, reason: str) -> str:
        return (
            f"<b>🚨 FTMO ALERTA DE RIESGO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⛔ {reason}\n"
            f"El bot se pausó automáticamente."
        )

    def calculate_monthly_income(self, funded_account: float,
                                  monthly_return_pct: float,
                                  profit_split_pct: float) -> dict:
        """
        Calculate potential monthly income from funded account.
        Returns dict with gross, net (after split), and yearly.
        """
        gross = funded_account * monthly_return_pct
        net   = gross * profit_split_pct
        return {
            "funded_account": funded_account,
            "monthly_return_pct": monthly_return_pct,
            "gross_monthly": gross,
            "net_monthly": net,
            "yearly": net * 12,
            "profit_split_pct": profit_split_pct,
        }
