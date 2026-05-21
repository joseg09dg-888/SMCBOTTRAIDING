"""
AxiSelectAgent - AXI SELECT funded account program tracker.

Free funded account program up to $1M. No challenge fee.
Based on Edge Score (skill, consistency, risk management).
Only stdlib dependencies: datetime, dataclasses, enum.
"""
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Optional


class AxiStage(Enum):
    PRE_SEED     = "pre_seed"
    SEED         = "seed"          # $5K, score 50
    INCUBATION   = "incubation"    # $25K, score 60
    ACCELERATION = "acceleration"  # $100K, score 70
    PRO          = "pro"           # $300K, score 80
    PRO_500      = "pro_500"       # $500K, score 85
    PRO_M        = "pro_m"         # $1M, score 90


@dataclass
class AxiStageConfig:
    stage: AxiStage
    edge_score_required: int
    capital_funded: float
    profit_split: float = 0.80
    min_trades: int = 20


@dataclass
class AxiEdgeScore:
    habilidad: int    # 0-40
    consistencia: int # 0-30
    riesgo: int       # 0-30
    total: int        # 0-100

    @property
    def is_eligible_seed(self) -> bool:
        return self.total >= 50


@dataclass
class AxiState:
    stage: AxiStage
    edge_score: AxiEdgeScore
    trades_closed: int
    wins: int
    losses: int
    daily_records: list  # list[dict] with date, pnl, trades
    current_balance: float
    initial_balance: float
    max_drawdown_pct: float
    consecutive_losses: int

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        # Simplified: wins/losses ratio as proxy
        return self.wins / max(self.losses, 1)

    @property
    def total_pnl(self) -> float:
        return self.current_balance - self.initial_balance

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.initial_balance if self.initial_balance > 0 else 0.0


class AxiSelectAgent:
    """
    Tracks AXI SELECT program progress and enforces rules.
    Free funded account program up to $1M.
    No challenge fee -- based on Edge Score.
    """

    STAGES = [
        AxiStageConfig(AxiStage.SEED,         50, 5_000),
        AxiStageConfig(AxiStage.INCUBATION,   60, 25_000),
        AxiStageConfig(AxiStage.ACCELERATION, 70, 100_000),
        AxiStageConfig(AxiStage.PRO,          80, 300_000),
        AxiStageConfig(AxiStage.PRO_500,      85, 500_000),
        AxiStageConfig(AxiStage.PRO_M,        90, 1_000_000),
    ]

    # Risk rules
    MAX_RISK_PER_TRADE_PCT = 0.005   # 0.5%
    MAX_DAILY_DRAWDOWN_PCT = 0.03    # 3%
    MAX_TOTAL_DRAWDOWN_PCT = 0.08    # 8%
    MAX_TRADES_PER_DAY     = 3
    MIN_SCORE_TO_TRADE     = 120     # out of 275 (bot's internal score)

    @staticmethod
    def new_state(initial_balance: float = 500.0) -> AxiState:
        """Create fresh Axi Select tracking state."""
        empty_score = AxiEdgeScore(0, 0, 0, 0)
        return AxiState(
            stage=AxiStage.PRE_SEED,
            edge_score=empty_score,
            trades_closed=0,
            wins=0,
            losses=0,
            daily_records=[],
            current_balance=initial_balance,
            initial_balance=initial_balance,
            max_drawdown_pct=0.0,
            consecutive_losses=0,
        )

    # ── Edge Score calculation ─────────────────────────────────────────────

    def calculate_habilidad(self, state: AxiState) -> int:
        """
        0-40 pts based on trading skill:
        - Win rate > 50%: +10
        - Profit factor > 1.2: +10
        - Sharpe > 0.5 (proxy: win_rate * 2 - 1 > 0.5): +10
        - Positive return: +10
        """
        pts = 0
        if state.win_rate > 0.50: pts += 10
        if state.profit_factor > 1.2: pts += 10
        if (state.win_rate * 2 - 1) > 0.5: pts += 10
        if state.total_pnl > 0: pts += 10
        return pts

    def calculate_consistencia(self, state: AxiState) -> int:
        """
        0-30 pts based on consistency:
        - No day > 30% of total profit: +10
        - Min 3 days traded per week (if enough records): +10
        - No 5 consecutive losses: +10
        """
        pts = 0
        # No day > 30% of total profit
        if state.total_pnl > 0:
            best_day = max(
                (r.get('pnl', 0) for r in state.daily_records if r.get('pnl', 0) > 0),
                default=0
            )
            if best_day == 0 or best_day / state.total_pnl <= 0.30:
                pts += 10
        else:
            pts += 10  # no profit yet, rule doesn't apply

        # Trading frequency (if >5 records, check)
        if len(state.daily_records) >= 5:
            recent = state.daily_records[-7:]
            active_days = sum(1 for r in recent if r.get('trades', 0) > 0)
            if active_days >= 3:
                pts += 10
        else:
            pts += 10  # not enough data, give benefit of doubt

        # No 5 consecutive losses
        if state.consecutive_losses < 5:
            pts += 10
        return pts

    def calculate_riesgo(self, state: AxiState) -> int:
        """
        0-30 pts based on risk management:
        - Max drawdown < 10%: +10
        - Positive balance (proxy for risk < 2% per trade): +10
        - Has trades (has SL, since bot always uses SL): +10
        """
        pts = 0
        if state.max_drawdown_pct < 0.10: pts += 10
        if state.current_balance >= state.initial_balance * 0.95: pts += 10
        if state.trades_closed > 0: pts += 10  # has trades with SL
        return pts

    def calculate_edge_score(self, state: AxiState) -> AxiEdgeScore:
        hab  = self.calculate_habilidad(state)
        cons = self.calculate_consistencia(state)
        risk = self.calculate_riesgo(state)
        return AxiEdgeScore(hab, cons, risk, hab + cons + risk)

    # ── Stage progression ──────────────────────────────────────────────────

    def get_current_stage(self, state: AxiState) -> AxiStage:
        """Determine current stage based on edge score and trades."""
        if state.trades_closed < 20:
            return AxiStage.PRE_SEED
        score = state.edge_score.total
        for cfg in reversed(self.STAGES):
            if score >= cfg.edge_score_required:
                return cfg.stage
        return AxiStage.PRE_SEED

    def get_next_stage(self, current: AxiStage) -> Optional[AxiStageConfig]:
        """Return the next stage config."""
        stage_order = [AxiStage.PRE_SEED] + [s.stage for s in self.STAGES]
        idx = stage_order.index(current) if current in stage_order else 0
        if idx < len(self.STAGES):
            return self.STAGES[min(idx, len(self.STAGES) - 1)]
        return None

    def get_funded_capital(self, stage: AxiStage) -> float:
        for cfg in self.STAGES:
            if cfg.stage == stage:
                return cfg.capital_funded
        return 0.0

    # ── Trade recording ────────────────────────────────────────────────────

    def record_trade(self, state: AxiState, pnl: float,
                     trade_date: date = None) -> AxiState:
        """Record a completed trade and update all metrics."""
        d = trade_date or date.today()
        state.current_balance += pnl
        state.trades_closed   += 1

        if pnl > 0:
            state.wins += 1
            state.consecutive_losses = 0
        else:
            state.losses += 1
            state.consecutive_losses += 1

        # Update drawdown
        dd = (state.initial_balance - state.current_balance) / state.initial_balance
        if dd > state.max_drawdown_pct:
            state.max_drawdown_pct = dd

        # Update daily record
        existing = next((r for r in state.daily_records if r.get('date') == str(d)), None)
        if existing:
            existing['pnl']    = existing.get('pnl', 0) + pnl
            existing['trades'] = existing.get('trades', 0) + 1
        else:
            state.daily_records.append({'date': str(d), 'pnl': pnl, 'trades': 1})

        # Recalculate edge score
        state.edge_score = self.calculate_edge_score(state)
        state.stage = self.get_current_stage(state)
        return state

    # ── Risk checks ────────────────────────────────────────────────────────

    def can_trade(self, state: AxiState, as_of: datetime = None) -> tuple:
        """Returns (allowed: bool, reason: str)."""
        try:
            dt = as_of or datetime.now(timezone.utc)

            # Check daily loss limit
            today = str(dt.date())
            today_rec = next((r for r in state.daily_records if r.get('date') == today), None)
            if today_rec:
                daily_loss = min(today_rec.get('pnl', 0), 0)
                if abs(daily_loss) / state.initial_balance > self.MAX_DAILY_DRAWDOWN_PCT:
                    return False, f"Daily loss limit hit: {abs(daily_loss)/state.initial_balance*100:.1f}%"

            # Total drawdown
            if state.max_drawdown_pct >= self.MAX_TOTAL_DRAWDOWN_PCT:
                return False, f"Max drawdown reached: {state.max_drawdown_pct*100:.1f}%"

            # Consecutive losses
            if state.consecutive_losses >= 5:
                return False, "5 consecutive losses -- pause required"

            # Friday after 16 UTC
            if dt.weekday() == 4 and dt.hour >= 16:
                return False, "Friday after 16:00 UTC -- no trading"

            return True, "OK to trade"
        except Exception:
            return True, "OK to trade"

    # ── Comparison with FTMO ───────────────────────────────────────────────

    def compare_with_ftmo(self) -> dict:
        """Compare Axi Select vs FTMO."""
        return {
            "axi_select": {
                "cost": "FREE",
                "max_funded": 1_000_000,
                "profit_split": 0.80,
                "challenge_required": False,
                "monthly_5pct_1M": 1_000_000 * 0.05 * 0.80,
            },
            "ftmo": {
                "cost": "$155+ per challenge",
                "max_funded": 200_000,
                "profit_split": 0.90,
                "challenge_required": True,
                "monthly_5pct_200k": 200_000 * 0.05 * 0.90,
            },
            "recommendation": "Axi Select -- FREE + 5x more capital",
        }

    # ── Reporting ──────────────────────────────────────────────────────────

    def format_telegram(self, state: AxiState) -> str:
        """HTML Telegram report for /axi command."""
        es = state.edge_score
        next_stage = self.get_next_stage(state.stage)
        next_capital = next_stage.capital_funded if next_stage else 0
        next_score   = next_stage.edge_score_required if next_stage else 100

        score_needed  = max(0, 50 - es.total)

        return (
            f"<b>AXI SELECT ESTADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>EDGE SCORE: {es.total}/100</b>\n"
            f"Habilidad:    {es.habilidad}/40\n"
            f"Consistencia: {es.consistencia}/30\n"
            f"Riesgo:       {es.riesgo}/30\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>PROGRESO:</b>\n"
            f"Trades cerrados: {state.trades_closed}/20\n"
            f"Win Rate: {state.win_rate*100:.1f}%\n"
            f"P&L: {'+' if state.total_pnl >= 0 else ''}${state.total_pnl:.2f}\n"
            f"Max DD: {state.max_drawdown_pct*100:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>ETAPA: {state.stage.value.upper().replace('_',' ')}</b>\n"
            f"Para siguiente: score {next_score} ({score_needed} mas)\n"
            f"Capital siguiente: ${next_capital:,.0f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>POTENCIAL PRO M:</b>\n"
            f"$1M x 2%/mes x 80% = ${1_000_000*0.02*0.80:,.0f}/mes\n"
            f"vs FTMO $200K x 5%/mes x 90% = ${200_000*0.05*0.90:,.0f}/mes\n"
            f"<b>Axi Select es GRATIS y 5x mas capital</b>"
        )
