# core/mode_manager.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict
from datetime import datetime, timezone


class TradingMode(Enum):
    AUTO   = "auto"
    SEMI   = "semi"
    HYBRID = "hybrid"


class ModeRule(Enum):
    PREMIUM_SCORE       = "premium_score"
    CRITICAL_GLINT      = "critical_glint"
    HIGH_VOLATILITY     = "high_volatility"
    WARMUP              = "warmup"
    NIGHT_MODE          = "night_mode"
    GOOD_STREAK         = "good_streak"
    WIN_RATE_HIGH       = "win_rate_high"
    NORMAL_CONDITIONS   = "normal_conditions"


@dataclass
class ModeDecision:
    mode: TradingMode
    reason: str
    night_mode: bool = False
    rules_triggered: List[str] = field(default_factory=list)
    previous_mode: TradingMode = TradingMode.HYBRID


# Night mode: 23:00 - 06:00 UTC
NIGHT_HOURS = set(range(23, 24)) | set(range(0, 6))
NIGHT_MIN_SCORE = 85   # Only A+ setups during night


class ModeManager:
    """
    Intelligently switches between AUTO and SEMI modes based on:
    - Trade score (premium = > 90 always SEMI for confirmation)
    - Glint critical signals
    - ATR volatility
    - Warmup period (first 10 trades)
    - Night mode (23pm-6am UTC)
    - Win streak (5/5 → promote to AUTO)
    """

    def __init__(self, base_mode: TradingMode = TradingMode.HYBRID):
        self.base_mode    = base_mode
        self.current_mode = base_mode
        self.history: List[Dict] = []

    def decide(
        self,
        score: int,
        glint_critical: bool,
        atr_ratio: float,
        trades_today: int,
        last_5_wins: int,
        win_rate_today: float,
        atr_normal: bool,
        hour_utc: int,
    ) -> ModeDecision:
        """
        Evaluates all rules and returns the recommended mode with reason.
        Rules are checked in priority order (highest priority first).
        """
        rules_triggered = []
        night_mode = hour_utc in NIGHT_HOURS

        # 1. Night mode (highest priority)
        if night_mode:
            rules_triggered.append(ModeRule.NIGHT_MODE.value)
            reason = f"Modo noche ({hour_utc:02d}:00 UTC) — solo setups A+ (score > {NIGHT_MIN_SCORE})"
            decision = ModeDecision(
                mode=TradingMode.SEMI,
                reason=reason,
                night_mode=True,
                rules_triggered=rules_triggered,
                previous_mode=self.current_mode,
            )
            self._record(decision)
            self.current_mode = TradingMode.SEMI
            return decision

        # 2. Warmup: first 10 trades of the day
        if trades_today < 10:
            rules_triggered.append(ModeRule.WARMUP.value)
            reason = f"Calentamiento — menos de 10 trades ({trades_today}/10). Usando SEMI para calibrar."
            decision = ModeDecision(
                mode=TradingMode.SEMI,
                reason=reason,
                night_mode=False,
                rules_triggered=rules_triggered,
                previous_mode=self.current_mode,
            )
            self._record(decision)
            self.current_mode = TradingMode.SEMI
            return decision

        # 3. Premium score > 90 → SEMI (confirm before big trade)
        if score > 90:
            rules_triggered.append(ModeRule.PREMIUM_SCORE.value)
            reason = f"Setup PREMIUM (score {score}/100) — confirmacion recomendada antes de ejecutar."
            decision = ModeDecision(
                mode=TradingMode.SEMI,
                reason=reason,
                night_mode=False,
                rules_triggered=rules_triggered,
                previous_mode=self.current_mode,
            )
            self._record(decision)
            self.current_mode = TradingMode.SEMI
            return decision

        # 4. Critical Glint signal → SEMI
        if glint_critical:
            rules_triggered.append(ModeRule.CRITICAL_GLINT.value)
            reason = "Señal Glint CRITICA activa — requiere confirmacion manual antes de operar."
            decision = ModeDecision(
                mode=TradingMode.SEMI,
                reason=reason,
                night_mode=False,
                rules_triggered=rules_triggered,
                previous_mode=self.current_mode,
            )
            self._record(decision)
            self.current_mode = TradingMode.SEMI
            return decision

        # 5. High volatility (ATR > 3x) → SEMI
        if atr_ratio > 3.0 or not atr_normal:
            rules_triggered.append(ModeRule.HIGH_VOLATILITY.value)
            reason = f"Volatilidad elevada (ATR {atr_ratio:.1f}x promedio) — gestion manual recomendada."
            decision = ModeDecision(
                mode=TradingMode.SEMI,
                reason=reason,
                night_mode=False,
                rules_triggered=rules_triggered,
                previous_mode=self.current_mode,
            )
            self._record(decision)
            self.current_mode = TradingMode.SEMI
            return decision

        # 6. Good streak: 5/5 wins + win rate > 60% → promote to AUTO
        if (self.base_mode == TradingMode.SEMI and
                last_5_wins >= 5 and win_rate_today > 60.0 and atr_normal):
            rules_triggered.append(ModeRule.GOOD_STREAK.value)
            reason = f"Racha positiva ({last_5_wins}/5 wins, {win_rate_today:.0f}% WR) — promoviendo a AUTO."
            decision = ModeDecision(
                mode=TradingMode.AUTO,
                reason=reason,
                night_mode=False,
                rules_triggered=rules_triggered,
                previous_mode=self.current_mode,
            )
            self._record(decision)
            self.current_mode = TradingMode.AUTO
            return decision

        # 7. Normal conditions → stay in base mode
        rules_triggered.append(ModeRule.NORMAL_CONDITIONS.value)
        reason = f"Condiciones normales — manteniendo modo {self.base_mode.value.upper()}."
        decision = ModeDecision(
            mode=self.base_mode,
            reason=reason,
            night_mode=False,
            rules_triggered=rules_triggered,
            previous_mode=self.current_mode,
        )
        self._record(decision)
        self.current_mode = self.base_mode
        return decision

    def _record(self, decision: ModeDecision):
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode":   decision.mode.value,
            "reason": decision.reason,
            "rules":  decision.rules_triggered,
        })

    def summary(self) -> str:
        recent = self.history[-5:] if self.history else []
        lines = [f"Mode Manager — Modo actual: {self.current_mode.value.upper()}"]
        for h in recent:
            lines.append(f"  [{h['timestamp'][:16]}] {h['mode'].upper()}: {h['reason'][:60]}")
        return "\n".join(lines)
