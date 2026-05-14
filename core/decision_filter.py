from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd

from core.config import Config
from core.risk_manager import RiskManager
from core.session_manager import session_score
from smc.structure import MarketStructure
from smc.orderblocks import OrderBlockDetector, FVGDetector
from smc.ml_predictor import MLPredictor
from smc.sentiment import SentimentAnalyzer


class TradeGrade(Enum):
    NO_TRADE = "no_trade"   # score < 60
    REDUCED  = "reduced"    # score 60-74 → 25% risk
    FULL     = "full"       # score 75-89 → 100% risk
    PREMIUM  = "premium"    # score 90+  → 100% risk + alert


@dataclass
class DecisionResult:
    score: int
    grade: TradeGrade
    risk_multiplier: float   # 0.0, 0.25, or 1.0
    premium_alert: bool
    reason: str
    breakdown: Dict[str, int] = field(default_factory=dict)
    detail: Dict[str, str]   = field(default_factory=dict)


# ── Score weights ──────────────────────────────────────────────────────────
# SMC technical quality:   0-30 pts
# ML / LSTM prediction:    0-25 pts
# Sentiment (Glint):       0-20 pts
# Risk / session / DD:     0-25 pts
# Total:                    100 pts


class DecisionFilter:
    """
    Aggregates SMC, ML, sentiment, risk, and historical signals into a
    score 0-100.

    Routing:
      < 60  → NO_TRADE   (risk_multiplier=0.0)
      60-74 → REDUCED    (risk_multiplier=0.25)
      75-89 → FULL       (risk_multiplier=1.0)
      90+   → PREMIUM    (risk_multiplier=1.0, premium_alert=True)
    """

    def __init__(self, config: Config, risk_manager: RiskManager,
                 historical_agent=None):
        self.config    = config
        self.rm        = risk_manager
        self._ml       = MLPredictor()
        self._sa       = SentimentAnalyzer()
        self._hist     = historical_agent  # optional HistoricalDataAgent

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        bias: str,
        glint_signals: Optional[List[Dict]] = None,
    ) -> DecisionResult:

        # Hard risk rules first — if they fail, score = 0 regardless
        allowed, risk_reason = self.rm.can_open_trade()
        if not allowed:
            return DecisionResult(
                score=0,
                grade=TradeGrade.NO_TRADE,
                risk_multiplier=0.0,
                premium_alert=False,
                reason=risk_reason,
                breakdown={"smc": 0, "ml": 0, "sentiment": 0, "risk": 0},
            )

        smc_score, smc_detail   = self._score_smc(df, bias)
        ml_result               = self._ml.predict(df, bias=bias)
        ml_score                = ml_result.score
        sentiment_result        = self._sa.analyze(symbol, glint_signals or [], bias=bias)
        sentiment_score         = sentiment_result.component_score
        risk_score, risk_detail = self._score_risk(entry, stop_loss, take_profit)

        # Historical context bonus (+0 to +20)
        hist_score = 0
        hist_detail = "sin agente histórico"
        if self._hist is not None:
            try:
                from datetime import datetime as _dt
                bonus = self._hist.score_adjustment(
                    symbol, bias, _dt.utcnow().month, entry
                )
                hist_score  = bonus.points
                hist_detail = bonus.breakdown_str()
            except Exception:
                pass

        total = smc_score + ml_score + sentiment_score + risk_score + hist_score
        total = min(max(total, 0), 100)

        breakdown = {
            "smc":        smc_score,
            "ml":         ml_score,
            "sentiment":  sentiment_score,
            "risk":       risk_score,
            "historical": hist_score,
        }
        detail = {
            "smc":        smc_detail,
            "ml":         f"dir={ml_result.direction} conf={ml_result.confidence:.0%}",
            "sentiment":  sentiment_result.reason,
            "risk":       risk_detail,
            "historical": hist_detail,
        }

        result = self._score_to_result(total, symbol, breakdown)
        result.detail = detail
        return result

    # ── Internal scorers ──────────────────────────────────────────────────

    def _score_smc(self, df: pd.DataFrame, bias: str) -> tuple:
        """SMC technical quality — max 30 pts."""
        ms = MarketStructure(df)
        structure = ms.analyze()

        ob_det  = OrderBlockDetector(df)
        fvg_det = FVGDetector(df)

        bull_obs  = ob_det.find_bullish_obs()
        bear_obs  = ob_det.find_bearish_obs()
        bull_fvgs = fvg_det.find_bullish_fvg()
        bear_fvgs = fvg_det.find_bearish_fvg()

        bos_list   = ms.detect_bos()
        choch_list = ms.detect_choch()

        score = 0
        parts = []

        # Clear trend structure: 10 pts
        is_trending = structure.structure_type.value in ("bullish_trend", "bearish_trend")
        if is_trending:
            score += 10
            parts.append("tendencia clara +10")

        # Trend matches bias: +3 bonus
        if structure.bias == bias and bias != "neutral":
            score += 3
            parts.append("bias alineado +3")

        # BOS confirmed: 8 pts
        if bos_list:
            score += 8
            parts.append(f"BOS confirmado +8 ({bos_list[-1]['direction']})")

        # CHoCH: 4 pts (potential reversal — use with bias)
        if choch_list:
            score += 4
            parts.append(f"CHoCH detectado +4")

        # OB present: 5 pts
        relevant_obs = bull_obs if bias == "bullish" else bear_obs
        if relevant_obs:
            score += 5
            parts.append(f"Order Block +5 ({len(relevant_obs)} zonas)")

        # FVG present: 5 pts (extra confluence)
        relevant_fvgs = bull_fvgs if bias == "bullish" else bear_fvgs
        if relevant_fvgs:
            score += 5
            parts.append(f"FVG +5 ({len(relevant_fvgs)} gaps)")

        # Neutral bias penalty: -5
        if bias == "neutral":
            score -= 5
            parts.append("bias neutral -5")

        score = min(max(score, 0), 30)
        return score, " | ".join(parts) if parts else "sin confluencias SMC"

    def _score_risk(self, entry: float, stop_loss: float, take_profit: float) -> tuple:
        """Risk quality — max 25 pts."""
        score = 0
        parts = []

        # Session timing: 0-8 pts
        sess_pts, sess_reason = session_score()
        score += sess_pts
        if sess_pts > 0:
            parts.append(f"sesión +{sess_pts}")

        # Risk/Reward quality: up to 9 pts
        if stop_loss and entry != stop_loss:
            rr = abs(take_profit - entry) / abs(entry - stop_loss)
            if rr >= 3.0:
                score += 9
                parts.append(f"RR 1:{rr:.1f} +9")
            elif rr >= 2.5:
                score += 7
                parts.append(f"RR 1:{rr:.1f} +7")
            elif rr >= 2.0:
                score += 5
                parts.append(f"RR 1:{rr:.1f} +5")
            else:
                parts.append(f"RR 1:{rr:.1f} insuficiente +0")

        # Drawdown health: up to 8 pts
        max_daily = self.rm.capital * self.config.max_daily_loss
        used_daily = abs(min(self.rm.daily_pnl, 0))
        dd_pct = used_daily / max_daily if max_daily > 0 else 0

        if dd_pct < 0.25:
            score += 8
            parts.append("drawdown saludable +8")
        elif dd_pct < 0.50:
            score += 5
            parts.append(f"drawdown moderado ({dd_pct:.0%}) +5")
        elif dd_pct < 0.75:
            score += 2
            parts.append(f"drawdown elevado ({dd_pct:.0%}) +2")
        else:
            parts.append(f"drawdown critico ({dd_pct:.0%}) +0")

        score = min(max(score, 0), 25)
        return score, " | ".join(parts) if parts else "riesgo sin puntaje"

    # ── Score routing ─────────────────────────────────────────────────────

    def _score_to_result(self, score: int, symbol: str, breakdown: Dict) -> DecisionResult:
        if score >= 90:
            return DecisionResult(
                score=score,
                grade=TradeGrade.PREMIUM,
                risk_multiplier=1.0,
                premium_alert=True,
                reason=f"Setup PREMIUM — score {score}/100 en {symbol}",
                breakdown=breakdown,
            )
        elif score >= 75:
            return DecisionResult(
                score=score,
                grade=TradeGrade.FULL,
                risk_multiplier=1.0,
                premium_alert=False,
                reason=f"Setup confirmado — score {score}/100",
                breakdown=breakdown,
            )
        elif score >= 60:
            return DecisionResult(
                score=score,
                grade=TradeGrade.REDUCED,
                risk_multiplier=0.25,
                premium_alert=False,
                reason=f"Setup marginal — operando al 25% del riesgo (score {score}/100)",
                breakdown=breakdown,
            )
        else:
            # Build specific rejection reason from weakest components
            if breakdown:
                weakest = min(breakdown, key=breakdown.get)
                max_pts = {"smc": 30, "ml": 25, "sentiment": 20, "risk": 25}
                pct = breakdown[weakest] / max_pts.get(weakest, 25) * 100
                reason = (
                    f"Score insuficiente ({score}/100). "
                    f"Componente más débil: {weakest.upper()} ({breakdown[weakest]}/{max_pts.get(weakest,25)} pts, {pct:.0f}%). "
                    "Si no hay setup claro, no se opera."
                )
            else:
                reason = f"Score insuficiente ({score}/100) — no se opera."

            return DecisionResult(
                score=score,
                grade=TradeGrade.NO_TRADE,
                risk_multiplier=0.0,
                premium_alert=False,
                reason=reason,
                breakdown=breakdown,
            )
