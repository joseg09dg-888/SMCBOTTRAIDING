from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class SignalType(Enum):
    LONG  = "long"
    SHORT = "short"
    WAIT  = "wait"


@dataclass
class TradeSignal:
    # Required fields (no defaults)
    symbol: str
    signal_type: SignalType
    entry: float
    stop_loss: Optional[float]
    take_profit: float
    timeframe: str
    trigger: str
    confidence: float
    # Optional fields (with defaults)
    glint_context: str = ""
    notes: str = ""
    decision_score: int = 0
    decision_grade: str = ""
    risk_multiplier: float = 1.0
    premium_alert: bool = False
    score_breakdown: Dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.stop_loss is None or self.entry == self.stop_loss:
            return 0.0
        return abs(self.take_profit - self.entry) / abs(self.entry - self.stop_loss)

    def is_valid(self) -> bool:
        """SMC rules: mandatory SL + RR ≥ 1:2 + confidence > 60%."""
        if self.stop_loss is None:
            return False
        if self.risk_reward < 2.0:
            return False
        if self.confidence < 0.60:
            return False
        if self.signal_type == SignalType.WAIT:
            return False
        return True

    def format_telegram(self) -> str:
        direction = "🟢 LONG" if self.signal_type == SignalType.LONG else "🔴 SHORT"
        valid_mark = "✅ SETUP VÁLIDO" if self.is_valid() else "⛔ SIN SETUP — No se opera"

        score_line = ""
        if self.decision_score:
            grade_emoji = {"premium": "🔥", "full": "✅", "reduced": "⚠️", "no_trade": "❌"}
            em = grade_emoji.get(self.decision_grade, "")
            risk_pct = int(self.risk_multiplier * 100)
            score_line = f"\nScore: `{self.decision_score}/100` {em} | Riesgo: `{risk_pct}%`"
            if self.score_breakdown:
                bd = self.score_breakdown
                score_line += f"\n  SMC:{bd.get('smc',0)} ML:{bd.get('ml',0)} Sent:{bd.get('sentiment',0)} Risk:{bd.get('risk',0)}"

        lines = [
            f"*{direction} — {self.symbol} | {self.timeframe}*",
            f"Entrada: `{self.entry}`",
            f"Stop Loss: `{self.stop_loss}`",
            f"Take Profit: `{self.take_profit}`",
            f"R:R = `1:{self.risk_reward:.1f}`",
            f"Trigger: {self.trigger}",
            f"Confianza: {self.confidence*100:.0f}%",
            valid_mark,
        ]
        if score_line:
            lines.insert(1, score_line)
        if self.glint_context:
            lines.append(f"Contexto: {self.glint_context}")
        return "\n".join(lines)


class SignalAgent:
    """
    Takes SMCAnalysisAgent output and generates concrete trade signals
    with entry, SL, and TP based on structure.
    """

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.signal_history: List[TradeSignal] = []

    def evaluate(
        self,
        analysis_text: str,
        symbol: str,
        timeframe: str,
        current_price: float,
        poi_zones: list,
        glint_context: str = "",
    ) -> TradeSignal:
        is_bullish = "bullish" in analysis_text.lower() or "alcista" in analysis_text.lower()
        is_bearish = "bearish" in analysis_text.lower() or "bajista" in analysis_text.lower()
        has_setup  = "setup válido" in analysis_text.lower() or "✅" in analysis_text

        if not has_setup or (not is_bullish and not is_bearish):
            return TradeSignal(
                symbol=symbol, signal_type=SignalType.WAIT,
                entry=current_price, stop_loss=None,
                take_profit=current_price, timeframe=timeframe,
                trigger="Sin setup claro — paciencia",
                confidence=0.0,
            )

        if poi_zones:
            poi = poi_zones[0]
            if is_bullish:
                entry = poi.get("zone_low", current_price)
                sl    = entry * 0.995
                tp    = entry + (entry - sl) * 3
            else:
                entry = poi.get("zone_high", current_price)
                sl    = entry * 1.005
                tp    = entry - (sl - entry) * 3
        else:
            entry = current_price
            sl    = entry * (0.995 if is_bullish else 1.005)
            tp    = entry + (entry - sl) * 2.5

        signal = TradeSignal(
            symbol       = symbol,
            signal_type  = SignalType.LONG if is_bullish else SignalType.SHORT,
            entry        = round(entry, 5),
            stop_loss    = round(sl, 5),
            take_profit  = round(tp, 5),
            timeframe    = timeframe,
            trigger      = "CHoCH + OB retest" if is_bullish else "BOS + FVG bajista",
            confidence   = 0.75 if has_setup else 0.5,
            glint_context= glint_context,
        )
        self.signal_history.append(signal)
        return signal
