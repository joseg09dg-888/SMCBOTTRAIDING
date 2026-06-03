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

    # Minimum SL distance per symbol — realistic minimums based on typical daily range
    # Forex pairs: ~20-30 pips minimum to avoid being stopped by normal noise
    _MIN_SL_DIST: Dict[str, float] = {
        "EURUSD": 0.0020,   # 20 pips min
        "GBPUSD": 0.0025,   # 25 pips min
        "AUDUSD": 0.0020,   # 20 pips min
        "USDCHF": 0.0020,   # 20 pips min
        "NZDUSD": 0.0020,   # 20 pips min
        "EURGBP": 0.0020,   # 20 pips min
        "USDJPY": 0.20,     # 20 pips min
        "GBPJPY": 0.30,     # 30 pips min
        "EURJPY": 0.25,     # 25 pips min
        "XAUUSD": 200.0,    # $200 min (gold moves $50+ in minutes)
        "NAS100": 50.0,     # 50 pts min
        "US30":   80.0,     # 80 pts min
    }

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.signal_history: List[TradeSignal] = []
        self._last_df: Optional[object] = None  # injected by supervisor

    def _sl_distance(self, symbol: str, entry: float, df=None) -> float:
        """Return SL distance using ATR(14) when df available, else 1% of price.

        Using 1.5x ATR keeps SL outside normal noise so it's not hit prematurely.
        Floor is always the symbol minimum to stay above broker stop distance.
        """
        import pandas as pd
        # ATR-based SL (preferred)
        _df = df if df is not None else self._last_df
        if _df is not None and not getattr(_df, 'empty', True) and len(_df) >= 15:
            try:
                highs  = _df["high"].astype(float)
                lows   = _df["low"].astype(float)
                closes = _df["close"].astype(float)
                prev_close = closes.shift(1)
                tr = pd.concat([
                    highs - lows,
                    (highs - prev_close).abs(),
                    (lows  - prev_close).abs(),
                ], axis=1).max(axis=1)
                atr14 = float(tr.rolling(14).mean().iloc[-1])
                if atr14 > 0:
                    atr_sl = atr14 * 2.0  # 2x ATR: more room to breathe, fewer premature stops
                    min_dist = self._MIN_SL_DIST.get(symbol, entry * 0.008)
                    return max(atr_sl, min_dist)
            except Exception:
                pass
        # Fallback: 1% of price (was 0.5% — doubled to reduce premature stops)
        pct_dist = entry * 0.01
        min_dist = self._MIN_SL_DIST.get(symbol, pct_dist)
        return max(pct_dist, min_dist)

    def _nearest_swing(
        self,
        entry: float,
        sl_dist: float,
        is_bullish: bool,
        tp_raw: float,
        df=None,
    ) -> float:
        """Return TP price: nearest swing high (LONG) or swing low (SHORT) in the chart,
        constrained between 1×SL and 3×SL from entry. Falls back to tp_raw."""
        try:
            _df = df if df is not None else self._last_df
            if _df is None or getattr(_df, 'empty', True) or len(_df) < 20:
                return tp_raw
            highs = _df["high"].astype(float).values[-50:]
            lows  = _df["low"].astype(float).values[-50:]
            min_tp_dist = sl_dist * 1.5   # at least 1.5:1 RR
            max_tp_dist = sl_dist * 3.5   # cap at 3.5:1 RR
            if is_bullish:
                # Nearest swing high above entry + min_tp_dist
                candidates = [h for h in highs if h > entry + min_tp_dist]
                if candidates:
                    nearest = min(candidates)
                    if nearest <= entry + max_tp_dist:
                        return round(nearest, 5)
            else:
                # Nearest swing low below entry - min_tp_dist
                candidates = [lo for lo in lows if lo < entry - min_tp_dist]
                if candidates:
                    nearest = max(candidates)
                    if nearest >= entry - max_tp_dist:
                        return round(nearest, 5)
        except Exception:
            pass
        return tp_raw

    def evaluate(
        self,
        analysis_text: str,
        symbol: str,
        timeframe: str,
        current_price: float,
        poi_zones: list,
        glint_context: str = "",
        df=None,
    ) -> TradeSignal:
        # Store df for ATR-based SL calculation
        if df is not None:
            self._last_df = df

        is_bullish = "bullish" in analysis_text.lower() or "alcista" in analysis_text.lower()
        is_bearish = "bearish" in analysis_text.lower() or "bajista" in analysis_text.lower()
        at_lower   = analysis_text.lower()
        has_setup  = ("setup" in at_lower and ("valid" in at_lower or "valido" in at_lower or "válido" in at_lower)) or "✅" in analysis_text

        if is_bullish and is_bearish:
            return TradeSignal(
                symbol=symbol, signal_type=SignalType.WAIT,
                entry=current_price, stop_loss=None,
                take_profit=current_price, timeframe=timeframe,
                trigger="Conflicto bullish+bearish — sin entrada",
                confidence=0.0,
            )

        if not has_setup or (not is_bullish and not is_bearish):
            return TradeSignal(
                symbol=symbol, signal_type=SignalType.WAIT,
                entry=current_price, stop_loss=None,
                take_profit=current_price, timeframe=timeframe,
                trigger="Sin setup claro — paciencia",
                confidence=0.0,
            )

        _df = df if df is not None else self._last_df

        # Count confluence factors to set TP multiplier (more confluence = bolder target)
        n_confluence = sum([
            "BOS" in analysis_text,
            "CHoCH" in analysis_text,
            "order block" in analysis_text,
            "FVG" in analysis_text,
        ])
        tp_mult = 3.0 if n_confluence >= 3 else (2.5 if n_confluence == 2 else 2.0)

        if poi_zones:
            poi = poi_zones[0]
            if is_bullish:
                entry    = poi.get("zone_low", current_price)
                sl_dist  = self._sl_distance(symbol, entry, _df)
                sl       = entry - sl_dist
                tp_raw   = entry + sl_dist * tp_mult
                # Use nearest swing high if available and closer than tp_raw
                tp = self._nearest_swing(entry, sl_dist, is_bullish=True, tp_raw=tp_raw, df=_df)
                sl       = entry - sl_dist
            else:
                entry    = poi.get("zone_high", current_price)
                sl_dist  = self._sl_distance(symbol, entry, _df)
                sl       = entry + sl_dist
                tp_raw   = entry - sl_dist * tp_mult
                tp = self._nearest_swing(entry, sl_dist, is_bullish=False, tp_raw=tp_raw, df=_df)
        else:
            entry   = current_price
            sl_dist = self._sl_distance(symbol, entry, _df)
            sl      = (entry - sl_dist) if is_bullish else (entry + sl_dist)
            tp_raw  = (entry + sl_dist * tp_mult) if is_bullish else (entry - sl_dist * tp_mult)
            tp = self._nearest_swing(entry, sl_dist, is_bullish=is_bullish, tp_raw=tp_raw, df=_df)

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
