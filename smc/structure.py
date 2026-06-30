from dataclasses import dataclass
from enum import Enum
from typing import List
import pandas as pd


class StructureType(Enum):
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    ACCUMULATION  = "accumulation"
    DISTRIBUTION  = "distribution"
    RANGING       = "ranging"


@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: str  # "HH", "HL", "LH", "LL"


@dataclass
class StructureResult:
    structure_type: StructureType
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    swing_points: List[SwingPoint]
    bias: str  # "bullish" | "bearish" | "neutral"


class MarketStructure:
    def __init__(self, df: pd.DataFrame, swing_lookback: int = 5):
        self.df = df.copy()
        self.lookback = swing_lookback
        self._swings: List[SwingPoint] = []

    def _find_swings(self) -> List[SwingPoint]:
        swings = []
        highs = self.df["high"].values
        lows  = self.df["low"].values
        n = len(highs)
        lb = self.lookback

        prev_high = highs[0]
        prev_low  = lows[0]

        for i in range(lb, n - lb):
            is_sh = all(highs[i] > highs[j] for j in range(i - lb, i + lb + 1) if j != i)
            is_sl = all(lows[i]  < lows[j]  for j in range(i - lb, i + lb + 1) if j != i)

            if is_sh:
                sp_type = "HH" if (prev_high is None or highs[i] > prev_high) else "LH"
                swings.append(SwingPoint(i, highs[i], sp_type))
                prev_high = highs[i]

            if is_sl:
                sp_type = "HL" if (prev_low is None or lows[i] > prev_low) else "LL"
                swings.append(SwingPoint(i, lows[i], sp_type))
                prev_low = lows[i]

        self._swings = sorted(swings, key=lambda x: x.index)
        return self._swings

    def analyze(self) -> StructureResult:
        swings = self._find_swings()
        hh = sum(1 for s in swings if s.swing_type == "HH")
        hl = sum(1 for s in swings if s.swing_type == "HL")
        lh = sum(1 for s in swings if s.swing_type == "LH")
        ll = sum(1 for s in swings if s.swing_type == "LL")

        # Bullish: más HH+HL que LH+LL (tendencia alcista dominante)
        # Bearish: más LH+LL que HH+HL (tendencia bajista dominante)
        # Condición anterior (lh==0 y ll==0) era imposible en 200 velas reales → siempre neutral
        bull_score = hh + hl
        bear_score = lh + ll
        if bull_score > bear_score and hh >= 1 and hl >= 1:
            stype, bias = StructureType.BULLISH_TREND, "bullish"
        elif bear_score > bull_score and lh >= 1 and ll >= 1:
            stype, bias = StructureType.BEARISH_TREND, "bearish"
        elif hh > 0 and ll > 0:
            stype, bias = StructureType.DISTRIBUTION, "neutral"
        elif hl > 0 and lh > 0:
            stype, bias = StructureType.ACCUMULATION, "neutral"
        else:
            stype, bias = StructureType.RANGING, "neutral"

        return StructureResult(stype, hh, hl, lh, ll, swings, bias)

    def _atr(self, period: int = 14) -> float:
        """Average True Range — mide rango promedio de velas."""
        df = self.df
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def detect_bos(self) -> List[dict]:
        swings = self._find_swings() if not self._swings else self._swings
        closes = self.df["close"].values
        highs  = self.df["high"].values
        lows   = self.df["low"].values
        opens  = self.df["open"].values if "open" in self.df.columns else closes
        atr    = self._atr()
        bos_events = []

        for swing in swings:
            if swing.swing_type == "HH":
                for j in range(swing.index + 1, len(closes)):
                    if closes[j] > swing.price:
                        candle_range = highs[j] - lows[j]
                        candle_body  = abs(closes[j] - opens[j])
                        # Displacement: rango expandido Y cierre fuerte (cuerpo > 60% del rango)
                        is_disp = (candle_range >= atr * 1.5) and (candle_body >= candle_range * 0.6)
                        bos_events.append({
                            "type": "BOS",
                            "direction": "bullish",
                            "level": swing.price,
                            "confirmed_at": j,
                            "is_displacement": is_disp,
                        })
                        break
            elif swing.swing_type == "LL":
                for j in range(swing.index + 1, len(closes)):
                    if closes[j] < swing.price:
                        candle_range = highs[j] - lows[j]
                        candle_body  = abs(closes[j] - opens[j])
                        is_disp = (candle_range >= atr * 1.5) and (candle_body >= candle_range * 0.6)
                        bos_events.append({
                            "type": "BOS",
                            "direction": "bearish",
                            "level": swing.price,
                            "confirmed_at": j,
                            "is_displacement": is_disp,
                        })
                        break

        return bos_events

    def detect_choch(self) -> List[dict]:
        swings = self._find_swings() if not self._swings else self._swings
        closes = self.df["close"].values
        choch_events = []

        for swing in swings:
            if swing.swing_type == "LH":
                for j in range(swing.index + 1, len(closes)):
                    if closes[j] > swing.price:
                        choch_events.append({
                            "type": "CHoCH",
                            "direction": "bullish",
                            "level": swing.price,
                            "confirmed_at": j,
                        })
                        break
            elif swing.swing_type == "HL":
                for j in range(swing.index + 1, len(closes)):
                    if closes[j] < swing.price:
                        choch_events.append({
                            "type": "CHoCH",
                            "direction": "bearish",
                            "level": swing.price,
                            "confirmed_at": j,
                        })
                        break

        return choch_events

    def summary(self) -> str:
        result = self.analyze()
        bos = self.detect_bos()
        choch = self.detect_choch()
        lines = [
            f"Estructura: {result.structure_type.value.upper()} | Bias: {result.bias.upper()}",
            f"HH:{result.higher_highs} HL:{result.higher_lows} LH:{result.lower_highs} LL:{result.lower_lows}",
            f"BOS: {len(bos)} | CHoCH: {len(choch)}",
        ]
        if bos:
            lines.append(f"Último BOS: {bos[-1]['direction']} @ {bos[-1]['level']:.5f}")
        if choch:
            lines.append(f"CHoCH: {choch[-1]['direction']} @ {choch[-1]['level']:.5f}")
        return "\n".join(lines)
