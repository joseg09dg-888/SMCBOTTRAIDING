from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
import numpy as np


@dataclass
class PredictionResult:
    direction: str       # "bullish" | "bearish" | "neutral"
    confidence: float    # 0.0 - 1.0
    score: int           # 0 - 25 (contribution to DecisionFilter)
    features: Dict[str, float] = field(default_factory=dict)


class MLPredictor:
    """
    Feature-based predictor that scores technical quality for the DecisionFilter.
    Architecture is designed to be replaced by a real LSTM: swap predict() with
    a model.predict() call and keep the same PredictionResult interface.

    Score breakdown (max 25):
      - Direction match:    15 pts
      - Confidence > 70%:   10 pts
    """

    def predict(self, df: pd.DataFrame, bias: str = "neutral") -> PredictionResult:
        features = self._extract_features(df)
        direction = self._infer_direction(features)
        confidence = self._calc_confidence(features)

        score = 0
        if direction == bias and bias != "neutral":
            score += 15
        if confidence > 0.70:
            score += 10
        elif confidence > 0.55:
            score += 5

        score = min(score, 25)
        return PredictionResult(
            direction=direction,
            confidence=round(confidence, 4),
            score=score,
            features=features,
        )

    def _extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        closes  = df["close"].values
        highs   = df["high"].values
        lows    = df["low"].values
        volumes = df["volume"].values
        n = len(closes)

        # Momentum: % change over last 5 bars vs previous 5 bars
        if n >= 10:
            recent = closes[-1] / closes[-5] - 1
            prior  = closes[-5] / closes[-10] - 1
            momentum = recent - prior
        elif n >= 2:
            momentum = (closes[-1] / closes[0] - 1)
        else:
            momentum = 0.0

        # Trend consistency: % of bars moving in dominant direction
        if n >= 3:
            up_bars   = sum(1 for i in range(1, n) if closes[i] > closes[i - 1])
            consistency = up_bars / (n - 1)
            # Normalize: 0.5 = ranging, >0.6 = bullish, <0.4 = bearish
            trend_consistency = (consistency - 0.5) * 2  # -1 to +1
        else:
            trend_consistency = 0.0

        # Volume confirmation: recent volume vs average
        if n >= 5:
            avg_vol = volumes[:-2].mean() if n > 2 else volumes.mean()
            vol_confirm = (volumes[-1] / avg_vol - 1) if avg_vol > 0 else 0.0
        else:
            vol_confirm = 0.0

        # Price action quality: ratio of body to wick on last 3 bars
        if n >= 3:
            bodies = [abs(df["close"].iloc[i] - df["open"].iloc[i]) for i in range(-3, 0)]
            wicks  = [df["high"].iloc[i] - df["low"].iloc[i] for i in range(-3, 0)]
            pa_quality = (
                sum(bodies) / sum(wicks) if sum(wicks) > 0 else 0.5
            )
        else:
            pa_quality = 0.5

        # Higher timeframe bias proxy: slope of last N closes
        if n >= 5:
            x = np.arange(5)
            y = closes[-5:]
            slope = np.polyfit(x, y, 1)[0]
            htf_bias = slope / closes[-1]  # normalized slope
        else:
            htf_bias = 0.0

        return {
            "momentum":          round(float(momentum), 6),
            "trend_consistency": round(float(trend_consistency), 4),
            "volume_confirmation": round(float(min(vol_confirm, 2.0)), 4),
            "price_action_quality": round(float(min(pa_quality, 1.0)), 4),
            "htf_bias":          round(float(htf_bias), 6),
        }

    def _infer_direction(self, features: Dict[str, float]) -> str:
        score = (
            features["momentum"] * 10
            + features["trend_consistency"]
            + features["htf_bias"] * 20
        )
        if score > 0.15:
            return "bullish"
        elif score < -0.15:
            return "bearish"
        return "neutral"

    def _calc_confidence(self, features: Dict[str, float]) -> float:
        """Confidence = alignment of all bullish/bearish signals."""
        signals = [
            features["momentum"],
            features["trend_consistency"],
            features["htf_bias"] * 5,
        ]
        # Confidence = % of signals agreeing with the dominant direction
        dominant = 1 if sum(signals) >= 0 else -1
        agreeing = sum(1 for s in signals if (s >= 0) == (dominant >= 0))
        base_conf = agreeing / len(signals)

        # Boost by volume confirmation and PA quality
        boost = min(features["volume_confirmation"] * 0.1 + features["price_action_quality"] * 0.1, 0.2)
        return min(base_conf + boost, 1.0)
