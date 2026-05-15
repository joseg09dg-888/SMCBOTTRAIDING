"""Market regime detection using heuristics (HMM optional)."""
from dataclasses import dataclass
from enum import Enum
import numpy as np

class MarketRegime(Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    HIGH_VOL      = "high_vol"
    RANGING       = "ranging"

@dataclass
class RegimeAnalysis:
    regime: MarketRegime
    confidence: float
    volatility: float
    trend_strength: float
    win_rate_in_regime: float
    recommended_rr: float
    recommended_risk_multiplier: float

class RegimeDetector:
    def __init__(self, lookback=50):
        self.lookback = lookback
        self._has_hmm = self._check_hmm()
    def _check_hmm(self):
        try:
            import hmmlearn; return True
        except ImportError:
            return False
    def detect(self, prices, as_of_index=-1):
        arr = np.array(prices, dtype=float)
        if len(arr) < 3:
            return RegimeAnalysis(MarketRegime.RANGING,0.5,0.0,0.0,0.55,1.5,0.5)
        window = arr[-self.lookback:] if as_of_index == -1 else arr[max(0,as_of_index-self.lookback):as_of_index if as_of_index>0 else len(arr)]
        if len(window) < 2: window = arr[-min(self.lookback,len(arr)):]
        returns = np.diff(window) / np.where(window[:-1]==0, 1e-8, window[:-1])
        if len(returns) == 0:
            return RegimeAnalysis(MarketRegime.RANGING,0.5,0.0,0.0,0.55,1.5,0.5)
        std = float(np.std(returns,ddof=1)) if len(returns)>1 else 0.0
        vol_annual = std * np.sqrt(252)
        mean_r = float(np.mean(returns))
        trend = mean_r/std if std > 0 else 0.0
        if std > 0.03:
            regime = MarketRegime.HIGH_VOL; conf = min(std/0.06,1.0); rr=3.0; rm=0.25
        elif trend > 0.5:
            regime = MarketRegime.TRENDING_UP; conf = min(abs(trend)/1.5,1.0); rr=2.5; rm=1.0
        elif trend < -0.5:
            regime = MarketRegime.TRENDING_DOWN; conf = min(abs(trend)/1.5,1.0); rr=2.5; rm=1.0
        else:
            regime = MarketRegime.RANGING; conf = float(np.clip(1.0-abs(trend)/0.5,0.3,0.9)); rr=1.5; rm=0.5
        return RegimeAnalysis(regime,float(np.clip(conf,0,1)),vol_annual,abs(trend),
            RegimeDetector.regime_win_rate_estimate(regime),rr,rm)
    def get_regime_history(self, prices, step=10):
        result = []
        for i in range(self.lookback, len(prices), step):
            result.append(self.detect(prices, as_of_index=i))
        if not result and len(prices)>=3: result.append(self.detect(prices))
        return result
    def get_dominant_regime(self, prices):
        history = self.get_regime_history(prices)
        if not history: return self.detect(prices).regime
        counts = {}
        for h in history: counts[h.regime] = counts.get(h.regime,0)+1
        return max(counts, key=counts.get)
    @staticmethod
    def regime_win_rate_estimate(regime):
        return {MarketRegime.TRENDING_UP:0.65,MarketRegime.TRENDING_DOWN:0.62,
                MarketRegime.HIGH_VOL:0.45,MarketRegime.RANGING:0.55}.get(regime,0.55)
