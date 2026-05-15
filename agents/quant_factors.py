"""Factor analysis using IC/IR framework."""
from dataclasses import dataclass
import numpy as np

@dataclass
class FactorResult:
    factor_name: str
    ic: float
    ir: float
    t_stat: float
    significant: bool
    direction: str

class FactorAnalyzer:
    @staticmethod
    def calculate_ic(factor_values, forward_returns):
        if len(factor_values) < 3 or len(factor_values) != len(forward_returns): return 0.0
        f = np.array(factor_values); r = np.array(forward_returns)
        if np.std(f)==0 or np.std(r)==0: return 0.0
        fr = np.argsort(np.argsort(f)).astype(float); rr = np.argsort(np.argsort(r)).astype(float)
        n = len(fr); mean_r = (n-1)/2
        cov = np.mean((fr-mean_r)*(rr-mean_r))
        std_f = np.std(fr); std_r = np.std(rr)
        return float(cov/(std_f*std_r)) if std_f>0 and std_r>0 else 0.0

    @staticmethod
    def calculate_ir(ic_series):
        if len(ic_series) < 2: return 0.0
        arr = np.array(ic_series)
        std = np.std(arr, ddof=1)
        return 0.0 if std < 1e-10 else float(np.mean(arr)/std)

    @staticmethod
    def calculate_t_stat(ic, n_obs):
        if n_obs < 2: return 0.0
        denom = 1 - ic**2
        return 0.0 if denom <= 0 else float(ic * np.sqrt(n_obs) / np.sqrt(denom))

    @staticmethod
    def calculate_momentum_factor(prices, period=20):
        if len(prices) <= period: return 0.0
        return (prices[-1] - prices[-1-period]) / prices[-1-period] if prices[-1-period] != 0 else 0.0

    @staticmethod
    def calculate_reversal_factor(prices, period=5):
        return -FactorAnalyzer.calculate_momentum_factor(prices, period)

    @staticmethod
    def calculate_volatility_factor(returns, window=20):
        if len(returns) < window: return 0.0
        return float(np.std(returns[-window:], ddof=1))

    def analyze_factor(self, factor_name, factor_values, forward_returns):
        ic = self.calculate_ic(factor_values, forward_returns)
        n = len(factor_values)
        ir = self.calculate_ir([ic])
        t = self.calculate_t_stat(ic, n)
        direction = "long" if ic > 0.05 else "short" if ic < -0.05 else "neutral"
        return FactorResult(factor_name, ic, ir, t, abs(t)>1.96, direction)

    def analyze_all_factors(self, prices, forward_returns):
        results = []
        for p in [20, 60]:
            fv = [self.calculate_momentum_factor(prices[:i+1], p) for i in range(len(prices))]
            if len(fv) >= len(forward_returns):
                fv = fv[-len(forward_returns):]
            elif len(forward_returns) > len(fv):
                forward_returns = forward_returns[-len(fv):]
            results.append(self.analyze_factor(f"momentum_{p}", fv, list(forward_returns)))
        rev_v = [self.calculate_reversal_factor(prices[:i+1], 5) for i in range(len(prices))]
        rev_v = rev_v[-len(forward_returns):] if len(rev_v) >= len(forward_returns) else rev_v
        fr2 = list(forward_returns)[-len(rev_v):]
        results.append(self.analyze_factor("reversal_5", rev_v, fr2))
        return results

    @staticmethod
    def get_best_factor(results):
        if not results: return None
        return max(results, key=lambda r: abs(r.ir))
