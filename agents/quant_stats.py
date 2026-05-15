"""Core quantitative statistics. Requires only numpy."""
from dataclasses import dataclass
import numpy as np

@dataclass
class MonteCarloResult:
    final_equities: list
    mean_return: float
    std_return: float
    var_95: float
    cvar_95: float
    ruin_probability: float
    best_case: float
    worst_case: float
    median: float

@dataclass
class WalkForwardResult:
    in_sample_sharpe: float
    out_sample_sharpe: float
    degradation_ratio: float
    n_splits: int

class QuantStats:
    @staticmethod
    def calculate_expectancy(trades):
        if not trades: return 0.0
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = abs(float(np.mean(losses))) if losses else 0.0
        return wr * avg_win - (1 - wr) * avg_loss

    @staticmethod
    def calculate_kelly_fraction(win_rate, avg_win, avg_loss):
        if avg_loss == 0: return 0.0
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        return float(np.clip(kelly, 0.0, 0.25))

    @staticmethod
    def calculate_var(returns, confidence=0.95):
        if not returns: return 0.0
        return float(-np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        if not returns: return 0.0
        arr = np.array(returns)
        var = np.percentile(arr, (1 - confidence) * 100)
        tail = arr[arr <= var]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-var)

    @staticmethod
    def calculate_sharpe(returns, risk_free=0.0, periods_per_year=252):
        if not returns: return 0.0
        arr = np.array(returns) - risk_free
        std = np.std(arr, ddof=1)
        return 0.0 if std == 0 else float(np.mean(arr) / std * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_sortino(returns, risk_free=0.0, periods_per_year=252):
        if not returns: return 0.0
        arr = np.array(returns) - risk_free
        down = arr[arr < 0]
        if len(down) == 0: return 0.0
        dstd = np.std(down, ddof=1)
        return 0.0 if dstd == 0 else float(np.mean(arr) / dstd * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        if len(equity_curve) < 2: return 0.0
        arr = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = np.where(peak == 0, 0.0, (peak - arr) / peak)
        return float(np.max(dd))

    @staticmethod
    def calculate_calmar(returns, equity_curve):
        if not returns or not equity_curve: return 0.0
        mdd = QuantStats.calculate_max_drawdown(equity_curve)
        ann = float(np.mean(returns)) * 252
        if mdd == 0: return ann if ann > 0 else 0.0
        return ann / mdd

    @staticmethod
    def calculate_ulcer_index(equity_curve):
        if len(equity_curve) < 2: return 0.0
        arr = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd_pct = np.where(peak == 0, 0.0, (peak - arr) / peak * 100)
        return float(np.sqrt(np.mean(dd_pct ** 2)))

    @staticmethod
    def calculate_profit_factor(trades):
        if not trades: return 0.0
        pnls = [t['pnl'] for t in trades]
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        return float(gw) if gl == 0 else float(gw / gl)

    @staticmethod
    def run_monte_carlo(returns, n_sims=10000, n_periods=252, initial_equity=1000.0, seed=42):
        if not returns:
            return MonteCarloResult([],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        rng = np.random.default_rng(seed)
        arr = np.array(returns)
        samples = rng.choice(arr, size=(n_sims, n_periods), replace=True)
        threshold = initial_equity * 0.01
        final_equities = []
        ruin_count = 0
        for sim in samples:
            eq = initial_equity
            ruined = False
            for r in sim:
                eq *= (1 + r)
                if eq <= threshold:
                    ruined = True; break
            if ruined:
                ruin_count += 1; final_equities.append(0.0)
            else:
                final_equities.append(eq)
        fe = np.array(final_equities)
        rets = (fe - initial_equity) / initial_equity
        var95 = float(-np.percentile(rets, 5))
        tail = rets[rets <= -var95]
        cvar95 = float(-np.mean(tail)) if len(tail) > 0 else var95
        return MonteCarloResult(
            final_equities=list(fe), mean_return=float(np.mean(rets)),
            std_return=float(np.std(rets)), var_95=var95, cvar_95=cvar95,
            ruin_probability=ruin_count/n_sims,
            best_case=float(np.percentile(fe,95)), worst_case=float(np.percentile(fe,5)),
            median=float(np.median(fe)))

    @staticmethod
    def run_walk_forward(returns, n_splits=5):
        if len(returns) < n_splits * 2:
            return WalkForwardResult(0.0, 0.0, 1.0, n_splits)
        arr = np.array(returns)
        block = len(arr) // (n_splits + 1)
        def ss(r):
            if len(r) < 2: return 0.0
            std = np.std(r, ddof=1)
            return float(np.mean(r)/std*np.sqrt(252)) if std > 0 else 0.0
        in_s, out_s = [], []
        for i in range(n_splits):
            in_s.append(ss(arr[:block*(i+1)]))
            out_block = arr[block*(i+1):block*(i+2)]
            if len(out_block) == 0: break
            out_s.append(ss(out_block))
        if not in_s: return WalkForwardResult(0.0,0.0,1.0,n_splits)
        mi = float(np.mean([v for v in in_s if np.isfinite(v)] or [0.0]))
        mo = float(np.mean([v for v in out_s if np.isfinite(v)] or [0.0]))
        deg = mo/mi if mi != 0 else 1.0
        return WalkForwardResult(mi, mo, deg, n_splits)

    @staticmethod
    def calculate_ruin_probability(returns, n_sims=10000, seed=42):
        if not returns: return 0.0
        return QuantStats.run_monte_carlo(returns, n_sims=n_sims, seed=seed).ruin_probability

    @staticmethod
    def calculate_omega_ratio(returns, threshold=0.0):
        if not returns: return 1.0
        arr = np.array(returns)
        gains = arr[arr > threshold] - threshold
        losses = threshold - arr[arr <= threshold]
        sg, sl = float(np.sum(gains)), float(np.sum(losses))
        return float('inf') if sl == 0 else sg/sl
