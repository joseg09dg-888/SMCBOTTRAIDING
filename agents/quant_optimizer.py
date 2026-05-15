"""Bayesian optimization with Optuna (random search fallback)."""
from dataclasses import dataclass
import numpy as np

@dataclass
class OptimizationResult:
    best_params: dict
    best_value: float
    n_trials: int
    improvement_pct: float
    convergence_trial: int

class BayesianOptimizer:
    DEFAULT_PARAMS = {
        "score_threshold":60.0,"ob_lookback":2.0,"ob_threshold":0.5,
        "min_rr":2.0,"risk_pct":0.5,"smc_weight":30.0,
        "ml_weight":25.0,"sentiment_weight":20.0,"risk_weight":25.0,
    }
    PARAM_BOUNDS = {
        "score_threshold":(55.0,85.0),"ob_lookback":(1.0,5.0),"ob_threshold":(0.2,1.5),
        "min_rr":(1.5,4.0),"risk_pct":(0.1,1.5),"smc_weight":(15.0,50.0),
        "ml_weight":(10.0,40.0),"sentiment_weight":(5.0,35.0),"risk_weight":(10.0,40.0),
    }
    def __init__(self, seed=42):
        self.seed = seed
        self._has_optuna = self._check_optuna()
    def _check_optuna(self):
        try:
            import optuna; return True
        except ImportError:
            return False
    def _random_search(self, objective, param_names, n_trials):
        rng = np.random.default_rng(self.seed)
        best_val = float('-inf'); best_params = {}; history = []
        names = param_names or list(self.PARAM_BOUNDS.keys())
        for t in range(n_trials):
            params = dict(self.DEFAULT_PARAMS)
            for k in names:
                lo, hi = self.PARAM_BOUNDS.get(k,(0.0,1.0))
                params[k] = float(rng.uniform(lo, hi))
            try: val = float(objective(params))
            except Exception: val = float('-inf')
            history.append(val)
            if val > best_val: best_val = val; best_params = dict(params)
        window = max(1, int(0.20*n_trials))
        conv = n_trials-1
        for i in range(len(history)-window):
            if max(history[i+1:i+window+1]) <= history[i]:
                conv = i; break
        default_val = float(objective(self.DEFAULT_PARAMS))
        imp = ((best_val-default_val)/abs(default_val)*100) if default_val!=0 else 0.0
        return OptimizationResult(best_params, best_val, n_trials, imp, conv)
    def optimize(self, objective, param_names=None, n_trials=50, direction="maximize"):
        names = param_names or list(self.PARAM_BOUNDS.keys())
        if self._has_optuna:
            try:
                import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
                sampler = optuna.samplers.TPESampler(seed=self.seed)
                study = optuna.create_study(direction=direction, sampler=sampler)
                history = []
                def wrapped(trial):
                    params = dict(self.DEFAULT_PARAMS)
                    for k in names:
                        lo, hi = self.PARAM_BOUNDS.get(k,(0.0,1.0))
                        params[k] = trial.suggest_float(k, lo, hi)
                    val = float(objective(params))
                    history.append(val)
                    return val
                study.optimize(wrapped, n_trials=n_trials, show_progress_bar=False)
                bp = dict(self.DEFAULT_PARAMS)
                bp.update(study.best_params)
                bv = study.best_value
                window = max(1,int(0.20*n_trials))
                conv = n_trials-1
                for i in range(len(history)-window):
                    if max(history[i+1:i+window+1]) <= history[i]:
                        conv=i; break
                default_val = float(objective(self.DEFAULT_PARAMS))
                imp = ((bv-default_val)/abs(default_val)*100) if default_val!=0 else 0.0
                return OptimizationResult(bp, bv, n_trials, imp, conv)
            except Exception:
                pass
        return self._random_search(objective, names, n_trials)
    def optimize_sharpe(self, returns_generator, n_trials=50):
        def obj(params):
            rets = returns_generator(params)
            if not rets: return -999.0
            arr = np.array(rets)
            std = np.std(arr,ddof=1)
            return float(np.mean(arr)/std*np.sqrt(252)) if std>0 else 0.0
        return self.optimize(obj, n_trials=n_trials)
    @staticmethod
    def clip_params(params):
        result = dict(params)
        for k,v in params.items():
            if k in BayesianOptimizer.PARAM_BOUNDS:
                lo,hi = BayesianOptimizer.PARAM_BOUNDS[k]
                result[k] = float(np.clip(v,lo,hi))
        return result
