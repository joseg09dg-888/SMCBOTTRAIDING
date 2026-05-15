"""ML ensemble for trade prediction."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EnsemblePrediction:
    probability: float
    confidence: str
    model_votes: dict
    feature_importance: dict
    should_trade: bool

class FeatureExtractor:
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period+1: return 50.0
        arr = np.array(prices, dtype=float)
        delta = np.diff(arr)
        gains = np.where(delta>0, delta, 0.0)
        losses = np.where(delta<0, -delta, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0: return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return float(100 - 100/(1+rs))

    @staticmethod
    def from_prices(prices, volumes=None):
        zero = {k: 0.0 for k in ['ret_1','ret_3','ret_5','ret_10','ret_20',
            'vol_5','vol_10','vol_20','skew_20','kurt_20','above_ma20','above_ma50',
            'rsi_14','price_range_pct','momentum_score','vol_ratio']}
        if len(prices) < 50: return zero
        arr = np.array(prices, dtype=float)
        def ret(n): return (arr[-1]-arr[-1-n])/arr[-1-n] if arr[-1-n]!=0 else 0.0
        def vol(n):
            r = np.diff(arr[-n-1:])/arr[-n-1:-1]
            return float(np.std(r,ddof=1)) if len(r)>1 else 0.0
        r20 = np.diff(arr[-21:])/arr[-21:-1] if len(arr)>=21 else np.array([])
        skew = float(np.mean(((r20-np.mean(r20))/np.std(r20))**3)) if len(r20)>2 and np.std(r20)>0 else 0.0
        kurt = float(np.mean(((r20-np.mean(r20))/np.std(r20))**4)-3) if len(r20)>2 and np.std(r20)>0 else 0.0
        ma20 = float(np.mean(arr[-20:])) if len(arr)>=20 else arr[-1]
        ma50 = float(np.mean(arr[-50:])) if len(arr)>=50 else arr[-1]
        v5,v20 = vol(5),vol(20)
        signs = sum(1 if ret(n)>0 else -1 for n in [1,3,5])/3
        return {
            'ret_1':ret(1),'ret_3':ret(3),'ret_5':ret(5),'ret_10':ret(10),'ret_20':ret(20),
            'vol_5':v5,'vol_10':vol(10),'vol_20':v20,
            'skew_20':skew,'kurt_20':kurt,
            'above_ma20':1.0 if arr[-1]>ma20 else 0.0,
            'above_ma50':1.0 if arr[-1]>ma50 else 0.0,
            'rsi_14':FeatureExtractor.calculate_rsi(prices),
            'price_range_pct':0.0,'momentum_score':float(signs),
            'vol_ratio':v5/v20 if v20>0 else 1.0,
        }

def _sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))

class MLEnsemble:
    THRESHOLD = 0.65
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self._models = {}
        self._is_fitted = False
        self._feature_names = ['ret_1','ret_5','vol_20','rsi_14','above_ma20',
                               'above_ma50','momentum_score','vol_ratio','skew_20','kurt_20']
        self._scaler = None
        self._train_X = None; self._train_y = None
        self._has_sklearn = self._check_sklearn()

    def _check_sklearn(self):
        try:
            import sklearn; return True
        except ImportError:
            return False

    def _feats_to_vec(self, features):
        return np.array([features.get(k,0.0) for k in self._feature_names])

    def fit(self, X, y):
        self._is_fitted = True
        if not self._has_sklearn: return
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
            Xarr = np.array([self._feats_to_vec(x) for x in X])
            yarr = np.array(y)
            self._scaler = StandardScaler().fit(Xarr)
            Xs = self._scaler.transform(Xarr)
            self._train_X = Xs; self._train_y = yarr
            for name, clf in [
                ('logistic', LogisticRegression(C=1.0,max_iter=200,random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50,max_depth=5,random_state=42)),
                ('gboost', GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=42)),
                ('extra_trees', ExtraTreesClassifier(n_estimators=50,max_depth=5,random_state=42)),
            ]:
                try: clf.fit(Xs,yarr); self._models[name]=clf
                except Exception: pass
        except Exception: pass

    def predict(self, features):
        vec = self._feats_to_vec(features)
        if not self._is_fitted or not self._models:
            ms = features.get('momentum_score',0.0)
            ma = features.get('above_ma20',0.5)
            prob = float(_sigmoid(ms*3+(ma-0.5)*1.0))
            votes = {'momentum_heuristic': prob}
        else:
            probs = []
            votes = {}
            Xs = self._scaler.transform(vec.reshape(1,-1))
            for name, clf in self._models.items():
                try:
                    p = float(clf.predict_proba(Xs)[0,1])
                    probs.append(p); votes[name]=p
                except Exception: pass
            prob = float(np.mean(probs)) if probs else 0.5
        if prob < 0.55: conf = "LOW"
        elif prob < 0.65: conf = "MEDIUM"
        elif prob < 0.75: conf = "HIGH"
        else: conf = "VERY_HIGH"
        return EnsemblePrediction(
            probability=float(np.clip(prob,0,1)), confidence=conf,
            model_votes=votes, feature_importance={},
            should_trade=prob >= self.threshold)

    def predict_from_prices(self, prices, volumes=None):
        return self.predict(FeatureExtractor.from_prices(prices, volumes))

    def get_model_accuracies(self):
        if not self._is_fitted or not self._models or self._train_X is None: return {}
        result = {}
        for name, clf in self._models.items():
            try:
                preds = clf.predict(self._train_X)
                result[name] = float(np.mean(preds == self._train_y))
            except Exception: pass
        return result
