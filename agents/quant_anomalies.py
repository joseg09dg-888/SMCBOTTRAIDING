"""Calendar effects and crypto anomaly detection."""
from dataclasses import dataclass
from datetime import datetime, date, timedelta

@dataclass
class AnomalySignal:
    anomaly_type: str
    direction: str
    strength: float
    pts: int
    description: str

_US_HOLIDAYS = {(1,1),(12,25),(7,4)}

class AnomalyDetector:
    @staticmethod
    def check_monday_effect(dt):
        if dt.weekday() == 0:
            return AnomalySignal("monday_effect","bearish",0.6,-3,"Lunes: efecto negativo en apertura")
        return None

    @staticmethod
    def check_turn_of_month(dt):
        if dt.day in (28,29,30,31,1,2,3):
            return AnomalySignal("turn_of_month","bullish",0.7,5,"Fin/inicio de mes: rebalanceo institucional")
        return None

    @staticmethod
    def check_end_of_quarter(dt):
        if dt.month in (3,6,9,12) and dt.day >= 28:
            return AnomalySignal("end_of_quarter","bullish",0.65,3,"Fin de trimestre: window dressing alcista")
        return None

    @staticmethod
    def check_pre_holiday(dt, holidays=None):
        tomorrow = (dt + timedelta(days=1)).date() if isinstance(dt, datetime) else dt + timedelta(days=1)
        if holidays is None:
            if (tomorrow.month, tomorrow.day) in _US_HOLIDAYS:
                return AnomalySignal("pre_holiday","bullish",0.7,4,"Pre-festivo: rally alcista")
        else:
            if tomorrow in holidays:
                return AnomalySignal("pre_holiday","bullish",0.7,4,"Pre-festivo: rally alcista")
        return None

    @staticmethod
    def check_funding_rate(funding_rate):
        if funding_rate > 0.003:
            return AnomalySignal("funding_extreme_positive","bearish",0.85,-5,
                f"Funding extremo positivo ({funding_rate:.4f}): reversión bajista")
        elif funding_rate < -0.001:
            return AnomalySignal("funding_extreme_negative","bullish",0.85,8,
                f"Funding extremo negativo ({funding_rate:.4f}): reversión alcista")
        return AnomalySignal("funding_normal","neutral",0.3,0,"Funding rate normal")

    @staticmethod
    def check_halving_cycle_phase(days_since_halving):
        if days_since_halving < 0:
            return AnomalySignal("halving_unknown","neutral",0.3,0,"Ciclo halving desconocido")
        if days_since_halving <= 365:
            return AnomalySignal("halving_phase_1","bullish",0.7,5,"Fase 1 halving: acumulacion")
        elif days_since_halving <= 730:
            return AnomalySignal("halving_phase_2","bullish",0.9,10,"Fase 2 halving: bull run")
        elif days_since_halving <= 1094:
            return AnomalySignal("halving_phase_3","bearish",0.7,-5,"Fase 3 halving: distribucion")
        return AnomalySignal("halving_phase_4","bearish",0.8,-8,"Fase 4 halving: bear market")

    @staticmethod
    def check_gap_fill_probability(gap_pct, is_gap_up):
        if gap_pct < 0.01:
            fp = 0.75; strength = 0.75
        elif gap_pct < 0.03:
            fp = 0.60; strength = 0.60
        else:
            fp = 0.40; strength = 0.40
        direction = "bearish" if is_gap_up else "bullish"
        pts = -2 if is_gap_up else 2
        if gap_pct >= 0.03:
            pts = -3 if is_gap_up else 3
        return AnomalySignal("gap_fill",direction,strength,pts,
            f"Gap {'up' if is_gap_up else 'down'} {gap_pct*100:.1f}%: fill prob {fp*100:.0f}%")

    def get_all_signals(self, dt, symbol="", funding_rate=0.0, gap_pct=0.0,
                        is_gap_up=True, days_since_halving=-1):
        signals = []
        for fn in (self.check_monday_effect, self.check_turn_of_month,
                   self.check_end_of_quarter):
            s = fn(dt)
            if s: signals.append(s)
        signals.append(self.check_funding_rate(funding_rate))
        if gap_pct > 0:
            signals.append(self.check_gap_fill_probability(gap_pct, is_gap_up))
        if days_since_halving >= 0 and ("BTC" in symbol.upper() or symbol == ""):
            signals.append(self.check_halving_cycle_phase(days_since_halving))
        return [s for s in signals if s is not None]

    def get_anomaly_score(self, dt, symbol="", funding_rate=0.0, gap_pct=0.0,
                          is_gap_up=True, days_since_halving=-1):
        signals = self.get_all_signals(dt, symbol, funding_rate, gap_pct, is_gap_up, days_since_halving)
        total = sum(s.pts for s in signals)
        return max(-15, min(15, total))
