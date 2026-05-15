import pytest, numpy as np
from agents.quant_factors import FactorAnalyzer, FactorResult
from agents.quant_anomalies import AnomalyDetector, AnomalySignal
from datetime import datetime, timezone, timedelta

def correlated(n=100,corr=0.7,seed=42):
    rng=np.random.default_rng(seed); x=rng.normal(0,1,n)
    y=corr*x+np.sqrt(1-corr**2)*rng.normal(0,1,n)
    return list(x),list(y)

# FACTORS
def test_ic_positive(): f,r=correlated(corr=0.6); assert FactorAnalyzer.calculate_ic(f,r) > 0.3
def test_ic_negative(): f,r=correlated(corr=-0.6); assert FactorAnalyzer.calculate_ic(f,r) < -0.3
def test_ic_range(): f,r=correlated(); assert -1<=FactorAnalyzer.calculate_ic(f,r)<=1
def test_ic_empty(): assert FactorAnalyzer.calculate_ic([],[]) == 0.0
def test_ic_constant(): assert FactorAnalyzer.calculate_ic([1.0]*10,[1.0]*10) == 0.0
def test_ir_stable(): assert FactorAnalyzer.calculate_ir([0.08,0.09,0.07,0.10,0.08]) > 0.5
def test_ir_empty(): assert FactorAnalyzer.calculate_ir([]) == 0.0
def test_ir_zero_std(): assert FactorAnalyzer.calculate_ir([0.1,0.1,0.1]) == 0.0
def test_t_stat_sig(): assert abs(FactorAnalyzer.calculate_t_stat(0.2,100)) > 1.96
def test_t_stat_insig(): assert abs(FactorAnalyzer.calculate_t_stat(0.01,10)) < 1.96
def test_momentum_up(): assert FactorAnalyzer.calculate_momentum_factor([100+i for i in range(50)],period=20) > 0
def test_momentum_down(): assert FactorAnalyzer.calculate_momentum_factor([200-i for i in range(50)],period=20) < 0
def test_momentum_insuf(): assert FactorAnalyzer.calculate_momentum_factor([100,101],period=20) == 0.0
def test_vol_factor():
    np.random.seed(42)
    r=list(np.random.normal(0,0.05,50))
    assert FactorAnalyzer.calculate_volatility_factor(r,window=20) > 0.02
def test_analyze_factor():
    f,r=correlated()
    res=FactorAnalyzer().analyze_factor("test",f,r)
    assert isinstance(res,FactorResult) and res.factor_name=="test"
def test_analyze_all():
    np.random.seed(42)
    prices=list(np.cumprod(1+np.random.normal(0.001,0.02,100))*100)
    rets=list(np.diff(prices)/prices[:-1])
    results=FactorAnalyzer().analyze_all_factors(prices,rets)
    assert len(results) >= 3
def test_best_factor():
    r1=FactorResult("mom",0.05,0.8,2.0,True,"long")
    r2=FactorResult("rev",-0.03,0.3,1.0,False,"short")
    assert FactorAnalyzer.get_best_factor([r1,r2]).factor_name=="mom"
def test_best_factor_empty(): assert FactorAnalyzer.get_best_factor([]) is None

# ANOMALIES
def make_dt(year,month,day): return datetime(year,month,day,10,0,tzinfo=timezone.utc)
def find_weekday(target_wd):
    d=datetime(2026,5,11,tzinfo=timezone.utc)
    while d.weekday()!=target_wd: d+=timedelta(days=1)
    return d

def test_monday_effect(): assert AnomalyDetector.check_monday_effect(find_weekday(0)) is not None
def test_not_monday(): assert AnomalyDetector.check_monday_effect(find_weekday(1)) is None
def test_turn_of_month_end(): assert AnomalyDetector.check_turn_of_month(make_dt(2026,5,30)) is not None
def test_turn_of_month_start(): assert AnomalyDetector.check_turn_of_month(make_dt(2026,5,2)) is not None
def test_turn_of_month_mid(): assert AnomalyDetector.check_turn_of_month(make_dt(2026,5,15)) is None
def test_end_of_quarter(): assert AnomalyDetector.check_end_of_quarter(make_dt(2026,3,30)) is not None
def test_not_quarter_end(): assert AnomalyDetector.check_end_of_quarter(make_dt(2026,5,15)) is None
def test_funding_extreme_pos():
    s=AnomalyDetector.check_funding_rate(0.005)
    assert s.direction=="bearish" and s.pts<0 and s.strength>0.8
def test_funding_extreme_neg():
    s=AnomalyDetector.check_funding_rate(-0.002)
    assert s.direction=="bullish" and s.pts>0
def test_funding_normal():
    s=AnomalyDetector.check_funding_rate(0.0005)
    assert s.direction=="neutral" and s.pts==0
def test_halving_phase_1(): assert AnomalyDetector.check_halving_cycle_phase(180).pts>=5
def test_halving_phase_2(): assert AnomalyDetector.check_halving_cycle_phase(500).pts>=10
def test_halving_phase_3(): assert AnomalyDetector.check_halving_cycle_phase(900).direction=="bearish"
def test_halving_phase_4(): assert AnomalyDetector.check_halving_cycle_phase(1200).pts<=-5
def test_gap_small(): assert AnomalyDetector.check_gap_fill_probability(0.005,True).strength==pytest.approx(0.75)
def test_gap_large_up(): assert AnomalyDetector.check_gap_fill_probability(0.04,True).direction=="bearish"
def test_gap_large_down(): assert AnomalyDetector.check_gap_fill_probability(0.04,False).direction=="bullish"
def test_anomaly_score_clamped():
    d=make_dt(2026,5,2)
    s=AnomalyDetector().get_anomaly_score(d,funding_rate=-0.003,days_since_halving=500)
    assert -15<=s<=15
def test_get_all_signals():
    assert isinstance(AnomalyDetector().get_all_signals(make_dt(2026,5,2),funding_rate=0.005),list)
