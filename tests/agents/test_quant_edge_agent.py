import pytest, numpy as np
from agents.quant_ensemble import MLEnsemble, FeatureExtractor, EnsemblePrediction
def prices(n=200,d=0.001,seed=42):
    rng=np.random.default_rng(seed); r=rng.normal(d,0.02,n)
    p=[100.0]
    for x in r: p.append(p[-1]*(1+x))
    return p
def dataset(n=200,seed=42):
    rng=np.random.default_rng(seed)
    X=[{"ret_1":rng.normal(0,0.02),"ret_5":rng.normal(0,0.03),"vol_20":rng.uniform(0.01,0.05),
        "rsi_14":rng.uniform(20,80),"above_ma20":float(rng.integers(0,2)),"momentum_score":rng.normal(0,0.5)} for _ in range(n)]
    y=[1 if rng.random()>0.4 else 0 for _ in range(n)]
    return X,y
def test_features_dict(): f=FeatureExtractor.from_prices(prices()); assert "ret_1" in f and "rsi_14" in f
def test_features_short(): f=FeatureExtractor.from_prices([100.0,101.0,102.0]); assert f["ret_1"]==0.0
def test_rsi_flat(): assert FeatureExtractor.calculate_rsi([100.0]*50)==pytest.approx(50.0)
def test_rsi_range(): r=FeatureExtractor.calculate_rsi(prices()); assert 0<=r<=100
def test_above_ma20():
    p=[100.0+i for i in range(60)]
    assert FeatureExtractor.from_prices(p)["above_ma20"]==1.0
def test_predict_no_fit():
    r=MLEnsemble().predict({"ret_1":0.01,"momentum_score":0.3,"above_ma20":1.0})
    assert isinstance(r,EnsemblePrediction) and 0<=r.probability<=1
def test_predict_threshold():
    e=MLEnsemble(threshold=0.65)
    r=e.predict({"ret_1":0.03,"momentum_score":2.0,"above_ma20":1.0,"above_ma50":1.0})
    assert r.should_trade==(r.probability>=0.65)
def test_confidence_valid():
    r=MLEnsemble().predict({"momentum_score":1.5,"above_ma20":1.0})
    assert r.confidence in ("LOW","MEDIUM","HIGH","VERY_HIGH")
def test_predict_from_prices(): assert isinstance(MLEnsemble().predict_from_prices(prices(d=0.002)),EnsemblePrediction)
def test_fit_predict():
    X,y=dataset()
    e=MLEnsemble(); e.fit(X,y)
    assert e._is_fitted
    r=e.predict(X[0])
    assert isinstance(r,EnsemblePrediction)
def test_model_votes_after_fit():
    X,y=dataset(); e=MLEnsemble(); e.fit(X,y)
    r=e.predict(X[0]); assert len(r.model_votes)>=1
def test_accuracies_after_fit():
    X,y=dataset(); e=MLEnsemble(); e.fit(X,y)
    accs=e.get_model_accuracies()
    if e._has_sklearn:
        for v in accs.values(): assert 0<=v<=1
def test_accuracies_not_fitted(): assert MLEnsemble().get_model_accuracies()=={}
def test_predict_empty(): r=MLEnsemble().predict({}); assert isinstance(r,EnsemblePrediction) and 0<=r.probability<=1
def test_deterministic():
    e=MLEnsemble(); f={"momentum_score":0.5,"above_ma20":1.0}
    assert e.predict(f).probability==pytest.approx(e.predict(f).probability)

# agents/quant_optimizer.py (BayesianOptimizer) ELIMINADO 2026-07-26 (panel
# SMC/quant): nunca se importaba desde statistical_edge_agent.py, solo
# aparecia en un comentario de docstring -- cero uso en produccion, mismo
# criterio que Elliott/Chaos. Ver agents/statistical_edge_agent.py.

from agents.quant_flow import OrderFlowAnalyzer, OrderFlowSignal
def test_imbalance_all_bids(): assert OrderFlowAnalyzer.calculate_imbalance([100,200,150],[])==pytest.approx(1.0)
def test_imbalance_all_asks(): assert OrderFlowAnalyzer.calculate_imbalance([],[100,200])==pytest.approx(-1.0)
def test_imbalance_balanced(): assert OrderFlowAnalyzer.calculate_imbalance([100,100],[100,100])==pytest.approx(0.0)
def test_imbalance_empty(): assert OrderFlowAnalyzer.calculate_imbalance([],[])==0.0
def test_vpin_informed(): assert OrderFlowAnalyzer.calculate_vpin([100.0]*50,[0.0]*50,bucket_size=10)==pytest.approx(1.0)
def test_vpin_balanced(): assert OrderFlowAnalyzer.calculate_vpin([100.0]*50,[100.0]*50,bucket_size=10)==pytest.approx(0.0)
def test_vpin_empty(): assert OrderFlowAnalyzer.calculate_vpin([],[])==0.0
def test_spread(): assert OrderFlowAnalyzer.calculate_spread_pct(100.0,100.1)==pytest.approx(0.001,abs=0.0001)
def test_spread_zero(): assert OrderFlowAnalyzer.calculate_spread_pct(0.0,100.0)==0.0
def test_strong_buy(): p,pts=OrderFlowAnalyzer.classify_pressure(0.6); assert p=="strong_buy" and pts==8
def test_strong_sell(): p,pts=OrderFlowAnalyzer.classify_pressure(-0.6); assert p=="strong_sell" and pts==-8
def test_neutral(): p,pts=OrderFlowAnalyzer.classify_pressure(0.0); assert p=="neutral" and pts==0
def test_analyze(): assert isinstance(OrderFlowAnalyzer().analyze([100,200],[50,30],best_bid=99.9,best_ask=100.1),OrderFlowSignal)
def test_toxic_flow():
    sig=OrderFlowAnalyzer().analyze([200]*20,[0]*20,buy_volumes=[200.0]*50,sell_volumes=[0.0]*50)
    assert sig.toxic_flow is True
def test_impact_small(): assert OrderFlowAnalyzer.estimate_market_impact(1000,1_000_000,0.02)<0.01
def test_impact_zero(): assert OrderFlowAnalyzer.estimate_market_impact(1000,0,0.02)==0.0
def test_iceberg_high(): assert OrderFlowAnalyzer.detect_iceberg_probability(10,1000)>0.9

from agents.quant_stress import StressTester, StressResult, StressReport, StressScenario
def test_scenarios_count(): assert len(StressTester.SCENARIOS)>=8
def test_run_scenario(): assert isinstance(StressTester().run_scenario(StressTester.SCENARIOS[0],1000.0),StressResult)
def test_loss_positive(): r=StressTester().run_scenario(StressTester.SCENARIOS[0],1000.0); assert r.portfolio_loss_usd>=0
def test_circuit_breaker():
    s=StressScenario("t","2026-01-01",-30.0,1,90,"all",5.0)
    assert StressTester().run_scenario(s,1000.0,open_positions=[{"size":500}]).circuit_breaker_triggered
def test_survived_small():
    s=StressScenario("t","2026-01-01",-2.0,1,5,"equity",1.5)
    assert StressTester().run_scenario(s,1000.0).survived
def test_run_all(): r=StressTester().run_all_scenarios(10000.0); assert r.scenarios_run==len(StressTester.SCENARIOS)
def test_survival_range(): r=StressTester().run_all_scenarios(10000.0); assert 0<=r.survival_rate<=1
def test_recommendations(): assert len(StressTester().run_all_scenarios(1000.0).recommendations)>0
def test_filter_asset(): r=StressTester().run_all_scenarios(10000.0,filter_asset_class="crypto"); assert r.scenarios_run<len(StressTester.SCENARIOS)
def test_find_scenario(): assert StressTester().get_scenario_by_name("luna") is not None
def test_not_found(): assert StressTester().get_scenario_by_name("doesnotexist") is None
def test_max_loss(): assert StressTester.estimate_max_loss_pct(-20.0,1.0,0.1)==pytest.approx(0.02)
def test_max_loss_clamped(): assert StressTester.estimate_max_loss_pct(-200.0,5.0,1.0)<=1.0

# agents/quant_intel.py (CollectiveIntelligence) ELIMINADO 2026-07-26 (panel
# SMC/quant): calculate_collective_score() nunca se llamaba desde ningun
# lado (ci_pts ya estaba hardcodeado a 0), y get_consensus_bias()/
# get_insider_activity() derivaban de hash(symbol)%100 -- una constante por
# simbolo sin relacion con datos reales. Mismo criterio que Elliott/Chaos.
# Ver agents/statistical_edge_agent.py.

from agents.statistical_edge_agent import QuantEdgeAgent, EdgeResult
def test_edge_returns_result(): assert isinstance(QuantEdgeAgent().calculate_full_edge("BTCUSDT"),EdgeResult)
def test_edge_score_range(): r=QuantEdgeAgent().calculate_full_edge(); assert 0<=r.edge_score<=100
def test_edge_confirmed_bool(): assert isinstance(QuantEdgeAgent().calculate_full_edge().edge_confirmed,bool)
def test_edge_ensemble_range(): r=QuantEdgeAgent().calculate_full_edge(); assert 0<=r.ensemble_probability<=1
def test_edge_with_trades():
    trades=[{"pnl":20}]*6+[{"pnl":-10}]*4
    r=QuantEdgeAgent(1000).calculate_full_edge(trades=trades)
    assert r.expectancy>0
def test_edge_stress_test_can_fail_with_real_exposure():
    """BUG-STRESS-ALWAYS-OK (2026-07-26): calculate_full_edge() used to never
    pass open_positions to run_all_scenarios(), and without it quant_stress.py
    cancels out `equity` in its loss_pct formula -- stress_test_passed was
    True on every single call, forever, regardless of real exposure. With
    real open_positions notional passed through, a tiny account heavily
    exposed relative to its capital must be able to fail the stress test."""
    tiny_capital = 100.0
    huge_exposure = [{"size": 100_000.0}]  # far larger than capital
    r = QuantEdgeAgent(tiny_capital).calculate_full_edge(open_positions=huge_exposure)
    assert r.stress_test_passed is False


def test_edge_score_not_inflated_by_synthetic_price_fallback():
    """BUG-SYNTHETIC-PRICE-FALSE-SIGNAL (2026-07-26): when prices=None (real
    data outage), calculate_full_edge() used to fall back to a perfectly
    linear price ramp -- regime detection saw that as TRENDING_UP with high
    confidence and the ML ensemble heuristic saw pure positive momentum, both
    awarding their full score bonus exactly when the score should be neutral.
    The same ramp passed explicitly as real data should score higher than
    the no-data case, proving the fallback's bonus is now gated off."""
    agent = QuantEdgeAgent()
    r_no_data = agent.calculate_full_edge(prices=None)
    ramp = [100.0 + i * 0.5 for i in range(200)]
    r_explicit = agent.calculate_full_edge(prices=ramp)
    assert r_explicit.edge_score > r_no_data.edge_score


def test_decision_pts_high():
    from agents.statistical_edge_agent import EdgeResult
    e=EdgeResult(True,0.5,0.1,0.01,"trending_up","OB",2.5,0.75,True,0.6,5,"VERY_HIGH",0.005,90)
    assert QuantEdgeAgent().get_decision_pts(e)==50
def test_decision_pts_low():
    from agents.statistical_edge_agent import EdgeResult
    e=EdgeResult(False,0.0,0.0,0.5,"ranging","",1.5,0.4,False,0.0,0,"LOW",0.005,30)
    assert QuantEdgeAgent().get_decision_pts(e)==0
def test_format_telegram_has_score():
    r=QuantEdgeAgent().calculate_full_edge()
    msg=QuantEdgeAgent().format_telegram(r,"BTCUSDT")
    assert "BTCUSDT" in msg and "Score" in msg
