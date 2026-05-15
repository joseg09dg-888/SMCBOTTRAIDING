"""
QuantEdgeAgent — integrates all 9 quant modules into a unified edge score.
Usage: from agents.statistical_edge_agent import QuantEdgeAgent
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np

@dataclass
class EdgeResult:
    edge_confirmed: bool
    expectancy: float
    kelly_fraction: float
    ruin_probability: float
    regime: str
    best_setup: str
    optimal_rr: float
    ensemble_probability: float
    stress_test_passed: bool
    factor_ir: float
    anomaly_score: int
    confidence: str
    recommended_risk: float
    edge_score: int
    # Breakdown
    monte_carlo_var: float = 0.0
    walk_forward_degradation: float = 1.0
    collective_intelligence_pts: int = 0
    order_flow_imbalance: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0

class QuantEdgeAgent:
    """
    PhD-level quantitative edge analysis. Integrates:
    SP2: QuantStats, SP3: RegimeDetector, SP4: FactorAnalyzer
    SP5: AnomalyDetector, SP6: MLEnsemble, SP7: BayesianOptimizer
    SP8: OrderFlowAnalyzer, SP9: StressTester, SP10: CollectiveIntelligence
    """

    def __init__(self, capital: float = 1000.0):
        self.capital = capital
        self._load_modules()

    def _load_modules(self):
        try:
            from agents.quant_stats import QuantStats
            self.stats = QuantStats
        except Exception:
            self.stats = None
        try:
            from agents.quant_regime import RegimeDetector
            self.regime_detector = RegimeDetector()
        except Exception:
            self.regime_detector = None
        try:
            from agents.quant_factors import FactorAnalyzer
            self.factor_analyzer = FactorAnalyzer()
        except Exception:
            self.factor_analyzer = None
        try:
            from agents.quant_anomalies import AnomalyDetector
            self.anomaly_detector = AnomalyDetector()
        except Exception:
            self.anomaly_detector = None
        try:
            from agents.quant_ensemble import MLEnsemble
            self.ensemble = MLEnsemble()
        except Exception:
            self.ensemble = None
        try:
            from agents.quant_stress import StressTester
            self.stress_tester = StressTester()
        except Exception:
            self.stress_tester = None
        try:
            from agents.quant_intel import CollectiveIntelligence
            self.collective = CollectiveIntelligence()
        except Exception:
            self.collective = None
        try:
            from agents.quant_flow import OrderFlowAnalyzer
            self.flow_analyzer = OrderFlowAnalyzer()
        except Exception:
            self.flow_analyzer = None

    def calculate_full_edge(
        self,
        symbol: str = "BTCUSDT",
        setup: str = "CHoCH+OB+FVG",
        prices: Optional[list] = None,
        trades: Optional[list] = None,
        bid_volumes: Optional[list] = None,
        ask_volumes: Optional[list] = None,
        funding_rate: float = 0.0,
        days_since_halving: int = -1,
        as_of: Optional[datetime] = None,
    ) -> EdgeResult:

        dt = as_of or datetime.now(timezone.utc)
        prices = prices or [100.0 + i * 0.5 for i in range(200)]
        trades = trades or []

        # SP2 — Core statistics
        expectancy = self.stats.calculate_expectancy(trades) if trades and self.stats else 0.0
        returns = [t["pnl"] / self.capital for t in trades] if trades else [0.001] * 100
        sharpe = self.stats.calculate_sharpe(returns) if self.stats else 0.0
        kelly = self.stats.calculate_kelly_fraction(0.6, 2.0, 1.0) if self.stats else 0.01
        mc = self.stats.run_monte_carlo(returns, n_sims=1000, seed=42) if self.stats else None
        ruin_prob = mc.ruin_probability if mc else 0.0
        mc_var = mc.var_95 if mc else 0.0
        wf = self.stats.run_walk_forward(returns) if self.stats else None
        wf_deg = wf.degradation_ratio if wf else 1.0
        mdd = self.stats.calculate_max_drawdown([1000*(1+r) for r in np.cumsum(returns)]) if self.stats else 0.0

        # SP3 — Regime
        regime_analysis = self.regime_detector.detect(prices) if self.regime_detector else None
        regime = regime_analysis.regime.value if regime_analysis else "ranging"
        optimal_rr = regime_analysis.recommended_rr if regime_analysis else 2.0

        # SP4 — Factors
        factor_ir = 0.0
        if self.factor_analyzer and len(prices) > 60:
            price_returns = list(np.diff(prices) / np.array(prices[:-1]))
            results = self.factor_analyzer.analyze_all_factors(prices, price_returns)
            if results:
                best = self.factor_analyzer.get_best_factor(results)
                factor_ir = best.ir if best else 0.0

        # SP5 — Anomalies
        anomaly_score = 0
        if self.anomaly_detector:
            anomaly_score = self.anomaly_detector.get_anomaly_score(
                dt, symbol, funding_rate, days_since_halving=days_since_halving)

        # SP6 — ML Ensemble
        ensemble_prob = 0.5
        if self.ensemble:
            pred = self.ensemble.predict_from_prices(prices)
            ensemble_prob = pred.probability

        # SP8 — Order Flow
        of_imb = 0.0
        of_pts = 0
        if self.flow_analyzer and bid_volumes and ask_volumes:
            sig = self.flow_analyzer.analyze(bid_volumes, ask_volumes)
            of_imb = sig.imbalance
            of_pts = sig.pts

        # SP9 — Stress test
        stress_passed = True
        if self.stress_tester:
            report = self.stress_tester.run_all_scenarios(self.capital)
            stress_passed = report.survival_rate >= 0.6

        # SP10 — Collective intelligence
        ci_pts = 0
        if self.collective:
            setup_tags = setup.split("+")
            ci_pts = self.collective.calculate_collective_score(symbol, setup_tags)

        # Composite edge score 0-100
        score = 50  # base

        # Regime adjustment
        regime_bonus = {"trending_up": 10, "trending_down": 8, "ranging": -5, "high_vol": -10}
        score += regime_bonus.get(regime, 0)

        # Ensemble probability
        if ensemble_prob >= 0.70: score += 15
        elif ensemble_prob >= 0.65: score += 10
        elif ensemble_prob >= 0.55: score += 5
        elif ensemble_prob < 0.45: score -= 10

        # Anomaly score (-15 to +15) → scale to ±10
        score += int(anomaly_score * 0.67)

        # Factor IR
        if factor_ir > 0.5: score += 8
        elif factor_ir > 0.3: score += 4

        # Collective intelligence
        score += ci_pts

        # Order flow
        score += of_pts // 2

        # Stress test
        if not stress_passed: score -= 15

        # Walk forward degradation
        if wf_deg < 0.3: score -= 10  # severe overfitting

        # Ruin probability
        if ruin_prob > 0.20: score -= 15

        score = int(np.clip(score, 0, 100))

        # Confidence
        if score >= 80: confidence = "VERY_HIGH"
        elif score >= 70: confidence = "HIGH"
        elif score >= 60: confidence = "MEDIUM"
        else: confidence = "LOW"

        edge_confirmed = score >= 65 and ensemble_prob >= 0.55 and stress_passed
        recommended_risk = kelly if kelly > 0 else 0.005

        return EdgeResult(
            edge_confirmed=edge_confirmed,
            expectancy=round(expectancy, 4),
            kelly_fraction=round(kelly, 4),
            ruin_probability=round(ruin_prob, 4),
            regime=regime,
            best_setup=setup,
            optimal_rr=round(optimal_rr, 2),
            ensemble_probability=round(ensemble_prob, 3),
            stress_test_passed=stress_passed,
            factor_ir=round(factor_ir, 3),
            anomaly_score=anomaly_score,
            confidence=confidence,
            recommended_risk=round(recommended_risk, 4),
            edge_score=score,
            monte_carlo_var=round(mc_var, 4),
            walk_forward_degradation=round(wf_deg, 3),
            collective_intelligence_pts=ci_pts,
            order_flow_imbalance=round(of_imb, 3),
            sharpe=round(sharpe, 3),
            max_drawdown=round(mdd, 4),
        )

    def get_decision_pts(self, edge_result: EdgeResult) -> int:
        """
        Convert edge_score to additional DecisionFilter points.
        New scale adds up to +50 pts to the existing 100pt system.
        """
        score = edge_result.edge_score
        if score >= 90: return 50
        elif score >= 80: return 40
        elif score >= 70: return 30
        elif score >= 60: return 20
        elif score >= 50: return 10
        return 0

    def format_telegram(self, edge_result: EdgeResult, symbol: str = "") -> str:
        e = edge_result
        status = "CONFIRMADO" if e.edge_confirmed else "NO CONFIRMADO"
        conf_emoji = {"VERY_HIGH":"💎","HIGH":"🔥","MEDIUM":"✅","LOW":"⚠️"}.get(e.confidence,"")
        return (
            f"{conf_emoji} *EDGE CUANTITATIVO — {symbol or 'MERCADO'}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Edge Score: {e.edge_score}/100 | {e.confidence}\n"
            f"{'✅' if e.edge_confirmed else '❌'} Edge: {status}\n"
            f"🎯 Expectancy: {e.expectancy:+.4f}\n"
            f"🎰 Kelly: {e.kelly_fraction*100:.1f}%\n"
            f"💀 Ruin Prob: {e.ruin_probability*100:.1f}%\n"
            f"🌊 Régimen: {e.regime}\n"
            f"🤖 ML Ensemble: {e.ensemble_probability*100:.1f}%\n"
            f"📈 Factor IR: {e.factor_ir:.3f}\n"
            f"⚡ Anomalías: {e.anomaly_score:+d} pts\n"
            f"🛡️ Stress Test: {'✅ OK' if e.stress_test_passed else '❌ FALLA'}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📐 RR Óptimo: 1:{e.optimal_rr}\n"
            f"💰 Riesgo rec: {e.recommended_risk*100:.2f}%\n"
            f"Setup: {e.best_setup}"
        )
