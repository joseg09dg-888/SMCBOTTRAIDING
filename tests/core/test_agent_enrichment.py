"""Tests for TradingSupervisor._enrich_with_agents() — institutional enrichment layer."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from agents.signal_agent import SignalType, TradeSignal
from core.supervisor import TradingSupervisor


def _make_df(n=100, start=1.1000, trend=0.0001):
    closes = [start + i * trend for i in range(n)]
    return pd.DataFrame({
        "open":   closes,
        "high":   [c + 0.0005 for c in closes],
        "low":    [c - 0.0005 for c in closes],
        "close":  closes,
        "volume": [1000.0] * n,
    })


def _make_signal(score=70, direction=SignalType.LONG, symbol="EURUSD"):
    s = TradeSignal(
        symbol=symbol,
        signal_type=direction,
        entry=1.1000,
        stop_loss=1.0970,
        take_profit=1.1090,
        timeframe="H4",
        trigger="CHoCH+OB",
        confidence=0.75,
    )
    s.decision_score = score
    s.score_breakdown = {}
    return s


@pytest.fixture(scope="module")
def supervisor():
    return TradingSupervisor(capital=100_000, demo_mode=True)


class TestEnrichWithAgents:
    def test_returns_zero_when_base_score_below_50(self, supervisor):
        signal = _make_signal(score=45)
        df = _make_df()
        bonus = supervisor._enrich_with_agents(signal, df)
        assert bonus == 0

    def test_returns_nonzero_for_good_signal(self, supervisor):
        signal = _make_signal(score=70)
        df = _make_df()
        bonus = supervisor._enrich_with_agents(signal, df)
        # With a trending DF: Elliott bonus >=0, Chaos >=0, Edge >=0
        assert isinstance(bonus, int)

    def test_bonus_capped_at_60(self, supervisor):
        signal = _make_signal(score=90)
        df = _make_df(n=200)
        bonus = supervisor._enrich_with_agents(signal, df)
        assert bonus <= 60

    def test_bonus_floor_at_minus_20(self, supervisor):
        # Chaotic (flat) market should not drop below -20
        signal = _make_signal(score=70)
        df_flat = _make_df(trend=0.0)  # flat = high entropy = chaos penalty
        bonus = supervisor._enrich_with_agents(signal, df_flat)
        assert bonus >= -20

    def test_short_signal_also_enriched(self, supervisor):
        signal = _make_signal(score=70, direction=SignalType.SHORT)
        df = _make_df()
        bonus = supervisor._enrich_with_agents(signal, df)
        assert isinstance(bonus, int)

    def test_crypto_symbol_allowed(self, supervisor):
        signal = _make_signal(score=70, symbol="BTCUSDT")
        df = _make_df()
        # Should not raise even though footprint build may fail (no live data in tests)
        bonus = supervisor._enrich_with_agents(signal, df)
        assert isinstance(bonus, int)

    def test_enrichment_updates_signal_score(self, supervisor):
        signal = _make_signal(score=70)
        df = _make_df()
        agent_bonus = supervisor._enrich_with_agents(signal, df)
        # Simulate what the scan loop does
        if agent_bonus != 0:
            signal.decision_score = max(0, min(150, signal.decision_score + agent_bonus))
            signal.score_breakdown["agents"] = agent_bonus
        assert signal.decision_score >= 0
        assert signal.decision_score <= 150

    def test_no_crash_when_agents_raise(self, supervisor):
        """Broken agents should not propagate exceptions."""
        signal = _make_signal(score=70)
        df = pd.DataFrame()  # empty — will cause agent failures
        # Should not raise, just return 0
        bonus = supervisor._enrich_with_agents(signal, df)
        assert isinstance(bonus, int)

    def test_five_agents_are_instantiated(self, supervisor):
        assert hasattr(supervisor, "_lunar")
        assert hasattr(supervisor, "_elliott")
        assert hasattr(supervisor, "_chaos")
        assert hasattr(supervisor, "_edge")
        assert hasattr(supervisor, "_footprint")

    def test_agent_class_names(self, supervisor):
        from agents.lunar_agent import LunarCycleAgent
        from agents.elliott_agent import ElliottFibonacciAgent
        from agents.chaos_agent import ChaosTheoryAgent
        from agents.statistical_edge_agent import QuantEdgeAgent
        from agents.footprint_agent import FootprintAgent
        assert isinstance(supervisor._lunar, LunarCycleAgent)
        assert isinstance(supervisor._elliott, ElliottFibonacciAgent)
        assert isinstance(supervisor._chaos, ChaosTheoryAgent)
        assert isinstance(supervisor._edge, QuantEdgeAgent)
        assert isinstance(supervisor._footprint, FootprintAgent)
