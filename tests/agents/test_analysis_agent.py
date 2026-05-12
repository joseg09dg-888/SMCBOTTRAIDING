import pytest
import pandas as pd
from agents.analysis_agent import SMCAnalysisAgent, SMCAnalysis


@pytest.fixture
def sample_ohlcv():
    data = {
        "open":   [100, 102, 104, 101, 110, 108],
        "high":   [103, 106, 107, 102, 115, 112],
        "low":    [99,  101, 102, 98,  108, 106],
        "close":  [102, 105, 103, 110, 112, 109],
        "volume": [1000, 1500, 800, 3000, 2000, 900],
    }
    return pd.DataFrame(data)


def test_smc_analysis_has_required_fields(sample_ohlcv):
    agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
    analysis = agent._run_technical_analysis(sample_ohlcv, "BTCUSDT", "1h")
    assert isinstance(analysis, SMCAnalysis)
    assert hasattr(analysis, "structure")
    assert hasattr(analysis, "order_blocks")
    assert hasattr(analysis, "fvgs")
    assert hasattr(analysis, "volume_profile")
    assert hasattr(analysis, "bias")
    assert analysis.bias in ("bullish", "bearish", "neutral")


def test_checklist_format(sample_ohlcv):
    agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
    analysis = agent._run_technical_analysis(sample_ohlcv, "BTCUSDT", "1h")
    checklist = agent._build_checklist(analysis)
    assert isinstance(checklist, str)
    assert "✅" in checklist or "❌" in checklist
    assert "Estructura" in checklist


def test_poi_zones_populated(sample_ohlcv):
    agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
    analysis = agent._run_technical_analysis(sample_ohlcv, "BTCUSDT", "1h")
    assert isinstance(analysis.poi_zones, list)
