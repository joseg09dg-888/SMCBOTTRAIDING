# tests/agents/test_reasoning_prompt.py
import json
import pytest
from unittest.mock import MagicMock
from agents.analysis_agent import SMCAnalysisAgent


def _mock_claude(json_body: dict):
    mock_content = MagicMock()
    mock_content.text = json.dumps(json_body)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


GOOD_JSON = {
    "smart_money_action": "Accumulating long positions",
    "historical_support": "supports",
    "regime_fit": "favorable",
    "lesson_applied": None,
    "decision": "LONG",
    "confidence": 80,
    "justification": "Strong CHoCH with OB, trending regime",
}

CONTRADICTS_JSON = {
    "smart_money_action": "Distributing",
    "historical_support": "contradicts",
    "regime_fit": "unfavorable",
    "lesson_applied": None,
    "decision": "WAIT",
    "confidence": 35,
    "justification": "Historical losses dominate this setup",
}


class TestReasonWithContext:
    def test_reason_with_context_returns_dict(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude(GOOD_JSON)
        episodes = [{"symbol": "USDJPY", "direction": "BUY",
                     "setup_type": "CHoCH+OB", "result": "WIN",
                     "pnl": 50.0, "ts": "2026-05-25"}]
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH confirmed at 155.50",
            similar_episodes=episodes,
            regime="trending", base_score=65,
        )
        assert isinstance(result, dict)
        assert "confidence" in result

    def test_reason_boosts_score_on_support(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude(GOOD_JSON)
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] >= 65

    def test_reason_reduces_score_on_unfavorable_regime(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude(CONTRADICTS_JSON)
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="high_vol", base_score=70,
        )
        assert result["adjusted_score"] < 70

    def test_low_confidence_triggers_wait(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "confidence": 30, "historical_support": "neutral"
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["wait_override"] is True

    def test_api_failure_returns_fallback(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.side_effect = Exception("API timeout")
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] == 65
        assert result.get("fallback") is True

    def test_contradicts_3_losses_triggers_wait(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **CONTRADICTS_JSON, "historical_support": "contradicts"
        })
        losses = [{"result": "LOSS"} for _ in range(3)]
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=losses,
            regime="trending", base_score=65,
        )
        assert result["wait_override"] is True

    def test_high_confidence_support_adds_10_pts(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "confidence": 80, "historical_support": "supports"
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] == min(100, 65 + 10)

    def test_unfavorable_regime_subtracts_15(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "regime_fit": "unfavorable", "confidence": 70
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="high_vol", base_score=70,
        )
        assert result["adjusted_score"] == max(0, 70 - 15)

    def test_json_decode_error_returns_fallback(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "NOT VALID JSON {{{"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        agent.client.messages.create.return_value = mock_response
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] == 65
        assert result.get("fallback") is True

    def test_build_prompt_includes_episodes(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        episodes = [{"symbol": "USDJPY", "direction": "BUY",
                     "setup_type": "CHoCH+OB", "result": "WIN",
                     "pnl": 50.0, "ts": "2026-05-25"}]
        prompt = agent._build_reasoning_prompt(
            smc_summary="CHoCH at 155.50",
            similar_episodes=episodes,
            regime="trending",
        )
        assert "CHoCH+OB" in prompt
        assert "WIN" in prompt

    def test_build_prompt_includes_regime(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        prompt = agent._build_reasoning_prompt(
            smc_summary="Test", similar_episodes=[], regime="high_vol",
        )
        assert "high_vol" in prompt

    def test_adjusted_score_capped_at_100(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "confidence": 80, "historical_support": "supports"
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="trending", base_score=95,
        )
        assert result["adjusted_score"] <= 100

    def test_adjusted_score_floored_at_0(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "regime_fit": "unfavorable",
            "historical_support": "contradicts", "confidence": 80
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH", similar_episodes=[],
            regime="high_vol", base_score=10,
        )
        assert result["adjusted_score"] >= 0

    def test_neutral_historical_no_override(self):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "historical_support": "neutral", "confidence": 80
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[{"result": "WIN"}, {"result": "WIN"}],
            regime="trending", base_score=65,
        )
        assert result.get("wait_override") is not True
