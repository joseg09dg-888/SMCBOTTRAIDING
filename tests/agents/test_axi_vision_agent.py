"""Tests for AxiVisionAgent — Claude Vision-based MT5 screen analysis."""
from __future__ import annotations

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from agents.axi_vision_agent import AxiVisionAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_b64_jpeg() -> str:
    """Return a tiny valid JPEG encoded as base64."""
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _mock_claude_response(text: str) -> MagicMock:
    msg = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    msg.content = [content_block]
    return msg


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestAxiVisionAgentInit:
    def test_instantiates_without_error(self):
        with patch("agents.axi_vision_agent.anthropic.Anthropic"):
            agent = AxiVisionAgent()
            assert agent is not None

    def test_has_client_attribute(self):
        with patch("agents.axi_vision_agent.anthropic.Anthropic") as mock_cls:
            agent = AxiVisionAgent()
            assert agent.client is mock_cls.return_value


# ---------------------------------------------------------------------------
# capture_mt5_screen
# ---------------------------------------------------------------------------

class TestCaptureScreen:
    def test_returns_base64_string(self):
        agent = AxiVisionAgent.__new__(AxiVisionAgent)
        agent.client = MagicMock()

        with patch("agents.axi_vision_agent.mss.mss") as mock_mss:
            # Build a fake screenshot
            fake_img = Image.new("RGB", (10, 10), color=(0, 0, 0))
            fake_buf = BytesIO()
            fake_img.save(fake_buf, format="JPEG")
            raw_bytes = fake_buf.getvalue()

            mock_sct = MagicMock()
            mock_sct.__enter__ = lambda s: s
            mock_sct.__exit__ = MagicMock(return_value=False)
            mock_sct.monitors = [None, {"left": 0, "top": 0, "width": 10, "height": 10}]

            grabbed = MagicMock()
            grabbed.size = (10, 10)
            grabbed.rgb = fake_img.tobytes()
            mock_sct.grab.return_value = grabbed
            mock_mss.return_value = mock_sct

            result = agent.capture_mt5_screen()

        assert isinstance(result, str)
        # Verify it is valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_uses_primary_monitor(self):
        agent = AxiVisionAgent.__new__(AxiVisionAgent)
        agent.client = MagicMock()

        with patch("agents.axi_vision_agent.mss.mss") as mock_mss:
            fake_img = Image.new("RGB", (4, 4))
            fake_buf = BytesIO()
            fake_img.save(fake_buf, format="JPEG")

            mock_sct = MagicMock()
            mock_sct.__enter__ = lambda s: s
            mock_sct.__exit__ = MagicMock(return_value=False)
            mock_sct.monitors = [None]  # index 0 = all monitors combined

            grabbed = MagicMock()
            grabbed.size = (4, 4)
            grabbed.rgb = fake_img.tobytes()
            mock_sct.grab.return_value = grabbed
            mock_mss.return_value = mock_sct

            agent.capture_mt5_screen()

        mock_sct.grab.assert_called_once()


# ---------------------------------------------------------------------------
# analyze_mt5_screen
# ---------------------------------------------------------------------------

class TestAnalyzeMt5Screen:
    def _make_agent(self) -> AxiVisionAgent:
        agent = AxiVisionAgent.__new__(AxiVisionAgent)
        agent.client = MagicMock()
        return agent

    def test_returns_dict(self):
        agent = self._make_agent()
        payload = json.dumps({
            "balance": 99000,
            "patrimonio": 99200,
            "posiciones": [],
            "alertas": [],
            "setup_visible": False,
            "accion_recomendada": "esperar",
        })
        agent.client.messages.create.return_value = _mock_claude_response(payload)

        with patch.object(agent, "capture_mt5_screen", return_value=_make_b64_jpeg()):
            result = agent.analyze_mt5_screen()

        assert isinstance(result, dict)

    def test_parses_balance(self):
        agent = self._make_agent()
        payload = json.dumps({
            "balance": 98500.0,
            "patrimonio": 98700.0,
            "posiciones": [],
            "alertas": [],
            "setup_visible": False,
            "accion_recomendada": "",
        })
        agent.client.messages.create.return_value = _mock_claude_response(payload)

        with patch.object(agent, "capture_mt5_screen", return_value=_make_b64_jpeg()):
            result = agent.analyze_mt5_screen()

        assert result["balance"] == 98500.0

    def test_falls_back_to_raw_on_invalid_json(self):
        agent = self._make_agent()
        agent.client.messages.create.return_value = _mock_claude_response("not json {{")

        with patch.object(agent, "capture_mt5_screen", return_value=_make_b64_jpeg()):
            result = agent.analyze_mt5_screen()

        assert "raw" in result

    def test_calls_claude_with_image(self):
        agent = self._make_agent()
        agent.client.messages.create.return_value = _mock_claude_response(
            json.dumps({"balance": 0, "patrimonio": 0, "posiciones": [],
                        "alertas": [], "setup_visible": False, "accion_recomendada": ""})
        )

        b64 = _make_b64_jpeg()
        with patch.object(agent, "capture_mt5_screen", return_value=b64):
            agent.analyze_mt5_screen()

        call_kwargs = agent.client.messages.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        content = messages[0]["content"]
        types = [c["type"] for c in content]
        assert "image" in types

    def test_uses_correct_model(self):
        agent = self._make_agent()
        agent.client.messages.create.return_value = _mock_claude_response("{}")

        with patch.object(agent, "capture_mt5_screen", return_value=_make_b64_jpeg()):
            try:
                agent.analyze_mt5_screen()
            except Exception:
                pass

        call_kwargs = agent.client.messages.create.call_args
        model = call_kwargs[1].get("model") or call_kwargs[0][0]
        assert "claude" in model.lower()


# ---------------------------------------------------------------------------
# should_close_position
# ---------------------------------------------------------------------------

class TestShouldClosePosition:
    def setup_method(self):
        self.agent = AxiVisionAgent.__new__(AxiVisionAgent)
        self.agent.client = MagicMock()

    def test_close_on_large_loss(self):
        assert self.agent.should_close_position("EURUSD", -600) is True

    def test_no_close_on_small_loss(self):
        assert self.agent.should_close_position("EURUSD", -100) is False

    def test_no_close_on_profit(self):
        assert self.agent.should_close_position("US30", 200) is False

    def test_threshold_boundary(self):
        # exactly at threshold — no close
        assert self.agent.should_close_position("XAUUSD", -500) is False
        # one below — close
        assert self.agent.should_close_position("XAUUSD", -501) is True


# ---------------------------------------------------------------------------
# monitor_and_protect
# ---------------------------------------------------------------------------

class TestMonitorAndProtect:
    def _make_agent(self) -> AxiVisionAgent:
        agent = AxiVisionAgent.__new__(AxiVisionAgent)
        agent.client = MagicMock()
        return agent

    def test_returns_dict_with_keys(self):
        agent = self._make_agent()
        analysis = {
            "balance": 99000,
            "patrimonio": 99200,
            "posiciones": [],
            "alertas": [],
            "setup_visible": False,
            "accion_recomendada": "",
        }
        with patch.object(agent, "analyze_mt5_screen", return_value=analysis):
            result = agent.monitor_and_protect()

        assert "analysis" in result
        assert "alerts" in result
        assert "balance" in result

    def test_alert_on_losing_position(self):
        agent = self._make_agent()
        analysis = {
            "balance": 99000,
            "posiciones": [{"symbol": "EURUSD", "pnl": -150}],
        }
        with patch.object(agent, "analyze_mt5_screen", return_value=analysis):
            result = agent.monitor_and_protect()

        assert any("EURUSD" in a for a in result["alerts"])

    def test_critical_alert_on_large_loss(self):
        agent = self._make_agent()
        analysis = {
            "balance": 99000,
            "posiciones": [{"symbol": "US30", "pnl": -600}],
        }
        with patch.object(agent, "analyze_mt5_screen", return_value=analysis):
            result = agent.monitor_and_protect()

        assert any("CERRAR" in a or "US30" in a for a in result["alerts"])

    def test_no_alerts_when_profitable(self):
        agent = self._make_agent()
        analysis = {
            "balance": 100000,
            "posiciones": [{"symbol": "XAUUSD", "pnl": 250}],
        }
        with patch.object(agent, "analyze_mt5_screen", return_value=analysis):
            result = agent.monitor_and_protect()

        assert result["alerts"] == []

    def test_returns_balance_from_analysis(self):
        agent = self._make_agent()
        analysis = {"balance": 98765, "posiciones": []}
        with patch.object(agent, "analyze_mt5_screen", return_value=analysis):
            result = agent.monitor_and_protect()

        assert result["balance"] == 98765
