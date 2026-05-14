"""Tests for LunarCycleAgent — no real ephem calls needed."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from agents.lunar_agent import LunarCycleAgent, LunarSignal


@pytest.fixture
def agent():
    return LunarCycleAgent()


# ── Phase detection ────────────────────────────────────────────────────────────

def test_get_current_phase_returns_signal(agent):
    s = agent.get_current_phase()
    assert isinstance(s, LunarSignal)


def test_phase_pct_between_0_and_1(agent):
    s = agent.get_current_phase()
    assert 0.0 <= s.phase_pct <= 1.0


def test_phase_name_is_valid(agent):
    s = agent.get_current_phase()
    assert s.phase_name in ("nueva", "creciente", "llena", "menguante")


def test_new_moon_is_bearish(agent):
    # phase_pct=0.05 → new moon
    with patch.object(agent, "_phase_pct", return_value=0.05):
        s = agent.get_current_phase()
    assert s.bias == "bearish"
    assert s.phase_name == "nueva"


def test_full_moon_is_bullish(agent):
    with patch.object(agent, "_phase_pct", return_value=0.50):
        s = agent.get_current_phase()
    assert s.bias == "bullish"
    assert s.phase_name == "llena"


def test_waxing_crescent_bullish(agent):
    with patch.object(agent, "_phase_pct", return_value=0.25):
        s = agent.get_current_phase()
    assert s.bias == "bullish"
    assert s.phase_name == "creciente"


def test_waning_is_bearish(agent):
    with patch.object(agent, "_phase_pct", return_value=0.75):
        s = agent.get_current_phase()
    assert s.bias == "bearish"
    assert s.phase_name == "menguante"


# ── Score adjustment ───────────────────────────────────────────────────────────

def test_score_adjustment_matching_bias_gives_5pts(agent):
    with patch.object(agent, "_phase_pct", return_value=0.50):  # full moon = bullish
        pts = agent.score_adjustment("bullish")
    assert pts == 5


def test_score_adjustment_mismatched_gives_0pts(agent):
    with patch.object(agent, "_phase_pct", return_value=0.50):  # full moon = bullish
        pts = agent.score_adjustment("bearish")
    assert pts == 0


def test_score_adjustment_neutral_gives_0(agent):
    pts = agent.score_adjustment("neutral")
    assert pts == 0


# ── Eclipse detection ──────────────────────────────────────────────────────────

def test_eclipse_returns_bool(agent):
    result = agent._check_eclipse()
    assert isinstance(result, bool)


def test_no_ephem_eclipse_returns_false(agent):
    with patch.dict("sys.modules", {"ephem": None}):
        result = agent._check_eclipse()
    assert result is False


# ── Telegram format ────────────────────────────────────────────────────────────

def test_format_telegram_contains_phase_name(agent):
    text = agent.format_telegram()
    assert any(name in text.lower() for name in ("nueva", "creciente", "llena", "menguante"))


def test_format_telegram_has_bonus_info(agent):
    text = agent.format_telegram()
    assert "pts" in text.lower() or "bonus" in text.lower()
