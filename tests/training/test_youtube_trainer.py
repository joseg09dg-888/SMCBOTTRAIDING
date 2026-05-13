import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from training.youtube_trainer import (
    YouTubeTrainer, ExtractedStrategy, StrategyStyle, StrategySelector
)

# 6 test cases:

def test_strategy_has_required_fields():
    strat = ExtractedStrategy(
        title="ICT Breaker Block",
        entry_rule="Enter at OB after CHoCH on 15m",
        exit_rule="TP at 1:3 RR, SL below OB",
        market_condition="trending",
        style=StrategyStyle.SMC,
        channel="InnerCircleTrader",
        confidence=0.85,
    )
    assert strat.entry_rule != ""
    assert strat.style == StrategyStyle.SMC
    assert 0 <= strat.confidence <= 1.0

def test_selector_chooses_smc_for_trend():
    selector = StrategySelector()
    strategies = [
        ExtractedStrategy("S1", "OB entry", "RR 1:3", "trending", StrategyStyle.SMC, "ICT", 0.9),
        ExtractedStrategy("S2", "Wyckoff spring", "re-accumulate", "ranging", StrategyStyle.WYCKOFF, "WyckoffAnalytics", 0.8),
        ExtractedStrategy("S3", "Breakout", "momentum", "breakout", StrategyStyle.PRICE_ACTION, "Rayner", 0.75),
    ]
    result = selector.select(strategies, market_condition="trending")
    assert result.style == StrategyStyle.SMC

def test_selector_chooses_wyckoff_for_range():
    selector = StrategySelector()
    strategies = [
        ExtractedStrategy("S1", "OB entry", "RR 1:3", "trending", StrategyStyle.SMC, "ICT", 0.9),
        ExtractedStrategy("S2", "Wyckoff spring", "re-accumulate", "ranging", StrategyStyle.WYCKOFF, "WyckoffAnalytics", 0.8),
    ]
    result = selector.select(strategies, market_condition="ranging")
    assert result.style == StrategyStyle.WYCKOFF

def test_selector_chooses_pa_for_breakout():
    selector = StrategySelector()
    strategies = [
        ExtractedStrategy("S1", "OB entry", "RR 1:3", "trending", StrategyStyle.SMC, "ICT", 0.9),
        ExtractedStrategy("S3", "Breakout", "momentum", "breakout", StrategyStyle.PRICE_ACTION, "Rayner", 0.85),
    ]
    result = selector.select(strategies, market_condition="breakout")
    assert result.style == StrategyStyle.PRICE_ACTION

def test_extract_strategy_from_text():
    trainer = YouTubeTrainer.__new__(YouTubeTrainer)
    trainer.strategies = []
    # Mock the Claude response
    mock_response = '{"title":"ICT OB","entry_rule":"Enter at OB","exit_rule":"TP 1:3","market_condition":"trending","style":"smc","confidence":0.85}'
    strategy = trainer._parse_strategy_json(mock_response, channel="ICT", video_id="abc123")
    assert strategy is not None
    assert strategy.entry_rule == "Enter at OB"
    assert strategy.style == StrategyStyle.SMC

def test_parse_invalid_json_returns_none():
    trainer = YouTubeTrainer.__new__(YouTubeTrainer)
    result = trainer._parse_strategy_json("not valid json", channel="X", video_id="y")
    assert result is None

def test_trainer_has_known_channels():
    trainer = YouTubeTrainer.__new__(YouTubeTrainer)
    assert len(trainer.CHANNELS) >= 5
    channel_ids = list(trainer.CHANNELS.keys())
    assert any("ict" in c.lower() or "inner" in c.lower() or "smc" in c.lower()
               for c in channel_ids)
