# tests/agents/test_screen_vision_agent.py

import pytest
import json
import base64
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
from agents.screen_vision_agent import (
    ScreenVisionAgent, ScreenCapture, VisionAnalysis,
    VisionAlert, MirrorSession
)

# ── ScreenCapture ──────────────────────────────────────────────────────────

def test_screen_capture_to_base64():
    cap = ScreenCapture(
        image_bytes=b"PNG_DATA", timestamp=datetime.now(timezone.utc),
        window_title="test", width=100, height=100
    )
    b64 = cap.to_base64()
    assert base64.b64decode(b64) == b"PNG_DATA"

def test_create_mock_capture_returns_screencapture():
    agent = ScreenVisionAgent()
    cap = agent.create_mock_capture()
    assert isinstance(cap, ScreenCapture)
    assert cap.image_bytes is not None
    assert len(cap.image_bytes) > 0
    assert cap.window_title == "mock"

def test_create_mock_capture_custom_size():
    agent = ScreenVisionAgent()
    cap = agent.create_mock_capture(width=200, height=150)
    assert cap.width == 200
    assert cap.height == 150

# ── capture_full_screen ────────────────────────────────────────────────────

def test_capture_full_screen_without_mss_returns_none_or_capture():
    agent = ScreenVisionAgent()
    result = agent.capture_full_screen()
    # Si mss no disponible → None; si disponible → ScreenCapture
    assert result is None or isinstance(result, ScreenCapture)

def test_capture_window_never_raises():
    agent = ScreenVisionAgent()
    result = agent.capture_window("NonExistentWindowTitle12345")
    # Puede retornar None o ScreenCapture, nunca lanza
    assert result is None or isinstance(result, ScreenCapture)

# ── parse_vision_response ──────────────────────────────────────────────────

def test_parse_vision_response_valid_json():
    agent = ScreenVisionAgent()
    raw = '{"symbol": "BTCUSDT", "price": 67450.0, "has_valid_setup": true}'
    result = agent.parse_vision_response(raw)
    assert result["symbol"] == "BTCUSDT"
    assert result["price"] == 67450.0

def test_parse_vision_response_json_embedded_in_text():
    agent = ScreenVisionAgent()
    raw = 'Aqui esta el analisis:\n{"symbol": "ETHUSDT", "price": 3200.0}\nEso es todo.'
    result = agent.parse_vision_response(raw)
    assert result["symbol"] == "ETHUSDT"

def test_parse_vision_response_invalid_returns_empty():
    agent = ScreenVisionAgent()
    result = agent.parse_vision_response("No hay JSON aqui")
    assert result == {}

def test_parse_vision_response_empty_string():
    agent = ScreenVisionAgent()
    result = agent.parse_vision_response("")
    assert result == {}

# ── build_vision_analysis ──────────────────────────────────────────────────

def test_build_vision_analysis_full():
    agent = ScreenVisionAgent()
    parsed = {
        "symbol": "BTCUSDT", "timeframe": "1H", "price": 67450.0,
        "market_structure": "bullish", "order_blocks": ["OB en 67000"],
        "fvgs": ["FVG 67100-67200"], "has_valid_setup": True,
        "entry": 67100.0, "stop_loss": 66800.0, "take_profit": 67800.0,
        "visual_score": 82
    }
    va = agent.build_vision_analysis(parsed, "raw text")
    assert va.symbol == "BTCUSDT"
    assert va.visual_score == 82
    assert va.has_valid_setup is True
    assert va.error is None

def test_build_vision_analysis_empty_dict_uses_defaults():
    agent = ScreenVisionAgent()
    va = agent.build_vision_analysis({}, "raw text")
    assert va.symbol == "" or va.symbol == "UNKNOWN"
    assert va.price == 0.0
    assert va.has_valid_setup is False
    assert va.visual_score == 0

def test_build_vision_analysis_partial_dict():
    agent = ScreenVisionAgent()
    parsed = {"symbol": "SOLUSDT"}
    va = agent.build_vision_analysis(parsed, "")
    assert va.symbol == "SOLUSDT"
    assert va.stop_loss == 0.0

# ── analyze_capture ────────────────────────────────────────────────────────

def test_analyze_capture_no_api_key_returns_error():
    agent = ScreenVisionAgent(api_key="")
    cap = agent.create_mock_capture()
    result = agent.analyze_capture(cap)
    assert isinstance(result, VisionAnalysis)
    assert result.error is not None

def test_analyze_capture_with_mock_api(mocker):
    mock_response = json.dumps({
        "symbol": "BTCUSDT", "timeframe": "1H", "price": 67450.0,
        "market_structure": "bullish", "order_blocks": [],
        "fvgs": [], "has_valid_setup": True,
        "entry": 67100.0, "stop_loss": 66800.0, "take_profit": 67800.0,
        "visual_score": 75
    })
    agent = ScreenVisionAgent(api_key="fake_key")
    mocker.patch.object(agent, '_call_vision_api', return_value=mock_response)
    cap = agent.create_mock_capture()
    result = agent.analyze_capture(cap)
    assert result.symbol == "BTCUSDT"
    assert result.visual_score == 75
    assert result.error is None

def test_analyze_capture_invalid_json_response(mocker):
    agent = ScreenVisionAgent(api_key="fake_key")
    mocker.patch.object(agent, '_call_vision_api', return_value="No es JSON")
    cap = agent.create_mock_capture()
    result = agent.analyze_capture(cap)
    assert result.error is not None

# ── Mirror mode ────────────────────────────────────────────────────────────

def test_start_mirror_mode():
    agent = ScreenVisionAgent()
    session = agent.start_mirror_mode()
    assert isinstance(session, MirrorSession)
    assert agent._mirror_active is True
    assert session.is_active is True

def test_stop_mirror_mode():
    agent = ScreenVisionAgent()
    agent.start_mirror_mode()
    session = agent.stop_mirror_mode()
    assert agent._mirror_active is False
    assert session.is_active is False

def test_record_mirror_action():
    agent = ScreenVisionAgent()
    agent.start_mirror_mode()
    agent.record_mirror_action("click en BUY")
    agent.record_mirror_action("setear SL")
    assert agent._mirror_session.actions_recorded == 2

def test_stop_mirror_without_start():
    agent = ScreenVisionAgent()
    result = agent.stop_mirror_mode()
    assert result is None

# ── Toggle y estado ────────────────────────────────────────────────────────

def test_toggle_enables():
    agent = ScreenVisionAgent(enabled=False)
    result = agent.toggle()
    assert result is True
    assert agent._enabled is True

def test_toggle_disables():
    agent = ScreenVisionAgent(enabled=True)
    result = agent.toggle()
    assert result is False

def test_get_status_message():
    agent = ScreenVisionAgent(enabled=True)
    msg = agent.get_status_message()
    assert isinstance(msg, str)
    assert len(msg) > 0

def test_get_last_analysis_empty():
    agent = ScreenVisionAgent()
    assert agent.get_last_analysis() is None

def test_get_analysis_history_empty():
    agent = ScreenVisionAgent()
    assert agent.get_analysis_history() == []

def test_get_analysis_history_with_data():
    agent = ScreenVisionAgent()
    va = VisionAnalysis(
        symbol="BTC", timeframe="1H", price=67000.0,
        market_structure="bullish", order_blocks=[], fvgs=[],
        has_valid_setup=False, entry=0.0, stop_loss=0.0, take_profit=0.0,
        visual_score=50, raw_response=""
    )
    agent._analysis_history.append(va)
    history = agent.get_analysis_history(n=5)
    assert len(history) == 1
    assert history[0].symbol == "BTC"

# ── build_alert_message ────────────────────────────────────────────────────

def test_build_alert_message_contains_symbol():
    agent = ScreenVisionAgent()
    va = VisionAnalysis(
        symbol="ETHUSDT", timeframe="4H", price=3200.0,
        market_structure="bullish", order_blocks=[], fvgs=[],
        has_valid_setup=True, entry=3200.0, stop_loss=3100.0, take_profit=3400.0,
        visual_score=78, raw_response=""
    )
    msg = agent.build_alert_message(va, "Binance")
    assert "ETHUSDT" in msg

def test_build_alert_message_contains_score():
    agent = ScreenVisionAgent()
    va = VisionAnalysis(
        symbol="BTCUSDT", timeframe="1H", price=67000.0,
        market_structure="bearish", order_blocks=[], fvgs=[],
        has_valid_setup=False, entry=0.0, stop_loss=0.0, take_profit=0.0,
        visual_score=45, raw_response=""
    )
    msg = agent.build_alert_message(va, "full")
    assert "45" in msg

# ── Async tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_alert_with_telegram(mocker):
    mock_telegram = MagicMock()
    mock_telegram.send_glint_alert = AsyncMock()
    agent = ScreenVisionAgent(telegram_bot=mock_telegram)
    va = VisionAnalysis(
        symbol="BTCUSDT", timeframe="1H", price=67000.0,
        market_structure="bullish", order_blocks=[], fvgs=[],
        has_valid_setup=True, entry=67000.0, stop_loss=66000.0, take_profit=68000.0,
        visual_score=80, raw_response=""
    )
    await agent.send_alert(va, "Binance")
    mock_telegram.send_glint_alert.assert_called_once()

@pytest.mark.asyncio
async def test_run_vision_loop_stops_when_not_running(mocker):
    agent = ScreenVisionAgent(enabled=True)
    agent._is_running = False  # ya parado antes de empezar
    mocker.patch.object(agent, 'capture_full_screen', return_value=None)
    # Debe terminar sin colgar
    await agent.run_vision_loop()

@pytest.mark.asyncio
async def test_run_vision_loop_one_cycle(mocker):
    agent = ScreenVisionAgent(enabled=True, capture_interval=0)
    call_count = 0

    async def fake_sleep(n):
        nonlocal call_count
        call_count += 1
        agent._is_running = False  # para el loop despues de 1 ciclo

    mocker.patch('asyncio.sleep', fake_sleep)
    mocker.patch.object(agent, 'capture_full_screen', return_value=agent.create_mock_capture())
    mock_va = VisionAnalysis(
        symbol="BTC", timeframe="", price=0.0, market_structure="unknown",
        order_blocks=[], fvgs=[], has_valid_setup=False,
        entry=0.0, stop_loss=0.0, take_profit=0.0, visual_score=0, raw_response=""
    )
    mocker.patch.object(agent, 'analyze_capture', return_value=mock_va)
    agent._is_running = True
    await agent.run_vision_loop()
    assert call_count >= 1
