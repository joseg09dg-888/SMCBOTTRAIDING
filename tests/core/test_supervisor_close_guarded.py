import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_supervisor():
    from core.supervisor import TradingSupervisor
    sup = TradingSupervisor.__new__(TradingSupervisor)
    sup.mt5 = MagicMock()
    sup.telegram = MagicMock()
    sup.telegram.send_glint_alert = AsyncMock()
    sup._position_peaks = {123: 45.0}
    return sup


@pytest.mark.asyncio
async def test_close_guarded_success_pops_peak_and_alerts():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = True
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "PEAK-GUARD", "<b>test</b> msg")

    assert ok is True
    sup.mt5.close_position.assert_called_once_with(123, "PEAK-GUARD")
    assert 123 not in sup._position_peaks
    sup.telegram.send_glint_alert.assert_awaited_once_with("<b>test</b> msg")


@pytest.mark.asyncio
async def test_close_guarded_failure_keeps_peak_no_alert():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = False
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "PEAK-GUARD", "<b>test</b> msg")

    assert ok is False
    assert 123 in sup._position_peaks
    sup.telegram.send_glint_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_guarded_telegram_error_does_not_raise():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = True
    sup.telegram.send_glint_alert.side_effect = Exception("network down")
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "PEAK-GUARD", "<b>test</b> msg")

    assert ok is True  # close itself still succeeded


@pytest.mark.asyncio
async def test_close_guarded_no_telegram_html_skips_alert():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = True
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "SCALP-TP")

    assert ok is True
    sup.telegram.send_glint_alert.assert_not_awaited()
