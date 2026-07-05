"""Tests for real spread/slippage capture in MT5Connector.place_order.

Context: the backtester assumes zero transaction cost. To calibrate a
realistic cost model later, place_order must capture the real bid/ask
spread and requested price at order-placement time, alongside MT5's
actual fill price.
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_tick(bid=1.10000, ask=1.10015):
    tick = MagicMock()
    tick.bid = bid
    tick.ask = ask
    return tick


def _make_symbol_info(point=0.00001, digits=5, trade_stops_level=0,
                       filling_mode=0x02):
    info = MagicMock()
    info.point = point
    info.digits = digits
    info.trade_stops_level = trade_stops_level
    info.filling_mode = filling_mode
    return info


def _make_order_result(order=12345, price=1.10017, retcode=10009):
    result = MagicMock()
    result.order = order
    result.price = price
    result.retcode = retcode
    result.comment = "ok"
    return result


@pytest.fixture
def fake_mt5():
    with patch("connectors.metatrader_connector.HAS_MT5", True):
        import connectors.metatrader_connector as _m
        fake = MagicMock()
        fake.ORDER_TYPE_BUY = 0
        fake.ORDER_TYPE_SELL = 1
        fake.TRADE_ACTION_DEAL = 1
        fake.TRADE_ACTION_SLTP = 2
        fake.ORDER_TIME_GTC = 0
        fake.ORDER_FILLING_FOK = 0
        fake.ORDER_FILLING_IOC = 1
        fake.ORDER_FILLING_RETURN = 2
        fake.TRADE_RETCODE_DONE = 10009
        fake.symbol_select.return_value = True
        fake.symbol_info_tick.return_value = _make_tick()
        fake.symbol_info.return_value = _make_symbol_info()
        fake.order_send.return_value = _make_order_result()
        fake.positions_get.return_value = []  # no SL/TP follow-up needed
        with patch.object(_m, "mt5", fake):
            from connectors.metatrader_connector import MT5Connector
            conn = MT5Connector.__new__(MT5Connector)
            yield conn, fake


class TestPlaceOrderCostCapture:
    def test_returns_spread_pips(self, fake_mt5):
        conn, fake = fake_mt5
        result = conn.place_order("EURUSD", "BUY", 0.10, sl=1.09800, tp=1.10500)
        # bid=1.10000 ask=1.10015 point=0.00001 -> spread = 0.00015 / 0.0001 = 1.5 pips
        assert result["spread_pips"] == pytest.approx(1.5, abs=0.01)

    def test_returns_bid_ask_and_requested_price(self, fake_mt5):
        conn, fake = fake_mt5
        result = conn.place_order("EURUSD", "BUY", 0.10, sl=1.09800, tp=1.10500)
        assert result["bid"] == pytest.approx(1.10000)
        assert result["ask"] == pytest.approx(1.10015)
        # BUY uses ask as the requested price
        assert result["requested_price"] == pytest.approx(1.10015)

    def test_fill_price_still_present(self, fake_mt5):
        conn, fake = fake_mt5
        result = conn.place_order("EURUSD", "BUY", 0.10, sl=1.09800, tp=1.10500)
        assert result["price"] == pytest.approx(1.10017)
        assert result["ticket"] == 12345

    def test_sell_uses_bid_as_requested_price(self, fake_mt5):
        conn, fake = fake_mt5
        result = conn.place_order("EURUSD", "SELL", 0.10, sl=1.10500, tp=1.09800)
        assert result["requested_price"] == pytest.approx(1.10000)

    def test_no_symbol_info_spread_is_none(self, fake_mt5):
        conn, fake = fake_mt5
        fake.symbol_info.return_value = None
        result = conn.place_order("EURUSD", "BUY", 0.10, sl=1.09800, tp=1.10500)
        assert result["spread_pips"] is None

    def test_zero_point_does_not_crash(self, fake_mt5):
        conn, fake = fake_mt5
        info = _make_symbol_info(point=0.0)
        fake.symbol_info.return_value = info
        result = conn.place_order("EURUSD", "BUY", 0.10, sl=1.09800, tp=1.10500)
        assert result["spread_pips"] is None
        assert result["ticket"] == 12345
