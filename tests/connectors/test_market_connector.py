import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from connectors.market_connector import MarketConnector, Market, OrderSide, MarketOrder


def test_market_enum_values():
    assert Market.BINANCE.value == "binance"
    assert Market.MT5.value == "mt5"


def test_order_has_required_fields():
    order = MarketOrder(
        market=Market.BINANCE,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.001,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=53000.0,
    )
    assert order.risk_reward == pytest.approx(3.0, rel=0.1)
    assert order.side == OrderSide.BUY


def test_order_rr_short():
    order = MarketOrder(
        market=Market.MT5,
        symbol="EURUSD",
        side=OrderSide.SELL,
        quantity=0.01,
        entry_price=1.1000,
        stop_loss=1.1050,
        take_profit=1.0900,
    )
    assert order.risk_reward == pytest.approx(2.0, rel=0.1)


def test_connector_routes_binance(tmp_path):
    mc = MarketConnector()
    mc._binance = MagicMock()
    mc._binance.get_ohlcv.return_value = pd.DataFrame({
        "open": [100.0], "high": [105.0], "low": [99.0],
        "close": [103.0], "volume": [1000.0]
    })
    df = mc.get_data(Market.BINANCE, "BTCUSDT", "1h", limit=1)
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    mc._binance.get_ohlcv.assert_called_once()


def test_connector_routes_mt5(tmp_path):
    mc = MarketConnector()
    mc._mt5 = MagicMock()
    mc._mt5.get_ohlcv.return_value = pd.DataFrame({
        "open": [1.1000], "high": [1.1050], "low": [1.0990],
        "close": [1.1020], "volume": [500.0]
    })
    df = mc.get_data(Market.MT5, "EURUSD", "H1", limit=1)
    assert isinstance(df, pd.DataFrame)
    mc._mt5.get_ohlcv.assert_called_once()


def test_get_portfolio_combines_markets():
    mc = MarketConnector()
    mc._binance = MagicMock()
    mc._mt5 = MagicMock()
    mc._binance.get_balance.return_value = 500.0
    mc._mt5.get_account_info.return_value = {"balance": 500.0, "equity": 520.0}
    portfolio = mc.get_portfolio()
    assert "binance_balance" in portfolio
    assert "mt5_balance" in portfolio
    assert portfolio["total_equity"] > 0


def test_binance_symbols_list():
    from connectors.binance_connector import BinanceConnector
    bc = BinanceConnector.__new__(BinanceConnector)
    assert "BTCUSDT" in bc.SYMBOLS
    assert "ETHUSDT" in bc.SYMBOLS
    assert len(bc.SYMBOLS) >= 6


def test_mt5_symbols_list():
    from connectors.metatrader_connector import MT5Connector
    mt5 = MT5Connector.__new__(MT5Connector)
    assert "EURUSD" in mt5.SYMBOLS
    assert "XAUUSD" in mt5.SYMBOLS
    assert len(mt5.SYMBOLS) >= 6


def test_order_side_enum():
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"
