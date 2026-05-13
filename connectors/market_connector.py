# connectors/market_connector.py
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
import pandas as pd

from connectors.binance_connector import BinanceConnector
from connectors.metatrader_connector import MT5Connector

logger = logging.getLogger(__name__)


class Market(Enum):
    BINANCE = "binance"
    MT5     = "mt5"


class OrderSide(Enum):
    BUY  = "buy"
    SELL = "sell"


@dataclass
class MarketOrder:
    market: Market
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    order_id: str = ""
    status: str = "pending"

    @property
    def risk_reward(self) -> float:
        if self.entry_price == self.stop_loss:
            return 0.0
        return abs(self.take_profit - self.entry_price) / abs(self.entry_price - self.stop_loss)


class MarketConnector:
    """Unified interface for Binance and MT5."""

    def __init__(
        self,
        binance_key: str = "",
        binance_secret: str = "",
        binance_testnet: bool = True,
        mt5_login: int = 0,
        mt5_password: str = "",
        mt5_server: str = "MetaQuotes-Demo",
    ):
        self._binance = BinanceConnector(binance_key, binance_secret, binance_testnet)
        self._mt5     = MT5Connector(mt5_login, mt5_password, mt5_server)

    def get_data(self, market: Market, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        if market == Market.BINANCE:
            return self._binance.get_ohlcv(symbol, timeframe, limit)
        elif market == Market.MT5:
            return self._mt5.get_ohlcv(symbol, timeframe, limit)
        return pd.DataFrame()

    def execute(self, order: MarketOrder) -> dict:
        if order.market == Market.BINANCE:
            return self._binance.place_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        elif order.market == Market.MT5:
            return self._mt5.place_order(
                symbol=order.symbol,
                order_type=order.side.value,
                volume=order.quantity,
                sl=order.stop_loss,
                tp=order.take_profit,
            )
        return {"error": "Unknown market"}

    def get_portfolio(self) -> Dict:
        binance_bal = 0.0
        mt5_info    = {"balance": 0.0, "equity": 0.0}
        try:
            binance_bal = self._binance.get_balance()
        except Exception:
            pass
        try:
            mt5_info = self._mt5.get_account_info()
        except Exception:
            pass

        return {
            "binance_balance": binance_bal,
            "mt5_balance":     mt5_info.get("balance", 0.0),
            "mt5_equity":      mt5_info.get("equity", 0.0),
            "total_equity":    binance_bal + mt5_info.get("equity", 0.0),
        }

    def get_all_positions(self) -> List[dict]:
        binance_pos = []
        mt5_pos     = []
        try:
            binance_pos = [{"market": "binance", **p} for p in self._binance.get_open_positions()]
        except Exception:
            pass
        try:
            mt5_pos = [{"market": "mt5", **p} for p in self._mt5.get_positions()]
        except Exception:
            pass
        return binance_pos + mt5_pos
