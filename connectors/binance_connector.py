# connectors/binance_connector.py
import logging
from typing import Optional, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False
    BinanceAPIException = Exception

INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
}


class BinanceConnector:
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    ]

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.testnet    = testnet
        self._client    = None

    def _get_client(self):
        if self._client is None:
            if not HAS_BINANCE:
                raise RuntimeError("python-binance not installed")
            self._client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )
        return self._client

    def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
        try:
            client = self._get_client()
            klines = client.get_klines(
                symbol=symbol.upper(),
                interval=INTERVAL_MAP.get(interval, interval),
                limit=limit,
            )
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Binance get_ohlcv error: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def get_balance(self) -> float:
        try:
            client = self._get_client()
            account = client.get_account()
            for asset in account["balances"]:
                if asset["asset"] == "USDT":
                    return float(asset["free"])
            return 0.0
        except Exception as e:
            logger.error(f"Binance get_balance error: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, quantity: float,
                    stop_loss: float = None, take_profit: float = None) -> dict:
        try:
            client = self._get_client()
            order = client.create_order(
                symbol=symbol.upper(),
                side=side.upper(),
                type="MARKET",
                quantity=quantity,
            )
            return {"order_id": order["orderId"], "status": order["status"]}
        except Exception as e:
            logger.error(f"Binance place_order error: {e}")
            return {"error": str(e)}

    def get_open_positions(self) -> List[dict]:
        try:
            client = self._get_client()
            orders = client.get_open_orders()
            return [{"symbol": o["symbol"], "order_id": o["orderId"],
                     "side": o["side"], "quantity": float(o["origQty"])} for o in orders]
        except Exception as e:
            logger.error(f"Binance get_open_positions error: {e}")
            return []

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        try:
            client = self._get_client()
            client.cancel_order(symbol=symbol, orderId=order_id)
            return True
        except Exception as e:
            logger.error(f"Binance cancel_order error: {e}")
            return False
