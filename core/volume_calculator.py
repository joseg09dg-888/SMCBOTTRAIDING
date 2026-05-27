# core/volume_calculator.py
from __future__ import annotations


class VolumeCalculator:
    AXI_STAGES = {
        "seed":       {"capital": 5_000,     "volume": 0.01},
        "incubation": {"capital": 25_000,    "volume": 0.03},
        "demo":       {"capital": 100_000,   "volume": 0.10},
        "pro":        {"capital": 300_000,   "volume": 0.30},
        "pro_500":    {"capital": 500_000,   "volume": 0.50},
        "pro_m":      {"capital": 1_000_000, "volume": 1.00},
    }

    # Size of 1 pip in price units
    _PIP_SIZE = {
        "EURUSD": 0.0001, "GBPUSD": 0.0001,
        "USDCHF": 0.0001, "AUDUSD": 0.0001,
        "NZDUSD": 0.0001, "EURGBP": 0.0001,
        "USDJPY": 0.01,   "GBPJPY": 0.01,
        "EURJPY": 0.01,
        "XAUUSD": 1.0,    # 1 price unit = 1 pip ($1 move)
    }

    # USD value per lot per pip: XAUUSD 1lot=100oz, 1pip=$1 -> $100/lot/pip
    _PIP_VALUE = {
        "EURUSD": 10.0,  "GBPUSD": 10.0,
        "USDCHF": 10.0,  "AUDUSD": 10.0,
        "NZDUSD": 10.0,  "EURGBP": 10.0,
        "USDJPY": 6.3,   "GBPJPY": 6.3,
        "EURJPY": 6.3,
        "XAUUSD": 100.0, # 100oz × $1/oz = $100 per lot per $1 move
    }

    # Hard caps per symbol (lots) — safety net regardless of calculation
    _MAX_VOL_BY_SYMBOL = {
        "XAUUSD": 0.05,
        "NAS100": 0.05,
        "US30":   0.05,
    }

    _MIN_VOL = 0.01
    _MAX_VOL = 10.0

    def calculate_volume(
        self,
        capital: float,
        entry: float,
        stop_loss: float,
        symbol: str,
        risk_pct: float = 0.005,
    ) -> float:
        sl_distance = abs(entry - stop_loss)
        if sl_distance == 0.0:
            return self._MIN_VOL

        pip_size  = self._PIP_SIZE.get(symbol, 0.0001)
        pip_value = self._PIP_VALUE.get(symbol, 10.0)

        pips     = sl_distance / pip_size
        risk_usd = capital * risk_pct
        volume   = risk_usd / (pips * pip_value)

        max_vol = self._MAX_VOL_BY_SYMBOL.get(symbol, self._MAX_VOL)
        volume  = max(self._MIN_VOL, min(max_vol, volume))
        return round(volume, 2)

    def get_stage_volume(self, capital: float) -> float:
        sorted_stages = sorted(self.AXI_STAGES.values(), key=lambda s: s["capital"])
        matched = sorted_stages[0]
        for stage in sorted_stages:
            if capital >= stage["capital"]:
                matched = stage
        return matched["volume"]

    def project_monthly_profit(
        self,
        capital: float,
        win_rate: float = 0.62,
        trades_per_month: int = 40,
        rr: float = 2.0,
        risk_pct: float = 0.005,
    ) -> dict:
        risk_per_trade = capital * risk_pct
        win_trades     = trades_per_month * win_rate
        loss_trades    = trades_per_month * (1.0 - win_rate)

        net_profit = win_trades * risk_per_trade * rr - loss_trades * risk_per_trade

        return {
            "net_profit_usd":   round(net_profit, 2),
            "your_share_80pct": round(net_profit * 0.80, 2),
            "monthly_roi_pct":  round(net_profit / capital * 100, 3),
            "trades_per_month": trades_per_month,
            "win_rate":         win_rate,
        }
