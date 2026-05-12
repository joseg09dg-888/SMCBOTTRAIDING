from dataclasses import dataclass, field
from typing import Tuple, Optional
from core.config import Config


@dataclass
class RiskManager:
    config: Config
    capital: float
    open_positions: int = 0
    daily_pnl: float = 0.0
    monthly_pnl: float = 0.0
    trade_history: list = field(default_factory=list)

    def can_open_trade(self) -> Tuple[bool, str]:
        max_daily = self.capital * self.config.max_daily_loss
        max_monthly = self.capital * self.config.max_monthly_loss

        if self.open_positions >= self.config.max_open_positions:
            return False, f"Límite de posiciones alcanzado ({self.config.max_open_positions})"
        if self.daily_pnl <= -abs(max_daily):
            return False, f"Pérdida diaria máxima alcanzada ({max_daily:.2f} USD)"
        if self.monthly_pnl <= -abs(max_monthly):
            return False, f"Pérdida mensual máxima alcanzada ({max_monthly:.2f} USD)"
        return True, "OK"

    def calculate_position_size(
        self,
        entry: float,
        stop_loss: float,
        pip_value: float = 0.0001,
    ) -> float:
        risk_usd = self.capital * self.config.max_risk_per_trade
        pips_sl = abs(entry - stop_loss) / pip_value
        if pips_sl == 0:
            return 0.0
        size = risk_usd / (pips_sl * pip_value * 10_000)
        return round(min(size, self.capital * 0.02), 4)

    def validate_trade(
        self,
        entry: float,
        stop_loss: Optional[float],
        take_profit: float,
    ) -> dict:
        if stop_loss is None:
            raise ValueError("Stop Loss es obligatorio en cada operación")
        if entry == stop_loss:
            raise ValueError("Entry y Stop Loss no pueden ser iguales")

        rr = abs(take_profit - entry) / abs(entry - stop_loss)
        return {
            "valid": True,
            "risk_reward": round(rr, 2),
            "min_rr_met": rr >= 2.0,
            "risk_usd": round(self.capital * self.config.max_risk_per_trade, 2),
        }

    def record_trade(self, pnl: float):
        self.daily_pnl += pnl
        self.monthly_pnl += pnl
        self.trade_history.append(pnl)
        if self.open_positions > 0:
            self.open_positions -= 1

    def reset_daily(self):
        self.daily_pnl = 0.0

    def reset_monthly(self):
        self.monthly_pnl = 0.0
