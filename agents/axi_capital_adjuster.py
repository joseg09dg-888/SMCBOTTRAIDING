"""
AxiCapitalAdjuster — Detecta cuando Axi asigna nuevo capital.

Monitorea el balance MT5. Cuando Axi escala el capital (salto > 10%
o > $1,000 en un ciclo), recalcula MAX_DOLLAR_RISK y volumenes.
Persiste capital conocido en memory/axi_select_state.json.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from core.atomic_json import read_json, write_json_atomic

STATE_FILE = os.path.join("memory", "axi_select_state.json")

# Parametros de sizing por capital Axi asignado
# (capital_min, risk_pct, max_dollar_risk_swing, max_dollar_risk_scalp)
SIZING_TABLE = [
    (0,           0.020, 10,    5),
    (5_000,       0.015, 50,   20),
    (10_000,      0.012, 120,  40),
    (50_000,      0.010, 500, 150),
    (150_000,     0.008, 1_200, 350),
    (500_000,     0.006, 3_000, 750),
    (2_000_000,   0.005, 10_000, 2_000),
    (4_000_000,   0.005, 20_000, 4_000),
]

JUMP_THRESHOLD_PCT = 0.10   # 10% de salto
JUMP_THRESHOLD_USD = 1_000  # o $1K absoluto


@dataclass
class AdjustResult:
    adjusted:            bool
    prev_capital:        float
    new_capital:         float
    jump_usd:            float
    jump_pct:            float
    new_risk_pct:        float
    new_max_risk_swing:  float
    new_max_risk_scalp:  float
    reason:              str


class AxiCapitalAdjuster:
    """Detects Axi capital scaling events and returns new sizing parameters."""

    def __init__(self) -> None:
        self._known_capital: float = self._load_capital()

    def _load_capital(self) -> float:
        return float(read_json(STATE_FILE, {}).get("capital", 500.0))

    def _save_capital(self, capital: float) -> None:
        state = read_json(STATE_FILE, {})
        state["capital"] = capital
        write_json_atomic(STATE_FILE, state)

    def _get_sizing(self, capital: float) -> tuple[float, float, float]:
        risk_pct = max_swing = max_scalp = 0.0
        for cap_min, rp, ms, msc in SIZING_TABLE:
            if capital >= cap_min:
                risk_pct, max_swing, max_scalp = rp, ms, msc
        return risk_pct, max_swing, max_scalp

    def check(self, current_balance: float) -> AdjustResult:
        """
        current_balance: balance actual de la cuenta MT5.
        Detecta si Axi asigno capital adicional.
        """
        prev   = self._known_capital
        jump   = current_balance - prev
        jump_p = jump / prev if prev > 0 else 0.0

        is_jump = (jump_p >= JUMP_THRESHOLD_PCT or jump >= JUMP_THRESHOLD_USD)
        adjusted = is_jump and jump > 0

        if adjusted:
            self._known_capital = current_balance
            self._save_capital(current_balance)

        cap = current_balance if adjusted else prev
        risk_pct, max_swing, max_scalp = self._get_sizing(cap)

        if adjusted:
            reason = (f"CAPITAL ESCALADO: ${prev:,.0f} → ${current_balance:,.0f} "
                      f"(+${jump:,.0f}, +{jump_p*100:.1f}%) "
                      f"| nuevo risk={risk_pct*100:.1f}% max_swing=${max_swing}")
        else:
            reason = f"Sin cambio. Capital conocido: ${prev:,.0f}"

        return AdjustResult(
            adjusted           = adjusted,
            prev_capital       = prev,
            new_capital        = current_balance,
            jump_usd           = jump,
            jump_pct           = jump_p,
            new_risk_pct       = risk_pct,
            new_max_risk_swing = max_swing,
            new_max_risk_scalp = max_scalp,
            reason             = reason,
        )

    @property
    def known_capital(self) -> float:
        return self._known_capital

    def format_telegram(self, result: AdjustResult) -> str:
        if not result.adjusted:
            return f"Capital sin cambio: ${result.prev_capital:,.0f}"
        return (
            f"<b>AXI CAPITAL ESCALADO</b>\n"
            f"${result.prev_capital:,.0f} → <b>${result.new_capital:,.0f}</b>\n"
            f"+${result.jump_usd:,.0f} (+{result.jump_pct*100:.1f}%)\n\n"
            f"Nuevo sizing:\n"
            f"  Risk: {result.new_risk_pct*100:.1f}%\n"
            f"  Max swing: ${result.new_max_risk_swing:,.0f}\n"
            f"  Max scalp: ${result.new_max_risk_scalp:,.0f}"
        )
