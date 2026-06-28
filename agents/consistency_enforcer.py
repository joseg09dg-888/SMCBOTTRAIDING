"""
ConsistencyEnforcer — Ningún dia supera el 30% del profit mensual.

Regla critica de Axi Select: si un solo dia representa > 30%
de las ganancias del mes, el perfil parece de "suerte" en vez
de sistema. Bloquea ordenes nuevas si se supera ese umbral.
"""
from __future__ import annotations
from dataclasses import dataclass

MAX_DAY_PCT_OF_MONTH = 0.30   # 30%
CONSERVATIVE_THRESHOLD = 0.25  # avisar a partir del 25%


@dataclass
class EnforceResult:
    is_conservative:       bool   # solo scalps pequeños permitidos
    should_block_new:      bool   # bloquear nuevas ordenes grandes
    today_pct_of_monthly:  float  # cuanto representa hoy del mes
    today_pnl:             float
    monthly_pnl:           float
    max_allowed_today:     float  # maximo que puede ganar hoy sin violar regla
    reason:                str


class ConsistencyEnforcer:
    """
    Enforces the Axi Select consistency rule:
    No single day > 30% of total monthly profit.
    """

    def check(self, today_pnl: float, monthly_pnl: float) -> EnforceResult:
        """
        today_pnl:    P&L realizado + flotante del dia actual
        monthly_pnl:  P&L total del mes (incluyendo hoy)
        """
        # Si el mes aun no tiene profit, la regla no aplica
        if monthly_pnl <= 0:
            return EnforceResult(
                is_conservative      = False,
                should_block_new     = False,
                today_pct_of_monthly = 0.0,
                today_pnl            = today_pnl,
                monthly_pnl          = monthly_pnl,
                max_allowed_today    = float("inf"),
                reason               = "Mes sin profit — regla 30% no aplica",
            )

        # Cuanto representa hoy del total mensual
        pct = today_pnl / monthly_pnl if monthly_pnl > 0 else 0.0

        # Maximo que puede ganar hoy sin violar 30%
        # Si monthly_pnl ya incluye today_pnl, resolvemos:
        # today / (monthly_sin_hoy + today) <= 0.30
        # today <= 0.30 * monthly_sin_hoy / 0.70
        monthly_sin_hoy   = monthly_pnl - today_pnl
        max_allowed_today = (MAX_DAY_PCT_OF_MONTH * monthly_sin_hoy
                             / (1 - MAX_DAY_PCT_OF_MONTH)
                             if monthly_sin_hoy > 0 else float("inf"))

        should_block_new  = pct >= MAX_DAY_PCT_OF_MONTH and today_pnl > 0
        is_conservative   = pct >= CONSERVATIVE_THRESHOLD and today_pnl > 0

        if should_block_new:
            reason = (f"BLOQUEO CONSISTENCIA: dia ${today_pnl:+.0f} = "
                      f"{pct*100:.0f}% del mes (max 30%). "
                      f"Solo scalps <= ${max_allowed_today:.0f}")
        elif is_conservative:
            reason = (f"AVISO: dia ${today_pnl:+.0f} = {pct*100:.0f}% del mes. "
                      f"Cerca del limite 30%. Modo conservador.")
        else:
            remaining = max_allowed_today - today_pnl
            reason = (f"OK: dia ${today_pnl:+.0f} = {pct*100:.0f}% del mes. "
                      f"Puede ganar ${remaining:,.0f} mas hoy.")

        return EnforceResult(
            is_conservative      = is_conservative,
            should_block_new     = should_block_new,
            today_pct_of_monthly = pct,
            today_pnl            = today_pnl,
            monthly_pnl          = monthly_pnl,
            max_allowed_today    = max_allowed_today,
            reason               = reason,
        )

    def format_telegram(self, result: EnforceResult) -> str:
        pct_str = f"{result.today_pct_of_monthly*100:.0f}%"
        if result.should_block_new:
            icon = "🔴"
        elif result.is_conservative:
            icon = "🟡"
        else:
            icon = "🟢"
        return (
            f"{icon} <b>CONSISTENCIA AXI</b>\n"
            f"Hoy: ${result.today_pnl:+.0f} = <b>{pct_str}</b> del mes\n"
            f"Mes total: ${result.monthly_pnl:+.0f}\n"
            f"Limite 30%: ${result.max_allowed_today:,.0f} max hoy\n"
            f"{result.reason}"
        )
