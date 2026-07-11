"""
AxiSelectGuard — Protege el capital diario de Axi Select.

Monitorea P&L en tiempo real. Si el dia cae -4% del capital asignado
cierra TODAS las posiciones y pausa el bot hasta el dia siguiente.
Alerta Telegram a -3% (warning) y -4% (emergency close).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from core.atomic_json import read_json, write_json_atomic

STATE_FILE = os.path.join("memory", "axi_select_state.json")


@dataclass
class GuardResult:
    should_close:    bool
    warning_level:   bool      # True si supero -3% pero aun no -4%
    daily_pnl_usd:   float
    daily_pnl_pct:   float
    capital:         float
    limit_usd:       float     # -4% del capital
    warning_usd:     float     # -3% del capital
    reason:          str


class AxiSelectGuard:
    """
    Guarda el balance al inicio del dia y compara contra equity actual.
    Limites:
      -3%: WARNING — reducir tamanio de posiciones
      -4%: EMERGENCY CLOSE — cierra todo, pausa bot
    """

    WARN_PCT  = 0.03   # 3% warning
    LIMIT_PCT = 0.04   # 4% emergency close

    def __init__(self) -> None:
        self._day_start_balance: float | None = None
        self._day_start_date: str | None      = None
        self._paused_today: bool              = False
        self._load()

    # ── persistence ──────────────────────────────────────────────────
    # BUG-AXI-GUARD-RESTART (2026-07-09): day_start_balance/paused_today
    # used to live only in memory. A pm2 restart mid-day (crash or watch
    # reload) reset day_start_balance to whatever the balance happened to
    # be at that moment AND silently unpaused the bot, letting it burn
    # through another full -4% before the guard fired again -- across
    # several restarts in one day this could blow well past the intended
    # daily loss limit. Now persisted to the same state file the tracker
    # and capital adjuster already use.
    def _load(self) -> None:
        state = read_json(STATE_FILE, {})
        self._day_start_balance = state.get("guard_day_start_balance")
        self._day_start_date    = state.get("guard_day_start_date")
        self._paused_today      = bool(state.get("guard_paused_today", False))

    def _save(self) -> None:
        state = read_json(STATE_FILE, {})
        state["guard_day_start_balance"] = self._day_start_balance
        state["guard_day_start_date"]    = self._day_start_date
        state["guard_paused_today"]      = self._paused_today
        write_json_atomic(STATE_FILE, state)

    @property
    def paused_today(self) -> bool:
        return self._paused_today

    def set_day_start(self, balance: float) -> None:
        """Llamar una vez al inicio de cada dia (o primer ciclo del dia)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._day_start_date != today:
            self._day_start_balance = balance
            self._day_start_date    = today
            self._paused_today      = False
            self._save()

    def check(self, equity: float, capital_assigned: float | None = None) -> GuardResult:
        """
        equity:           equity actual de la cuenta MT5
        capital_assigned: capital que Axi asigno (si None usa day_start_balance)
        """
        if self._day_start_balance is None:
            self.set_day_start(equity)

        base     = capital_assigned if capital_assigned else self._day_start_balance
        base     = base or equity

        daily_pnl     = equity - self._day_start_balance
        daily_pnl_pct = daily_pnl / base if base > 0 else 0.0

        limit_usd   = -base * self.LIMIT_PCT
        warning_usd = -base * self.WARN_PCT

        should_close  = daily_pnl <= limit_usd
        warning_level = (not should_close) and (daily_pnl <= warning_usd)

        if should_close and not self._paused_today:
            self._paused_today = True
            self._save()

        if should_close:
            reason = (f"EMERGENCY: dia {daily_pnl_pct*100:.1f}% "
                      f"(${daily_pnl:+.0f}) supera limite -4% (${limit_usd:.0f})")
        elif warning_level:
            reason = (f"WARNING: dia {daily_pnl_pct*100:.1f}% "
                      f"(${daily_pnl:+.0f}) supera aviso -3% (${warning_usd:.0f})")
        else:
            reason = f"OK: dia {daily_pnl_pct*100:.1f}% (${daily_pnl:+.0f})"

        return GuardResult(
            should_close  = should_close,
            warning_level = warning_level,
            daily_pnl_usd = daily_pnl,
            daily_pnl_pct = daily_pnl_pct,
            capital       = base,
            limit_usd     = limit_usd,
            warning_usd   = warning_usd,
            reason        = reason,
        )

    def format_telegram(self, result: GuardResult) -> str:
        bar = int(abs(result.daily_pnl_pct) * 100 / 4 * 10)
        bar = min(bar, 10)
        pct_str = f"{result.daily_pnl_pct*100:+.2f}%"
        if result.should_close:
            icon = "🚨"
        elif result.warning_level:
            icon = "⚠️"
        else:
            icon = "✅"

        return (
            f"{icon} <b>AXI GUARD</b>\n"
            f"P&L dia: <b>{pct_str}</b> (${result.daily_pnl_usd:+.0f})\n"
            f"Limite: ${result.limit_usd:.0f} (-4%) | Aviso: ${result.warning_usd:.0f} (-3%)\n"
            f"[{'|'*bar}{'.'*(10-bar)}] {pct_str}\n"
            f"{result.reason}"
        )
