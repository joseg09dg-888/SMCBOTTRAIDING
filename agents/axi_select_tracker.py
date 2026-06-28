"""
AxiSelectTracker — Seguimiento mensual vs objetivo 5% Axi Select.

Registra P&L diario, calcula % mensual acumulado, proyecta
si el mes va a pasar el umbral de 5% con los dias restantes.
Persiste en memory/axi_select_state.json.
Comando Telegram: /axi
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List

STATE_FILE = os.path.join("memory", "axi_select_state.json")
MONTHLY_TARGET_PCT = 0.05   # 5%
TRADING_DAYS_MONTH = 22


@dataclass
class DayRecord:
    date:     str
    pnl:      float
    pnl_pct:  float


@dataclass
class TrackResult:
    monthly_pnl:        float
    monthly_pct:        float
    target_pct:         float
    days_traded:        int
    days_remaining:     int
    on_track:           bool
    projected_pct:      float
    daily_avg_needed:   float   # pnl/dia requerido para llegar al 5%
    capital:            float
    stage_name:         str


class AxiSelectTracker:
    """Tracks monthly P&L progress toward Axi Select 5% target."""

    STAGE_THRESHOLDS = [
        (5_000,     "PRE-SEED $500"),
        (10_500,    "SEED $10K"),
        (50_500,    "INCUBATION $50K"),
        (150_500,   "ACCELERATION $150K"),
        (500_500,   "PRO $500K"),
        (2_000_500, "PRO-M $2M"),
        (4_000_500, "PRO-MAX $4M"),
    ]

    def __init__(self) -> None:
        self._state: dict = self._load()

    # ── persistence ──────────────────────────────────────────────────
    def _load(self) -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"month": None, "records": [], "capital": 500.0,
                "initial_month_balance": 500.0}

    def _save(self) -> None:
        os.makedirs("memory", exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    # ── public API ───────────────────────────────────────────────────
    def set_capital(self, capital: float) -> None:
        """Llamar cuando Axi asigna nuevo capital."""
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        if self._state.get("month") != current_month:
            self._state["month"] = current_month
            self._state["records"] = []
            self._state["initial_month_balance"] = capital
        self._state["capital"] = capital
        self._save()

    def record_day(self, pnl: float, capital: float | None = None) -> None:
        """Registra el P&L del dia actual."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_month = today[:7]

        if self._state.get("month") != current_month:
            cap = capital or self._state.get("capital", 500.0)
            self._state["month"] = current_month
            self._state["records"] = []
            self._state["initial_month_balance"] = cap

        if capital:
            self._state["capital"] = capital

        cap = self._state.get("capital", 500.0)
        pnl_pct = pnl / cap if cap > 0 else 0.0

        # update or insert today
        records = self._state["records"]
        existing = [r for r in records if r["date"] == today]
        if existing:
            existing[0]["pnl"]     = pnl
            existing[0]["pnl_pct"] = pnl_pct
        else:
            records.append({"date": today, "pnl": pnl, "pnl_pct": pnl_pct})

        self._save()

    def get_status(self) -> TrackResult:
        """Calcula estado actual del mes."""
        cap     = self._state.get("capital", 500.0)
        init_b  = self._state.get("initial_month_balance", cap)
        records = self._state.get("records", [])

        monthly_pnl = sum(r["pnl"] for r in records)
        monthly_pct = monthly_pnl / init_b if init_b > 0 else 0.0
        target_pct  = MONTHLY_TARGET_PCT
        target_usd  = init_b * target_pct

        days_traded    = len(records)
        days_remaining = max(0, TRADING_DAYS_MONTH - days_traded)

        # proyeccion lineal
        if days_traded > 0:
            daily_avg      = monthly_pnl / days_traded
            projected_pnl  = monthly_pnl + daily_avg * days_remaining
            projected_pct  = projected_pnl / init_b if init_b > 0 else 0.0
        else:
            projected_pct  = 0.0
            daily_avg      = 0.0

        # cuanto necesita ganar por dia para llegar al 5%
        remaining_needed  = target_usd - monthly_pnl
        daily_avg_needed  = remaining_needed / days_remaining if days_remaining > 0 else 0.0

        on_track = projected_pct >= target_pct

        # stage name
        stage_name = "PRE-SEED"
        for threshold, name in self.STAGE_THRESHOLDS:
            if cap >= threshold:
                stage_name = name

        return TrackResult(
            monthly_pnl      = monthly_pnl,
            monthly_pct      = monthly_pct,
            target_pct       = target_pct,
            days_traded      = days_traded,
            days_remaining   = days_remaining,
            on_track         = on_track,
            projected_pct    = projected_pct,
            daily_avg_needed = daily_avg_needed,
            capital          = cap,
            stage_name       = stage_name,
        )

    def format_telegram(self) -> str:
        r = self.get_status()
        bar_filled = int(r.monthly_pct / r.target_pct * 10) if r.target_pct > 0 else 0
        bar_filled = max(0, min(10, bar_filled))
        bar = "=" * bar_filled + "." * (10 - bar_filled)

        on_track_icon = "✅" if r.on_track else "⚠️"
        pct_str       = f"{r.monthly_pct*100:+.2f}%"
        proj_str      = f"{r.projected_pct*100:+.2f}%"

        return (
            f"<b>AXI SELECT — {r.stage_name}</b>\n"
            f"Capital: <b>${r.capital:,.0f}</b>\n\n"
            f"Mes actual: <b>{pct_str}</b> / objetivo 5%\n"
            f"[{bar}] {r.monthly_pct*100:.1f}%\n\n"
            f"Dias operados: {r.days_traded} | Restantes: {r.days_remaining}\n"
            f"P&L mes: ${r.monthly_pnl:+,.0f}\n"
            f"Proyeccion fin de mes: {proj_str} {on_track_icon}\n"
            f"Necesita ${r.daily_avg_needed:,.0f}/dia para llegar al 5%"
        )
