"""
PortfolioTracker — Plan Financiero 70-20-10 de Jose David.

Rastrea en tiempo real como el capital del bot se distribuye
entre los 3 buckets. Persiste en memory/portfolio_state.json.
Comando Telegram: /plan
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

STATE_FILE = os.path.join("memory", "portfolio_state.json")

# Umbrales para activar cada bucket (USD acumulados del bot)
BUCKET_70_START   =  3_000   # empezar a invertir en ETFs/acciones
BUCKET_20_START   = 10_000   # empezar plataforma distribucion musical
BUCKET_DEBT_CLEAR =  2_000   # saldar deudas primero
BUCKET_EXPENSES   =  3_000   # 1 mes gastos cubiertos

# Allocation ratios una vez superados los umbrales base
ALLOC_70 = 0.70
ALLOC_20 = 0.20
ALLOC_10 = 0.10   # se reinvierte en bot / capital propio

# Metas de capital propio (independiente del fondeo)
OWN_CAPITAL_TARGETS = [
    (100_000,  "Primer $100K propio — inicio inversion seria 70%"),
    (500_000,  "$500K — plataforma musical escalada globalmente"),
    (1_000_000,"$1M propio — independencia total del fondeo"),
    (5_000_000,"$5M — operar capital propio sin prop firms"),
]


@dataclass
class BucketState:
    bot_earnings_total:  float   # todo lo que ha ganado el bot
    debts_paid:          float   # deudas saldadas
    expenses_covered:    float   # meses de gastos cubiertos * 3000
    bucket_70_invested:  float   # invertido en ETFs/bonos/funerarias/energia
    bucket_20_invested:  float   # invertido en negocios (musica/plataforma)
    bucket_10_capital:   float   # capital propio en el bot
    own_capital_total:   float   # capital propio acumulado (no fondeo)
    axi_funded_capital:  float   # capital que Axi tiene asignado
    last_updated:        str


class PortfolioTracker:

    def __init__(self) -> None:
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "bot_earnings_total": 0.0,
            "debts_paid": 0.0,
            "expenses_covered": 0.0,
            "bucket_70_invested": 0.0,
            "bucket_20_invested": 0.0,
            "bucket_10_capital": 500.0,
            "own_capital_total": 500.0,
            "axi_funded_capital": 500.0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _save(self) -> None:
        os.makedirs("memory", exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def record_earnings(self, amount: float, axi_capital: float | None = None) -> None:
        s = self._state
        s["bot_earnings_total"] += amount
        if axi_capital:
            s["axi_funded_capital"] = axi_capital

        remaining = amount

        # 1. Saldar deudas primero
        if s["debts_paid"] < 2_000:
            debt_payment = min(remaining, 2_000 - s["debts_paid"])
            s["debts_paid"] += debt_payment
            remaining -= debt_payment

        # 2. Cubrir 2 meses de gastos (buffer de seguridad)
        if s["expenses_covered"] < 6_000 and remaining > 0:
            exp_payment = min(remaining, 6_000 - s["expenses_covered"])
            s["expenses_covered"] += exp_payment
            remaining -= exp_payment

        # 3. Una vez cubiertos debt + expenses: distribuir 70-20-10
        if remaining > 0:
            s["bucket_70_invested"]  += remaining * ALLOC_70
            s["bucket_20_invested"]  += remaining * ALLOC_20
            s["bucket_10_capital"]   += remaining * ALLOC_10
            s["own_capital_total"]   += remaining * ALLOC_10  # solo el 10% es capital propio

        s["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_next_milestone(self) -> tuple[float, str]:
        own = self._state["own_capital_total"]
        for target, label in OWN_CAPITAL_TARGETS:
            if own < target:
                return target, label
        return OWN_CAPITAL_TARGETS[-1]

    def format_telegram(self, axi_monthly_income: float | None = None) -> str:
        s = self._state
        total_own = s["own_capital_total"]
        next_target, next_label = self.get_next_milestone()
        pct_to_next = min(total_own / next_target * 100, 100) if next_target > 0 else 100
        bar = int(pct_to_next / 10)
        bar_str = "=" * bar + "." * (10 - bar)

        # Estado deudas y gastos
        debts_ok  = "✅" if s["debts_paid"] >= 2_000 else f"⏳ ${s['debts_paid']:,.0f}/$2,000"
        exp_ok    = "✅" if s["expenses_covered"] >= 6_000 else f"⏳ ${s['expenses_covered']:,.0f}/$6,000"

        # Proyeccion a siguiente meta
        if axi_monthly_income and axi_monthly_income > 0:
            months_to_next = (next_target - total_own) / (axi_monthly_income * ALLOC_10)
            proj_str = f"~{months_to_next:.0f} meses al siguiente hito"
        else:
            proj_str = "Bot activo — proyeccion pendiente"

        lines = [
            f"<b>PLAN FINANCIERO 70-20-10</b>",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"<b>PRIORIDADES BASE:</b>",
            f"  Deudas saldadas: {debts_ok}",
            f"  Buffer gastos 2 meses: {exp_ok}",
            f"",
            f"<b>CAPITAL PROPIO ACUMULADO:</b>",
            f"  ${total_own:,.0f}",
            f"  [{bar_str}] {pct_to_next:.0f}%",
            f"  Meta: ${next_target:,.0f} ({next_label})",
            f"  {proj_str}",
            f"",
            f"<b>PORTAFOLIO 70-20-10:</b>",
            f"  70% Inversiones pasivas: ${s['bucket_70_invested']:,.0f}",
            f"     (ETFs, funerarias, energia IA, bonos, RaizToken)",
            f"  20% Negocios activos:    ${s['bucket_20_invested']:,.0f}",
            f"     (Plataforma dist. musical, sello, YouTube)",
            f"  10% Capital bot propio:  ${s['bucket_10_capital']:,.0f}",
            f"",
            f"<b>AXI SELECT:</b>",
            f"  Capital fondeado: ${s['axi_funded_capital']:,.0f}",
            f"  Ganancias totales bot: ${s['bot_earnings_total']:,.0f}",
            f"━━━━━━━━━━━━━━━━━━━━",
        ]

        # Siguiente accion concreta
        if s["debts_paid"] < 2_000:
            lines.append(f"<b>AHORA:</b> Saldar ${2_000 - s['debts_paid']:,.0f} en deudas")
        elif s["expenses_covered"] < 6_000:
            lines.append(f"<b>AHORA:</b> Completar buffer gastos ${6_000 - s['expenses_covered']:,.0f}")
        elif s["bucket_70_invested"] < 5_000:
            lines.append(f"<b>AHORA:</b> Abrir cuenta inversion ETFs con ${s['bucket_70_invested']:,.0f}")
        elif s["bucket_20_invested"] < 10_000:
            lines.append(f"<b>AHORA:</b> Iniciar plataforma distribucion musical")
        else:
            lines.append(f"<b>AHORA:</b> Escalar — todo en marcha")

        return "\n".join(lines)
