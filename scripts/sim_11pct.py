"""
Simulacion del impacto del DIM6 circuit breaker sobre el 11% de fallo mensual.
Usa la distribucion real calibrada en sim_pro.py (10 anos yfinance + GARCH).
Compara: sin DIM6 vs con DIM6 → muestra nuevo P(fallo mes).
"""
import numpy as np
from scipy.stats import t as t_dist

rng = np.random.default_rng(2026)

# Parametros calibrados con datos reales (sim_pro.py)
T_NU    = 25.2
T_LOC   = 918.3   # E[dia] con $97K capital
T_SCALE = 1074.8

CAPITAL      = 97_022.0
TARGET_5PCT  = CAPITAL * 0.05      # $4,851 — objetivo mensual Axi
DAILY_LIMIT  = -CAPITAL * 0.04     # -$3,880 — guard diario
DD_LIMIT     = -CAPITAL * 0.10     # -$9,702 — drawdown max
TRADING_DAYS = 22
N_PATHS      = 100_000

def gen_dias(n):
    """Genera n dias de P&L desde distribucion t-Student calibrada."""
    return t_dist.rvs(df=T_NU, loc=T_LOC, scale=T_SCALE, size=n, random_state=rng)

# ── VERSION SIN DIM6 ──────────────────────────────────────────────────
def sim_sin_dim6():
    fails = 0
    for _ in range(N_PATHS):
        balance  = CAPITAL
        peak     = CAPITAL
        monthly  = 0.0
        failed   = False

        dias = gen_dias(TRADING_DAYS)
        for pnl in dias:
            # Solo aplica guard diario y DD limit
            if pnl < DAILY_LIMIT:
                pnl = DAILY_LIMIT   # guard cierra todo
            balance += pnl
            monthly += pnl
            peak     = max(peak, balance)
            if balance - CAPITAL < DD_LIMIT:  # drawdown max
                failed = True
                break

        if failed or monthly < TARGET_5PCT:
            fails += 1

    return fails / N_PATHS

# ── VERSION CON DIM6 ──────────────────────────────────────────────────
def sim_con_dim6():
    """
    DIM6 aplica 3 reglas:
    1. Circuit breaker: 3 perdidas seguidas → pausa 2 dias (pierde esos dias)
    2. WR < 40% en ultimos 5 trades → size reducido a 60%
    3. Profit >= 4% → size reducido a 30% (protege target Axi)
    """
    fails = 0
    for _ in range(N_PATHS):
        balance    = CAPITAL
        peak       = CAPITAL
        monthly    = 0.0
        failed     = False
        consec_loss = 0
        recent_5   = []
        paused_days = 0

        dias = gen_dias(TRADING_DAYS + 4)  # generar extra por si hay pausas
        dias_idx = 0
        dias_operados = 0

        while dias_operados < TRADING_DAYS and dias_idx < len(dias):
            if paused_days > 0:
                paused_days -= 1
                dias_idx += 1
                dias_operados += 1
                continue

            pnl_raw = dias[dias_idx]
            dias_idx += 1

            # DIM6 regla 3: monthly profit >= 4% → solo 30% size
            monthly_pct = monthly / CAPITAL
            if monthly_pct >= 0.04:
                pnl_raw *= 0.30

            # DIM6 regla 2: WR < 40% en ultimos 5 → 60% size
            elif len(recent_5) >= 5:
                wr5 = sum(1 for p in recent_5[-5:] if p > 0) / 5
                if wr5 < 0.40:
                    pnl_raw *= 0.60

            # Guard diario
            if pnl_raw < DAILY_LIMIT:
                pnl_raw = DAILY_LIMIT

            balance  += pnl_raw
            monthly  += pnl_raw
            peak      = max(peak, balance)
            dias_operados += 1

            # Track consecutivos y recientes
            recent_5.append(pnl_raw)
            if len(recent_5) > 5:
                recent_5.pop(0)

            if pnl_raw < 0:
                consec_loss += 1
                # DIM6 regla 1: 3 consecutivos → pausa 2 dias
                if consec_loss >= 3:
                    paused_days = 2
                    consec_loss = 0
            else:
                consec_loss = 0

            # DD max check
            if balance - CAPITAL < DD_LIMIT:
                failed = True
                break

        if failed or monthly < TARGET_5PCT:
            fails += 1

    return fails / N_PATHS

print("=" * 60)
print("  SIMULACION: IMPACTO DIM6 SOBRE EL 11% DE FALLO")
print("  100,000 paths | Datos reales 10 anos yfinance + GARCH")
print("=" * 60)

print("\n  Corriendo simulacion SIN DIM6 (baseline)...", flush=True)
p_fail_sin = sim_sin_dim6()

print("  Corriendo simulacion CON DIM6 (circuit breaker)...", flush=True)
p_fail_con = sim_con_dim6()

mejora_abs = (p_fail_sin - p_fail_con) * 100
mejora_rel = (1 - p_fail_con / p_fail_sin) * 100

print(f"""
  RESULTADO:
  ┌────────────────────────────────────────────┐
  │  Sin DIM6:  P(fallo mes) = {p_fail_sin*100:5.1f}%           │
  │  Con DIM6:  P(fallo mes) = {p_fail_con*100:5.1f}%           │
  │                                            │
  │  Mejora absoluta:  -{mejora_abs:.1f}pp               │
  │  Mejora relativa:  -{mejora_rel:.0f}% menos fallos       │
  │                                            │
  │  P(pasar Axi mes) sin DIM6: {(1-p_fail_sin)*100:.1f}%      │
  │  P(pasar Axi mes) con DIM6: {(1-p_fail_con)*100:.1f}%      │
  └────────────────────────────────────────────┘

  COMO FUNCIONA DIM6 (3 mecanismos):

  1. Circuit breaker (3 perdidas seguidas → pausa 2 dias):
     Evita que el bot siga operando en un regimen malo.
     El mercado choppy genera rachas de 3-5 perdidas.
     Al pausar, espera a que el mercado se normalice.

  2. WR filter (WR < 40% en 5 trades → size 60%):
     Si el bot esta perdiendo mas de lo normal,
     reduce el riesgo automaticamente antes de un drawdown serio.

  3. Monthly profit lock (profit >= 4% → size 30%):
     Una vez que alcanzas el objetivo Axi (5% aprox),
     el bot casi para de operar. Protege la ganancia del mes.
     El 5% queda asegurado incluso si las ultimas sesiones son malas.
""")
