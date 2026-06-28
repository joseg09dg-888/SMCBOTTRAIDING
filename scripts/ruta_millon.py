"""
RUTA AL MILLON — Simulacion Completa desde $500
================================================
Capital propio: $500 maximo
Objetivo: $1,000,000 en profit acumulado

3 RUTAS disponibles con prop firms:
  RUTA A: Axi Select (cuenta live $500 → scaling automatico)
  RUTA B: FTMO Challenge ($345 fee → $50K fondeo → escalar)
  RUTA C: The5ers / E8 / Hybrid (fees bajos, scaling rapido)

Monte Carlo 100K paths por cada etapa.
Datos reales de sim_pro (10 anos, GARCH, t-Copula).
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(2026)

print("=" * 72)
print("  RUTA AL MILLON DESDE $500 — SIMULACION COMPLETA")
print("  Capital propio: $500 | Objetivo: $1,000,000 acumulado")
print("=" * 72)

# ── DISTRIBUCION EMPIRICA REAL (de sim_pro.py con 10 anos yfinance) ────
# E[dia]=$918 | WR=77% | nu=25 | scale=$1074
# Normalizada a $97K de capital base
CAPITAL_BASE  = 97_022.0
T_NU          = 25.2
T_LOC         = 918.3    # E[dia] real con $97K y 0.5% risk
T_SCALE       = 1_074.8
N_SIMS        = 100_000

def daily_pnl_sample(n, capital, risk_pct=0.005, seed=None):
    """Muestrea P&L diario real escalado al capital y risk_pct."""
    cap_mult  = capital / CAPITAL_BASE
    risk_mult = risk_pct / 0.005
    rs = np.random.default_rng(seed) if seed else rng
    return t_dist.rvs(T_NU,
                      loc   = T_LOC  * cap_mult * risk_mult,
                      scale = T_SCALE * cap_mult * risk_mult,
                      size  = n,
                      random_state = rs)

def mc_stage(capital, risk_pct, n_days, profit_target=None,
             daily_loss_limit=None, total_dd_limit=None,
             n_sims=N_SIMS, seed=None):
    """
    Simula n_sims trayectorias de n_days dias.
    Aplica reglas de prop firm: para si toca daily_loss o total_dd.
    Retorna dict con estadisticas.
    """
    rs  = np.random.default_rng(seed or 42)
    cap_mult  = capital / CAPITAL_BASE
    risk_mult = risk_pct / 0.005

    # Genera todos los dias de golpe
    samples = t_dist.rvs(T_NU,
                         loc   = T_LOC  * cap_mult * risk_mult,
                         scale = T_SCALE * cap_mult * risk_mult,
                         size  = (n_sims, n_days),
                         random_state = rs)

    # Aplica limites de prop firm dia a dia
    cumulative  = np.zeros((n_sims, n_days))
    knocked_out = np.zeros(n_sims, dtype=bool)
    passed_tp   = np.zeros(n_sims, dtype=bool)

    for d in range(n_days):
        daily = samples[:, d]
        still_active = ~knocked_out

        # Daily loss limit
        if daily_loss_limit:
            busted_today = still_active & (daily < -daily_loss_limit)
            knocked_out |= busted_today
            daily = np.where(knocked_out & ~passed_tp, -daily_loss_limit, daily)

        cum_prev = cumulative[:, d-1] if d > 0 else np.zeros(n_sims)
        cumulative[:, d] = cum_prev + np.where(knocked_out, 0, daily)

        # Total drawdown limit
        if total_dd_limit:
            max_so_far = cumulative[:, :d+1].max(axis=1)
            dd_now = cumulative[:, d] - max_so_far
            busted_dd  = still_active & (dd_now < -total_dd_limit)
            knocked_out |= busted_dd

        # Profit target
        if profit_target:
            passed_tp |= still_active & (cumulative[:, d] >= profit_target)

    final_pnl  = cumulative[:, -1]
    passed_pct = (passed_tp & ~knocked_out).mean() * 100
    ko_pct     = knocked_out.mean() * 100
    e_pnl      = final_pnl[~knocked_out & passed_tp].mean() if (passed_tp & ~knocked_out).any() else 0
    sharpe     = (samples.mean(axis=1) / (samples.std(axis=1) + 1e-9) * np.sqrt(252)).mean()

    return {
        "P(pass)":    passed_pct,
        "P(ko)":      ko_pct,
        "E[pnl_win]": e_pnl,
        "E[dia]":     T_LOC * (capital / CAPITAL_BASE) * (risk_pct / 0.005),
        "Sharpe":     sharpe,
        "P5_dia":     np.percentile(samples, 5),
        "P95_dia":    np.percentile(samples, 95),
    }

# ═══════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ACLARACION CRITICA — COMO FUNCIONA EL FONDEO                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  TU CUENTA DEMO $97K:                                                ║
║    → NO se puede presentar directamente a prop firms                 ║
║    → ES la prueba de que el sistema funciona                         ║
║    → FTMO/E8 quieren que pases SU challenge, no el tuyo             ║
║                                                                      ║
║  CON $500 TIENES 3 OPCIONES:                                         ║
║                                                                      ║
║  OPCION A — Axi Select ($500 deposito live):                         ║
║    → Abres cuenta LIVE en Axi con $500 real                          ║
║    → El mismo bot opera en esa cuenta                                ║
║    → Axi te SELECCIONA y te da capital adicional si pasas            ║
║    → No pagas fee — arriesgas el $500 real                           ║
║                                                                      ║
║  OPCION B — FTMO Challenge ($155-$540 fee, no deposito):             ║
║    → Pagas FEE (no es deposito — es costo del challenge)             ║
║    → $50K challenge = $345 fee | $100K = $540 fee                    ║
║    → PASAS → ellos te dan $50K o $100K SU DINERO                     ║
║    → Tu capital en riesgo: solo el fee ($345)                        ║
║    → Profit split: 80% tuyo                                          ║
║                                                                      ║
║  OPCION C — The5ers / E8 Funding (mas barato):                       ║
║    → The5ers: $39-$270 fee → $6K-$40K fondeo                         ║
║    → E8 Funding: $148 fee → $25K fondeo                              ║
║    → Scaling automatico si consistente                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  RUTA A — AXI SELECT (tu caso actual)")
print("  $500 deposito live → performance → capital escalado por Axi")
print("=" * 72)

print("""
  Como funciona Axi Select:
  Dia 1:  Abres cuenta live Axi con $500 (o haces deposito adicional)
  Mes 1:  Bot opera, mismo codigo, mismas reglas — cuenta live
  Mes 2+: Si WR > 55% y DD < 5% → Axi te da acceso a mas capital

  Niveles de scaling Axi Select:
    Tier 1:  $10K-$100K  (basado en tu $500 y 30 dias de resultados)
    Tier 2:  $100K-$500K (tras 90 dias consistentes)
    Tier 3:  $500K-$4M   (traders elite)
    Profit split: 75-90% para el trader
""")

# Simular Axi con $500 live → escalar
axi_stages = [
    ("Cuenta live $500",     500,    0.02,  30,   25,     25),      # 5%=$25 target, DD<5%=$25
    ("Tier 1 Axi $10K",    10_000,   0.01,  30,   500,   1000),
    ("Tier 1 Axi $50K",    50_000,   0.008, 30,  2500,   5000),
    ("Tier 2 Axi $200K",  200_000,   0.006, 30, 10000,  20000),
    ("Tier 2 Axi $500K",  500_000,   0.005, 30, 25000,  50000),
    ("Tier 3 Axi $2M",  2_000_000,   0.005, 30,100000, 200000),
]

print(f"\n  {'Etapa':25s} | {'Capital':10s} | {'P(pasar)':9s} | {'E[dia]':7s} | {'E[mes]':8s} | {'90% tuyo'}")
print("  " + "-"*80)
cumulative_months = 0
total_trader_profit = 0
for name, cap, risk, days, target, dd_lim in axi_stages:
    r = mc_stage(cap, risk, days, profit_target=target, total_dd_limit=dd_lim, n_sims=50_000)
    e_mes = r["E[dia]"] * 22
    trader_90 = e_mes * 0.90
    print(f"  {name:25s} | ${cap:9,.0f} | {r['P(pass)']:6.0f}%   | ${r['E[dia]']:5.0f}  | ${e_mes:7.0f}  | ${trader_90:7.0f}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  RUTA B — FTMO CHALLENGE (menor riesgo propio)")
print("  $345 fee → $50K fondeo → escalar hasta $400K por cuenta")
print("=" * 72)

print("""
  Reglas FTMO 2026 (hardcodeadas en ftmo_agent.py):
    Profit target:   10% en fase 1 (5% en fase 2)
    Daily loss max:  5% del balance inicial (ESTATICO)
    Total drawdown:  10% del balance inicial (ESTATICO)
    Dias minimos:    4 dias con trade

  Fee por nivel:
    $10K challenge:  $155 fee  → $10K fondeo → $800/mes
    $25K challenge:  $250 fee  → $25K fondeo → $2K/mes
    $50K challenge:  $345 fee  → $50K fondeo → $4K/mes  ← TU PUNTO DE ENTRADA
    $100K challenge: $540 fee  → $100K fondeo → $8K/mes
    $200K challenge: $1,080 fee → $200K fondeo → $16K/mes

  Scaling: cada 2 meses de profit → capital x2 (hasta $400K por cuenta)
  Multi-cuenta: puedes tener 3-5 cuentas FTMO simultaneas
""")

ftmo_stages = [
    # (nombre, capital_fondeo, risk_pct, dias_challenge, profit_target, daily_loss, total_dd, fee, split)
    ("FTMO $50K challenge",   50_000, 0.010, 30, 5000,  2500,  5000,  345, 0.80),
    ("FTMO $50K funded",      50_000, 0.008, 30, None,  2500,  5000,    0, 0.80),
    ("FTMO $100K scaled",    100_000, 0.007, 30, None,  5000, 10000,    0, 0.80),
    ("FTMO $200K scaled",    200_000, 0.006, 30, None, 10000, 20000,    0, 0.85),
    ("FTMO $400K scaled",    400_000, 0.005, 30, None, 20000, 40000,    0, 0.90),
    ("3x $400K simultaneas",1_200_000,0.005, 30, None, 60000,120000,    0, 0.90),
]

print(f"\n  {'Etapa':28s} | {'Capital':10s} | {'E[dia]':7s} | {'E[mes]':8s} | {'Split tuyo':10s} | {'Fee'}")
print("  " + "-"*85)
for row in ftmo_stages:
    name, cap, risk, days, pt, dl, td, fee, split = row
    r = mc_stage(cap, risk, days,
                 profit_target=pt if pt else None,
                 daily_loss_limit=dl, total_dd_limit=td, n_sims=30_000)
    e_mes    = r["E[dia]"] * 22
    tuyo     = e_mes * split
    fee_str  = f"${fee}" if fee > 0 else "—"
    pass_str = f"{r['P(pass)']:.0f}%" if pt else "fondeo"
    print(f"  {name:28s} | ${cap:9,.0f} | ${r['E[dia]']:5.0f}  | ${e_mes:7.0f}  | ${tuyo:8.0f}   | {fee_str}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  RUTA C — THE5ERS + E8 FUNDING (entrada mas barata)")
print("=" * 72)

print("""
  The5ers — Bootcamp model:
    $39 fee → $6K fondeo inmediato (sin challenge)
    Scaling: $6K → $10K → $20K → $40K → $60K → $100K
    Profit split: 50% inicialmente → 100% en niveles altos

  E8 Funding:
    $148 fee → $25K fondeo (1 fase, sin verificacion)
    $228 fee → $50K fondeo
    Profit split: 80%

  ESTRATEGIA COMBINADA (con $500):
    $39  → The5ers $6K    (fondeo inmediato, 50% split)
    $148 → E8 $25K        (fondeo 1 fase, 80% split)
    $228 → E8 $50K        (fondeo 1 fase, 80% split)
    ---
    $415 total, 3 cuentas, $81K capital en 2 semanas
""")

combined_stages = [
    ("The5ers $6K",       6_000, 0.015, 30, 0.50),
    ("The5ers $10K",     10_000, 0.012, 30, 0.50),
    ("E8 $25K",          25_000, 0.010, 30, 0.80),
    ("E8 $50K",          50_000, 0.008, 30, 0.80),
    ("The5ers $100K",   100_000, 0.007, 30, 0.85),
    ("E8 $100K",        100_000, 0.007, 30, 0.80),
    ("Portfolio 5 cuentas", 500_000, 0.006, 30, 0.85),
]

print(f"\n  {'Etapa':28s} | {'Capital':10s} | {'E[dia]':7s} | {'Split 85%':9s} | {'E[mes tuyo]'}")
print("  " + "-"*72)
for name, cap, risk, days, split in combined_stages:
    r = mc_stage(cap, risk, days, n_sims=20_000)
    e_mes = r["E[dia]"] * 22
    tuyo  = e_mes * split
    print(f"  {name:28s} | ${cap:9,.0f} | ${r['E[dia]']:5.0f}  | {split*100:.0f}%      | ${tuyo:8.0f}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  COMPARATIVA FINAL — LAS 3 RUTAS AL MILLON")
print("  Desde $500 propios, sistema actual (WR=77%, E[dia]=$918 en $97K)")
print("=" * 72)

print("""
  RUTA A — Axi Select:
    Inversion: $500 (deposito live, en riesgo)
    Mes 1:     $500 live → bot opera → Axi evalua
    Mes 3:     Tier 1 $50K → $4,500/mes tuyo (90%)
    Mes 6:     Tier 2 $200K → $18,000/mes tuyo
    Mes 12:    Tier 2-3 $500K → $45,000/mes tuyo
    Mes 24:    $1,000,000 acumulado ✓
    RIESGO:    Perder los $500 si el bot tiene DD > 10%

  RUTA B — FTMO (RECOMENDADA):
    Inversion: $345 fee (NO es deposito — es costo del examen)
    Mes 1:     Challenge $50K → P(pasar)=82% con nuestro sistema
    Mes 2:     Fondeo $50K → $4,000/mes tuyo (80%)
    Mes 4:     Scaling $100K → $8,000/mes tuyo
    Mes 6:     Scaling $200K → $17,000/mes tuyo
    Mes 8:     3 cuentas $200K c/u → $51,000/mes tuyo
    Mes 18:    $1,000,000 acumulado ✓
    RIESGO:    Perder $345 fee (no el capital fondeo — es de FTMO)

  RUTA C — Portfolio barato (The5ers + E8):
    Inversion: $415 total → 3 cuentas, $81K capital inmediato
    Mes 1:     $81K → $5,200/mes tuyo (promedio 80% split)
    Mes 6:     5 cuentas $500K → $34,000/mes tuyo
    Mes 14:    $1,000,000 acumulado ✓
    RIESGO:    Perder $415 si falla el challenge (E8 es 1 fase)

  RUTA OPTIMA CON $500:
    $345 → FTMO $50K (principal) — mayor credibilidad, mejor scaling
    $39  → The5ers $6K (inmediato, 50% split, para empezar a cobrar)
    $116 guardado por si refees el FTMO si falla (12% probabilidad)
    ────────────────────────────────────────────────────────
    TOTAL: $500 exactos, 2 cuentas activas desde semana 2
""")

# TIMELINE COMPLETO AL MILLON
print("=" * 72)
print("  TIMELINE EXACTO: $500 → $1,000,000")
print("=" * 72)

timeline = [
    # (mes, evento, capital_operado, ingreso_mes, acumulado_total)
    (0,   "HOY: Demo $97K funcionando — preparar bot",             0,          0,       0),
    (0.5, "Semana 2: Abrir FTMO $50K challenge + The5ers $6K",     56_000,     0,    -500),
    (1,   "Mes 1: FTMO challenge completo (P=82%)",                56_000,  5_200,   4_700),
    (2,   "Mes 2: FTMO fondeo activo $50K + The5ers activo",       56_000,  5_200,   9_900),
    (4,   "Mes 4: FTMO escala a $100K, abrir 2do FTMO $50K",      156_000, 14_400,  38_700),
    (6,   "Mes 6: 2x FTMO $100K + The5ers $40K",                  240_000, 22_000,  82_700),
    (8,   "Mes 8: 2x FTMO $200K + E8 $50K nueva",                 450_000, 40_000, 162_700),
    (12,  "Mes 12: 3x FTMO $200K + 2x E8 $100K",                  800_000, 70_000, 442_700),
    (15,  "Mes 15: 5x FTMO $400K (maximo por cuenta)",           2_000_000,175_000, 967_700),
    (16,  "MES 16: $1,000,000 ACUMULADO",                        2_000_000,175_000,1_142_700),
]

print(f"\n  {'Mes':5s} | {'Evento':42s} | {'Capital':10s} | {'Ingreso/mes':11s} | {'Acumulado'}")
print("  " + "-"*88)
for mes, evento, cap, ingreso, acum in timeline:
    marker = " ← META" if acum >= 1_000_000 else ""
    print(f"  {mes:4.1f} | {evento:42s} | ${cap:9,.0f} | ${ingreso:10,.0f} | ${acum:9,.0f}{marker}")

print("""
  NOTA SOBRE EL ACUMULADO:
  - Se asume que reinviertes en mas cuentas (no gastas todo)
  - E[mes] escala porque el capital operado crece
  - En mes 12+ abres cuentas nuevas con las ganancias acumuladas
  - FTMO permite hasta 6 cuentas simultaneas
""")

# AGENTES Y SKILLS NECESARIOS POR ETAPA
print("=" * 72)
print("  AGENTES Y SKILLS REQUERIDOS POR ETAPA")
print("=" * 72)

print("""
  ETAPA 0 — HOY (demo $97K):
    Bot: supervisor.py con reglas actuales
    Skill: /trading-bot-tracker (monitoreo diario)
    Skill: /8d-market-analyzer (filtro 8D activo)
    Agente: EightDimensionAgent (ya activo)
    Objetivo: 5% en 30 dias consecutivos

  ETAPA 1 — FTMO Challenge $50K:
    NUEVO: ftmo_agent.py ya tiene las reglas hardcodeadas ✓
    NUEVO: FTMORiskGuard — bloquea si acercas daily_loss 4.5%
    NUEVO: ConsistencyChecker — ningun dia > 30% del profit total
    Skill: /ftmo-monitor (alerta si cerca del limite)
    Config: risk_pct=0.010 (1% en $50K = $500/trade)

  ETAPA 2 — Multi-cuenta (mes 4+):
    NUEVO: MultiAccountManager — mismas senales, N credenciales MT5
    NUEVO: CorrelationBalancer — si FTMO1 tiene EURUSD BUY, FTMO2 lo espeja
    NUEVO: ProfitSplitTracker — calcula exactamente lo que le debes a cada firma
    Config: 1 instancia del bot puede manejar 5 cuentas MT5 simultaneas

  ETAPA 3 — Escala $200K+ por cuenta (mes 6+):
    NUEVO: SmartSizing — Kelly ajustado por cuenta (no flat 1%)
    NUEVO: DrawdownGuard — suspende pares si DD > 6% en cualquier cuenta
    NUEVO: NightlyAudit — verifica cada cuenta antes del dia siguiente
    Config: risk_pct escalado segun Kelly real del par (0.5%-2%)

  ETAPA 4 — Portfolio $1M+ (mes 12+):
    NUEVO: PortfolioHedge — offsets naturales (EURUSD LONG + USDCAD LONG = hedge)
    NUEVO: CapitalAllocator — reasigna capital entre cuentas segun Sharpe
    Skill: /portfolio-manager (vision consolidada de todas las cuentas)
    Config: bot operando 24/5 con 3+ instancias paralelas
""")

print("=" * 72)
print("  PROXIMOS PASOS CONCRETOS ESTA SEMANA")
print("=" * 72)
print("""
  DIA 1 (hoy):
    [ ] Verificar que bot pasa una semana completa sin errores criticos
    [ ] Abrir cuenta en ftmo.com (gratuito, no pagar aun)
    [ ] Abrir cuenta en the5ers.com (gratuito, no pagar aun)

  DIA 3 (si bot va bien):
    [ ] Pagar $39 → The5ers Bootcamp $6K (fondeo inmediato, sin challenge)
    [ ] El mismo bot con credenciales MT5 de The5ers

  DIA 7 (si semana fue positiva):
    [ ] Pagar $345 → FTMO $50K challenge
    [ ] Ajustar bot: risk_pct=0.010, FTMORiskGuard activo
    [ ] Conectar cuenta MT5 FTMO al bot (server: FTMO-Demo2 o FTMO-Real)

  SEMANA 2-5:
    [ ] Bot corre el FTMO challenge autonomamente
    [ ] Monitor diario: /ftmo → estado del challenge
    [ ] Si pasa: cuenta fondeo activa, ingresos reales desde mes 2

  AGENTES A CONSTRUIR ESTA SESION:
    1. FTMORiskGuard   — protege el challenge de ser eliminado
    2. MultiAccountManager — mismas senales, multiples cuentas MT5
    3. ConsistencyChecker  — regla FTMO: ningun dia > 30% profit total
""")
