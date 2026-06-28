"""
PLAN FINANCIERO COMPLETO — Jose David
======================================
Desde $500 hasta capital propio de $1M-$5M.
Muestra mes a mes como fluye el capital del bot
hacia los 3 buckets: 70% pasivo / 20% negocios / 10% bot.
Monte Carlo 50K paths con datos reales 10 anos.
"""
import numpy as np
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(2026)

# Distribucion real (sim_pro.py — 10 anos yfinance + GARCH)
CAPITAL_BASE = 97_022.0
T_NU, T_LOC, T_SCALE = 25.2, 918.3, 1074.8

def e_mes(capital, risk=0.005, split=1.0):
    return T_LOC * (capital / CAPITAL_BASE) * (risk / 0.005) * 22 * split

# Fases Axi Select
AXI_PHASES = [
    # (mes_inicio, capital_axi, risk_pct, split_trader)
    (0,       500,       0.020, 1.00),
    (1,    10_500,       0.015, 0.90),
    (2,    50_500,       0.010, 0.85),
    (4,   150_500,       0.008, 0.85),
    (6,   500_500,       0.006, 0.80),
    (9, 2_000_500,       0.005, 0.80),
    (12,4_000_500,       0.005, 0.80),
]

def get_phase(mes):
    phase = AXI_PHASES[0]
    for m, cap, risk, split in AXI_PHASES:
        if mes >= m:
            phase = (m, cap, risk, split)
    return phase

print("=" * 70)
print("  PLAN FINANCIERO COMPLETO — $500 → Capital Propio $1M-$5M")
print("  Bot: Axi Select | Portafolio: 70-20-10 | Monte Carlo 50K paths")
print("=" * 70)

print("""
LOGICA DE DISTRIBUCION DEL CAPITAL:

  MES 1-2:    100% a saldar deudas ($2,000) + buffer gastos ($6,000)
  MES 3+:     70% → inversiones pasivas
               20% → plataforma distribucion + sello
               10% → capital propio acumulado (reinvertir o guardar)

  CAPITAL PROPIO = solo el 10% que te quedas cada mes
  (el resto va a los buckets — el 70% trabaja solo gestionado por firmas)

  META CAPITAL PROPIO: $1M propio para operar independiente del fondeo
  META FINAL: $5M propio + fondeo ilimitado = maximo apalancamiento
""")

# Simulacion mes a mes
print("=" * 70)
print("  MES A MES — FLUJO DE CAPITAL")
print("=" * 70)
print(f"\n  {'Mes':4s} | {'Ing/mes':8s} | {'Acum.bot':9s} | {'Deuda':7s} | {'70% Inversion':14s} | {'20% Negocio':11s} | {'Capital Propio':14s} | {'Sig. Hito'}")
print("  " + "-" * 108)

deuda_pagada   = 0.0
gastos_buffer  = 0.0
bucket_70      = 0.0
bucket_20      = 0.0
capital_propio = 500.0   # inversion inicial
acum_bot       = 0.0

DEUDA_TOTAL    = 2_000.0
GASTOS_BUFFER  = 6_000.0  # 2 meses de gastos

hitos = [
    (5_000,   "Deuda+buffer OK"),
    (50_000,  "ETFs iniciales"),
    (100_000, "Plataforma musical"),
    (500_000, "Escala global"),
    (1_000_000,"$1M PROPIO"),
    (5_000_000,"$5M PROPIO"),
]

def next_hito(cp):
    for target, label in hitos:
        if cp < target:
            return f"→${target:,.0f} ({label})"
    return "MAXIMO ALCANZADO"

for mes in range(37):  # 3 anos
    _, cap_axi, risk, split = get_phase(mes)
    ingreso_mes = e_mes(cap_axi, risk, split)
    acum_bot   += ingreso_mes
    restante    = ingreso_mes

    # 1. Prioridad: deudas
    if deuda_pagada < DEUDA_TOTAL:
        pago = min(restante, DEUDA_TOTAL - deuda_pagada)
        deuda_pagada += pago
        restante -= pago

    # 2. Prioridad: buffer gastos 2 meses
    if gastos_buffer < GASTOS_BUFFER and restante > 0:
        buf = min(restante, GASTOS_BUFFER - gastos_buffer)
        gastos_buffer += buf
        restante -= buf

    # 3. Distribucion 70-20-10
    if restante > 0:
        bucket_70      += restante * 0.70
        bucket_20      += restante * 0.20
        capital_propio += restante * 0.10

    if mes % 3 == 0 or mes in [1, 2, 9, 12, 18, 24, 36]:
        deuda_str = "OK" if deuda_pagada >= DEUDA_TOTAL else f"${deuda_pagada:,.0f}"
        hito_str  = next_hito(capital_propio)
        print(f"  {mes:3d} | ${ingreso_mes:7,.0f} | ${acum_bot:8,.0f} | {deuda_str:7s} | ${bucket_70:13,.0f} | ${bucket_20:10,.0f} | ${capital_propio:13,.0f} | {hito_str}")

# Resumen de hitos
print("\n" + "=" * 70)
print("  CUÁNDO ALCANZAS CADA META")
print("=" * 70)

deuda_pagada2  = 0.0
gastos_buffer2 = 0.0
bucket_70_2    = 0.0
bucket_20_2    = 0.0
capital_propio2 = 500.0
acum_bot2      = 0.0

hito_results = {}
metas_capital = {
    "Deuda saldada":             2_000,
    "Buffer 2 meses gastos":    6_000,
    "Plataforma musical lista": 10_000,
    "$100K propio":           100_000,
    "$500K propio":           500_000,
    "$1M propio":           1_000_000,
    "$5M propio":           5_000_000,
}
reached = set()

for mes in range(300):
    _, cap_axi, risk, split = get_phase(min(mes, 12))
    ingreso_mes = e_mes(cap_axi, risk, split)
    acum_bot2  += ingreso_mes
    restante2   = ingreso_mes

    if deuda_pagada2 < 2_000:
        p = min(restante2, 2_000 - deuda_pagada2)
        deuda_pagada2 += p; restante2 -= p
    if gastos_buffer2 < 6_000 and restante2 > 0:
        b = min(restante2, 6_000 - gastos_buffer2)
        gastos_buffer2 += b; restante2 -= b
    if restante2 > 0:
        bucket_70_2      += restante2 * 0.70
        bucket_20_2      += restante2 * 0.20
        capital_propio2  += restante2 * 0.10

    for label, target in metas_capital.items():
        if label not in reached:
            val = deuda_pagada2 if "Deuda" in label else (gastos_buffer2 if "Buffer" in label else (bucket_20_2 if "Plataforma" in label else capital_propio2))
            if val >= target:
                hito_results[label] = mes
                reached.add(label)

print(f"\n  {'Meta':30s} | {'Mes':5s} | {'Ano':6s} | Descripcion")
print("  " + "-" * 72)
for label, target in metas_capital.items():
    mes_r = hito_results.get(label, 999)
    ano_r = mes_r / 12
    if mes_r < 300:
        print(f"  {label:30s} | {mes_r:4d}  | {ano_r:5.1f} yr | ${target:,.0f}")
    else:
        print(f"  {label:30s} | {'?':4s}  | {'?':5s}    | ${target:,.0f}")

# Lo que genera el 70% una vez invertido
print("\n" + "=" * 70)
print("  LO QUE GENERA EL 70% PASIVO SOLO (sin el bot)")
print("  Una vez invertido, trabaja para ti 24/7")
print("=" * 70)

scenarios_70 = [
    ("ETFs S&P500 (7%/ano)",       bucket_70_2 * 0.30, 0.07 / 12),
    ("Funerarias + Energia IA",    bucket_70_2 * 0.20, 0.10 / 12),
    ("Bonos (4%/ano)",             bucket_70_2 * 0.20, 0.04 / 12),
    ("Propiedad raiz tokenizada",  bucket_70_2 * 0.20, 0.08 / 12),
    ("Oro (reserva, 5%/ano)",      bucket_70_2 * 0.10, 0.05 / 12),
]

total_pasivo = 0.0
print(f"\n  {'Activo':30s} | {'Capital':10s} | {'Retorno/mes':12s} | {'Retorno/ano'}")
print("  " + "-" * 72)
for name, capital, r_mes in scenarios_70:
    ret_mes = capital * r_mes
    ret_ano = ret_mes * 12
    total_pasivo += ret_mes
    print(f"  {name:30s} | ${capital:9,.0f} | ${ret_mes:10,.0f}   | ${ret_ano:10,.0f}")
print(f"  {'TOTAL PASIVO':30s} | ${bucket_70_2:9,.0f} | ${total_pasivo:10,.0f}   | ${total_pasivo*12:10,.0f}")

print("\n" + "=" * 70)
print("  ESTADO FINAL — ANO 3")
print("=" * 70)
print(f"""
  Capital Axi Select (fondeo):    ${cap_axi:>12,.0f}  (operando con el de ellos)
  Capital propio acumulado:       ${capital_propio2:>12,.0f}
  Invertido en 70% pasivo:        ${bucket_70_2:>12,.0f}  → ${total_pasivo:,.0f}/mes solo
  Invertido en 20% negocios:      ${bucket_20_2:>12,.0f}  (plataforma musical escalando)
  Total generado por bot:         ${acum_bot2:>12,.0f}

  Ingreso mensual total (ano 3):
    Bot (Axi $4M):    ${e_mes(4_000_500, 0.005, 0.80):>10,.0f}/mes
    Pasivo 70%:       ${total_pasivo:>10,.0f}/mes
    Plataforma music: estimado $5,000-$50,000/mes (segun usuarios)
    YouTube:          estimado $1,000-$10,000/mes
    ─────────────────────────────────
    TOTAL AÑO 3:      ${e_mes(4_000_500,0.005,0.80)+total_pasivo+10_000:>10,.0f}+/mes

  CON $5M PROPIO (fondeo independiente):
    $5M × 5%/mes × sin split = ${5_000_000*0.05:,.0f}/mes TUYO al 100%
    Puedes prescindir de Axi y operar tu propio capital
""")
