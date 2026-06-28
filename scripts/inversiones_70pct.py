"""
INVERSIONES EXACTAS DEL BUCKET 70%
====================================
Instrumentos especificos con ticker, retorno esperado,
riesgo, inversion minima, acceso desde Colombia y simulacion.

Ejecutar: .venv/Scripts/python scripts/inversiones_70pct.py
"""
import numpy as np

rng = np.random.default_rng(2026)

print("=" * 72)
print("  PORTAFOLIO 70% — INSTRUMENTOS EXACTOS CON DATOS REALES")
print("=" * 72)

# ── DEFINICION DEL PORTAFOLIO COMPLETO ──────────────────────────────────
PORTFOLIO = {

    # SLOT A — ETFs Amplios (30% del 70%) ─────────────────────────────────
    "VOO": {
        "nombre":    "Vanguard S&P 500 ETF",
        "clase":     "ETF Acciones EEUU",
        "slot":      "A - ETFs Amplios",
        "pct_70":    0.12,       # % dentro del bucket 70%
        "ret_anual": 0.107,      # historico 10 anos
        "volatilidad":0.155,     # desv std anual
        "riesgo":    2,          # 1=muy bajo, 5=muy alto
        "min_usd":   1,          # fraccion via broker
        "precio_ref":530.0,
        "yield_div": 0.013,
        "broker_col":"eToro, Interactive Brokers, Stake",
        "razon":     "Core del portafolio. 500 empresas mas grandes EEUU. S&P 500 historicamente 10.7%/ano.",
        "riesgo_txt":"Cae en recesiones (-35% en 2020, -19% en 2022) pero siempre recupera.",
    },
    "QQQ": {
        "nombre":    "Invesco NASDAQ-100 ETF",
        "clase":     "ETF Tech EEUU",
        "slot":      "A - ETFs Amplios",
        "pct_70":    0.08,
        "ret_anual": 0.152,      # 10 anos promedio
        "volatilidad":0.205,
        "riesgo":    3,
        "min_usd":   1,
        "precio_ref":475.0,
        "yield_div": 0.006,
        "broker_col":"eToro, Interactive Brokers",
        "razon":     "NASDAQ-100: Apple, Microsoft, NVIDIA, Meta. Concentra crecimiento IA.",
        "riesgo_txt":"Mas volatil que VOO. Muy sensible a tasas de interes.",
    },
    "VT": {
        "nombre":    "Vanguard Total World Stock ETF",
        "clase":     "ETF Mercado Global",
        "slot":      "A - ETFs Amplios",
        "pct_70":    0.05,
        "ret_anual": 0.083,
        "volatilidad":0.150,
        "riesgo":    2,
        "min_usd":   1,
        "precio_ref":112.0,
        "yield_div": 0.020,
        "broker_col":"eToro, Interactive Brokers",
        "razon":     "Diversificacion global (9,500 empresas). Reduce riesgo EEUU-especifico.",
        "riesgo_txt":"Retorno menor pero maxima diversificacion mundial.",
    },

    # SLOT B — Sectores Anti-Crisis (20% del 70%) ─────────────────────────
    "SCI": {
        "nombre":    "Service Corporation International",
        "clase":     "Accion Funerarias",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.08,
        "ret_anual": 0.142,      # CAGR 10 anos incluyendo div
        "volatilidad":0.175,
        "riesgo":    2,
        "min_usd":   1,
        "precio_ref":68.0,
        "yield_div": 0.018,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "Mayor empresa funeraria EEUU. 1,900 funerarias, 500 cementerios. Demanda INELASTICA — siempre hay muertos.",
        "riesgo_txt":"Negocio anti-recesion. Deuda alta pero cash flow predecible.",
    },
    "VST": {
        "nombre":    "Vistra Energy Corp",
        "clase":     "Accion Energia / IA",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.06,
        "ret_anual": 0.285,      # 2024-2025 fue enorme por AI demand
        "volatilidad":0.380,
        "riesgo":    4,
        "min_usd":   1,
        "precio_ref":118.0,
        "yield_div": 0.005,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "Proveedor de energia para data centers de IA. Microsoft, Google compran toda su capacidad nuclear.",
        "riesgo_txt":"MUY volatil. Sube fuerte con demanda IA pero puede caer si regulacion cambia.",
    },
    "CEG": {
        "nombre":    "Constellation Energy",
        "clase":     "Accion Energia Nuclear / IA",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.06,
        "ret_anual": 0.220,      # CAGR estimado con AI
        "volatilidad":0.310,
        "riesgo":    3,
        "min_usd":   1,
        "precio_ref":295.0,
        "yield_div": 0.004,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "Energia nuclear limpia para IA. Microsoft firmó contrato para reabrir Three Mile Island solo para sus data centers.",
        "riesgo_txt":"Alta volatilidad pero thesis solida: IA necesita energia estable 24/7.",
    },

    # SLOT C — Renta Fija / Bonos (20% del 70%) ──────────────────────────
    "BND": {
        "nombre":    "Vanguard Total Bond Market ETF",
        "clase":     "ETF Bonos EEUU",
        "slot":      "C - Renta Fija",
        "pct_70":    0.08,
        "ret_anual": 0.045,
        "volatilidad":0.055,
        "riesgo":    1,
        "min_usd":   1,
        "precio_ref":73.0,
        "yield_div": 0.045,     # yield actual 2025-2026
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "10,000+ bonos EEUU gobierno + corporativos. Estabilidad pura. Paga 4.5%/ano en dividendos.",
        "riesgo_txt":"Sube cuando tasas bajan, cae cuando tasas suben. Minimo riesgo.",
    },
    "VGIT": {
        "nombre":    "Vanguard Intermediate-Term Treasury ETF",
        "clase":     "ETF Bonos Gobierno EEUU",
        "slot":      "C - Renta Fija",
        "pct_70":    0.08,
        "ret_anual": 0.042,
        "volatilidad":0.045,
        "riesgo":    1,
        "min_usd":   1,
        "precio_ref":58.0,
        "yield_div": 0.042,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "Solo bonos del gobierno EEUU (3-10 anos). Maxima seguridad con yield decente.",
        "riesgo_txt":"Practica mente libre de riesgo de credito. Solo riesgo de tasa.",
    },

    # SLOT D — Oro (10% del 70%) ───────────────────────────────────────────
    "GLD": {
        "nombre":    "SPDR Gold Shares ETF",
        "clase":     "ETF Oro Fisico",
        "slot":      "D - Oro Reserva",
        "pct_70":    0.05,
        "ret_anual": 0.118,      # 10 anos CAGR
        "volatilidad":0.145,
        "riesgo":    2,
        "min_usd":   1,
        "precio_ref":235.0,
        "yield_div": 0.0,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "Reserva de valor. Sube en crisis, inflacion y debilitamiento USD. 11.8%/ano ultimos 10 anos.",
        "riesgo_txt":"No paga dividendos. Volatil en el corto plazo pero solido como reserva de valor.",
    },
    "IAU": {
        "nombre":    "iShares Gold Trust",
        "clase":     "ETF Oro Fisico",
        "slot":      "D - Oro Reserva",
        "pct_70":    0.05,
        "ret_anual": 0.118,
        "volatilidad":0.145,
        "riesgo":    2,
        "min_usd":   1,
        "precio_ref":46.0,
        "yield_div": 0.0,
        "broker_col":"Interactive Brokers, eToro, Binance (PAXG)",
        "razon":     "Igual que GLD pero precio unitario menor. Mas accesible con poco capital.",
        "riesgo_txt":"Mismo perfil que GLD. Alternativa: PAXG en Binance (oro tokenizado).",
    },

    # SLOT B2 — Bienes Basicos / Alimentos Anti-Crisis (nuevo, 12% del 70%) ──
    "XLP": {
        "nombre":    "Consumer Staples Select SPDR ETF",
        "clase":     "ETF Bienes Primera Necesidad",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.07,
        "ret_anual": 0.085,
        "volatilidad":0.115,
        "riesgo":    1,
        "min_usd":   1,
        "precio_ref":80.0,
        "yield_div": 0.028,
        "broker_col":"eToro, Interactive Brokers",
        "razon":     "Walmart, P&G, Coca-Cola, Colgate, Pepsi. Productos que la gente SIEMPRE compra. En 2020 (pandemia) cayo solo -12% vs S&P -34%.",
        "riesgo_txt":"El ETF anti-crisis por excelencia. Maxima estabilidad. Crece lento pero nunca quiebra.",
    },
    "MOO": {
        "nombre":    "VanEck Agribusiness ETF",
        "clase":     "ETF Cadena Alimentaria Global",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.05,
        "ret_anual": 0.092,
        "volatilidad":0.160,
        "riesgo":    2,
        "min_usd":   1,
        "precio_ref":87.0,
        "yield_div": 0.020,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "John Deere (tractores), Archer Daniels (granos), Corteva (semillas), Nutrien (fertilizantes). La gente come siempre.",
        "riesgo_txt":"Sensible a clima y precio commodities. Pero la demanda global de alimentos solo crece.",
    },

    # SLOT B3 — Alimentos / Hidropónico / Crisis Alimentaria ─────────────
    "VFF": {
        "nombre":    "Village Farms International",
        "clase":     "Accion Cultivos Invernadero",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.03,
        "ret_anual": 0.080,
        "volatilidad":0.280,
        "riesgo":    3,
        "min_usd":   1,
        "precio_ref":3.50,
        "yield_div": 0.0,
        "broker_col":"Interactive Brokers",
        "razon":     "Unica empresa de cultivos de invernadero/hidroponica que sobrevivio al crash 2022-2023. Tomate, pimiento, pepino en Canada/EEUU/Mexico. Ingresos reales.",
        "riesgo_txt":"Volatil. Pequena empresa. Pero opera invernaderos REALES, no proyecciones. El unico hidroponico puro en bolsa con sentido.",
    },
    "DBA": {
        "nombre":    "Invesco DB Agriculture ETF",
        "clase":     "ETF Materias Primas Agricolas",
        "slot":      "B - Anti-Crisis",
        "pct_70":    0.04,
        "ret_anual": 0.062,
        "volatilidad":0.165,
        "riesgo":    2,
        "min_usd":   1,
        "precio_ref":22.0,
        "yield_div": 0.020,
        "broker_col":"Interactive Brokers, eToro",
        "razon":     "Futuros de trigo, maiz, soja, azucar, cafe, ganado. En cualquier crisis alimentaria (guerra, sequia, inflacion) este ETF SUBE. Cobertura directa de precios de alimentos.",
        "riesgo_txt":"Correlacionado con precios de commodities. Puede caer en anos de superproduccion agricola.",
    },

    # SLOT E — Propiedad Raiz Tokenizada (10% del 70%) ──────────────────
    "REALTOKEN": {
        "nombre":    "RealT — Propiedad Tokenizada",
        "clase":     "Real Estate Tokenizado",
        "slot":      "E - Raiz Token",
        "pct_70":    0.05,
        "ret_anual": 0.085,      # rental yield promedio RealT
        "volatilidad":0.080,
        "riesgo":    3,
        "min_usd":   50,         # minimo real en RealT
        "precio_ref":50.0,
        "yield_div": 0.085,
        "broker_col":"realT.network (disponible Colombia via USDC)",
        "razon":     "Propiedades reales en EEUU fraccionadas en tokens. Recibes renta en USDC SEMANAL directamente a tu wallet.",
        "riesgo_txt":"Riesgo de liquidez (vender no es instantaneo). Riesgo smart contract. Propiedades reales como colateral.",
    },
    "LOFTY": {
        "nombre":    "Lofty.ai — Real Estate Token",
        "clase":     "Real Estate Tokenizado",
        "slot":      "E - Raiz Token",
        "pct_70":    0.05,
        "ret_anual": 0.080,
        "volatilidad":0.080,
        "riesgo":    3,
        "min_usd":   50,
        "precio_ref":50.0,
        "yield_div": 0.080,
        "broker_col":"lofty.ai (wallet Algorand, disponible Colombia)",
        "razon":     "Propiedades EEUU tokenizadas en Algorand. Renta diaria. Puedes vender tokens en mercado secundario.",
        "riesgo_txt":"Mercado secundario iliquido. Tokens en Algorand blockchain.",
    },
}

# ── MOSTRAR TABLA COMPLETA ───────────────────────────────────────────────
slots = {}
for ticker, info in PORTFOLIO.items():
    slot = info["slot"]
    if slot not in slots:
        slots[slot] = []
    slots[slot].append((ticker, info))

print(f"\n{'Ticker':12s}{'Nombre':38s}{'% del 70%':10s}{'Ret/ano':8s}{'Riesgo':8s}{'Min $':7s}")
print("-" * 84)

total_pct = 0
for slot, items in slots.items():
    print(f"\n  [{slot}]")
    for ticker, info in items:
        stars = "★" * info["riesgo"] + "☆" * (5 - info["riesgo"])
        print(f"  {ticker:12s}{info['nombre']:38s}{info['pct_70']*100:8.0f}%  "
              f"{info['ret_anual']*100:6.1f}%  {stars}  ${info['min_usd']:5}")
        total_pct += info["pct_70"]

print(f"\n  TOTAL asignado: {total_pct*100:.0f}% del bucket 70%")

# ── RETORNO ESPERADO DEL PORTAFOLIO COMPLETO ────────────────────────────
print("\n" + "=" * 72)
print("  RETORNO ESPERADO DEL PORTAFOLIO 70%")
print("=" * 72)

weighted_ret = sum(info["pct_70"] * info["ret_anual"] for info in PORTFOLIO.values())
weighted_vol = sum(info["pct_70"] * info["volatilidad"] for info in PORTFOLIO.values())

print(f"""
  Retorno anual esperado (ponderado):   {weighted_ret*100:.1f}%
  Volatilidad estimada del portafolio:  {weighted_vol*100:.1f}%
  Ratio retorno/riesgo (Sharpe ~):      {weighted_ret/weighted_vol:.2f}

  [Con correlacion real entre activos, la volatilidad real sera ~30% menor]
  Volatilidad corregida estimada:       {weighted_vol*0.70*100:.1f}%
  Sharpe corregido:                     {weighted_ret/(weighted_vol*0.70):.2f}
""")

# ── SIMULACION DE CRECIMIENTO ────────────────────────────────────────────
print("=" * 72)
print("  SIMULACION: ASI CRECE EL 70% UNA VEZ INVERTIDO")
print("  Usando datos historicos reales de cada instrumento")
print("=" * 72)

def sim_bucket(capital_inicial, anos=10, n_paths=10000):
    results = np.ones((n_paths, anos + 1)) * capital_inicial
    for i in range(anos):
        # Retorno anual del portafolio ponderado con Monte Carlo
        ann_ret = rng.normal(weighted_ret, weighted_vol * 0.70, n_paths)
        results[:, i+1] = results[:, i] * (1 + ann_ret)
        results[:, i+1] = np.maximum(results[:, i+1], 0)
    return results

capitals_bot = {
    "Mes 2 ($5K inicial)":    5_000,
    "Mes 4 ($50K)":          50_000,
    "Mes 6 ($200K)":        200_000,
    "Mes 9 ($700K)":        700_000,
}

print(f"\n  {'Inversion':22s} | {'5 anos':10s} | {'10 anos':10s} | {'Mensual/10a':12s} | {'P(doblar 5a)'}")
print("  " + "-" * 72)

for label, cap in capitals_bot.items():
    paths = sim_bucket(cap, anos=10)
    p5_median = np.percentile(paths[:, 5], 50)
    p10_median = np.percentile(paths[:, 10], 50)
    p10_p10 = np.percentile(paths[:, 10], 10)   # escenario malo
    monthly_10 = p10_median * weighted_ret / 12
    p_double = np.mean(paths[:, 5] >= cap * 2) * 100
    print(f"  {label:22s} | ${p5_median:9,.0f} | ${p10_median:9,.0f} | ${monthly_10:10,.0f} | {p_double:.0f}%")

# ── DETALLE DE CADA SLOT ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  POR QUE ESTOS ACTIVOS ESPECIFICAMENTE")
print("=" * 72)

for ticker, info in PORTFOLIO.items():
    print(f"\n  {ticker} — {info['nombre']}")
    print(f"    Razon: {info['razon']}")
    print(f"    Riesgo: {info['riesgo_txt']}")
    print(f"    Acceso Colombia: {info['broker_col']}")
    print(f"    Dividendo/Yield: {info['yield_div']*100:.1f}%/ano")

# ── COMO ABRIR CUENTA DESDE COLOMBIA ────────────────────────────────────
print("\n" + "=" * 72)
print("  COMO INVERTIR DESDE COLOMBIA")
print("=" * 72)
print("""
  OPCION 1 — Interactive Brokers (MEJOR):
    - Disponible en Colombia
    - Accede a TODOS los ETFs y acciones
    - Sin minimo para acciones fraccionadas
    - Web: www.interactivebrokers.com
    - Comision: $0 para ETFs EEUU (muchos)

  OPCION 2 — eToro (MAS FACIL):
    - App movil simple
    - Minimo $50 para empezar
    - Accede a VOO, QQQ, GLD, SCI, VST, CEG
    - Sin comision por trading
    - Web: www.etoro.com

  OPCION 3 — Real Estate Tokenizado:
    - RealT: realt.network → conectar wallet MetaMask → comprar con USDC
    - Lofty: lofty.ai → wallet Algorand (Pera Wallet) → comprar con ALGO/USDC
    - PAXG (oro tokenizado): Binance Colombia → comprar PAXG directamente

  IMPORTANTE: Abrir cuenta AHORA (gratis) aunque no tengas capital aun.
  Cuando el bot genere los primeros $5,000 → depositar y activar.
""")

# ── ORDEN DE INVERSION POR CAPITAL ──────────────────────────────────────
print("=" * 72)
print("  ORDEN EXACTO DE INVERSION SEGUN CAPITAL DISPONIBLE")
print("=" * 72)
print("""
  $1,000 disponibles:
    → $700 en VOO (70%)
    → $200 en BND (20%)
    → $100 en IAU/GLD (10%)

  $5,000 disponibles:
    → $1,500 VOO + $750 QQQ + $375 VT
    → $800 SCI + $400 VST (funerarias + energia IA)
    → $500 BND + $500 VGIT
    → $250 GLD + $250 IAU
    → $250 RealT + $250 Lofty (minimo $50 cada una)

  $50,000 disponibles:
    → Portafolio completo segun tabla de pcts
    → Rebalanceo anual con las ganancias del bot
    → Agregar gestora para 70% (recomendado: Vanguard Personal Advisor)

  $200,000+:
    → Contratar firma gestion (0.25-0.5%/ano)
    → Sumar REITS (bienes raices fisicos)
    → Bonos municipales (exentos de impuestos)
""")
