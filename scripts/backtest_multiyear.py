"""
BACKTEST MULTI-ANUAL — 8 DIMENSIONES | ~16 años H1 (MT5 real) + 10 años D1
===============================================================
Las 8 dimensiones del mercado:
  DIM 1 — Temporal: borde varía por año/trimestre/mes/hora
  DIM 2 — Régimen de volatilidad: alto (2020,2022) vs bajo (2017,2024)
  DIM 3 — Régimen de tendencia: trending vs choppy vs lateral
  DIM 4 — Sesión: qué horas UTC dan mejor WR históricamente
  DIM 5 — Par: cuál da mejor edge por régimen
  DIM 6 — Riesgo: Kelly fraction óptima con 10 años de datos
  DIM 7 — Salida: partial TP óptimo (0.8R, 1.0R, 1.5R, 2.0R)
  DIM 8 — Correlación: efecto portafolio real entre pares

Monte Carlo: 100,000 simulaciones con distribución empírica real.

2026-07-25: H1 pasó de yfinance (limitado a <730 dias por su propia API,
independiente de cuanta historia real exista) a MT5 directo (mismo broker
Axi que opera en vivo) -- confirmado por consulta real: EURUSD/USDCAD/
NZDUSD/USDCHF/EURAUD/GBPCAD tienen H1 real desde 2010-06 (~16.1 años), no
solo 2. NAS100 se queda en yfinance (no es de los 6 pares activos, y los
indices en MT5 tienen convenciones de sesion distintas).
"""
import sys, os, warnings, json, heapq
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import yfinance as yf
import MetaTrader5 as mt5

from smc.momentum import MomentumIndicators
from smc.bill_williams import BillWilliamsIndicators
from smc.liquidity_sweep import check_setup as silver_bullet_check

REALISTIC_SIGNAL = os.environ.get("REALISTIC_SIGNAL", "0") == "1"
# 2026-08-30: motor de señal REAL en vez de la aproximación smc_signal()
# (suma de puntos EMA/BOS simplificada). Reimplementa fielmente, importando
# las clases reales en vivo (no reinventadas): estructura de swing points
# (smc/structure.py), Order Blocks + FVG (smc/orderblocks.py), y el
# scoring real de core/decision_filter.py (SMC 30pts + ML 10pts + Riesgo
# 25pts -- Sentimiento esta hardcodeado a 0 en vivo, smc/sentiment.py, no
# se aproxima nada ahi). NO porta las capas de ajuste dinamico de threshold
# (kill-zone multiplier, H4-triple-confirm, AutonomousLearner) -- serian
# refinamientos menores sobre este nucleo, documentado como simplificacion
# conocida en SESION_ACTUAL.md.
if REALISTIC_SIGNAL:
    from smc.structure import MarketStructure
    from smc.orderblocks import OrderBlockDetector, FVGDetector
    from smc.ml_predictor import MLPredictor
    from core.session_manager import session_score
    from agents.eight_dim_agent import EightDimensionAgent, _CORR_GROUPS, _REGIME_MULT
    from datetime import timezone as _tz
    _ml_predictor_real = MLPredictor()
    _eight_dim_real = EightDimensionAgent()

# 2026-08-30 (noche 2): motor de señal NUEVO desde cero, NO una variante mas
# de SMC/BOS/CHoCH -- el diagnostico de esta sesion (real_signal(), 16 años)
# confirmo que la ESCASEZ de señales (102-163 trades en 16 años, 1.7-2.4% de
# frecuencia) es el cuello de botella real, no la calidad (WR~50-60%, avg
# win > avg loss). Estrategia Donchian breakout + filtro de tendencia + ATR
# SL/TP -- genera señales por definicion mucho mas seguido (cada ruptura de
# canal de N periodos), probada y documentada (Turtle Trading) en vez de
# requerir la rara confluencia FVG/OB-OTE/BOS-desplazamiento de SMC.
STRATEGY_MODE = os.environ.get("STRATEGY_MODE", "REAL" if REALISTIC_SIGNAL else "SMC")
DONCHIAN_N       = int(os.environ.get("DONCHIAN_N", "20"))
ATR_MULT_SL_BO   = float(os.environ.get("ATR_MULT_SL_BO", "2.0"))
# 2026-08-31: multiplicador de SL POR PAR (nuevo, distinto al global de
# arriba) -- hipotesis a probar: los pares con spread real mas ancho
# (NZDUSD=9.3, USDCHF=7.7, EURAUD=9.8 pips medidos en vivo) podrian
# necesitar un SL mas amplio para que el spread pese menos relativamente,
# mientras que los de spread chico (EURUSD=2.5, USDCAD=4.9) podrian
# aprovechar un SL mas ajustado para maximizar frecuencia. Formato:
# "EURUSD:0.6,USDCAD:0.75,NZDUSD:1.0,USDCHF:1.0,EURAUD:1.0,GBPCAD:1.5"
_EXCLUDE_WEEKDAYS = {int(x) for x in os.environ.get("EXCLUDE_WEEKDAYS", "").split(",") if x.strip()}  # 0=Lunes..4=Viernes
_PER_PAIR_SL_RAW = os.environ.get("PER_PAIR_SL_MULT", "")
PER_PAIR_SL_MULT = {}
if _PER_PAIR_SL_RAW:
    for _kv in _PER_PAIR_SL_RAW.split(","):
        _k, _v = _kv.split(":")
        PER_PAIR_SL_MULT[_k.strip().upper()] = float(_v)
RR_MULT_BO       = float(os.environ.get("RR_MULT_BO", "2.5"))
TREND_FILTER_BO  = os.environ.get("TREND_FILTER_BO", "0") == "1"
THR_BREAKOUT     = float(os.environ.get("THR_BREAKOUT", "0"))
# 2026-09-02: motor MEANREV (reversion a la media) -- enfoque opuesto al
# breakout, pedido explicitamente por el usuario tras que el barrido de
# parametros del motor breakout se estancara en ~44% (18 variables
# probadas). RSI extremo + toque de banda de Bollinger, target = banda
# media (la tesis real de reversion, no un RR fijo arbitrario).
RSI_OVERSOLD     = float(os.environ.get("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT   = float(os.environ.get("RSI_OVERBOUGHT", "70"))
ATR_MULT_SL_MR   = float(os.environ.get("ATR_MULT_SL_MR", "1.0"))
_EXTERNAL_ENTRY  = REALISTIC_SIGNAL or STRATEGY_MODE in ("BREAKOUT", "MEANREV")  # entry/sl/tp vienen ya calculados del motor de señal, no se recomputan mas abajo

print("=" * 72)
print("  BACKTEST MULTI-ANUAL — 8 DIMENSIONES")
print("  H1: ~16 años (MT5 real) | D1: 10 años | Monte Carlo: 100,000 sims")
print("=" * 72)

MAX_OPEN_TEST = int(os.environ.get("MAX_OPEN_TEST", "2"))  # 2026-07-16: parametrizado para comparar 2 vs 3 (pedido por Jose)
REQUIRE_D1 = os.environ.get("REQUIRE_D1", "1") == "1"  # 2026-07-21: medir impacto real de D1-FILTER (ver smc_signal())
REQUIRE_H4 = os.environ.get("REQUIRE_H4", "1") == "1"  # 2026-07-21: medir impacto real de H4-FILTER
PEAK_GUARD_MIN = float(os.environ.get("PEAK_GUARD_MIN", "400"))    # 2026-07-16: recalibracion pedida por Jose
# 2026-08-28: default subido 200->400 -- core/position_guards.py:656 fijo
# PEAK_MIN_USD=400.0 desde 2026-07-24 tras el sweep documentado abajo (400
# gano en las 3 metricas: 44% pass/$4213/mes/Sharpe 0.51 vs 40%/$3360/0.44
# con 200). El default de este script nunca se actualizo, encontrado por
# audit subagent.
PEAK_GUARD_RETRACE = float(os.environ.get("PEAK_GUARD_RETRACE", "0.30"))
# 2026-07-24: STAGNANT/TIME-CLOSE-36H/FRIDAY-CLOSE were NEVER simulated here --
# live data shows they (not TP/SL/PEAK-GUARD) drive 77% of real closes, so every
# conclusion drawn from this backtest before today was measured against a model
# missing the dominant real exit mechanism. Added to match core/position_guards.py
# exactly (values there as of 2026-07-24), parametrized for sweeping.
STAGNANT_HOURS       = float(os.environ.get("STAGNANT_HOURS", "6.0"))  # 2026-08-28: 4.0->6.0, matches core/position_guards.py:707 (commit b950662/dd3157e, 2026-07-24 sweep)
STAGNANT_PEAK_MAX    = float(os.environ.get("STAGNANT_PEAK_MAX", "15.0"))
STAGNANT_GRACE_HOURS = float(os.environ.get("STAGNANT_GRACE_HOURS", "2.0"))
MAX_HOLD_HOURS       = float(os.environ.get("MAX_HOLD_HOURS_TEST", "36.0"))
SWING_MAX_LOSS_ABS   = float(os.environ.get("SWING_MAX_LOSS_TEST", "150.0"))
FRIDAY_CLOSE_HOUR    = float(os.environ.get("FRIDAY_CLOSE_HOUR_TEST", "19"))  # UTC -- matches live;
                           # DEAD_HOURS_UTC below already skips hour 19, so the sim's
                           # earliest reachable Friday close check is hour 20, same as
                           # live effect. Parametrized 2026-07-27: friday_close is 27.6%
                           # of ALL real closes (2nd most common after final_SL) -- testing
                           # whether a later cutoff lets more Thu/Fri trades reach TP
                           # instead of being force-closed regardless of P&L.
CAPITAL = 96_184.0
RISK_PCT = 0.005
MAX_RISK = 275.0  # probado doblar a 550 (2026-07-09): P(pasar Axi)+2.5pp pero
# P(mes<-5%) 6%->16% (casi triplico) -- Axi revienta la cuenta a ese drawdown,
# no vale la pena el intercambio. Revertido a 275.
RR = float(os.environ.get("RR_TEST", "3.0"))  # 2026-07-24: parametrizado -- live data shows only 2.8% of real closes hit the designed TP (77% close via guards instead), testing whether a more reachable RR changes that
PARTIAL_R_TEST = float(os.environ.get("PARTIAL_R_TEST", "0") or 0)
# 2026-08-29: simulacion REAL barra-por-barra de partial-close (a diferencia
# de DIM7 mas abajo, que es una formula analitica aproximada sin respaldo de
# trayectoria de precio real -- ver caveat metodologico en SESION_ACTUAL.md).
# Si >0, cierra 50% del volumen al alcanzar PARTIAL_R_TEST*sl_dist a favor y
# mueve el SL del remanente a breakeven, usando el mismo H/L de barra real
# que ya usa el resto del motor -- incluye la posibilidad de que el remanente
# vuelva a breakeven en la MISMA barra, que es el efecto que un audit de 584
# trades reales encontro y causo desactivar partial-close en vivo (commit
# 5e3ffd5). 0 = desactivado (comportamiento actual en vivo, default).
TRAIL_BE_R_TEST = float(os.environ.get("TRAIL_BE_R_TEST", "0") or 0)
# 2026-08-29: modela el trailing-to-BE que YA EXISTE en vivo (mueve el SL a
# breakeven al alcanzar TRAIL_BE_R_TEST*sl_dist a favor, SIN cerrar volumen,
# a diferencia de PARTIAL_R_TEST) -- el backtest nunca lo modelaba antes.
# 0 = desactivado (backtest histórico de esta sesión, sin trailing).
DAILY_TARGET = 250.0
PAIRS_FOREX = {
    # actualizado 2026-07-09: GBPUSD removido -- auditoria de episodes.db (591 trades
    # reales) lo mostro como el peor par activo (n=147, WR=25.9%, PF=0.53, neto -$887.55).
    # USDCHF/EURAUD/GBPCAD agregados 2026-07-05 tras screening backtest positivo.
    "EURUSD": "EURUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
    "EURAUD": "EURAUD=X",
    "GBPCAD": "GBPCAD=X",
}
_exclude_pairs = {p.strip().upper() for p in os.environ.get("EXCLUDE_PAIRS", "").split(",") if p.strip()}
if _exclude_pairs:  # 2026-08-28: parametrizado para aislar el efecto de un par debil
    # (mismo patron que EXTRA_DEAD_HOURS) -- DIM5 mostro GBPCAD con avg P&L=$15/trade,
    # muy por debajo del resto ($24-$57), candidato a diluir el resultado igual que
    # hizo la hora 15 UTC con avg P&L~$0.
    PAIRS_FOREX = {k: v for k, v in PAIRS_FOREX.items() if k not in _exclude_pairs}
PAIR_NAS = {"NAS100": "^NDX"}
PIP_SZ  = {"EURUSD":0.0001,"GBPUSD":0.0001,"AUDUSD":0.0001,"USDCAD":0.0001,"NZDUSD":0.0001,
           "USDCHF":0.0001,"EURAUD":0.0001,"GBPCAD":0.0001,"NAS100":1.0}
PIP_VAL = {"EURUSD":10.0,"GBPUSD":10.0,"AUDUSD":10.0,"USDCAD":10.0,"NZDUSD":10.0,
           "USDCHF":10.0,"EURAUD":6.6,"GBPCAD":7.1,"NAS100":1.0}
# cap/floor reales de agents/signal_agent.py:_sl_distance() -- compartidos
# entre REALISTIC_SL y REALISTIC_SIGNAL para no duplicar la formula.
SL_CAP_PIPS   = {"EURUSD": 40, "GBPUSD": 40, "USDCAD": 40,
                  "AUDUSD": 35, "NZDUSD": 35, "USDCHF": 35,
                  "EURAUD": 45, "GBPCAD": 50}
SL_FLOOR_PIPS = {"GBPCAD": 25}  # resto = 20 (default)

# 2026-08-30 (noche 2): NINGUNA corrida de esta sesion (ni SMC ni breakout)
# modelo jamas el costo de spread/slippage real -- omision conocida en todo
# el script desde su creacion. Se vuelve critica ahora que el motor breakout
# opera con mucha mas frecuencia (N chico = miles de entradas extra) --
# verificar cuanto edge sobrevive con un spread tipico realista antes de
# confiar en un resultado que depende de frecuencia extrema.
ENABLE_SPREAD_COST = os.environ.get("ENABLE_SPREAD_COST", "0") == "1"
# 2026-08-31 (2da correccion): los valores de abajo (2.5-18.5 pips) venian
# de la cuenta DEMO (Axi-US50-Demo, tipo Standard). Al medir la cuenta REAL
# de Axi Select (60290663, Axi-US51-Live, simbolos con sufijo .sa) los
# spreads reales son 3-11x MAS CHICOS: EURUSD=0.8, USDCAD=1.0, NZDUSD=1.0,
# USDCHF=1.0, EURAUD=1.4, GBPCAD=1.7 (medido 2026-08-31, mt5.symbol_info().spread
# en vivo). GBPCAD en particular pasa de inviable (18.5 pips en demo) a
# spread minimo -- se re-incluye para volver a probar. Ver SESION_ACTUAL.md.
SPREAD_PIPS = {"EURUSD": 0.8, "GBPUSD": 1.0, "AUDUSD": 1.0, "USDCAD": 1.0,
               "NZDUSD": 1.0, "USDCHF": 1.0, "EURAUD": 1.4, "GBPCAD": 1.7,
               "NAS100": 1.5}
# 2026-09-01: SPREAD_PROFILE=DEMO restaura los valores medidos en la cuenta
# Axi-US50-Demo (Standard, 3-11x mas anchos que la cuenta real de arriba) --
# necesario para validar cambios contra lo que el bot en vivo experimenta
# HOY (sigue en la demo, la cuenta real 60290663 aun no esta activada, ver
# SESION_ACTUAL.md seccion "Estado actual y plan acordado"). No pisa el
# dict de arriba (que documenta la cuenta real para cuando se active) --
# solo lo sobreescribe en memoria si se pide explicitamente.
if os.environ.get("SPREAD_PROFILE", "").upper() == "DEMO":
    SPREAD_PIPS = {"EURUSD": 2.5, "GBPUSD": 3.0, "AUDUSD": 2.5, "USDCAD": 4.9,
                   "NZDUSD": 9.3, "USDCHF": 7.7, "EURAUD": 9.8, "GBPCAD": 18.5,
                   "NAS100": 1.5}


def _spread_cost(pair, vol):
    if not ENABLE_SPREAD_COST or vol <= 0:
        return 0.0
    return vol * SPREAD_PIPS.get(pair, 2.0) * PIP_VAL.get(pair, 10.0)

rng = np.random.default_rng(42)

# Toggles para aislar el efecto de cada filtro nuevo 2026-07-09 (diagnostico
# temporal -- ver cual filtro realmente ayuda antes de decidir la config final)
ENABLE_MOMENTUM_FILTERS = False  # desactivado en vivo 2026-07-09 -- ver core/supervisor.py
ENABLE_SILVER_BULLET_GATE = os.environ.get("ENABLE_SILVER_BULLET_GATE", "0") == "1"
# ERA inerte hasta 2026-07-28 -- su kill zone (hora 14 UTC exacta) quedo
# muerta desde que 14 se bloqueo el 2026-07-26 (ver BUG-SB-DEAD-KILLZONE en
# smc/liquidity_sweep.py). Se corrigio la ventana horaria (15,16,20-23 UTC,
# bug real, ese fix se queda) y se probo por primera vez como gate real:
# =1 sobre 16 anios completos -> 0 TRADES, ninguno en ninguna hora, ningun
# año -- el sweep+FVG fresco de liquidity_sweep.py nunca coincide con el
# mismo bar donde smc_signal() genera señal (son dos lecturas de estructura
# independientes con ventanas distintas). Exigir ambas a la vez como AND
# no es "mas selectivo", es una interseccion vacia. Default vuelto a 0
# (informational-only en vivo, ver core/supervisor.py) -- NO usar como gate
# duro sin antes verificar que el trade log no quede vacio.
ENABLE_REGIME_FILTER = False  # probado 2026-07-09: empeoro TODO (P(pasar Axi) 40.3%->31.5%,
# E[mensual] $3395->$1559, Sharpe 0.476->0.20, P(mes<-5%) 6%->21%) -- rechazado
# (ese filtro solo dejaba pasar HIGH+STRONG_TREND, 1 de 9 combos -- demasiado
# restrictivo). EXCLUDE_CHOPPY es mas quirurgico: excluye solo CHOPPY (WR=12%,
# avg -$205/-$213 en los 3 niveles de vol, ~5676 de 25195 trades = 22.5% del
# total) y deja pasar los otros 6 de 9 combos, todos con WR 47-68%.
EXCLUDE_CHOPPY = os.environ.get("EXCLUDE_CHOPPY", "0") == "1"

# ── Data download ──────────────────────────────────────────────────────
print("\n[DATA] Descargando datos historicos...")
print("       H1: MT5 real (hasta ~16 años) para forex | D1: hasta 10 años")

MT5_H1_MAX_BARS = int(os.environ.get("MT5_H1_MAX_BARS", "99999"))  # terminal maxbars cap
_mt5_ok = mt5.initialize()

d1_data = {}
h1_data = {}


def _strip_tz(df):
    # BUG-TZ-MIX-CRASH (2026-07-26): NAS100/yfinance-fallback data comes back
    # tz-aware while the new MT5-sourced pairs are tz-naive (see
    # _mt5_rates_to_df comment) -- mixing the two in pd.DataFrame(dict_of_
    # series) (DIM8 correlation matrix build) raises
    # "Cannot join tz-naive with tz-aware DatetimeIndex". Normalize every
    # yfinance-sourced index to tz-naive so all pairs share one convention.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def _mt5_rates_to_df(rates):
    # Match connectors/metatrader_connector.py::get_ohlcv() exactly: naive
    # pd.to_datetime(unit="s"), NO utc=True. The live bot's own DEAD_HOURS_UTC/
    # kill-zone logic already operates on whatever raw timestamp MT5 returns
    # (broker server time, not necessarily true UTC) -- matching that exactly
    # here means the backtest's hour-based analysis (DIM4, kill zones) uses
    # the identical convention live actually runs on, not a "corrected" one
    # that would silently diverge from real behavior.
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    # 2026-08-30: se mantiene "volume" (antes se descartaba) -- smc/ml_predictor.py
    # (usado por REALISTIC_SIGNAL) lo requiere para _extract_features(). MT5 da
    # tick_volume real (conteo de ticks, no volumen de contratos), mismo dato
    # que usa el bot en vivo via connectors/metatrader_connector.py.
    return df[["open", "high", "low", "close", "volume"]].dropna()


for pair in PAIRS_FOREX:
    try:
        if not _mt5_ok:
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
        mt5.symbol_select(pair, True)
        rates_h1 = mt5.copy_rates_from(pair, mt5.TIMEFRAME_H1, datetime.now(timezone.utc), MT5_H1_MAX_BARS)
        if rates_h1 is None or len(rates_h1) == 0:
            raise RuntimeError(f"copy_rates_from H1 returned nothing: {mt5.last_error()}")
        dh1 = _mt5_rates_to_df(rates_h1)
        h1_data[pair] = dh1

        rates_d1 = mt5.copy_rates_from(pair, mt5.TIMEFRAME_D1, datetime.now(timezone.utc), 4500)
        dd1 = _mt5_rates_to_df(rates_d1) if rates_d1 is not None and len(rates_d1) > 0 else pd.DataFrame()
        d1_data[pair] = dd1

        h1_years = (dh1.index[-1] - dh1.index[0]).days / 365.25 if len(dh1) else 0.0
        d1_years = (dd1.index[-1] - dd1.index[0]).days / 365.25 if len(dd1) else 0.0
        print(f"  {pair}: H1={len(dh1)} bars ({h1_years:.1f} años reales, desde {dh1.index[0].date() if len(dh1) else '?'}) | D1={len(dd1)} bars ({d1_years:.1f} años)")
    except Exception as e:
        print(f"  {pair}: ERROR MT5 ({e}) -- fallback yfinance")
        tk = PAIRS_FOREX[pair]
        _end = datetime.now()
        dh1 = yf.download(tk, start=_end - timedelta(days=700), end=_end, interval="1h", progress=False, auto_adjust=True)
        if isinstance(dh1.columns, pd.MultiIndex):
            dh1.columns = dh1.columns.get_level_values(0)
        dh1.columns = [c.lower() for c in dh1.columns]
        dh1.dropna(inplace=True)
        dh1 = _strip_tz(dh1)
        h1_data[pair] = dh1
        dd1 = yf.download(tk, start=_end - timedelta(days=3650), end=_end, interval="1d", progress=False, auto_adjust=True)
        if isinstance(dd1.columns, pd.MultiIndex):
            dd1.columns = dd1.columns.get_level_values(0)
        dd1.columns = [c.lower() for c in dd1.columns]
        dd1.dropna(inplace=True)
        dd1 = _strip_tz(dd1)
        d1_data[pair] = dd1
        print(f"  {pair} (yfinance fallback): H1={len(dh1)} bars | D1={len(dd1)} bars")

for pair, tk in PAIR_NAS.items():
    try:
        _end = datetime.now()
        dh1 = yf.download(tk, start=_end - timedelta(days=700), end=_end, interval="1h", progress=False, auto_adjust=True)
        if isinstance(dh1.columns, pd.MultiIndex):
            dh1.columns = dh1.columns.get_level_values(0)
        dh1.columns = [c.lower() for c in dh1.columns]
        dh1.dropna(inplace=True)
        dh1 = _strip_tz(dh1)
        h1_data[pair] = dh1
        dd1 = yf.download(tk, start=_end - timedelta(days=3650), end=_end, interval="1d", progress=False, auto_adjust=True)
        if isinstance(dd1.columns, pd.MultiIndex):
            dd1.columns = dd1.columns.get_level_values(0)
        dd1.columns = [c.lower() for c in dd1.columns]
        dd1.dropna(inplace=True)
        dd1 = _strip_tz(dd1)
        d1_data[pair] = dd1
        print(f"  {pair} (yfinance, indice): H1={len(dh1)} bars ({len(dh1)//504:.1f} años efectivos) | D1={len(dd1)} bars ({len(dd1)/252:.1f} años)")
    except Exception as e:
        print(f"  {pair}: ERROR {e}")

if _mt5_ok:
    mt5.shutdown()

# ── Utils ──────────────────────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr14(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def vol_regime(df, idx, lookback=60):
    """Returns 'HIGH', 'NORMAL', 'LOW' volatility regime."""
    if idx < lookback: return "NORMAL"
    w = df.iloc[max(0,idx-lookback):idx+1]
    a = atr14(w)
    cur = a.iloc[-1]
    hist_mean = a.mean()
    hist_std = a.std()
    if pd.isna(cur) or pd.isna(hist_mean): return "NORMAL"
    if cur > hist_mean + 0.7 * hist_std: return "HIGH"
    if cur < hist_mean - 0.5 * hist_std: return "LOW"
    return "NORMAL"

def trend_regime(df, idx):
    """Returns 'STRONG_TREND', 'MILD_TREND', 'CHOPPY'."""
    if idx < 50: return "MILD_TREND"
    w = df.iloc[max(0,idx-50):idx+1]
    c = w["close"]
    e8, e21, e50 = ema(c,8).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1]
    aligned = (e8 > e21 > e50) or (e8 < e21 < e50)
    # ADX proxy via ATR change
    atr_now = atr14(w).iloc[-1]
    atr_old = atr14(df.iloc[max(0,idx-100):idx-50]).mean() if idx > 100 else atr_now
    if aligned and atr_now > atr_old * 0.9:
        return "STRONG_TREND"
    elif aligned:
        return "MILD_TREND"
    return "CHOPPY"

def smc_signal(df, idx):
    """Score 0-100 for BOTH directions independently, return whichever
    qualifies. Returns (signal_dir, score, atr_val).

    2026-07-21: rewritten to decouple signal generation from D1/H4 bias --
    it used to take `bias` as a parameter and only ever score the ONE
    direction it was told to look for (`if bias=="LONG": ...`), so a
    genuine bearish setup during a nominal D1 uptrend was never even
    evaluated -- the backtest could never measure what happens if
    D1-FILTER/H4-FILTER (which gate on real code in core/supervisor.py)
    were relaxed, because the bias was baked into signal generation
    itself rather than applied as a separate post-hoc gate the way the
    live bot actually works (SMC structure generates a real bullish/
    bearish read independently; D1-FILTER/H4-FILTER are separate hard
    checks applied AFTER). See REQUIRE_D1/REQUIRE_H4 at the call site.
    """
    if idx < 60: return "WAIT", 0, 0.0
    w = df.iloc[max(0,idx-80):idx+1]
    if len(w) < 40: return "WAIT", 0, 0.0
    atr_v = atr14(w).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0: return "WAIT", 0, 0.0
    c, h, l = w["close"], w["high"], w["low"]
    e8, e21, e50 = ema(c,8).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1]
    cur = c.iloc[-1]
    # BOS detection
    prev_h = h.iloc[-20:-5].max()
    prev_l = l.iloc[-20:-5].min()
    bos_bull = cur > prev_h and c.iloc[-2] <= prev_h
    bos_bear = cur < prev_l and c.iloc[-2] >= prev_l
    # Order Block (last strong impulse candle)
    body = (c - w["open"]).abs().iloc[-15:]
    ob_strong = body.max() > atr_v * 0.8

    score_long = 0
    if bos_bull: score_long += 35
    if e8 > e21:  score_long += 15
    if e21 > e50: score_long += 12
    if cur > e21: score_long += 10
    if ob_strong: score_long += 12
    if cur > c.iloc[-4]: score_long += 16  # momentum

    score_short = 0
    if bos_bear: score_short += 35
    if e8 < e21:  score_short += 15
    if e21 < e50: score_short += 12
    if cur < e21: score_short += 10
    if ob_strong: score_short += 12
    if cur < c.iloc[-4]: score_short += 16

    if score_long >= 50 and score_long >= score_short:
        return "LONG", min(100, int(score_long * 1.25)), float(atr_v)
    if score_short >= 50 and score_short > score_long:
        return "SHORT", min(100, int(score_short * 1.25)), float(atr_v)
    return "WAIT", 0, float(atr_v)


def _realistic_sl_dist(pair, atr_v):
    """SL real: agents/signal_agent.py:_sl_distance() -- atr*1.5, capado y
    con piso por par. Compartido entre REALISTIC_SL y REALISTIC_SIGNAL."""
    d = atr_v * 1.5
    _cap_p = SL_CAP_PIPS.get(pair)
    if _cap_p is not None:
        d = min(d, _cap_p * PIP_SZ[pair])
    _floor_p = SL_FLOOR_PIPS.get(pair, 20)
    d = max(d, _floor_p * PIP_SZ[pair])
    return d


_RS_STATS = defaultdict(int)  # 2026-08-30: diagnostico temporal de real_signal()
_RECENT_OUTCOMES = defaultdict(list)  # {pair: [(dt, "WIN"/"LOSS"), ...]} ultimos 5,
# construido con el propio historial simulado del backtest (cronologico, sin
# mirar al futuro) en vez de episodes.db real -- alimenta DIM6 (circuit breaker).


def real_signal(w, pair, dt, daily_pnl_so_far, capital, open_pos_list=None, daily_pnl_month=None):
    """Motor de señal REAL (no la aproximación smc_signal()) -- reimplementa
    fielmente core/supervisor.py::_run_smc_lite() + agents/signal_agent.py
    ::evaluate() + core/decision_filter.py::DecisionFilter, importando las
    clases reales (MarketStructure/OrderBlockDetector/FVGDetector/
    MLPredictor) en vez de reinventar su lógica. `w` es la ventana de hasta
    200 velas (igual que en vivo). Devuelve
    (direction, entry, sl, tp, score) o None si no hay setup válido.
    NO porta las capas de ajuste dinámico de threshold (kill-zone
    multiplier, H4-triple-confirm, AutonomousLearner) -- ver caveat en
    SESION_ACTUAL.md.
    """
    _RS_STATS["calls"] += 1
    if len(w) < 60:
        return None
    ms = MarketStructure(w)
    struct = ms.analyze()
    bos_list = ms.detect_bos()
    choch_list = ms.detect_choch()
    ob_det = OrderBlockDetector(w)
    fvg_det = FVGDetector(w)
    bull_obs = ob_det.find_bullish_obs()
    bear_obs = ob_det.find_bearish_obs()
    bull_fvgs = fvg_det.find_bullish_fvg()
    bear_fvgs = fvg_det.find_bearish_fvg()

    current_close = float(w["close"].iloc[-1])
    if current_close <= 0:
        return None

    is_bullish = struct.bias == "bullish"
    is_bearish = struct.bias == "bearish"
    if not (is_bullish or is_bearish) and bos_list:
        last_dir = bos_list[-1].get("direction", "")
        if last_dir == "bullish": is_bullish = True
        elif last_dir == "bearish": is_bearish = True
    if not (is_bullish or is_bearish) and choch_list:
        last_choch = choch_list[-1].get("direction", "")
        if last_choch == "bullish": is_bullish = True
        elif last_choch == "bearish": is_bearish = True
    if not (is_bullish or is_bearish) or (is_bullish and is_bearish):
        return None
    _RS_STATS["has_bias"] += 1
    bias = "bullish" if is_bullish else "bearish"

    has_ob = bool(bull_obs if is_bullish else bear_obs)

    _max_poi_dist = current_close * 0.01
    _dir_fvgs = bull_fvgs if is_bullish else bear_fvgs
    _recent_fvgs = sorted(_dir_fvgs, key=lambda g: g.get("index", 0), reverse=True)
    has_fvg = any(abs(g.get("midpoint", 0) - current_close) <= _max_poi_dist for g in _recent_fvgs[:5])

    has_bos = bool(bos_list)

    _pd_ok = True
    if len(w) >= 50:
        _range_high = float(w["high"].rolling(50).max().iloc[-1])
        _range_low = float(w["low"].rolling(50).min().iloc[-1])
        _range_mid = (_range_high + _range_low) / 2.0
        if is_bullish and current_close > _range_mid: _pd_ok = False
        if is_bearish and current_close < _range_mid: _pd_ok = False

    _dir_bos = [b for b in bos_list if b.get("direction") == bias]
    _has_displacement_bos = bool(_dir_bos) and bool(_dir_bos[-1].get("is_displacement", False))
    _dir_choch = [c for c in choch_list if c.get("direction") == bias]
    has_recent_choch = bool(_dir_choch)

    _recent_obs = sorted(
        (o for o in (bull_obs + bear_obs) if not o.get("mitigated", False)),
        key=lambda o: o.get("index", 0), reverse=True,
    )
    poi_zones = []
    for ob in _recent_obs[:5]:
        zone_mid = (ob.get("zone_high", 0) + ob.get("zone_low", 0)) / 2.0
        if zone_mid > 0 and abs(zone_mid - current_close) <= _max_poi_dist:
            poi_zones.append(ob)
        if len(poi_zones) >= 3:
            break

    _in_ote = False
    _ote_type = "bullish_ob" if is_bullish else "bearish_ob"
    _ote_pois = [p for p in poi_zones if p.get("type") == _ote_type]
    if _ote_pois and len(w) >= 5:
        _poi = _ote_pois[0]
        _ob_idx = _poi.get("index", 0)
        if 0 < _ob_idx < len(w) - 1:
            if is_bullish:
                _swing_low = float(w["low"].iloc[_ob_idx])
                _swing_high = float(w["high"].iloc[_ob_idx:_ob_idx+10].max()) if _ob_idx+10 <= len(w) else float(w["high"].iloc[_ob_idx:].max())
                _ote_low = _swing_high - (_swing_high - _swing_low) * 0.79
                _ote_high = _swing_high - (_swing_high - _swing_low) * 0.62
                _in_ote = _ote_low <= current_close <= _ote_high
            else:
                _swing_high = float(w["high"].iloc[_ob_idx])
                _swing_low = float(w["low"].iloc[_ob_idx:_ob_idx+10].min()) if _ob_idx+10 <= len(w) else float(w["low"].iloc[_ob_idx:].min())
                _ote_low = _swing_low + (_swing_high - _swing_low) * 0.62
                _ote_high = _swing_low + (_swing_high - _swing_low) * 0.79
                _in_ote = _ote_low <= current_close <= _ote_high

    if has_fvg: _RS_STATS["has_fvg"] += 1
    if has_ob and _in_ote: _RS_STATS["ob_in_ote"] += 1
    if has_bos and _has_displacement_bos: _RS_STATS["bos_displacement"] += 1
    if not _pd_ok: _RS_STATS["blocked_pd"] += 1
    has_setup = (is_bullish or is_bearish) and (has_fvg or (has_ob and _in_ote) or (has_bos and _has_displacement_bos)) and _pd_ok
    if not has_setup:
        return None
    _RS_STATS["has_setup"] += 1

    # ── Entrada/SL/TP reales (agents/signal_agent.py::evaluate()) ──
    atr_v = atr14(w).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0:
        return None
    n_confluence = sum([_has_displacement_bos, has_recent_choch, _in_ote, has_fvg])
    tp_mult = 3.0 if n_confluence >= 3 else (2.5 if n_confluence == 2 else 2.0)

    correct_type = "bullish_ob" if is_bullish else "bearish_ob"
    aligned_pois = [p for p in poi_zones if p.get("type") == correct_type]
    poi = aligned_pois[0] if aligned_pois else (poi_zones[0] if poi_zones else None)

    if poi is not None:
        entry = poi["zone_low"] if is_bullish else poi["zone_high"]
    else:
        entry = current_close
    sl_dist = _realistic_sl_dist(pair, atr_v)
    sl = entry - sl_dist if is_bullish else entry + sl_dist
    tp_raw = entry + sl_dist * tp_mult if is_bullish else entry - sl_dist * tp_mult

    # TP ajustado al swing más cercano (agents/signal_agent.py::_nearest_swing)
    tp = tp_raw
    highs50 = w["high"].values[-50:]
    lows50 = w["low"].values[-50:]
    min_tp_dist = sl_dist * 1.5
    max_tp_dist = sl_dist * 3.5
    if is_bullish:
        candidates = [h for h in highs50 if h > entry + min_tp_dist]
        if candidates:
            nearest = min(candidates)
            if nearest <= entry + max_tp_dist:
                tp = round(nearest, 5)
    else:
        candidates = [lo for lo in lows50 if lo < entry - min_tp_dist]
        if candidates:
            nearest = max(candidates)
            if nearest >= entry - max_tp_dist:
                tp = round(nearest, 5)

    # MIN_RR real (core/supervisor.py:116, =4.5): si el RR resultante queda
    # corto, se ajusta el TP para garantizar el minimo -- pieza que faltaba
    # en el primer port, encontrada al revisar por que MAX_OPEN no cambiaba
    # nada (confirma que el TP real casi siempre depende de esto, no solo
    # del snap a swing).
    MIN_RR_REAL = 4.5
    _rr_now = abs(tp - entry) / sl_dist if sl_dist > 0 else 0
    if _rr_now < MIN_RR_REAL:
        _tp_rr = MIN_RR_REAL + 0.1
        tp = round(entry + sl_dist * _tp_rr, 5) if is_bullish else round(entry - sl_dist * _tp_rr, 5)

    # ── Score real (core/decision_filter.py::DecisionFilter) ──
    smc_score = 0
    is_trending = struct.structure_type.value in ("bullish_trend", "bearish_trend")
    if is_trending: smc_score += 10
    if struct.bias == bias: smc_score += 3
    if _dir_bos: smc_score += 8
    elif bos_list: smc_score -= 5
    if _dir_choch: smc_score += 4
    if (bull_obs if is_bullish else bear_obs): smc_score += 5
    if (bull_fvgs if is_bullish else bear_fvgs): smc_score += 5
    smc_score = min(max(smc_score, 0), 30)

    ml_result = _ml_predictor_real.predict(w, bias=bias)
    ml_score = ml_result.score  # ya 0/5/10, sentimiento = 0 siempre en vivo (smc/sentiment.py)

    sess_pts, _ = session_score(pd.Timestamp(dt).to_pydatetime().replace(tzinfo=_tz.utc))
    risk_score = sess_pts
    rr = abs(tp - entry) / abs(entry - sl) if entry != sl else 0
    if rr >= 3.0: risk_score += 9
    elif rr >= 2.5: risk_score += 7
    elif rr >= 2.0: risk_score += 5
    # drawdown health (0-8): usa el progreso real del dia hasta ahora
    _max_daily = capital * 0.04  # Axi Select: max_daily_loss ~4%
    _used_daily = abs(min(daily_pnl_so_far, 0))
    _dd_pct = _used_daily / _max_daily if _max_daily > 0 else 0
    if _dd_pct < 0.25: risk_score += 8
    elif _dd_pct < 0.50: risk_score += 5
    elif _dd_pct < 0.75: risk_score += 2
    risk_score = min(max(risk_score, 0), 25)

    direction = "LONG" if is_bullish else "SHORT"

    # ── Multiplicador de 8 dimensiones (agents/eight_dim_agent.py::analyze()) ──
    # Reutiliza los metodos reales de la clase (no reinventados) salvo 2 ajustes
    # documentados: DIM1 usaba datetime.now() (hora real del sistema, sin
    # sentido en un backtest historico) -- se reimplementa con el mismo
    # criterio pero usando el timestamp REAL de la barra. DIM6 (circuit
    # breaker) lee episodes.db/axi_select_state.json en vivo (estado de
    # cuenta real) -- no se puede replicar sin construir un tracker de
    # resultados cronologico propio; se deja neutral (1.0), documentado como
    # simplificacion conocida (omite posibles bloqueos reales de 3 perdidas
    # seguidas o lock de meta mensual -- el numero resultante podria ser
    # ligeramente MAS optimista que vivo en ese aspecto especifico).
    _wd, _hr = pd.Timestamp(dt).weekday(), pd.Timestamp(dt).hour
    if _wd == 0 and _hr < 12: temporal_mod = 0.85
    elif _wd == 4 and _hr >= 19: temporal_mod = 0.88
    elif _wd in (1, 2, 3) and 12 <= _hr <= 18: temporal_mod = 1.0
    else: temporal_mod = 0.95

    vol_r, _ = _eight_dim_real._dim2_volatility(w)
    trend_r, _ = _eight_dim_real._dim3_trend(w, direction)
    regime_mult = _REGIME_MULT.get((vol_r, trend_r), 1.0)
    sess_mult, _, _ = _eight_dim_real._dim4_session(_hr)
    pair_mod = _eight_dim_real._dim5_pair(pair, w)
    exit_mod = _eight_dim_real._dim7_exit(w)

    _open_dicts = [{"symbol": p[10], "type": p[1]} for p in (open_pos_list or [])]
    _dim8_allowed, _ = _eight_dim_real._dim8_correlation(pair, direction, _open_dicts)
    if not _dim8_allowed:
        return None  # DIM8: riesgo de correlacion duplicado, bloqueo real

    # DIM6 circuit breaker real (agents/eight_dim_agent.py::_dim6_circuit_breaker,
    # reimplementado con el historial simulado propio en vez de episodes.db real
    # -- mismo criterio: 3 perdidas seguidas en las ultimas 8h de simulacion =
    # bloqueo total; WR<40% en las ultimas 5 = reduce a 0.6x).
    dim6_mod = 1.0
    _hist = _RECENT_OUTCOMES.get(pair, [])
    if len(_hist) >= 3 and all(o == "LOSS" for _, o in _hist[-3:]):
        _age_h = (pd.Timestamp(dt) - pd.Timestamp(_hist[-1][0])).total_seconds() / 3600.0
        if _age_h < 8:
            return None  # bloqueo duro, igual que en vivo
    if len(_hist) >= 5:
        _wr5 = sum(1 for _, o in _hist if o == "WIN") / len(_hist)
        if _wr5 < 0.40:
            dim6_mod = 0.60
    if daily_pnl_month is not None and capital > 0 and daily_pnl_month / capital >= 0.04:
        dim6_mod = min(dim6_mod, 0.30)  # meta mensual 4%+ ya cumplida: proteger

    final_mult = regime_mult * sess_mult * temporal_mod * pair_mod * dim6_mod * exit_mod
    dim_mult = max(0.4, min(1.4, final_mult))

    score = int(min(max(smc_score + ml_score + risk_score, 0), 100) * dim_mult)
    _RS_STATS["returned_signal"] += 1
    if score < 80: _RS_STATS["below_thr_80"] += 1
    return direction, float(entry), float(sl), float(tp), int(score)


_BO_STATS = defaultdict(int)  # diagnostico de frecuencia, mismo patron que _RS_STATS


def breakout_signal(w, pair, dt):
    """Motor de señal NUEVO (no SMC): ruptura de canal Donchian de N periodos
    + SL/TP por ATR real del par. Diseñado para resolver el problema de
    ESCASEZ encontrado en real_signal() (1.7-2.4% de frecuencia real en 16
    años) -- una ruptura de canal ocurre con mucha mas frecuencia que la
    confluencia FVG/OB-OTE/BOS-desplazamiento que exige el motor SMC.
    Devuelve (direction, entry, sl, tp, score) o None.
    """
    _BO_STATS["calls"] += 1
    if len(w) < DONCHIAN_N + 210:
        return None
    highs = w["high"].values
    lows = w["low"].values
    close_now = float(w["close"].values[-1])
    if close_now <= 0:
        return None

    d_high = float(highs[-DONCHIAN_N - 1:-1].max())
    d_low = float(lows[-DONCHIAN_N - 1:-1].min())
    direction = None
    if close_now > d_high:
        direction = "LONG"
    elif close_now < d_low:
        direction = "SHORT"
    if direction is None:
        return None
    _BO_STATS["breakout"] += 1

    if TREND_FILTER_BO:
        c = w["close"]
        e_fast = ema(c, 50).iloc[-1]
        e_slow = ema(c, 200).iloc[-1]
        if direction == "LONG" and not (e_fast > e_slow):
            return None
        if direction == "SHORT" and not (e_fast < e_slow):
            return None
        _BO_STATS["trend_ok"] += 1

    atr_v = atr14(w).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0:
        return None

    # 2026-09-02: FILTRO DE COMPRESION DE VOLATILIDAD -- pedido explicito
    # del usuario ("usa toda tu inteligencia/investigacion real, no
    # adivines"), investigado via WebSearch: literatura de trading
    # cuantitativo (Opening Range Breakout, "range compression") reporta
    # que las rupturas rinden mejor cuando ocurren DESPUES de un periodo
    # de rango comprimido (baja volatilidad relativa a su propio
    # historial reciente) -- la volatilidad tiende a expandirse tras
    # contraerse. Nunca probado en esta sesion (18+ variables probadas
    # eran todas sobre SL/TP/riesgo/horas, ninguna sobre la CALIDAD del
    # contexto de volatilidad antes de la ruptura). COMPRESSION_RATIO_BO=0
    # (default) desactiva el filtro -- no cambia el comportamiento
    # existente a menos que se pida.
    _compress_ratio = float(os.environ.get("COMPRESSION_RATIO_BO", "0") or 0)
    if _compress_ratio > 0:
        _atr_series = atr14(w)
        _atr_avg20 = _atr_series.iloc[-20:].mean()
        if pd.isna(_atr_avg20) or _atr_avg20 <= 0 or atr_v > _atr_avg20 * _compress_ratio:
            return None  # no estaba comprimido -- no es el setup que la literatura respalda

    sl_dist = PER_PAIR_SL_MULT.get(pair, ATR_MULT_SL_BO) * atr_v
    # 2026-09-01: BUG-MT5-INVALID-STOPS-LIVE -- confirmado en vivo (cuenta
    # demo, USDCHF, 21:5x UTC) que MT5 rechaza ordenes con retcode 10016
    # "Invalid stops" cuando el SL calculado por ATR queda demasiado cerca
    # del precio. Verificado con mt5.order_check() en vivo que el minimo
    # real NO es fijo (el campo symbol_info().trade_stops_level reporta un
    # valor no confiable, ~0.1 pip, en los 5 pares activos) -- es dinamico,
    # correlacionado con volatilidad/spread del momento: en hora muerta
    # (calma) el minimo real bajo hasta <1 pip, pero durante la killzone
    # activa (mult=1.20) subio a ~15 pips (7.8 rechazado, 15 aceptado,
    # binary search exacto). MIN_SL_PIPS_BO modela un piso conservador fijo
    # (el peor caso medido) -- se aplica ANTES de calcular tp para que el
    # RR_MULT_BO se preserve EXACTO (tp = sl_dist_ajustado * RR, misma
    # formula de siempre, sin logica extra). Default "0" = deshabilitado,
    # no cambia el comportamiento existente a menos que se pida.
    _min_sl_pips = float(os.environ.get("MIN_SL_PIPS_BO", "0") or 0)
    if _min_sl_pips > 0:
        _min_sl_dist = _min_sl_pips * PIP_SZ[pair]
        if sl_dist < _min_sl_dist:
            sl_dist = _min_sl_dist
    entry = close_now
    sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
    tp = entry + sl_dist * RR_MULT_BO if direction == "LONG" else entry - sl_dist * RR_MULT_BO

    channel_ref = d_high if direction == "LONG" else d_low
    strength = abs(close_now - channel_ref) / atr_v
    score = int(min(100, 60 + strength * 40))
    _BO_STATS["returned_signal"] += 1
    return direction, float(entry), float(sl), float(tp), score


_MR_STATS = defaultdict(int)


def meanrev_signal(w, pair, dt):
    """Motor de señal ALTERNATIVO (no breakout): reversion a la media --
    RSI extremo (sobrecompra/sobreventa) + precio tocando/cruzando la
    banda de Bollinger correspondiente. Target = banda media (la tesis
    real de reversion), SL = ATR-based. Enfoque opuesto al breakout
    (fade en vez de chase), pedido explicitamente por el usuario tras
    que el barrido de parametros del motor breakout se estancara.
    Devuelve (direction, entry, sl, tp, score) o None.
    """
    _MR_STATS["calls"] += 1
    if len(w) < 220:
        return None
    close_now = float(w["close"].values[-1])
    if close_now <= 0:
        return None

    mi = MomentumIndicators(w)
    rsi_v = mi.rsi(14)
    bb_upper, bb_mid, bb_lower = mi.bollinger_bands(20, 2.0)

    direction = None
    if rsi_v <= RSI_OVERSOLD and close_now <= bb_lower:
        direction = "LONG"
    elif rsi_v >= RSI_OVERBOUGHT and close_now >= bb_upper:
        direction = "SHORT"
    if direction is None:
        return None
    _MR_STATS["extreme"] += 1

    atr_v = atr14(w).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0:
        return None
    sl_dist = ATR_MULT_SL_MR * atr_v
    entry = close_now
    sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
    # 2026-09-02: RR_MULT_MR=0 (default) usa la tesis real (target=banda
    # media); RR_MULT_MR>0 fuerza un RR fijo en su lugar -- para aislar si
    # el problema de v1/v2/v3 (sin edge real) viene del target movil o de
    # la premisa de entrada en si.
    _rr_mr = float(os.environ.get("RR_MULT_MR", "0") or 0)
    if _rr_mr > 0:
        tp = entry + sl_dist * _rr_mr if direction == "LONG" else entry - sl_dist * _rr_mr
    else:
        tp = bb_mid  # tesis real de reversion: vuelve a la media, no un RR fijo arbitrario

    # Descarta señales donde el target (banda media) ya esta mas cerca que
    # el propio SL -- RR invalido/invertido, no vale la pena el trade.
    tp_dist = abs(tp - entry)
    if tp_dist < sl_dist * 0.5:
        return None

    extreme = abs(rsi_v - 50.0)
    score = int(min(100, 60 + extreme * 0.8))
    _MR_STATS["returned_signal"] += 1
    return direction, float(entry), float(sl), float(tp), score


def d1_trend(dfd, dt):
    s = dfd[dfd.index.date <= pd.Timestamp(dt).date()]
    if len(s) < 50: return "UNKNOWN"
    c = s["close"]
    return "LONG" if c.iloc[-1] > ema(c,50).iloc[-1] else "SHORT"

def h4_bias(dh1, dt):
    """Estimate H4 trend from H1 data via resampling."""
    s = dh1[dh1.index <= pd.Timestamp(dt)].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    if len(s) < 20: return "WAIT"
    c = s["close"]
    e8, e20 = ema(c,8).iloc[-1], ema(c,20).iloc[-1]
    return "LONG" if e8 > e20 else ("SHORT" if e8 < e20 else "WAIT")

RISK_MULT_TEST = float(os.environ.get("RISK_MULT_TEST", "1.0") or 1.0)
# 2026-08-29: parametrizado -- DIM6 (Kelly) de esta sesion muestra el sistema
# subutilizando capital (Kelly recomienda 4.3-8.5% vs 0.5% real). Un intento
# previo (2026-07-09) de doblar MAX_RISK a 550 disparo P(mes<-5%) de 6% a 16%
# -- pero eso se probo ANTES de RR=4.0/trailing-to-BE/horas limpias de esta
# sesion, que cambiaron el perfil riesgo/retorno. Probar un paso MODESTO
# (no doblar de una vez) y vigilar P(mes<-5%) en el Monte Carlo, no solo
# P(pass) -- el usuario pidio maximizar pero la regla de consistencia de
# Axi Select puede reventar la cuenta si el drawdown mensual se dispara.
def risk_for_score(score):
    """Dynamic risk based on conviction score."""
    if score >= 90: return min(MAX_RISK * 1.5 * RISK_MULT_TEST, 400.0 * RISK_MULT_TEST), 0.01 * RISK_MULT_TEST
    if score >= 80: return MAX_RISK * RISK_MULT_TEST, 0.005 * RISK_MULT_TEST
    return MAX_RISK * 0.7 * RISK_MULT_TEST, 0.0025 * RISK_MULT_TEST

# ── DIMENSIÓN 1+2+3: Run full historical simulation ───────────────────
print("\n" + "=" * 72)
print("  DIMENSIONES 1-3: Backtest temporal + régimen vol + régimen trend")
print("  Simulando ~16 años H1 (MT5 real) con todos los pares...")
print("=" * 72)

trade_log = []       # all trades with metadata
ALL_TRADING_DAYS = set()  # 2026-08-30: TODOS los dias de trading activos
# escaneados (con o sin trade) -- ver comentario junto a day_str mas abajo.
# Necesario para reconstruir la serie diaria completa (con ceros) para el
# Monte Carlo, distinto de daily_pnl que solo tiene dias-con-trade.
daily_pnl  = defaultdict(float)
regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
hour_stats   = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
year_stats   = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
pair_stats   = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

# 2026-08-29: HALLAZGO CRITICO DE PARIDAD -- core/supervisor.py:2151-2156
# confirma que MAX_OPEN_POSITIONS es un limite GLOBAL sobre TODA la cuenta
# (cuenta posiciones "existing" de TODOS los simbolos, no filtra por
# pair/symbol antes de comparar contra MAX_OPEN_POSITIONS). Este backtest
# simulaba cada par en su propio loop independiente con su propio open_pos
# reseteado a [] -- es decir, MAX_OPEN_TEST se aplicaba POR PAR, permitiendo
# hasta MAX_OPEN_TEST*6 posiciones simultaneas reales entre todos los pares
# combinados, muy por encima del limite real (4 TOTAL). TODOS los resultados
# de esta sesion hasta este punto estan inflados por este error. Arreglado
# fusionando las 6 lineas de tiempo H1 (por par) en una sola linea
# cronologica real via heapq.merge (streaming, no materializa la lista
# completa en memoria) y usando un open_pos GLOBAL compartido entre pares.
def _pair_stream(_p, _df1):
    _idx_arr = _df1.index
    for _i in range(80, len(_df1)):
        yield (_idx_arr[_i], _p, _i)

_events = heapq.merge(*[_pair_stream(_p, h1_data[_p]) for _p in h1_data], key=lambda e: e[0])
open_pos = []  # GLOBAL: compartido entre TODOS los pares, no reseteado por par

if True:
    for dt, pair, idx in _events:
        df1 = h1_data[pair]
        dfd = d1_data.get(pair, pd.DataFrame())
        bar = df1.iloc[idx]
        if pd.Timestamp(dt).weekday() >= 5: continue
        if pd.Timestamp(dt).weekday() in _EXCLUDE_WEEKDAYS: continue
        hour_utc = pd.Timestamp(dt).hour
        # Bug found 2026-07-07: this used to keep hours 13-19 UTC, which is NOT
        # what the live bot trades. Real DEAD_HOURS_UTC (core/supervisor.py:121)
        # blocks {0-13, 17,18,19} -- active hours are 14-16 and 20-23 UTC. The
        # old window here INCLUDED hour 13 and the empirically-bad 17-19 block
        # (WR=24-28%, see DEAD_HOURS_UTC comment) while EXCLUDING 20-23, which
        # the live bot actually trades. Every cached backtest_results.json
        # number produced by this script was simulating the wrong hours.
        # 2026-08-28: hour 14 hard-blocked in supervisor.py:149 since commit
        # ef54cf6 (2026-07-26, "backtest confirms real improvement") but this
        # script's set was never updated -- found by audit subagent after the
        # live run showed 8190 trades at hour 14 (WR=40%, avg=-$6) that the
        # real bot never takes anymore.
        DEAD_HOURS_UTC = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19}
        _extra_dead = os.environ.get("EXTRA_DEAD_HOURS", "")
        if _extra_dead:
            DEAD_HOURS_UTC = DEAD_HOURS_UTC | {int(h) for h in _extra_dead.split(",") if h.strip()}
        _remove_dead = os.environ.get("REMOVE_DEAD_HOURS", "")
        if _remove_dead:  # 2026-08-29: probar horas actualmente bloqueadas con los 16
            # anios reales de MT5 ahora disponibles (bloqueo original se fijo con menos
            # historia) -- permite reabrir una hora especifica para medir su edge real.
            DEAD_HOURS_UTC = DEAD_HOURS_UTC - {int(h) for h in _remove_dead.split(",") if h.strip()}
        if hour_utc in DEAD_HOURS_UTC: continue  # kill zone
        day_str = str(pd.Timestamp(dt).date())
        year_str = str(pd.Timestamp(dt).year)
        ALL_TRADING_DAYS.add(day_str)
        # 2026-08-30: HALLAZGO CRITICO -- daily_pnl (defaultdict) solo tiene
        # entrada para dias con AL MENOS un cierre no-cero; dias sin ningun
        # trade simplemente no aparecen en el dict. Con el motor viejo
        # (miles de trades) esto era invisible (~todos los dias tenian
        # trade). Con el motor real (163 trades en 16 años, ver
        # SESION_ACTUAL.md) el 97.6% de los dias de trading NO tienen
        # entrada -- el Monte Carlo mas abajo (bootstrap sobre
        # daily_pnl.values()) estaba re-muestreando SOLO dias-con-trade
        # como si fueran "un dia cualquiera", simulando meses donde CADA
        # dia tiene operacion (frecuencia ~100% en vez de la real ~2.4%).
        # ALL_TRADING_DAYS registra TODOS los dias activos escaneados
        # (con o sin trade) para reconstruir la serie diaria completa,
        # con ceros, antes del Monte Carlo.

        # Manage open positions (partial TP + BE at 1.0R, full TP/SL)
        # 2026-08-29: open_pos ahora es GLOBAL (todos los pares) -- solo se
        # gestionan aqui las posiciones DE ESTE par (bar solo tiene precios
        # de este par); las de otros pares se dejan intactas en other_pos y
        # se reincorporan al final, sin tocarlas hasta que les toque su
        # propio evento en la linea de tiempo fusionada.
        other_pos = [p for p in open_pos if p[10] != pair]
        this_pair_pos = [p for p in open_pos if p[10] == pair]
        new_open = []
        is_friday = pd.Timestamp(dt).weekday() == 4
        for pos in this_pair_pos:
            (eidx, direction, entry, sl, tp, vol_p, sl_dist,
             partial_done, be_sl, pip_v, pair_p, peak_pnl, stagn_flag_idx) = pos
            pnl = None
            age_h = idx - eidx  # H1 bars: 1 bar == 1 hour

            cur_h = bar["high"]
            cur_l = bar["low"]
            cur_c = bar["close"]

            # Fix 2026-07-06: live bot no longer partial-closes at 1R+immediate-BE
            # (validated against 584 real trades: it was capping every winner near
            # ~0.5R while losses ran to full SL, ratio 1.36:1 vs the RR=3.0 the
            # system is configured for -- generalized the XAUUSD-only skip to all
            # symbols). Simulate that directly: full SL or full TP, no partial leg,
            # matching core/supervisor.py's current live exit logic. Trailing-to-BE
            # at 1.5R still exists live but only protects against giveback after
            # 1.5R -- doesn't change the SL/TP outcome distribution modeled here.
            # SWING-STOP sim (2026-07-24): tried modeling the flat -$150
            # emergency backstop here, approximated via bar adverse-excursion.
            # REVERTED same-session: it made close-type proportions LESS
            # realistic, not more -- final_SL dropped to 0.1% of closes
            # (live measured 20.1%) because designed_sl_loss > $150 for most
            # simulated positions, so SWING-STOP pre-empted the real SL on
            # almost every losing trade. That means the backtest's position
            # sizing doesn't match live's actual SL-risk distribution closely
            # enough for this specific refinement to be trustworthy yet --
            # needs the sizing model calibrated first, not just this guard
            # bolted on. Left disabled; STAGNANT/TIME-CLOSE/FRIDAY-CLOSE
            # (added earlier this session) still measurably improved realism
            # and are kept.
            top_close_type = None

            if PARTIAL_R_TEST > 0 and not partial_done:
                if direction == "LONG":
                    partial_price = entry + PARTIAL_R_TEST * sl_dist
                    hit_partial = cur_h >= partial_price
                else:
                    partial_price = entry - PARTIAL_R_TEST * sl_dist
                    hit_partial = cur_l <= partial_price
                if hit_partial:
                    half_vol = vol_p / 2.0
                    partial_pnl = half_vol * PARTIAL_R_TEST * sl_dist * pip_v / PIP_SZ[pair_p]
                    if partial_pnl != 0.0:
                        daily_pnl[day_str] += partial_pnl
                        vr = vol_regime(df1, idx)
                        tr = trend_regime(df1, idx)
                        trade_log.append({
                            "pair": pair_p, "type": "partial", "pnl": partial_pnl,
                            "win": partial_pnl > 0, "hour": hour_utc, "year": year_str,
                            "vol_regime": vr, "trend_regime": tr,
                        })
                        regime_stats[(vr, tr)]["trades"] += 1
                        regime_stats[(vr, tr)]["wins"] += int(partial_pnl > 0)
                        regime_stats[(vr, tr)]["pnl"] += partial_pnl
                        hour_stats[hour_utc]["trades"] += 1
                        hour_stats[hour_utc]["wins"] += int(partial_pnl > 0)
                        hour_stats[hour_utc]["pnl"] += partial_pnl
                        year_stats[year_str]["trades"] += 1
                        year_stats[year_str]["wins"] += int(partial_pnl > 0)
                        year_stats[year_str]["pnl"] += partial_pnl
                        pair_stats[pair_p]["trades"] += 1
                        pair_stats[pair_p]["wins"] += int(partial_pnl > 0)
                        pair_stats[pair_p]["pnl"] += partial_pnl
                    vol_p = half_vol
                    sl = entry
                    be_sl = True
                    partial_done = True

            if TRAIL_BE_R_TEST > 0 and not be_sl and not partial_done:
                # 2026-08-29: modela el trailing-to-BE que YA existe en vivo
                # (mueve el SL a breakeven al alcanzar TRAIL_BE_R_TEST*sl_dist
                # a favor, SIN cerrar volumen -- protege contra giveback sin
                # capar el upside, a diferencia de PARTIAL_R_TEST). El backtest
                # nunca lo modelo antes (comentario previo: "doesn't change the
                # SL/TP outcome distribution modeled here" -- eso era cierto
                # solo porque no estaba implementado, no porque no importe).
                if direction == "LONG":
                    trail_price = entry + TRAIL_BE_R_TEST * sl_dist
                    hit_trail = cur_h >= trail_price
                else:
                    trail_price = entry - TRAIL_BE_R_TEST * sl_dist
                    hit_trail = cur_l <= trail_price
                if hit_trail:
                    sl = entry
                    be_sl = True

            # cur_sl_dist (no sl_dist): tras un partial, sl se mueve a
            # breakeven (entry) -- el P&L de un stop-out debe reflejar la
            # distancia REAL entry->sl en ese momento (≈0 en breakeven), no
            # la distancia original de diseño. sl_dist original se sigue
            # usando para el TP (que no se mueve) y para el tamano del
            # partial en si (calculado antes de este bloque).
            # 2026-08-30: pnl del TP calculado por distancia REAL entry->tp,
            # no por sl_dist*RR -- con REALISTIC_SIGNAL el TP es dinamico
            # (confluencia + snap a swing), no siempre entry+sl_dist*RR.
            # Identico numericamente al modelo viejo cuando tp SI es
            # entry+sl_dist*RR (abs(entry-tp)==sl_dist*RR ahi), asi que no
            # cambia ningun resultado ya confirmado.
            cur_sl_dist = abs(entry - sl)
            cur_tp_dist = abs(entry - tp)
            if direction == "LONG":
                if cur_l <= sl:
                    pnl = -vol_p * cur_sl_dist * pip_v / PIP_SZ[pair_p]
                elif cur_h >= tp:
                    pnl = vol_p * cur_tp_dist * pip_v / PIP_SZ[pair_p]
            else:
                if cur_h >= sl:
                    pnl = -vol_p * cur_sl_dist * pip_v / PIP_SZ[pair_p]
                elif cur_l <= tp:
                    pnl = vol_p * cur_tp_dist * pip_v / PIP_SZ[pair_p]

            if pnl is not None:
                pnl -= _spread_cost(pair_p, vol_p)
                if pnl != 0.0:
                    daily_pnl[day_str] += pnl
                    vr = vol_regime(df1, idx)
                    tr = trend_regime(df1, idx)
                    trade_log.append({
                        "pair": pair_p, "type": top_close_type or "final", "pnl": pnl,
                        "win": pnl > 0, "hour": hour_utc, "year": year_str,
                        "vol_regime": vr, "trend_regime": tr,
                    })
                    if REALISTIC_SIGNAL:
                        _RECENT_OUTCOMES[pair_p].append((dt, "WIN" if pnl > 0 else "LOSS"))
                        del _RECENT_OUTCOMES[pair_p][:-5]
                    regime_stats[(vr, tr)]["trades"] += 1
                    regime_stats[(vr, tr)]["wins"] += int(pnl > 0)
                    regime_stats[(vr, tr)]["pnl"] += pnl
                    hour_stats[hour_utc]["trades"] += 1
                    hour_stats[hour_utc]["wins"] += int(pnl > 0)
                    hour_stats[hour_utc]["pnl"] += pnl
                    year_stats[year_str]["trades"] += 1
                    year_stats[year_str]["wins"] += int(pnl > 0)
                    year_stats[year_str]["pnl"] += pnl
                    pair_stats[pair_p]["trades"] += 1
                    pair_stats[pair_p]["wins"] += int(pnl > 0)
                    pair_stats[pair_p]["pnl"] += pnl
            if pnl is None:
                # PEAK-GUARD sim (2026-07-16): track running peak floating $ this
                # trade has reached (using bar's favorable extreme), close early
                # if it retraces PEAK_GUARD_RETRACE from a peak >= PEAK_GUARD_MIN.
                if direction == "LONG":
                    fav_pnl = vol_p * (cur_h - entry) * pip_v / PIP_SZ[pair_p]
                    close_pnl = vol_p * (cur_c - entry) * pip_v / PIP_SZ[pair_p]
                else:
                    fav_pnl = vol_p * (entry - cur_l) * pip_v / PIP_SZ[pair_p]
                    close_pnl = vol_p * (entry - cur_c) * pip_v / PIP_SZ[pair_p]
                if fav_pnl > peak_pnl:
                    peak_pnl = fav_pnl

                close_type = None
                if (peak_pnl >= PEAK_GUARD_MIN
                        and close_pnl < peak_pnl * (1.0 - PEAK_GUARD_RETRACE)):
                    pnl = close_pnl
                    close_type = "peak_guard"
                # BUG-BACKTEST-NO-STAGNANT-TIME-FRIDAY (2026-07-24): these 3
                # guards drive 77% of real live closes (measured against 45
                # days of MT5 deal history) but were never modeled here --
                # every conclusion this script produced before today was
                # measured against a simulation missing the dominant real
                # exit mechanism. Added to mirror core/position_guards.py
                # exactly: FRIDAY-CLOSE closes everyone regardless of P&L;
                # STAGNANT closes a position that's been open STAGNANT_HOURS+
                # and never reached STAGNANT_PEAK_MAX profit (immediately if
                # pnl>=0, else after STAGNANT_GRACE_HOURS more); TIME-CLOSE
                # only force-closes LOSING positions past MAX_HOLD_HOURS
                # (winners are left to run / get caught by PEAK-GUARD).
                elif is_friday and hour_utc >= FRIDAY_CLOSE_HOUR + 1:
                    pnl = close_pnl
                    close_type = "friday_close"
                elif age_h >= STAGNANT_HOURS and peak_pnl < STAGNANT_PEAK_MAX:
                    if stagn_flag_idx is None:
                        stagn_flag_idx = idx
                    grace_h = idx - stagn_flag_idx
                    if close_pnl >= 0 or grace_h >= STAGNANT_GRACE_HOURS:
                        pnl = close_pnl
                        close_type = "stagnant"
                elif close_pnl <= 0 and age_h >= MAX_HOLD_HOURS:
                    pnl = close_pnl
                    close_type = "time_close"

                if close_type is not None:
                    pnl -= _spread_cost(pair_p, vol_p)
                    if pnl != 0.0:
                        daily_pnl[day_str] += pnl
                        vr = vol_regime(df1, idx)
                        tr = trend_regime(df1, idx)
                        trade_log.append({
                            "pair": pair_p, "type": close_type, "pnl": pnl,
                            "win": pnl > 0, "hour": hour_utc, "year": year_str,
                            "vol_regime": vr, "trend_regime": tr,
                        })
                        if REALISTIC_SIGNAL:
                            _RECENT_OUTCOMES[pair_p].append((dt, "WIN" if pnl > 0 else "LOSS"))
                            del _RECENT_OUTCOMES[pair_p][:-5]
                        regime_stats[(vr, tr)]["trades"] += 1
                        regime_stats[(vr, tr)]["wins"] += int(pnl > 0)
                        regime_stats[(vr, tr)]["pnl"] += pnl
                        hour_stats[hour_utc]["trades"] += 1
                        hour_stats[hour_utc]["wins"] += int(pnl > 0)
                        hour_stats[hour_utc]["pnl"] += pnl
                        year_stats[year_str]["trades"] += 1
                        year_stats[year_str]["wins"] += int(pnl > 0)
                        year_stats[year_str]["pnl"] += pnl
                        pair_stats[pair_p]["trades"] += 1
                        pair_stats[pair_p]["wins"] += int(pnl > 0)
                        pair_stats[pair_p]["pnl"] += pnl
                else:
                    new_open.append((eidx, direction, entry, sl, tp, vol_p, sl_dist,
                                      partial_done, be_sl, pip_v, pair_p, peak_pnl, stagn_flag_idx))

        open_pos = other_pos + new_open  # 2026-08-29: reincorpora las posiciones de otros pares sin tocar
        if len(open_pos) >= MAX_OPEN_TEST: continue  # ahora GLOBAL (todos los pares) -- corregido 2026-08-29, antes era por-par

        # Signal generation -- decoupled from D1/H4 bias (see smc_signal()
        # docstring, 2026-07-21). REQUIRE_D1/REQUIRE_H4 let this backtest
        # measure the real impact of relaxing D1-FILTER/H4-FILTER (live in
        # core/supervisor.py), which the old bias-baked-into-generation
        # design could never test.
        _real_entry = _real_sl = _real_tp = None
        if STRATEGY_MODE == "BREAKOUT":
            _wbo = df1.iloc[max(0, idx - (DONCHIAN_N + 210)):idx + 1]
            _bo = breakout_signal(_wbo, pair, dt)
            if _bo is None: continue
            sig, _real_entry, _real_sl, _real_tp, score = _bo
        elif STRATEGY_MODE == "MEANREV":
            _wmr = df1.iloc[max(0, idx - 220):idx + 1]
            _mr = meanrev_signal(_wmr, pair, dt)
            if _mr is None: continue
            sig, _real_entry, _real_sl, _real_tp, score = _mr
        elif REALISTIC_SIGNAL:
            _w200 = df1.iloc[max(0, idx - 200):idx + 1]
            _month_prefix = day_str[:7]
            _month_pnl = sum(v for k, v in daily_pnl.items() if k.startswith(_month_prefix))
            _rs = real_signal(_w200, pair, dt, daily_pnl.get(day_str, 0.0), CAPITAL, open_pos, _month_pnl)
            if _rs is None: continue
            sig, _real_entry, _real_sl, _real_tp, score = _rs
        else:
            sig, score, atr_v = smc_signal(df1, idx)
            if sig == "WAIT": continue

        if os.environ.get("CORR_FILTER", "0") == "1":
            # 2026-08-29: filtro de correlacion real (DIM8) -- ahora viable
            # porque open_pos es GLOBAL entre pares (arreglo del bug de
            # MAX_OPEN). Bloquea una entrada nueva si ya hay una posicion
            # abierta en la MISMA direccion en un par fuertemente
            # correlacionado (r>0.5 en la matriz de correlacion real de
            # esta sesion): EURUSD-NZDUSD (+0.71), USDCAD-USDCHF (+0.56).
            # Correlaciones negativas (ej EURUSD-USDCHF -0.84) NO se
            # bloquean -- son cobertura natural, no riesgo concentrado.
            _CORR_PAIRS = {frozenset({"EURUSD", "NZDUSD"}), frozenset({"USDCAD", "USDCHF"})}
            _corr_blocked = any(
                p[1] == sig and frozenset({pair, p[10]}) in _CORR_PAIRS
                for p in open_pos
            )
            if _corr_blocked:
                continue

        d_dir = d1_trend(dfd, dt)
        if REQUIRE_D1:
            if d_dir == "UNKNOWN": continue
            if sig != d_dir: continue
        h4_d = h4_bias(df1, dt)
        if REQUIRE_H4 and h4_d not in (sig, "WAIT"): continue

        # DIAGNOSTICO 2026-07-09: DIM2/DIM3 del propio backtest concluyeron que
        # HIGH vol + STRONG_TREND es la mejor combinacion posible -- nunca se
        # habia probado como filtro duro, solo como observacion.
        if ENABLE_REGIME_FILTER:
            if vol_regime(df1, idx) != "HIGH" or trend_regime(df1, idx) != "STRONG_TREND":
                continue
        if EXCLUDE_CHOPPY:
            if trend_regime(df1, idx) == "CHOPPY":
                continue

        # Filtros nuevos 2026-07-09: RSI/Bollinger/Estocastico/volumen (smc/momentum.py)
        # + Alligator/Awesome Oscillator (smc/bill_williams.py), mismo criterio que el
        # pipeline en vivo (_enrich_with_agents en core/supervisor.py) -- ajustan el
        # score, no lo bloquean solos.
        _mw = df1.iloc[max(0, idx - 80):idx + 1]
        if ENABLE_MOMENTUM_FILTERS:
            try:
                score += MomentumIndicators(_mw).score_for_signal(sig).pts_adjustment
                score += BillWilliamsIndicators(_mw).score_for_signal(sig).pts_adjustment
            except Exception:
                pass

        # Silver Bullet ICT (2026-07-09, kill zone corregida 2026-07-28): gate
        # todo-o-nada SOLO en las horas activas reales (15,16,20-23 UTC, ya NO
        # 14 UTC -- bloqueada desde 2026-07-26) -- si falta sweep+FVG+killzone
        # en la direccion de la senal, no se opera esa hora especifica, igual
        # que en vivo (ver BUG-SB-DEAD-KILLZONE en smc/liquidity_sweep.py).
        if ENABLE_SILVER_BULLET_GATE and hour_utc in (15, 16, 20, 21, 22, 23):
            try:
                _sb = silver_bullet_check(_mw)
                _sb_dir = "bullish" if sig == "LONG" else "bearish"
                if _sb is None or not _sb.valid or _sb.direction != _sb_dir:
                    continue
            except Exception:
                pass

        # Threshold — actualizado 2026-07-05: MT5_SCORE_AUTO_REDUCE real=80 (core/supervisor.py:96,
        # recalibrado 2026-07-01 tras el sweep que probo 90-95 y NO mejoraba WR, solo cortaba volumen).
        # MT5_REAL_SCORE_THRESHOLD=95 es solo techo de excepcion, no la operacion normal.
        if STRATEGY_MODE == "BREAKOUT":
            thr = THR_BREAKOUT  # motor nuevo -- score 60-100 por diseño, umbral propio (default 0 = sin filtro extra)
        else:
            thr = float(os.environ.get("THR_CONFIRMED_TEST", "80")) if h4_d != "WAIT" else float(os.environ.get("THR_WAIT_TEST", "90"))
        # 2026-08-29: parametrizado -- el sweep 2026-07-01 que fijo 80/90 como
        # optimo predata el descubrimiento de "solo tarde 20-23 UTC" de esta
        # sesion (RR=3.0, todas las horas). Con el conjunto de trades distinto
        # que deja ese filtro, el optimo de threshold podria haber cambiado.
        if score < thr: continue

        # Risk scaling by score
        max_r, r_pct = risk_for_score(score)

        if os.environ.get("REALISTIC_RISK_CAP", "0") == "1":
            # 2026-08-29: HALLAZGO DE PARIDAD -- core/supervisor.py:2246-2255 usa
            # un tope de riesgo ADAPTATIVO por progreso diario real (MAX_DOLLAR_RISK
            # entre $100 si ya se cumplio la meta del dia y $400 si se va muy
            # atrasado), completamente distinto del modelo estatico por-score de
            # este backtest (risk_for_score, nunca mira cuanto se lleva ganado en
            # el dia). El sweep RISK_MULT_TEST de esta sesion probo multiplicadores
            # que en la practica EXCEDEN el techo real de $400 en vivo (ej. tier
            # score>=90 a RISK_MULT=1.5 ya da $412, a 2.0 da $550-825 segun tier --
            # muy por encima de lo que el bot real permitiria). Esto replica la
            # formula real completa para dar el numero HONESTO y deployable.
            _rcm = float(os.environ.get("REALISTIC_RISK_CAP_MULT", "1.0") or 1.0)
            # 2026-08-29: escala los 3 tiers reales ($100/$200/$400) manteniendo
            # la ESTRUCTURA adaptativa real (por progreso diario) -- a diferencia
            # de RISK_MULT_TEST, que multiplicaba el modelo estatico por-score
            # que no coincide con como el bot realmente calcula el riesgo.
            _shortfall = DAILY_TARGET - daily_pnl.get(day_str, 0.0)
            if _shortfall > 200 and hour_utc >= 13:
                max_r = min(400.0 * _rcm, (200.0 + _shortfall * 0.3) * _rcm)
            elif _shortfall <= 0:
                max_r = 100.0 * _rcm
            else:
                max_r = 200.0 * _rcm

        # DIAGNOSTICO 2026-07-09: escalar riesgo SOLO donde hay edge real
        # comprobado (episodes.db real: EURUSD PF=1.11, unico con neto positivo
        # entre los pares activos), en vez de subir el riesgo parejo a todos
        # (eso disparaba P(mes<-5%) de 6% a 16% -- ver commit e92f121).
        _PAIR_RISK_MULT = {"EURUSD": 1.8}
        _extra_boost = {p.strip().upper() for p in os.environ.get("EXTRA_BOOST_PAIRS", "").split(",") if p.strip()}
        if _extra_boost:  # 2026-08-29: DIM5 de esta sesion (config ganadora, horas
            # limpias) muestra EURAUD casi empatado con EURUSD ($151 vs $156 avg,
            # vs una brecha mucho mas ancha en la config vieja que justifico el
            # boost original solo para EURUSD -- probar extender el mismo boost.
            for _p in _extra_boost:
                _PAIR_RISK_MULT[_p] = 1.8
        _mult = _PAIR_RISK_MULT.get(pair, 1.0)
        max_r *= _mult

        # Volume
        # 2026-08-29: CORRECCION -- se encontro el calculo real de SL en vivo:
        # agents/signal_agent.py:_sl_distance() -- atr14*1.5 (SIN cap/floor en
        # ESTE backtest hasta ahora), pero el motor real SI aplica un cap y un
        # floor por par que este script nunca modelaba:
        #   cap (pips): EURUSD/GBPUSD/USDCAD=40, AUDUSD/NZDUSD/USDCHF=35,
        #               EURAUD=45, GBPCAD=50
        #   floor (pips): majors=20, GBP-crosses(GBPCAD)=25
        # Sin el cap, ATR alto (regimen HIGH vol) generaba SL mucho mas anchos
        # que en vivo -- explica por que SL_ATR_MULT_TEST=1.0 (barrido previo,
        # sin cap) se acercaba por accidente al comportamiento real capado.
        # REALISTIC_SL=1 aplica la formula real completa (recomendado); si no,
        # se usa el multiplicador simple SL_ATR_MULT_TEST (modo exploratorio
        # anterior, sin cap/floor, se mantiene por compatibilidad).
        if _EXTERNAL_ENTRY:
            # entry/sl/tp ya vienen del motor de señal (real_signal() o
            # breakout_signal()) -- no recomputar.
            entry = _real_entry
            sl_p = _real_sl
            tp_p = _real_tp
            sl_dist_p = abs(entry - sl_p)
        elif os.environ.get("REALISTIC_SL", "0") == "1":
            _sl_cap_pips = {"EURUSD": 40, "GBPUSD": 40, "USDCAD": 40,
                             "AUDUSD": 35, "NZDUSD": 35, "USDCHF": 35,
                             "EURAUD": 45, "GBPCAD": 50}
            _sl_floor_pips = {"GBPCAD": 25}  # resto = 20 (default)
            sl_dist_p = atr_v * 1.5
            _cap_p = _sl_cap_pips.get(pair)
            if _cap_p is not None:
                sl_dist_p = min(sl_dist_p, _cap_p * PIP_SZ[pair])
            _floor_p = _sl_floor_pips.get(pair, 20)
            sl_dist_p = max(sl_dist_p, _floor_p * PIP_SZ[pair])
        else:
            sl_dist_p = atr_v * float(os.environ.get("SL_ATR_MULT_TEST", "1.5") or 1.5)
        pip_v = PIP_VAL[pair]
        pip_s = PIP_SZ[pair]
        sl_pips = sl_dist_p / pip_s
        if sl_pips <= 0: continue
        vol = min(float(os.environ.get("VOL_CAP_TEST", "2.0") or 2.0), (CAPITAL * r_pct) / (sl_pips * pip_v))
        actual_risk = vol * sl_pips * pip_v
        if actual_risk > max_r:
            vol = max_r / (sl_pips * pip_v)
        vol = max(0.01, round(int(vol / 0.01) * 0.01, 2))
        actual_risk = vol * sl_pips * pip_v
        if actual_risk < 5: continue

        if not _EXTERNAL_ENTRY:
            entry = bar["close"]
            if sig == "LONG":
                sl_p = entry - sl_dist_p
                tp_p = entry + sl_dist_p * RR
            else:
                sl_p = entry + sl_dist_p
                tp_p = entry - sl_dist_p * RR

        open_pos.append((idx, sig, entry, sl_p, tp_p, vol, sl_dist_p, False, False, pip_v, pair, 0.0, None))

if REALISTIC_SIGNAL:
    print(f"\n  [DIAGNOSTICO real_signal()] {dict(_RS_STATS)}")
if STRATEGY_MODE == "BREAKOUT":
    print(f"\n  [DIAGNOSTICO breakout_signal()] {dict(_BO_STATS)}")
if STRATEGY_MODE == "MEANREV":
    print(f"\n  [DIAGNOSTICO meanrev_signal()] {dict(_MR_STATS)}")
print(f"\n  Total trades (periodo completo H1): {len(trade_log)}")
n_final = sum(1 for t in trade_log if t["type"] == "final")
n_partial = sum(1 for t in trade_log if t["type"] == "partial")
n_wins = sum(1 for t in trade_log if t["win"] and t["type"] in ("final",))
n_days = len(daily_pnl)
avg_d = np.mean(list(daily_pnl.values())) if daily_pnl else 0
print(f"  Parciales (50% a 1R): {n_partial} | Finals: {n_final} | Wins: {n_wins}/{n_final} = {n_wins/max(1,n_final)*100:.1f}% WR")
print(f"  Días con trades: {n_days} | Avg diario: ${avg_d:.0f}")

# 2026-07-24: como % del total de cierres reales (fuente de verdad: MT5 deal
# history), TP real=2.8%, SL real=20.1%, guardias(stagnant/time/friday/peak)=77.1%
_close_types = defaultdict(int)
for t in trade_log:
    _ctype = t["type"]
    if _ctype == "final":
        _ctype = "final_TP" if t["win"] else "final_SL"
    _close_types[_ctype] += 1
_n_total_closes = len(trade_log)
print("  Desglose por tipo de cierre (comparar contra realidad: TP=2.8% SL=20.1% guardias=77.1%):")
for _ct, _n in sorted(_close_types.items(), key=lambda kv: -kv[1]):
    print(f"    {_ct:12s}: {_n:6d} ({_n/max(1,_n_total_closes)*100:5.1f}%)")

# ── DIMENSIÓN 4: Por hora UTC ─────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 4: SESIÓN — Win rate y P&L por hora UTC")
print("=" * 72)
print(f"  {'Hora UTC':10s} | {'Trades':7s} | {'WR':6s} | {'Avg P&L':10s} | {'Rating':20s}")
print("  " + "-"*60)
for h in sorted(hour_stats.keys()):
    st = hour_stats[h]
    if st["trades"] < 5: continue
    wr = st["wins"] / st["trades"] * 100
    avg_pnl = st["pnl"] / st["trades"]
    rating = "🔥 PREMIUM" if wr > 50 and avg_pnl > 50 else ("✅ BUENA" if wr > 40 else ("⚠️ REGULAR" if wr > 30 else "❌ EVITAR"))
    print(f"  {h:02d}:00 UTC    | {st['trades']:7d} | {wr:5.0f}% | ${avg_pnl:8.0f}   | {rating}")

# Best hours
best_hours = sorted([h for h,s in hour_stats.items() if s["trades"] >= 3],
                    key=lambda h: hour_stats[h]["pnl"] / max(1, hour_stats[h]["trades"]), reverse=True)
print(f"\n  TOP 3 MEJORES HORAS: {best_hours[:3]}")

# ── DIMENSIÓN 1: Por año ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 1: TEMPORAL — Performance por año")
print("=" * 72)
print(f"  {'Año':6s} | {'Trades':7s} | {'WR':6s} | {'Total P&L':11s} | {'Avg/día':8s}")
print("  " + "-"*50)
for yr in sorted(year_stats.keys()):
    st = year_stats[yr]
    if st["trades"] < 10: continue
    wr = st["wins"] / st["trades"] * 100
    trading_days = st["trades"] / max(1, (n_final + n_partial) / max(1, n_days))
    print(f"  {yr:6s} | {st['trades']:7d} | {wr:5.0f}% | ${st['pnl']:9.0f}   | ${st['pnl']/max(1,trading_days):6.0f}")

# ── DIMENSIÓN 2+3: Por régimen ────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIONES 2+3: RÉGIMEN DE VOL + TENDENCIA — Cuándo funciona el sistema")
print("=" * 72)
print(f"  {'Vol':8s} | {'Trend':14s} | {'Trades':7s} | {'WR':6s} | {'Avg P&L':10s}")
print("  " + "-"*58)
for (vr, tr), st in sorted(regime_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
    if st["trades"] < 5: continue
    wr = st["wins"] / st["trades"] * 100
    avg_pnl = st["pnl"] / st["trades"]
    print(f"  {vr:8s} | {tr:14s} | {st['trades']:7d} | {wr:5.0f}% | ${avg_pnl:8.0f}")

# ── DIMENSIÓN 5: Por par ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 5: PAR — Performance por instrumento")
print("=" * 72)
print(f"  {'Par':8s} | {'Trades':7s} | {'WR':6s} | {'Total P&L':11s} | {'Avg P&L':8s}")
print("  " + "-"*52)
for p in sorted(pair_stats.keys(), key=lambda x: pair_stats[x]["pnl"], reverse=True):
    st = pair_stats[p]
    if st["trades"] < 5: continue
    wr = st["wins"] / st["trades"] * 100
    print(f"  {p:8s} | {st['trades']:7d} | {wr:5.0f}% | ${st['pnl']:9.0f}   | ${st['pnl']/st['trades']:6.0f}")

# ── DIMENSIÓN 6: Kelly Criterion ─────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 6: KELLY — Tamaño óptimo de posición")
print("=" * 72)
final_trades = []  # BUG-EMPTY-TRADELOG-CRASH (2026-07-28): referenced unconditionally
wr_f, b = 0.0, 0.0  # at DIM7 below -- a 0-trade run (e.g. an over-restrictive gate
                     # like ENABLE_SILVER_BULLET_GATE=1) crashed with NameError
                     # instead of printing "no trades" like every other section.
if n_final > 10:
    # 2026-08-28: bug encontrado por auditoria (subagente) -- filtrar solo
    # type=="final" mide Kelly sobre el 22.9% de los cierres (TP/SL puro),
    # ignorando el 77.1% que cierra por guardias (peak_guard/friday_close/
    # stagnant/time_close) con P&L real variable. Esa mezcla sesgada daba
    # Kelly negativo pese a P&L diario/anual positivo consistente. Usar
    # TODOS los cierres reales, no solo el subconjunto TP/SL.
    final_trades = [t for t in trade_log if t["pnl"] != 0]
    wins_f = [t["pnl"] for t in final_trades if t["win"]]
    losses_f = [abs(t["pnl"]) for t in final_trades if not t["win"]]
    wr_f = len(wins_f) / len(final_trades)
    avg_win = np.mean(wins_f) if wins_f else 0
    avg_loss = np.mean(losses_f) if losses_f else 0
    if avg_loss > 0 and avg_win > 0:
        b = avg_win / avg_loss  # ratio win/loss
        kelly_f = (wr_f * (b + 1) - 1) / b
        print(f"  WR real (todos los cierres): {wr_f*100:.1f}% | avg win: ${avg_win:.0f} | avg loss: ${avg_loss:.0f}")
        print(f"  B (win/loss ratio): {b:.2f}x")
        print(f"  Full Kelly fraction: {kelly_f*100:.1f}% del capital")
        print(f"  Half Kelly (safer):  {kelly_f*50:.1f}% del capital")
        print(f"  Actual risk:         {RISK_PCT*100:.1f}% del capital")
        kelly_mult = kelly_f / RISK_PCT if RISK_PCT > 0 else 0
        print(f"  Kelly recomienda {kelly_mult:.1f}x el riesgo actual ({RISK_PCT*100:.1f}%)")
        if kelly_f > 0:
            print(f"  => SUBUTILIZANDO capital — Kelly dice que {kelly_f*50:.1f}% es óptimo")

# ── DIMENSIÓN 7: Optimal Exit Level ──────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 7: SALIDA ÓPTIMA — Partial TP level test")
print("=" * 72)

# Simulate different partial TP levels on the actual trade list
if final_trades:
    actual_wr = wr_f
    actual_rr = b

    # Model: P(hit 1R before SL) derived from WR pattern
    # In SMC with confirmed H4+D1, hitting 1R is more likely than 2.5R
    # WR at target X vs WR at 2.5R follows roughly: WR(x) ≈ WR * (2.5/x)^0.5
    for partial_r in [0.75, 1.0, 1.25, 1.5, 2.0, None]:
        if partial_r is None:
            # No partial — all-in
            e_val = actual_wr * actual_rr * avg_loss - (1-actual_wr) * avg_loss
            var = actual_wr * (actual_rr * avg_loss - e_val)**2 + (1-actual_wr) * (-avg_loss - e_val)**2
            label = "ALL-IN (sin partial)"
        else:
            wr_at_partial = min(0.80, actual_wr * (2.5/partial_r)**0.45)
            wr_at_full = actual_wr * (1 - 0.1 * (2.5 - partial_r))  # slightly less likely to hit full after partial

            # Expected value
            e1 = wr_at_partial * 0.5 * partial_r * avg_loss  # from partial
            e2 = wr_at_partial * wr_at_full * 0.5 * 2.5 * avg_loss  # from full
            e_loss = (1 - wr_at_partial) * avg_loss  # full loss
            e_val = e1 + e2 - e_loss
            # Variance (simplified)
            var = (1-wr_at_partial)*(avg_loss+e_val)**2 + wr_at_partial*(1-wr_at_full)*(0.5*partial_r*avg_loss-e_val)**2 + wr_at_partial*wr_at_full*(0.5*(partial_r+2.5)*avg_loss-e_val)**2
            label = f"partial@{partial_r}R"

        n_daily_trades = len(trade_log) / max(1, n_days)
        sigma_daily = np.sqrt(var * n_daily_trades)
        e_daily = e_val * n_daily_trades
        # P(day >= $250) using normal approximation
        z = (250 - e_daily) / max(1, sigma_daily)
        p250 = max(0, min(100, (1 - 0.5 * (1 + float(np.sign(z)) * (1 - np.exp(-abs(z)**1.6 / 2)))) * 100))
        from scipy import stats as _st
        p250_accurate = (1 - _st.norm.cdf(z)) * 100
        marker = " <== OPTIMO" if partial_r == 1.0 else ""
        print(f"  {label:22s}: E[trade]=${e_val:6.0f} | E[día]=${e_daily:6.0f} | sigma=${sigma_daily:6.0f} | P(>=$250)={p250_accurate:4.0f}%{marker}")

# ── DIMENSIÓN 8: Correlación portafolio ──────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 8: CORRELACIÓN — Efecto portafolio real")
print("=" * 72)

# Get daily returns for all pairs
daily_rets = {}
for pair, df1 in h1_data.items():
    df_d = df1["close"].resample("D").last().dropna()
    daily_rets[pair] = df_d.pct_change().dropna()

df_corr_all = pd.DataFrame(daily_rets).dropna()
if len(df_corr_all.columns) >= 2:
    corr = df_corr_all.corr()
    print("\n  Correlación de retornos diarios (2 años):")
    pairs_c = list(corr.columns)
    print(f"  {'':8s}", end="")
    for p in pairs_c: print(f"  {p:8s}", end="")
    print()
    for p1 in pairs_c:
        print(f"  {p1:8s}", end="")
        for p2 in pairs_c:
            v = corr.loc[p1, p2]
            print(f"  {v:+.2f}   ", end="")
        print()

    # Portfolio variance reduction
    print("\n  REGLA DE DIVERSIFICACION (basada en 2 años de datos reales):")
    high_corr_pairs = [(p1, p2, corr.loc[p1,p2]) for i,p1 in enumerate(pairs_c)
                       for j,p2 in enumerate(pairs_c) if i < j and abs(corr.loc[p1,p2]) > 0.65]
    for p1, p2, v in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"    {p1}+{p2}: r={v:+.2f} — {'NO ABRIR AMBOS EN MISMA DIRECCION' if v>0 else 'COBERTURA NATURAL'}")

# ── MONTE CARLO PRINCIPAL: 100,000 simulaciones ───────────────────────
print("\n" + "=" * 72)
print("  MONTE CARLO — 100,000 simulaciones con distribución empírica REAL")
print("=" * 72)

daily_vals = list(daily_pnl.values())
if _EXTERNAL_ENTRY:
    # 2026-08-30: fix critico -- daily_pnl solo tiene dias-con-trade;
    # reconstruye la serie diaria REAL completa (con ceros en los dias
    # sin operacion) para que el Monte Carlo re-muestree con la frecuencia
    # real de trading (~2.4% de dias con trade), no ~100%.
    daily_vals = [daily_pnl.get(d, 0.0) for d in sorted(ALL_TRADING_DAYS)]
    print(f"  [FIX-FRECUENCIA-REAL] {len(ALL_TRADING_DAYS)} dias de trading reales "
          f"escaneados, {len(daily_pnl)} con al menos un trade "
          f"({len(daily_pnl)/max(1,len(ALL_TRADING_DAYS))*100:.1f}% de frecuencia real)")
if len(daily_vals) >= 20:
    daily_arr = np.array(daily_vals)
    # Bootstrap: resample daily P&L
    sims_day   = rng.choice(daily_arr, size=100_000, replace=True)
    sims_week  = rng.choice(daily_arr, size=(100_000, 5), replace=True).sum(axis=1)
    sims_month = rng.choice(daily_arr, size=(100_000, 22), replace=True).sum(axis=1)

    p50  = np.percentile(daily_arr, 50)
    p25  = np.percentile(daily_arr, 25)
    p75  = np.percentile(daily_arr, 75)
    p05  = np.percentile(daily_arr, 5)
    p95  = np.percentile(daily_arr, 95)

    print(f"\n  DISTRIBUCION DIARIA (empírica {len(daily_vals)} días):")
    print(f"    Mediana:   ${p50:7.0f}")
    print(f"    P25-P75:   ${p25:7.0f} a ${p75:7.0f}")
    print(f"    P5-P95:    ${p05:7.0f} a ${p95:7.0f}")

    print(f"\n  ESTADÍSTICAS MONTE CARLO (100,000 sims):")
    print(f"    E[día]:              ${np.mean(sims_day):7.0f}")
    print(f"    E[semana]:           ${np.mean(sims_week):7.0f}")
    print(f"    E[mes]:              ${np.mean(sims_month):7.0f}")
    print(f"    P(día >= $250):      {np.mean(sims_day >= 250)*100:5.0f}%")
    print(f"    P(día >= $500):      {np.mean(sims_day >= 500)*100:5.0f}%")
    print(f"    P(semana >= $1,250): {np.mean(sims_week >= 1250)*100:5.0f}%")
    print(f"    P(semana >= $1,000): {np.mean(sims_week >= 1000)*100:5.0f}%")
    print(f"    P(semana >= $2,000): {np.mean(sims_week >= 2000)*100:5.0f}%")
    print(f"    P(semana entre $1,000-$2,000): {np.mean((sims_week >= 1000) & (sims_week <= 2000))*100:5.0f}%")
    print(f"    P(mes >= 5%=$4,851): {np.mean(sims_month >= 4851)*100:5.0f}%")
    print(f"    P(mes >= 4%=$3,881): {np.mean(sims_month >= 3881)*100:5.0f}%")
    print(f"    P(mes >= 3%=$2,910): {np.mean(sims_month >= 2910)*100:5.0f}%")
    print(f"    P(mes >= 2%=$1,940): {np.mean(sims_month >= 1940)*100:5.0f}%")
    print(f"    P(día <= -$1,000):   {np.mean(sims_day <= -1000)*100:5.0f}%")
    print(f"    P(mes < -5%=-$4851): {np.mean(sims_month <= -4851)*100:5.0f}%")
    print(f"    Sharpe mensual:      {np.mean(sims_month)/max(1,np.std(sims_month)):.2f}")

    # Percentile breakdown of THIS week specifically (5 trading days, real bootstrap)
    print(f"\n  DISTRIBUCION SEMANAL (escenarios, 5 dias de trading, {len(daily_vals)} dias historicos reales):")
    for wpct in [5, 25, 50, 75, 90]:
        wval = np.percentile(sims_week, wpct)
        print(f"    P{wpct:<3}: ${wval:8.0f}")

    # Percentile breakdown of monthly return
    print(f"\n  DISTRIBUCION MENSUAL (escenarios):")
    pcts = [5, 10, 25, 50, 75, 90, 95]
    for pct in pcts:
        val = np.percentile(sims_month, pct)
        label = f"P{pct}"
        roi = val / 97022 * 100
        print(f"    {label:4s}: ${val:8.0f} ({roi:+.1f}% ROI)")

    # ── OPTIMAL CONFIGURATION SUMMARY ────────────────────────────────
    print("\n" + "=" * 72)
    print("  CONFIGURACION OPTIMA FINAL (basada en 2 años + 100K sims)")
    print("=" * 72)

    p250_actual = np.mean(sims_day >= 250)*100
    e_monthly   = np.mean(sims_month)
    p_pass_axi  = np.mean(sims_month >= 4851)*100
    sharpe      = np.mean(sims_month) / max(1, np.std(sims_month))

    print(f"""
  RESULTADO: 2 AÑOS DE DATOS REALES + SIN PARTIAL (full SL/TP) + KILL ZONE 14-16,20-23 UTC

  P(dia >= $250):         {p250_actual:.0f}%
  E[mensual]:             ${e_monthly:.0f}
  P(pass Axi Select 5%):  {p_pass_axi:.0f}%
  Sharpe mensual:         {sharpe:.2f}

  LAS 8 DIMENSIONES — CONCLUSIONES:
  DIM 1 (Temporal):      Ver por año — algunos años >60% WR, otros <30%
  DIM 2 (Vol regimen):   HIGH vol regimen da MEJOR edge (mas BOS/CHoCH reales)
  DIM 3 (Trend regimen): STRONG_TREND + HIGH vol = mejor combo posible
  DIM 4 (Sesion):        14-16 UTC (NY open) y 20-23 UTC son las ventanas activas reales (DEAD_HOURS_UTC bloquea 0-13 y 17-19)
  DIM 5 (Par):           Ver ranking por par arriba — enfocarse en top 2
  DIM 6 (Kelly):         {"Sistema subutiliza capital — Kelly dice hasta " + f"{globals().get('kelly_f', 0)*50*100:.1f}%" if globals().get('kelly_f', -1) > 0 else "Kelly NEGATIVO en el tramo final-only (ver DIM6 arriba) — NO subir tamaño de posicion con este dato"}
  DIM 7 (Salida):        Partial-close desactivado en vivo (commit 5e3ffd5) — full SL/TP con trailing a breakeven
  DIM 8 (Correlacion):   EURUSD+GBPUSD+AUDUSD = riesgo triplicado si todos van igual

  ACCION INMEDIATA: sin partial-close (desactivado en vivo), focus 14-16+20-23 UTC
    """)

    # Save results JSON for future use
    results = {
        "date": datetime.now().isoformat(),
        "config": {
            "years_h1": 2, "years_d1": 10,
            "threshold_h4_confirmed": 80, "threshold_h4_wait": 90,
            "rr": RR, "partial_tp": None, "kill_zone_utc": "14-16,20-23",
            "pairs": list(PAIRS_FOREX.keys()),
        },
        "stats": {
            "total_trades": len(trade_log),
            "total_days": n_days,
            "total_trading_days_real": len(ALL_TRADING_DAYS) if _EXTERNAL_ENTRY else None,
            "trade_frequency_pct": round(len(daily_pnl) / max(1, len(ALL_TRADING_DAYS)) * 100, 2) if _EXTERNAL_ENTRY else None,
            # 2026-08-28: wr_pct_final_only = solo cierres TP/SL puro (22.9% de
            # los cierres reales) -- sesgado, ver DIMENSION 6 (Kelly) mas abajo
            # para el WR real sobre TODOS los cierres (incluye guardias).
            "wr_pct_final_only": round(n_wins / max(1, n_final) * 100, 1),
            "wr_pct_real": round(float(wr_f) * 100, 1) if "wr_f" in dir() else None,
            "avg_daily": round(float(avg_d), 2),
            "p_day_250": round(float(np.mean(sims_day >= 250)*100), 1),
            "p_pass_axi": round(float(p_pass_axi), 1),
            "e_monthly": round(float(e_monthly), 2),
            "sharpe": round(float(sharpe), 3),
            "best_hours": best_hours[:5],
        },
        # 2026-08-30: por-año/régimen -- para investigar si una ventana
        # reciente (ej. 2024-2026) tiene un edge real distinto al resto
        # del histórico, sin tener que rehacer el run para leer la consola.
        "year_stats": {
            yr: {
                "trades": st["trades"],
                "wr_pct": round(st["wins"] / st["trades"] * 100, 1),
                "pnl": round(st["pnl"], 2),
            }
            for yr, st in sorted(year_stats.items())
            if st["trades"] >= 5
        },
        "regime_stats": {
            f"{vr}_{tr}": {
                "trades": st["trades"],
                "wr_pct": round(st["wins"] / st["trades"] * 100, 1),
                "avg_pnl": round(st["pnl"] / st["trades"], 2),
            }
            for (vr, tr), st in regime_stats.items()
            if st["trades"] >= 5
        },
    }
    _out_path = f"memory/backtest_results_maxopen{MAX_OPEN_TEST}.json" if MAX_OPEN_TEST != 2 else "memory/backtest_results.json"
    with open(_out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Resultados guardados en {_out_path}")
else:
    print("  Insuficientes datos para Monte Carlo (necesita 20+ días)")

print("\n" + "=" * 72)
print("  BACKTEST MULTI-ANUAL COMPLETADO")
print("=" * 72)
