"""
SMC Backtest — 6 months historical simulation.

Question: Does the current config statistically guarantee >= $250/day?
Answer: Run actual SMC signals on real historical data, simulate trades,
        compute win rate / signals-per-day / P(daily >= $250).

Pairs: EURUSD, GBPUSD, AUDUSD, USDCAD, NAS100 (via yfinance)
TF: H1 (primary), H4 (direction filter)
Period: 6 months lookback
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────
CAPITAL          = 97_022.0
RISK_PCT         = 0.005          # 0.5% per trade
MAX_DOLLAR_RISK  = 275.0          # adaptive cap (start of day)
RR               = 2.5            # TP/SL ratio
MIN_SCORE        = 80             # MT5_REAL_SCORE_THRESHOLD
H4_THRESHOLD     = 95
H1_THRESHOLD     = 100
MAX_POSITIONS    = 3
DEAD_HOURS_UTC   = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19}  # fix 2026-07-08: was missing 13 and the 17-19 block (core/supervisor.py:121)
DAILY_TARGET     = 250.0

PAIRS_YFINANCE = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NAS100":  "^NDX",
}

PIP_SIZE  = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001, "NAS100": 1.0}
PIP_VALUE = {"EURUSD": 10.0,   "GBPUSD": 10.0,   "AUDUSD": 10.0,   "USDCAD": 10.0,   "NAS100": 1.0}
MIN_VOL   = {"EURUSD": 0.01,   "GBPUSD": 0.01,   "AUDUSD": 0.01,   "USDCAD": 0.01,   "NAS100": 1.0}
MAX_VOL   = {"EURUSD": 2.0,    "GBPUSD": 2.0,    "AUDUSD": 2.0,    "USDCAD": 2.0,    "NAS100": 1.0}

# ── Data Download ────────────────────────────────────────────────────
print("=" * 65)
print("  SMC BACKTEST — 6 MESES HISTÓRICO")
print(f"  Target: $250/dia | Capital: ${CAPITAL:,.0f}")
print("=" * 65)

import yfinance as yf

END   = datetime.now()
START = END - timedelta(days=185)  # 6 months

print(f"\n[DATA] Descargando {len(PAIRS_YFINANCE)} pares {START.strftime('%Y-%m-%d')} → {END.strftime('%Y-%m-%d')}")

h1_data  = {}
h4_data  = {}
d1_data  = {}

for pair, ticker in PAIRS_YFINANCE.items():
    try:
        df1 = yf.download(ticker, start=START, end=END, interval="1h", progress=False, auto_adjust=True)
        df4 = yf.download(ticker, start=START, end=END, interval="1d", progress=False, auto_adjust=True)
        if df1.empty or df4.empty:
            print(f"  {pair}: sin datos")
            continue
        # Flatten MultiIndex columns if present
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.get_level_values(0)
        if isinstance(df4.columns, pd.MultiIndex):
            df4.columns = df4.columns.get_level_values(0)
        df1.columns = [c.lower() for c in df1.columns]
        df4.columns = [c.lower() for c in df4.columns]
        # Resample 1h -> 4h for H4
        df4h = df1.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        h1_data[pair] = df1
        h4_data[pair] = df4h
        d1_data[pair] = df4  # using daily as D1
        print(f"  {pair}: H1={len(df1)} bars, H4={len(df4h)} bars, D1={len(df4)} bars")
    except Exception as e:
        print(f"  {pair}: ERROR — {e}")

if not h1_data:
    print("\n[ERROR] No se pudo descargar datos. Abortando.")
    sys.exit(1)

# ── SMC Signal Functions ──────────────────────────────────────────────
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.DataFrame({"hl": h-l, "hc": (h - c.shift()).abs(), "lc": (l - c.shift()).abs()}).max(axis=1)
    return tr.rolling(n).mean()

def d1_direction(d1df, date):
    """D1 trend from EMA50."""
    d = d1df[d1df.index.date <= date.date()]
    if len(d) < 50:
        return "UNKNOWN"
    e = ema(d["close"], 50)
    return "LONG" if d["close"].iloc[-1] > e.iloc[-1] else "SHORT"

def h4_direction(h4df, dt):
    """H4 structural direction from last 50 bars EMA cross."""
    subset = h4df[h4df.index <= dt]
    if len(subset) < 20:
        return "WAIT"
    e_fast = ema(subset["close"], 8)
    e_slow = ema(subset["close"], 20)
    if e_fast.iloc[-1] > e_slow.iloc[-1]:
        return "LONG"
    elif e_fast.iloc[-1] < e_slow.iloc[-1]:
        return "SHORT"
    return "WAIT"

def smc_signal(df, idx, direction_bias):
    """
    Simplified SMC signal: BOS/CHoCH + Order Block check.
    Returns (has_signal: bool, signal_dir: str, score: int, sl_atr: float)
    """
    if idx < 50:
        return False, "WAIT", 0, 0.0

    window = df.iloc[max(0, idx-50):idx+1]
    if len(window) < 30:
        return False, "WAIT", 0, 0.0

    close = window["close"]
    high  = window["high"]
    low   = window["low"]

    # ATR for SL sizing
    atr_val = atr(window, 14).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return False, "WAIT", 0, 0.0

    # EMA trend for bias
    e8  = ema(close, 8).iloc[-1]
    e21 = ema(close, 21).iloc[-1]
    cur = close.iloc[-1]

    # Score components
    score = 0
    signal_dir = "WAIT"

    # Structure: last 5-bar swing high/low
    recent_high = high.iloc[-5:].max()
    recent_low  = low.iloc[-5:].min()
    prev_high   = high.iloc[-15:-5].max()
    prev_low    = low.iloc[-15:-5].min()

    # BOS detection (simplified)
    bullish_bos = cur > prev_high and close.iloc[-2] <= prev_high
    bearish_bos = cur < prev_low  and close.iloc[-2] >= prev_low

    # Momentum confirmation (last 3 bars)
    momentum_up   = cur > close.iloc[-4]
    momentum_down = cur < close.iloc[-4]

    if direction_bias == "LONG":
        if bullish_bos:
            score += 30
        if e8 > e21:
            score += 20
        if momentum_up:
            score += 15
        if cur > e21:
            score += 15
        # Order block proximity (simplified: last strong bullish candle)
        body = (window["close"] - window["open"]).iloc[-10:]
        if body.max() > atr_val * 0.8:
            score += 20
        signal_dir = "LONG" if score >= 50 else "WAIT"

    elif direction_bias == "SHORT":
        if bearish_bos:
            score += 30
        if e8 < e21:
            score += 20
        if momentum_down:
            score += 15
        if cur < e21:
            score += 15
        body = (window["open"] - window["close"]).iloc[-10:]
        if body.max() > atr_val * 0.8:
            score += 20
        signal_dir = "SHORT" if score >= 50 else "WAIT"

    has_signal = signal_dir in ("LONG", "SHORT")
    # Scale score to match bot's 0-100 range
    score = min(100, int(score * 1.2))

    return has_signal, signal_dir, score, float(atr_val)

def calc_volume(capital, sl_pips, pair):
    """Calculate position volume with risk cap."""
    pip_v = PIP_VALUE[pair]
    if sl_pips <= 0 or pip_v <= 0:
        return 0.0
    risk_usd = capital * RISK_PCT
    vol = risk_usd / (sl_pips * pip_v)
    vol = max(MIN_VOL[pair], min(MAX_VOL[pair], vol))
    # Apply MAX_DOLLAR_RISK cap
    actual = vol * sl_pips * pip_v
    if actual > MAX_DOLLAR_RISK:
        vol = MAX_DOLLAR_RISK / (sl_pips * pip_v)
        vol = round(int(vol / 0.01) * 0.01, 2)
        vol = max(MIN_VOL[pair], vol)
    return round(vol, 2)

# ── Backtest Engine ────────────────────────────────────────────────────
print("\n[BACKTEST] Simulando señales H1 con filtro H4+D1...")

all_trades  = []
daily_pnl   = defaultdict(float)
daily_count = defaultdict(int)
signals_per_day = defaultdict(int)

for pair in h1_data:
    df1 = h1_data[pair]
    df4 = h4_data[pair]
    d1  = d1_data[pair]

    open_positions = []  # (entry_idx, direction, entry_price, sl, tp, volume, sl_pips)

    for idx in range(50, len(df1)):
        bar = df1.iloc[idx]
        dt  = df1.index[idx]

        # Skip weekends and dead hours
        if pd.Timestamp(dt).weekday() >= 5:
            continue
        hour_utc = pd.Timestamp(dt).hour
        if hour_utc in DEAD_HOURS_UTC:
            continue

        day_str = pd.Timestamp(dt).date()

        # Check/close open positions
        closed_this_bar = []
        for pos in open_positions:
            entry_idx_p, direction, entry_price, sl, tp, vol, sl_pips = pos

            pip_v = PIP_VALUE[pair]
            if direction == "LONG":
                hit_tp = bar["high"] >= tp
                hit_sl = bar["low"]  <= sl
                if hit_tp:
                    pnl = vol * sl_pips * pip_v * RR
                elif hit_sl:
                    pnl = -vol * sl_pips * pip_v
                else:
                    continue
            else:
                hit_tp = bar["low"]  <= tp
                hit_sl = bar["high"] >= sl
                if hit_tp:
                    pnl = vol * sl_pips * pip_v * RR
                elif hit_sl:
                    pnl = -vol * sl_pips * pip_v
                else:
                    continue

            daily_pnl[day_str]   += pnl
            all_trades.append({
                "pair": pair, "dir": direction, "date": day_str,
                "pnl": pnl, "vol": vol, "win": pnl > 0
            })
            closed_this_bar.append(pos)

        for pos in closed_this_bar:
            open_positions.remove(pos)

        # Skip if max positions
        if len(open_positions) >= MAX_POSITIONS:
            continue

        # Get direction filters
        d1_dir = d1_direction(d1, pd.Timestamp(dt))
        h4_dir = h4_direction(df4, pd.Timestamp(dt))

        if d1_dir == "UNKNOWN":
            continue

        # H4 must confirm OR be WAIT (WAIT allows only with high score)
        if h4_dir not in (d1_dir, "WAIT"):
            continue  # H4 contradicts D1

        # Generate SMC signal
        bias = d1_dir
        has_sig, sig_dir, score, atr_val = smc_signal(df1, idx, bias)

        if not has_sig or sig_dir == "WAIT":
            continue

        # Apply H4 filter
        effective_threshold = H1_THRESHOLD
        if h4_dir == "WAIT":
            effective_threshold = 115

        if score < effective_threshold:
            continue

        # Check no conflicting direction
        for pos in open_positions:
            if pos[1] != sig_dir:
                # Opposite direction open — skip
                continue

        signals_per_day[day_str] += 1

        # Size position
        sl_atr = atr_val * 1.5
        pip_sz = PIP_SIZE[pair]
        sl_pips = sl_atr / pip_sz
        vol = calc_volume(CAPITAL, sl_pips, pair)

        if vol < MIN_VOL.get(pair, 0.01) * 0.99:
            continue

        entry = bar["close"]
        if sig_dir == "LONG":
            sl = entry - sl_atr
            tp = entry + sl_atr * RR
        else:
            sl = entry + sl_atr
            tp = entry - sl_atr * RR

        open_positions.append((idx, sig_dir, entry, sl, tp, vol, sl_pips))

    n_pair_trades = sum(1 for t in all_trades if t["pair"] == pair)
    n_pair_wins   = sum(1 for t in all_trades if t["pair"] == pair and t["win"])
    if n_pair_trades > 0:
        wr = n_pair_wins / n_pair_trades * 100
        print(f"  {pair:8s}: {n_pair_trades:3d} trades | WR={wr:.1f}%")

# ── Results Analysis ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTADOS BACKTEST")
print("=" * 65)

if not all_trades:
    print("  [ERROR] Sin trades — los filtros son demasiado restrictivos")
    sys.exit(1)

total_trades = len(all_trades)
total_wins   = sum(1 for t in all_trades if t["win"])
total_losses = total_trades - total_wins
win_rate     = total_wins / total_trades if total_trades > 0 else 0

# Daily stats
daily_pnl_list = list(daily_pnl.values())
avg_daily      = np.mean(daily_pnl_list) if daily_pnl_list else 0
median_daily   = np.median(daily_pnl_list) if daily_pnl_list else 0
std_daily      = np.std(daily_pnl_list) if daily_pnl_list else 0
days_above_250 = sum(1 for v in daily_pnl_list if v >= 250)
days_negative  = sum(1 for v in daily_pnl_list if v < 0)
n_trading_days = len(daily_pnl_list)

avg_signals   = np.mean(list(signals_per_day.values())) if signals_per_day else 0
max_daily_loss = min(daily_pnl_list) if daily_pnl_list else 0
max_daily_win  = max(daily_pnl_list) if daily_pnl_list else 0

# Per-trade stats
avg_win_usd  = np.mean([t["pnl"] for t in all_trades if t["win"]]) if total_wins > 0 else 0
avg_loss_usd = np.mean([t["pnl"] for t in all_trades if not t["win"]]) if total_losses > 0 else 0
profit_factor = (total_wins * avg_win_usd) / abs(total_losses * avg_loss_usd) if total_losses > 0 and avg_loss_usd != 0 else 0

print(f"\n  Periodo: {START.strftime('%Y-%m-%d')} → {END.strftime('%Y-%m-%d')}")
print(f"  Trades totales:    {total_trades}")
print(f"  Win rate:          {win_rate*100:.1f}%")
print(f"  Profit factor:     {profit_factor:.2f}")
print(f"  Avg win:           ${avg_win_usd:.0f}")
print(f"  Avg loss:          ${avg_loss_usd:.0f}")
print(f"\n  Trading days:      {n_trading_days}")
print(f"  Avg signals/day:   {avg_signals:.1f}")
print(f"  Avg daily P&L:     ${avg_daily:.0f}")
print(f"  Median daily P&L:  ${median_daily:.0f}")
print(f"  Std daily P&L:     ${std_daily:.0f}")
print(f"\n  Dias >= $250:      {days_above_250}/{n_trading_days} ({days_above_250/n_trading_days*100:.0f}%)")
print(f"  Dias negativos:    {days_negative}/{n_trading_days} ({days_negative/n_trading_days*100:.0f}%)")
print(f"  Mejor dia:         ${max_daily_win:.0f}")
print(f"  Peor dia:          ${max_daily_loss:.0f}")

# Probability analysis
p_above_250 = days_above_250 / n_trading_days if n_trading_days > 0 else 0
ftmo_daily_loss = 100_000 * 0.05  # $5,000
days_ftmo_breach = sum(1 for v in daily_pnl_list if v < -ftmo_daily_loss * 0.6)

print(f"\n  P(dia >= $250):    {p_above_250*100:.0f}%")
print(f"  P(FTMO breach):    {days_ftmo_breach}/{n_trading_days} dias")

# Top 5 worst and best days
sorted_days = sorted(daily_pnl.items(), key=lambda x: x[1])
print("\n  5 PEORES DIAS:")
for d, v in sorted_days[:5]:
    print(f"    {d}: ${v:.0f}")
print("  5 MEJORES DIAS:")
for d, v in sorted_days[-5:]:
    print(f"    {d}: ${v:.0f}")

# ── Monte Carlo ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MONTE CARLO (1000 semanas simuladas)")
print("=" * 65)

if daily_pnl_list:
    rng = np.random.default_rng(42)
    n_sim = 1000
    week_pnl = []
    for _ in range(n_sim):
        days = rng.choice(daily_pnl_list, size=5, replace=True)
        week_pnl.append(float(np.sum(days)))

    avg_week  = np.mean(week_pnl)
    days_sim  = rng.choice(daily_pnl_list, size=(n_sim, 22), replace=True)  # 22 trading days
    month_pnl = days_sim.sum(axis=1)

    p_week_positive   = np.mean(np.array(week_pnl) > 0) * 100
    p_month_5pct      = np.mean(month_pnl >= 100_000 * 0.05) * 100  # 5% = $5K on $100K
    expected_month    = np.mean(month_pnl)

    print(f"  Avg semana simulada:     ${avg_week:.0f}")
    print(f"  P(semana positiva):      {p_week_positive:.0f}%")
    print(f"  P(mes >= 5% FTMO):       {p_month_5pct:.0f}%")
    print(f"  Expected mensual:        ${expected_month:.0f}")
    print(f"  ROI mensual esperado:    {expected_month/CAPITAL*100:.2f}%")

# ── Diagnosis ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DIAGNOSTICO")
print("=" * 65)

issues = []
strengths = []

if win_rate >= 0.50:
    strengths.append(f"Win rate {win_rate*100:.0f}% >= 50% — edge positivo")
else:
    issues.append(f"Win rate {win_rate*100:.0f}% < 50% — con RR=2.5 sigue siendo positivo si > 28.6%")

if profit_factor >= 1.5:
    strengths.append(f"Profit factor {profit_factor:.2f} — robusto")
elif profit_factor >= 1.0:
    issues.append(f"Profit factor {profit_factor:.2f} — aceptable pero no robusto")
else:
    issues.append(f"Profit factor {profit_factor:.2f} — negativo!")

if avg_signals >= 2.0:
    strengths.append(f"{avg_signals:.1f} señales/dia — suficiente frecuencia")
else:
    issues.append(f"Solo {avg_signals:.1f} señales/dia — los filtros son muy restrictivos")

if p_above_250 >= 0.40:
    strengths.append(f"{p_above_250*100:.0f}% dias >= $250")
else:
    issues.append(f"Solo {p_above_250*100:.0f}% dias alcanzan $250 — target difícil con avg={avg_daily:.0f}")

if avg_daily >= 100:
    strengths.append(f"Daily avg ${avg_daily:.0f} es positivo")
else:
    issues.append(f"Daily avg ${avg_daily:.0f} insuficiente para target")

print("\n  FORTALEZAS:")
for s in strengths:
    print(f"    [+] {s}")

print("\n  PROBLEMAS:")
if not issues:
    print("    [+] Sin problemas identificados")
for i in issues:
    print(f"    [-] {i}")

# Final verdict
print("\n" + "=" * 65)
print("  VEREDICTO")
print("=" * 65)

if p_above_250 >= 0.60 and profit_factor >= 1.5 and avg_signals >= 1.5:
    verdict = "CONFIGURACION VIABLE — $250/dia objetivo alcanzable con margen"
elif p_above_250 >= 0.35 and profit_factor >= 1.0:
    verdict = "CONFIGURACION POSITIVA — $250/dia factible en dias con señales fuertes"
else:
    verdict = "CONFIGURACION NECESITA AJUSTES — ver diagnostico"

print(f"\n  {verdict}")
print(f"  Expected diario: ${avg_daily:.0f}")
print(f"  P(dia >= $250): {p_above_250*100:.0f}%")
print(f"  Win rate: {win_rate*100:.1f}% | Profit factor: {profit_factor:.2f}")
print()
