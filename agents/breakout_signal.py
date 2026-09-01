"""
Motor de señal Donchian Breakout — reemplaza el motor SMC/BOS/CHoCH para el
scan de forex en MT5 (2026-08-31, tras la sesión de backtesting que encontró
que el motor SMC generaba señales de calidad pero demasiado escasas: solo
~100-165 trades en 16 años completos, insuficiente para una meta MENSUAL).

Estrategia: ruptura del máximo/mínimo de la vela anterior (Donchian N=1) +
SL por ATR14 + TP a 10x esa distancia (rara vez se alcanza -- el cierre real
lo hacen los guards de core/position_guards.py, no el TP). Sin filtro de
tendencia (probado explícitamente: quitarlo mejoró el resultado, las rupturas
contra-tendencia resultaron igual de buenas).

Validado sobre 16 años reales de datos MT5 (los mismos 6 pares que
MT5_SYMBOLS en core/supervisor.py), 66,377 trades, con costo de spread real
modelado: P(pasar Axi Select 5% mensual) = 96% cuando se restringe a horario
20:00-21:00 UTC (ver DEAD_HOURS_UTC en core/supervisor.py). Detalle completo
en SESION_ACTUAL.md.

Esta es la implementación EXACTA de scripts/backtest_multiyear.py::breakout_signal()
(mismas fórmulas, mismos multiplicadores) -- cualquier cambio aquí debe
replicarse ahí también para no romper la paridad backtest-vivo que costó
descubrir el bug del motor SMC.
"""

from typing import Optional

import pandas as pd

from agents.signal_agent import SignalType, TradeSignal

DONCHIAN_N = 1
ATR_MULT_SL = 0.75
# 2026-08-31 (2da correccion, con spread real): RR subido de 10.0 a 20.0 --
# revalidado sobre 16 anios reales con el costo de spread REAL de esta
# cuenta (medido en vivo, 3-6x mas ancho de lo asumido originalmente):
# RR=10 daba P(pass)=72%, RR=20 dio 74-75% (RR=30 no mejoro mas, techo
# confirmado). Ver SESION_ACTUAL.md.
RR_MULT = 20.0
MIN_BARS = DONCHIAN_N + 20  # margen de seguridad sobre el lookback de ATR14


def _atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(14).mean()


def generate_breakout_signal(df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradeSignal]:
    """Devuelve un TradeSignal LONG/SHORT si hay ruptura de canal válida, o
    un TradeSignal WAIT si no. Devuelve None solo si faltan datos suficientes."""
    if df is None or len(df) < MIN_BARS:
        return None

    highs = df["high"].values
    lows = df["low"].values
    close_now = float(df["close"].values[-1])
    if close_now <= 0:
        return None

    # Donchian N=1: máximo/mínimo de la vela INMEDIATAMENTE anterior (excluye
    # la vela actual -- sin lookahead, misma slice que el backtest validado).
    d_high = float(highs[-DONCHIAN_N - 1:-1].max())
    d_low = float(lows[-DONCHIAN_N - 1:-1].min())

    direction = None
    if close_now > d_high:
        direction = SignalType.LONG
    elif close_now < d_low:
        direction = SignalType.SHORT

    if direction is None:
        return TradeSignal(
            symbol=symbol, signal_type=SignalType.WAIT, entry=close_now,
            stop_loss=None, take_profit=0.0, timeframe=timeframe,
            trigger="sin ruptura de canal", confidence=0.0,
        )

    atr_v = _atr14(df).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0:
        return TradeSignal(
            symbol=symbol, signal_type=SignalType.WAIT, entry=close_now,
            stop_loss=None, take_profit=0.0, timeframe=timeframe,
            trigger="ATR invalido", confidence=0.0,
        )

    sl_dist = ATR_MULT_SL * atr_v
    entry = close_now
    is_long = direction == SignalType.LONG
    sl = entry - sl_dist if is_long else entry + sl_dist
    tp = entry + sl_dist * RR_MULT if is_long else entry - sl_dist * RR_MULT

    channel_ref = d_high if is_long else d_low
    strength = abs(close_now - channel_ref) / atr_v
    score = int(min(100, 60 + strength * 40))

    return TradeSignal(
        symbol=symbol,
        signal_type=direction,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        timeframe=timeframe,
        trigger=f"donchian_breakout N={DONCHIAN_N} strength={strength:.2f}xATR",
        confidence=min(0.95, 0.5 + strength * 0.2),
        decision_score=score,
    )
