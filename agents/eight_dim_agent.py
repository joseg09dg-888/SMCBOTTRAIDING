"""
EightDimensionAgent — Análisis de mercado en 8 dimensiones simultáneas.

Las 8 dimensiones del mercado:
  DIM 1 — Temporal:      edge varía por año/trimestre/mes/hora
  DIM 2 — Volatilidad:   régimen HIGH/NORMAL/LOW → WR diferente
  DIM 3 — Tendencia:     STRONG_TREND vs CHOPPY vs LATERAL
  DIM 4 — Sesión:        horas UTC con mejor WR histórico (13-16 = gold)
  DIM 5 — Par:           performance relativa del par vs los demás
  DIM 6 — Kelly:         sizing óptimo por Kelly fraction
  DIM 7 — Salida:        nivel óptimo de partial TP (1.0R + BE)
  DIM 8 — Correlación:   portafolio — evitar riesgo duplicado entre pares

Resultado: score_multiplier (0.5–1.3) y allowed (bool) para la señal.
Se integra en el enrichment pipeline de supervisor.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class EightDimResult:
    """Result object returned by EightDimensionAgent.analyze()."""
    score_mult: float = 1.0        # multiply signal score by this
    allowed: bool = True           # False = block this trade (DIM 8 violation)
    vol_regime: str = "NORMAL"     # HIGH / NORMAL / LOW
    trend_regime: str = "UNKNOWN"  # STRONG_TREND / MILD_TREND / CHOPPY
    session_quality: str = "OK"    # GOLD / GOOD / OK / POOR
    dim_scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""               # why allowed=False


# ── Regime multiplier table (from 2-year historical backtest) ──────────
# (vol_regime, trend_regime) → score_multiplier
_REGIME_MULT: Dict[Tuple[str,str], float] = {
    ("HIGH",   "STRONG_TREND"):  1.25,
    ("NORMAL", "STRONG_TREND"):  1.10,
    ("HIGH",   "MILD_TREND"):    1.05,
    ("NORMAL", "MILD_TREND"):    1.00,
    ("LOW",    "STRONG_TREND"):  0.90,
    ("LOW",    "MILD_TREND"):    0.85,
    ("HIGH",   "CHOPPY"):        0.70,
    ("NORMAL", "CHOPPY"):        0.65,
    ("LOW",    "CHOPPY"):        0.50,
}

# ── Session (UTC hour) multiplier (from backtested data) ───────────────
# London/NY overlap 13-16 UTC = gold; late NY 16-20 = good; Asia = avoid
_SESSION_MULT: Dict[int, Tuple[float, str]] = {
    12: (0.85, "OK"),
    13: (0.60, "POOR"),   # Bloqueado en DEAD_HOURS: WR real=29%, avg=-$97/trade (backtest 2y)
    14: (1.25, "GOLD"),   # NY open — strongest momentum
    15: (1.15, "GOLD"),   # London close / NY mid
    16: (1.05, "GOOD"),   # London closed, NY still active
    17: (1.00, "GOOD"),
    18: (0.95, "GOOD"),
    19: (0.90, "OK"),
    20: (0.80, "OK"),
    21: (0.70, "POOR"),   # quiet pre-Asia
    22: (0.65, "POOR"),
    23: (0.60, "POOR"),
}

# ── Correlation groups (no more than 1 trade per group) ───────────────
_CORR_GROUPS: List[List[str]] = [
    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],   # USD-bear group (r > 0.65)
    ["USDCAD"],                                   # USD-bull solo (inverse)
    ["NAS100", "NAS100.fs", "US30", "US30.fs"],   # indices (uncorrelated with FX)
]


class EightDimensionAgent:
    """
    8-dimensional market filter for the SMC trading bot.

    Usage in supervisor.py:
        eight_d = EightDimensionAgent()
        result = eight_d.analyze(symbol, df_h1, open_positions_list, utc_hour)
        effective_score = int(base_score * result.score_mult)
        if not result.allowed:
            continue
    """

    def __init__(self) -> None:
        self._par_history: Dict[str, List[float]] = {}  # rolling WR per pair

    # ── Public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        df_h1: "pd.DataFrame",
        open_positions: Optional[List[dict]] = None,
        utc_hour: Optional[int] = None,
        direction: str = "LONG",
    ) -> EightDimResult:
        """
        Run all 8 dimensions and return combined result.

        Args:
            symbol:         instrument, e.g. 'EURUSD' or 'NAS100.fs'
            df_h1:          H1 OHLCV dataframe (needs 60+ bars)
            open_positions: list of dicts with 'symbol', 'type' keys (from MT5)
            utc_hour:       current UTC hour (0-23); auto-detected if None
            direction:      'LONG' or 'SHORT'
        """
        from datetime import datetime, timezone
        if utc_hour is None:
            utc_hour = datetime.now(timezone.utc).hour

        sym_base = symbol.split(".")[0].upper()

        dims: Dict[str, float] = {}
        result = EightDimResult()

        # DIM 1 — Temporal (hour of session)
        dims["DIM1_temporal"] = self._dim1_temporal(utc_hour)

        # DIM 2 — Volatility regime
        vol_r, dims["DIM2_volatility"] = self._dim2_volatility(df_h1)
        result.vol_regime = vol_r

        # DIM 3 — Trend regime
        trend_r, dims["DIM3_trend"] = self._dim3_trend(df_h1, direction)
        result.trend_regime = trend_r

        # DIM 4 — Session quality (UTC kill zone)
        sess_mult, sess_q, dims["DIM4_session"] = self._dim4_session(utc_hour)
        result.session_quality = sess_q

        # DIM 5 — Pair-specific strength index
        dims["DIM5_pair"] = self._dim5_pair(sym_base, df_h1)

        # DIM 6 — Consecutive-loss circuit breaker + monthly profit lock
        dims["DIM6_kelly"] = self._dim6_kelly(sym_base)
        if dims["DIM6_kelly"] == 0.0:
            result.allowed = False
            result.reason = "DIM6-BLOCK: 3 consecutive losses — 24h circuit breaker active"
            result.dim_scores = dims
            result.score_mult = 0.0
            return result

        # DIM 7 — Exit quality (current ATR suitable for partial TP?)
        dims["DIM7_exit"] = self._dim7_exit(df_h1)

        # DIM 8 — Correlation / portfolio risk
        allowed, reason = self._dim8_correlation(sym_base, direction, open_positions or [])
        if not allowed:
            result.allowed = False
            result.reason = reason
            result.dim_scores = dims
            result.score_mult = 0.0
            return result

        # ── Combine dimensions into final multiplier ────────────────
        # Regime mult (DIM 2 + 3 combined)
        regime_key = (vol_r, trend_r)
        regime_mult = _REGIME_MULT.get(regime_key, 1.0)

        # Session mult (DIM 4)
        # DIM 1 (temporal) is incorporated as a small modifier
        temporal_mod = dims["DIM1_temporal"]  # 0.9 – 1.0

        # Pair strength (DIM 5)
        pair_mod = dims["DIM5_pair"]  # 0.8 – 1.2

        # Circuit breaker / monthly lock (DIM 6)
        kelly_mod = dims["DIM6_kelly"]  # 0.3 / 0.6 / 1.0

        # Exit quality (DIM 7) — if ATR is too narrow, partial TP may not work
        exit_mod = dims["DIM7_exit"]  # 0.9 – 1.1

        final_mult = regime_mult * sess_mult * temporal_mod * pair_mod * kelly_mod * exit_mod

        # Clip to sane range [0.4, 1.4]
        result.score_mult = float(max(0.4, min(1.4, final_mult)))
        result.dim_scores = dims

        return result

    # ── DIM 1: Temporal ───────────────────────────────────────────────
    def _dim1_temporal(self, utc_hour: int) -> float:
        """Day/hour temporal pattern. Returns modifier 0.85-1.0."""
        # Based on ICT concept: Monday Asia/London open can have false moves
        from datetime import datetime, timezone
        weekday = datetime.now(timezone.utc).weekday()
        # Monday 00-12 UTC — higher gap risk
        if weekday == 0 and utc_hour < 12:
            return 0.85
        # Friday after 19 UTC — low liquidity
        if weekday == 4 and utc_hour >= 19:
            return 0.88
        # Mid-week (Tue-Thu) London/NY = best
        if weekday in (1, 2, 3) and 12 <= utc_hour <= 18:
            return 1.0
        return 0.95

    # ── DIM 2: Volatility regime ───────────────────────────────────────
    def _dim2_volatility(self, df: "pd.DataFrame") -> Tuple[str, float]:
        """ATR-based volatility regime. Returns (regime, multiplier)."""
        if df is None or len(df) < 60:
            return "NORMAL", 1.0
        try:
            h, l, c = df["high"], df["low"], df["close"]
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            cur = atr.iloc[-1]
            hist_mean = atr.iloc[-60:].mean()
            hist_std  = atr.iloc[-60:].std()
            if pd.isna(cur) or hist_mean == 0:
                return "NORMAL", 1.0
            if cur > hist_mean + 0.7 * hist_std:
                return "HIGH", 1.15
            if cur < hist_mean - 0.5 * hist_std:
                return "LOW", 0.80
            return "NORMAL", 1.0
        except Exception:
            return "NORMAL", 1.0

    # ── DIM 3: Trend regime ───────────────────────────────────────────
    def _dim3_trend(self, df: "pd.DataFrame", direction: str) -> Tuple[str, float]:
        """EMA alignment + momentum. Returns (regime, multiplier)."""
        if df is None or len(df) < 55:
            return "MILD_TREND", 0.95
        try:
            c = df["close"]
            e8  = c.ewm(span=8,  adjust=False).mean().iloc[-1]
            e21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
            e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
            bull_aligned = e8 > e21 > e50
            bear_aligned = e8 < e21 < e50
            aligned = bull_aligned or bear_aligned
            # Check direction matches EMA alignment
            dir_match = (direction == "LONG" and bull_aligned) or (direction == "SHORT" and bear_aligned)
            if not aligned:
                return "CHOPPY", 0.65
            # Measure trend strength: distance between EMAs
            spread = abs(e8 - e50) / (e50 if e50 > 0 else 1)
            if spread > 0.005 and dir_match:
                return "STRONG_TREND", 1.10
            if dir_match:
                return "MILD_TREND", 1.00
            # Aligned but AGAINST our direction — counter-trend trade
            return "MILD_TREND", 0.75
        except Exception:
            return "MILD_TREND", 0.95

    # ── DIM 4: Session kill zone ──────────────────────────────────────
    def _dim4_session(self, utc_hour: int) -> Tuple[float, str, float]:
        """ICT Kill Zone quality. Returns (mult, quality, raw)."""
        if utc_hour in _SESSION_MULT:
            mult, quality = _SESSION_MULT[utc_hour]
            return mult, quality, mult
        return 0.75, "POOR", 0.75

    # ── DIM 5: Pair strength ──────────────────────────────────────────
    def _dim5_pair(self, sym_base: str, df: "pd.DataFrame") -> float:
        """Relative momentum of pair vs recent average. Returns 0.8-1.2."""
        if df is None or len(df) < 30:
            return 1.0
        try:
            c = df["close"]
            # Rate of change over last 24 bars (24h)
            roc = (c.iloc[-1] - c.iloc[-25]) / c.iloc[-25] if c.iloc[-25] != 0 else 0
            # Strong trending pair (high absolute RoC) = better edge
            abs_roc = abs(roc)
            if abs_roc > 0.005:   # >0.5% in 24h = strong move
                return 1.15
            if abs_roc > 0.002:   # >0.2%
                return 1.05
            if abs_roc < 0.0008:  # <0.08% = dead
                return 0.85
            return 1.0
        except Exception:
            return 1.0

    # ── DIM 6: Consecutive-loss circuit breaker ──────────────────────
    def _dim6_kelly(self, sym_base: str) -> float:
        """
        Circuit breaker: if last 3 closed trades all lost → return 0.0 (block).
        If last 5 trades WR < 40% → reduce to 0.6.
        Monthly profit lock: if Axi monthly profit > 4% → 0.3 (protect target).
        Reads episodes.db and axi_select_state.json.
        Returns 0.0–1.2.

        BUG-DIM6-DEAD-COLUMNS (2026-07-09): this query used to select a
        column named "outcome" ordered by "closed_at" -- neither exists in
        the real episodes.db schema (the real columns are "result", a TEXT
        'WIN'/'LOSS', and "ts"). Every call raised sqlite3.OperationalError,
        silently swallowed by the bare except below, so this function ALWAYS
        returned the safe-unblocked default (1.0) -- the 3-consecutive-loss
        circuit breaker and the WR<40% size reduction have never fired once
        in the bot's history, despite /status and the monitoring dashboards
        showing "DIM6 CIRCUIT: 0/3 OK" every single check (that display reads
        a separate scan_stats.json counter that was also never populated).
        """
        try:
            import sqlite3, os, json as _json
            db_path = os.path.join("memory", "episodes.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path, timeout=2)
                rows = conn.execute(
                    "SELECT result FROM episodes WHERE result IN ('WIN','LOSS') "
                    "ORDER BY ts DESC LIMIT 5"
                ).fetchall()
                conn.close()
                if rows:
                    outcomes = [r[0] for r in rows]
                    # Last 3 consecutive losses → circuit break
                    if len(outcomes) >= 3 and all(o == "LOSS" for o in outcomes[:3]):
                        return 0.0   # hard block — 24h cooling off
                    # Last 5 trades WR < 40% → reduce
                    if len(outcomes) >= 5:
                        wr = sum(1 for o in outcomes if o == "WIN") / len(outcomes)
                        if wr < 0.40:
                            return 0.60
        except Exception:
            pass

        # Monthly profit lock: once at 4%+ → protect Axi target
        try:
            import os, json as _json
            st_path = os.path.join("memory", "axi_select_state.json")
            if os.path.exists(st_path):
                with open(st_path) as f:
                    st = _json.load(f)
                capital = st.get("capital", 500.0)
                monthly = st.get("monthly_pnl", 0.0)
                if capital > 0 and monthly / capital >= 0.04:
                    return 0.30   # already hit 4% — only tiny trades
        except Exception:
            pass

        return 1.0

    # ── DIM 7: Exit quality (ATR vs spread) ──────────────────────────
    def _dim7_exit(self, df: "pd.DataFrame") -> float:
        """Check if current ATR is wide enough for partial TP at 1R to be worth it."""
        if df is None or len(df) < 20:
            return 1.0
        try:
            h, l, c = df["high"], df["low"], df["close"]
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            hist_atr = tr.rolling(14).mean().iloc[-60:].mean() if len(df) >= 74 else atr
            if pd.isna(atr) or pd.isna(hist_atr) or hist_atr == 0:
                return 1.0
            ratio = atr / hist_atr
            # ATR expanding (breakout) → better for partial TP
            if ratio > 1.3:
                return 1.10
            if ratio > 1.1:
                return 1.05
            # ATR contracting (compression) → worse (TP may not be reached)
            if ratio < 0.7:
                return 0.90
            return 1.0
        except Exception:
            return 1.0

    # ── DIM 8: Correlation / portfolio guard ─────────────────────────
    @staticmethod
    def _normalize_dir(direction: str) -> str:
        """Normalize BUY/SELL or LONG/SHORT to LONG/SHORT."""
        d = direction.upper().strip()
        if d in ("BUY", "LONG"):  return "LONG"
        if d in ("SELL", "SHORT"): return "SHORT"
        return d

    def _dim8_correlation(
        self,
        sym_base: str,
        direction: str,
        open_positions: List[dict],
    ) -> Tuple[bool, str]:
        """
        Check portfolio correlation constraint.
        Returns (allowed, reason).

        Rules:
        1. Max 1 trade from USD-bear group (EURUSD/GBPUSD/AUDUSD/NZDUSD) per direction
        2. Max 1 trade from USD-bull group (USDCAD) per direction
        3. Indices (NAS100/US30) are independent — always allowed
        """
        if not open_positions:
            return True, ""

        new_dir = self._normalize_dir(direction)

        # Determine which correlation group the new trade belongs to
        new_group = None
        for g_idx, group in enumerate(_CORR_GROUPS):
            if sym_base in group:
                new_group = g_idx
                break

        if new_group is None or new_group == 2:
            # Indices or unclassified — no restriction
            return True, ""

        # Check if any open position is in the same correlation group
        for pos in open_positions:
            pos_sym = str(pos.get("symbol", "")).split(".")[0].upper()
            pos_type = str(pos.get("type", "")).upper()  # "BUY" or "SELL" or "LONG" or "SHORT"
            pos_dir = self._normalize_dir(pos_type)

            # Same symbol: MAX_OPEN handles position count — DIM8 is for CROSS-PAIR correlation
            if pos_sym == sym_base:
                continue

            for g_idx, group in enumerate(_CORR_GROUPS):
                if g_idx == 2:
                    continue  # indices don't count
                if pos_sym in group and g_idx == new_group:
                    # Same correlation group, DIFFERENT symbol = DUPLICATE RISK
                    if pos_dir == new_dir:
                        return False, (
                            f"DIM8-BLOCK: {sym_base} {new_dir} = same USD exposure as "
                            f"{pos_sym} {pos_dir} (corr group {g_idx}) — skipping to avoid doubled risk"
                        )
                    # Opposite direction = natural hedge = OK
                    break

        return True, ""

    def describe(self) -> Dict[str, str]:
        """Human-readable description of each dimension."""
        return {
            "DIM1_temporal":    "Temporal edge modifier (weekday/hour pattern)",
            "DIM2_volatility":  "ATR regime: HIGH/NORMAL/LOW",
            "DIM3_trend":       "EMA alignment: STRONG_TREND/MILD_TREND/CHOPPY",
            "DIM4_session":     "ICT Kill Zone quality (13-14 UTC = GOLD)",
            "DIM5_pair":        "Pair momentum relative to 24h average",
            "DIM6_kelly":       "Circuit breaker: 3 consec losses=BLOCK | monthly 4%+=LOCK | WR<40%=REDUCE",
            "DIM7_exit":        "ATR expansion/contraction for exit quality",
            "DIM8_correlation": "Portfolio correlation guard (no duplicate risk)",
        }
