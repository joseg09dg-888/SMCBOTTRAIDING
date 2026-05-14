"""
Module 9: Chaos Theory Agent
Uses R/S Hurst exponent, Shannon entropy, fractal dimension, and a Lyapunov proxy
to detect market regime (trending / random / mean-reverting / chaotic).
Degrades gracefully if nolds is not installed.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class HurstResult:
    exponent: float          # 0.0-1.0
    interpretation: str      # "trending"(>0.6), "random"(0.4-0.6), "mean_reverting"(<0.4)
    score_bonus: int         # +15 trending, 0 random, -10 mean_reverting


@dataclass
class EntropyResult:
    shannon_entropy: float
    normalized: float        # 0.0-1.0 (0=ordered, 1=chaotic)
    is_ordered: bool         # True if normalized < 0.4
    score_bonus: int         # +10 if ordered


@dataclass
class FractalResult:
    dimension: float         # 1.0-2.0
    timeframes_aligned: int  # 0-3
    is_fractal: bool         # True if timeframes_aligned >= 2
    score_bonus: int         # +10 if fractal confirmed


@dataclass
class ChaosSignal:
    hurst: HurstResult
    entropy: EntropyResult
    fractal: FractalResult
    is_chaotic: bool         # True if Lyapunov proxy is positive
    chaos_penalty: int       # -20 if chaotic, 0 otherwise
    total_bonus: int         # capped -20 to +35
    summary: str


class ChaosTheoryAgent:
    """
    Computes chaos-theory metrics on price series and returns a ChaosSignal
    with a composite score adjustment.
    """

    # ------------------------------------------------------------------
    # Hurst exponent (R/S analysis)
    # ------------------------------------------------------------------

    def _rs_single(self, series: np.ndarray) -> float:
        """Compute R/S for a single sub-series."""
        n = len(series)
        if n < 2:
            return 1.0
        mean = series.mean()
        deviations = series - mean
        cumdev = np.cumsum(deviations)
        R = cumdev.max() - cumdev.min()
        S = series.std(ddof=1)
        if S == 0:
            return 1.0
        return R / S

    def calculate_hurst(self, prices: np.ndarray) -> HurstResult:
        """
        Calculate Hurst exponent using R/S analysis.
        Falls back to H=0.5 (random) when fewer than 20 prices are supplied,
        or when scipy/nolds are unavailable.
        """
        prices = np.asarray(prices, dtype=float)

        if len(prices) < 20:
            return HurstResult(exponent=0.5, interpretation="random", score_bonus=0)

        # Try nolds first for a more accurate estimate
        try:
            import nolds
            h = float(nolds.hurst_rs(prices))
            h = float(np.clip(h, 0.0, 1.0))
        except Exception:
            # Fall back to manual multi-scale R/S
            try:
                h = self._manual_hurst(prices)
            except Exception:
                h = 0.5

        h = float(np.clip(h, 0.0, 1.0))

        if h > 0.6:
            interpretation = "trending"
            score_bonus = 15
        elif h < 0.4:
            interpretation = "mean_reverting"
            score_bonus = -10
        else:
            interpretation = "random"
            score_bonus = 0

        return HurstResult(exponent=round(h, 4), interpretation=interpretation,
                           score_bonus=score_bonus)

    def _manual_hurst(self, prices: np.ndarray) -> float:
        """Manual multi-scale R/S Hurst estimate."""
        n = len(prices)
        # Use at least 3 scales
        scales = []
        for power in range(3, int(np.log2(n)) + 1):
            scales.append(2 ** power)
        if not scales:
            scales = [max(4, n // 2)]

        log_rs = []
        log_n = []

        for scale in scales:
            if scale > n:
                break
            n_chunks = n // scale
            if n_chunks == 0:
                continue
            rs_vals = []
            for i in range(n_chunks):
                chunk = prices[i * scale:(i + 1) * scale]
                rs = self._rs_single(chunk)
                if rs > 0:
                    rs_vals.append(rs)
            if rs_vals:
                log_rs.append(np.log(np.mean(rs_vals)))
                log_n.append(np.log(scale))

        if len(log_n) < 2:
            return 0.5

        # Linear regression: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(log_n, log_rs, 1)
        return float(coeffs[0])

    # ------------------------------------------------------------------
    # Shannon entropy
    # ------------------------------------------------------------------

    def calculate_entropy(self, prices: np.ndarray, window: int = 50) -> EntropyResult:
        """
        Shannon entropy of the price-return distribution over the last *window* bars.
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return EntropyResult(shannon_entropy=0.0, normalized=0.0,
                                 is_ordered=True, score_bonus=10)

        # Use up to the last *window* prices
        subset = prices[-window:] if len(prices) > window else prices
        returns = np.diff(subset) / subset[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return EntropyResult(shannon_entropy=0.0, normalized=0.0,
                                 is_ordered=True, score_bonus=10)

        # If all returns are practically identical (perfect trend), entropy → 0
        rng = float(returns.max() - returns.min())
        if rng < 1e-10:
            return EntropyResult(shannon_entropy=0.0, normalized=0.0,
                                 is_ordered=True, score_bonus=10)

        n_buckets = 10
        counts, _ = np.histogram(returns, bins=n_buckets)
        total = counts.sum()
        if total == 0:
            return EntropyResult(shannon_entropy=0.0, normalized=0.0,
                                 is_ordered=True, score_bonus=10)

        probs = counts[counts > 0] / total
        entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(n_buckets)
        normalized = float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy > 0 else 0.0

        is_ordered = normalized < 0.4
        score_bonus = 10 if is_ordered else 0

        return EntropyResult(
            shannon_entropy=round(entropy, 6),
            normalized=round(normalized, 6),
            is_ordered=is_ordered,
            score_bonus=score_bonus,
        )

    # ------------------------------------------------------------------
    # Fractal pattern detection
    # ------------------------------------------------------------------

    def _pattern_correlation(self, prices: np.ndarray, length: int) -> float:
        """Pearson correlation between the last *length* bars and normalised short pattern."""
        n = len(prices)
        if n < length:
            return 0.0
        short_len = min(10, length)
        short = prices[n - short_len:]
        long_ = prices[n - length:n - length + short_len]
        if len(long_) < short_len or np.std(short) == 0 or np.std(long_) == 0:
            return 0.0
        corr = float(np.corrcoef(short, long_)[0, 1])
        return 0.0 if np.isnan(corr) else abs(corr)

    def detect_fractal_pattern(self, df: pd.DataFrame) -> FractalResult:
        """
        Compare the last 10-bar pattern against 30-bar and 60-bar windows.
        If correlation > 0.6 for 2+ scales → fractal confirmed.
        """
        if df is None or len(df) < 10:
            return FractalResult(dimension=1.5, timeframes_aligned=0,
                                 is_fractal=False, score_bonus=0)

        col = "close" if "close" in df.columns else df.columns[-1]
        prices = df[col].values.astype(float)

        aligned = 0
        for window in (30, 60):
            corr = self._pattern_correlation(prices, window)
            if corr > 0.6:
                aligned += 1

        # Simplified fractal dimension proxy (box-counting not implemented;
        # use the Hurst-based estimate: D ≈ 2 - H)
        h_result = self.calculate_hurst(prices)
        dimension = round(2.0 - h_result.exponent, 4)

        is_fractal = aligned >= 2
        score_bonus = 10 if is_fractal else 0

        return FractalResult(
            dimension=dimension,
            timeframes_aligned=aligned,
            is_fractal=is_fractal,
            score_bonus=score_bonus,
        )

    # ------------------------------------------------------------------
    # Lyapunov proxy
    # ------------------------------------------------------------------

    def estimate_lyapunov(self, prices: np.ndarray) -> float:
        """
        Simplified Lyapunov exponent proxy based on variance of log-returns.
        > 0.002  → chaotic (positive Lyapunov)
        < 0.0005 → predictable (negative Lyapunov)
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices[prices > 0]))
        if len(log_returns) == 0:
            return 0.0
        variance = float(np.var(log_returns))
        # Return a signed proxy: positive = chaotic, negative = predictable
        if variance > 0.002:
            return variance          # positive → chaotic
        elif variance < 0.0005:
            return -variance         # negative → predictable
        else:
            return 0.0               # ambiguous

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def get_signal(self, df: pd.DataFrame) -> ChaosSignal:
        """Build a full ChaosSignal from a price DataFrame."""
        col = "close" if "close" in df.columns else df.columns[-1]
        prices = df[col].values.astype(float)

        hurst = self.calculate_hurst(prices)
        entropy = self.calculate_entropy(prices)
        fractal = self.detect_fractal_pattern(df)

        lyapunov = self.estimate_lyapunov(prices)
        is_chaotic = lyapunov > 0
        chaos_penalty = -20 if is_chaotic else 0

        raw_bonus = (
            hurst.score_bonus
            + entropy.score_bonus
            + fractal.score_bonus
            + chaos_penalty
        )
        total_bonus = int(np.clip(raw_bonus, -20, 35))

        parts = [
            f"Hurst={hurst.exponent:.3f} ({hurst.interpretation}, {hurst.score_bonus:+d})",
            f"Entropy={entropy.normalized:.3f} ({'ordered' if entropy.is_ordered else 'chaotic'}, {entropy.score_bonus:+d})",
            f"Fractal={'yes' if fractal.is_fractal else 'no'} ({fractal.score_bonus:+d})",
            f"Lyapunov={'chaotic' if is_chaotic else 'stable'} ({chaos_penalty:+d})",
            f"Total={total_bonus:+d}",
        ]
        summary = " | ".join(parts)

        return ChaosSignal(
            hurst=hurst,
            entropy=entropy,
            fractal=fractal,
            is_chaotic=is_chaotic,
            chaos_penalty=chaos_penalty,
            total_bonus=total_bonus,
            summary=summary,
        )

    def score_adjustment(self, df: pd.DataFrame) -> int:
        """Return the composite chaos score adjustment for *df*."""
        return self.get_signal(df).total_bonus

    # ------------------------------------------------------------------
    # Telegram formatting
    # ------------------------------------------------------------------

    def format_telegram(self, symbol: str, df: pd.DataFrame) -> str:
        """Return a Telegram-ready chaos analysis string."""
        sig = self.get_signal(df)
        lines = [
            f"*Chaos Theory — {symbol}*",
            f"Hurst: `{sig.hurst.exponent:.3f}` → {sig.hurst.interpretation.upper()} ({sig.hurst.score_bonus:+d}pt)",
            f"Entropy: `{sig.entropy.normalized:.3f}` → {'ORDERED' if sig.entropy.is_ordered else 'CHAOTIC'} ({sig.entropy.score_bonus:+d}pt)",
            f"Fractal: `{'Confirmed' if sig.fractal.is_fractal else 'No'}` ({sig.fractal.score_bonus:+d}pt)",
            f"Lyapunov: `{'CHAOTIC' if sig.is_chaotic else 'STABLE'}` ({sig.chaos_penalty:+d}pt)",
            f"Chaos bonus: `{sig.total_bonus:+d}/35`",
        ]
        return "\n".join(lines)
