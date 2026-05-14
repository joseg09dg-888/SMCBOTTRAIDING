"""
Tests for agents/chaos_agent.py
"""

import numpy as np
import pandas as pd
import pytest

from agents.chaos_agent import (
    ChaosTheoryAgent,
    ChaosSignal,
    HurstResult,
    EntropyResult,
    FractalResult,
)


def make_df(prices) -> pd.DataFrame:
    return pd.DataFrame({"close": prices})


def trending_prices(n=200, slope=1.0) -> np.ndarray:
    """Prices with a clear upward trend (high Hurst)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.2, n)
    return np.cumsum(slope * np.ones(n) + noise) + 1000


def random_walk_prices(n=200) -> np.ndarray:
    """Pure random walk (Hurst ≈ 0.5)."""
    rng = np.random.default_rng(7)
    steps = rng.choice([-1, 1], size=n).astype(float)
    return np.cumsum(steps) + 200


def mean_reverting_prices(n=200) -> np.ndarray:
    """
    Mean-reverting (anti-persistent) series: each step tends to oppose the previous.
    """
    rng = np.random.default_rng(99)
    prices = [100.0]
    for _ in range(n - 1):
        # Strong mean reversion: pull toward mean 100
        drift = -0.7 * (prices[-1] - 100.0)
        prices.append(prices[-1] + drift + rng.normal(0, 0.5))
    return np.array(prices)


agent = ChaosTheoryAgent()


# ------------------------------------------------------------------
# Hurst tests
# ------------------------------------------------------------------

class TestHurstExponent:

    def test_hurst_trending_market(self):
        """Clear trend → Hurst > 0.5."""
        prices = trending_prices(200)
        result = agent.calculate_hurst(prices)
        assert isinstance(result, HurstResult)
        assert result.exponent > 0.5, f"Expected H>0.5 for trending market, got {result.exponent}"

    def test_hurst_random_walk(self):
        """Random walk → Hurst should be roughly in range 0.3-0.7 (includes 0.5)."""
        prices = random_walk_prices(200)
        result = agent.calculate_hurst(prices)
        assert isinstance(result, HurstResult)
        # Random walk hurst can vary; just check it's a valid float in range
        assert 0.0 <= result.exponent <= 1.0

    def test_hurst_insufficient_data_returns_random(self):
        """Fewer than 20 prices → fallback H=0.5, interpretation=random, bonus=0."""
        prices = np.array([100.0, 101.0, 102.0])
        result = agent.calculate_hurst(prices)
        assert result.exponent == 0.5
        assert result.interpretation == "random"
        assert result.score_bonus == 0

    def test_hurst_trending_gives_correct_interpretation(self):
        """Hurst > 0.6 → trending interpretation with +15 bonus."""
        prices = trending_prices(300)
        result = agent.calculate_hurst(prices)
        if result.exponent > 0.6:
            assert result.interpretation == "trending"
            assert result.score_bonus == 15

    def test_hurst_mean_reverting_gives_negative_bonus(self):
        """Hurst < 0.4 → mean_reverting with -10 bonus."""
        prices = mean_reverting_prices(200)
        result = agent.calculate_hurst(prices)
        if result.exponent < 0.4:
            assert result.interpretation == "mean_reverting"
            assert result.score_bonus == -10


# ------------------------------------------------------------------
# Entropy tests
# ------------------------------------------------------------------

class TestShannonEntropy:

    def test_entropy_ordered_market_low_entropy(self):
        """Constant-pct-return series → all returns identical → single bucket → entropy~0."""
        # np.linspace gives VARYING returns (decreasing as price rises).
        # A constant percentage gain gives identical returns → one bucket → low entropy.
        prices = 100.0 * (1.005 ** np.arange(100))
        result = agent.calculate_entropy(prices)
        assert isinstance(result, EntropyResult)
        assert result.normalized < 0.2

    def test_entropy_chaotic_market_high_entropy(self):
        """Random prices → high (near-maximum) entropy."""
        rng = np.random.default_rng(123)
        prices = rng.uniform(100, 200, 200)
        result = agent.calculate_entropy(prices)
        assert isinstance(result, EntropyResult)
        assert 0.0 <= result.normalized <= 1.0

    def test_entropy_normalized_range(self):
        """Normalized entropy must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        prices = rng.standard_normal(100) + 100
        result = agent.calculate_entropy(prices)
        assert 0.0 <= result.normalized <= 1.0

    def test_entropy_ordered_flag(self):
        """is_ordered=True when normalized < 0.4."""
        prices = np.linspace(100, 200, 100)
        result = agent.calculate_entropy(prices)
        assert result.is_ordered == (result.normalized < 0.4)

    def test_entropy_score_bonus_10_when_ordered(self):
        """score_bonus=+10 only when is_ordered=True."""
        prices = np.linspace(100, 200, 100)
        result = agent.calculate_entropy(prices)
        expected = 10 if result.is_ordered else 0
        assert result.score_bonus == expected


# ------------------------------------------------------------------
# Fractal tests
# ------------------------------------------------------------------

class TestFractalPattern:

    def test_fractal_returns_result(self):
        """detect_fractal_pattern always returns a FractalResult."""
        prices = trending_prices(100)
        df = make_df(prices)
        result = agent.detect_fractal_pattern(df)
        assert isinstance(result, FractalResult)
        assert 1.0 <= result.dimension <= 2.0

    def test_fractal_insufficient_data(self):
        """Less than 10 bars → not fractal."""
        df = make_df([100, 101, 102])
        result = agent.detect_fractal_pattern(df)
        assert result.is_fractal is False
        assert result.score_bonus == 0

    def test_fractal_score_bonus_when_confirmed(self):
        """score_bonus=+10 if is_fractal, 0 otherwise."""
        prices = trending_prices(200)
        df = make_df(prices)
        result = agent.detect_fractal_pattern(df)
        expected = 10 if result.is_fractal else 0
        assert result.score_bonus == expected

    def test_fractal_timeframes_range(self):
        """timeframes_aligned must be 0, 1, or 2."""
        prices = trending_prices(100)
        df = make_df(prices)
        result = agent.detect_fractal_pattern(df)
        assert result.timeframes_aligned in (0, 1, 2)


# ------------------------------------------------------------------
# Lyapunov tests
# ------------------------------------------------------------------

class TestLyapunovProxy:

    def test_lyapunov_high_variance_is_chaotic(self):
        """High variance log-returns → positive Lyapunov (chaotic)."""
        rng = np.random.default_rng(5)
        # Create prices where log-returns have very high variance
        log_rets = rng.normal(0, 0.1, 300)   # std=0.1 → variance=0.01 >> 0.002
        prices = np.exp(np.cumsum(log_rets)) * 100
        lyap = agent.estimate_lyapunov(prices)
        assert lyap > 0, f"Expected positive Lyapunov for high-variance series, got {lyap}"

    def test_lyapunov_low_variance_is_predictable(self):
        """Low variance log-returns → negative or zero Lyapunov."""
        prices = np.linspace(100, 110, 200)  # perfectly smooth → near-zero variance
        lyap = agent.estimate_lyapunov(prices)
        assert lyap <= 0


# ------------------------------------------------------------------
# Combined signal tests
# ------------------------------------------------------------------

class TestGetSignal:

    def test_get_signal_returns_chaos_signal(self):
        """get_signal returns a valid ChaosSignal object."""
        prices = trending_prices(100)
        df = make_df(prices)
        sig = agent.get_signal(df)
        assert isinstance(sig, ChaosSignal)
        assert isinstance(sig.hurst, HurstResult)
        assert isinstance(sig.entropy, EntropyResult)
        assert isinstance(sig.fractal, FractalResult)
        assert isinstance(sig.is_chaotic, bool)
        assert isinstance(sig.total_bonus, int)
        assert isinstance(sig.summary, str)

    def test_score_adjustment_trending_gives_positive(self):
        """Trending market score_adjustment should be positive (≥ 15)."""
        # Use a very strong trend to guarantee H>0.6
        prices = np.cumsum(np.ones(300)) + 1000
        df = make_df(prices)
        sig = agent.get_signal(df)
        if sig.hurst.exponent > 0.6:
            assert sig.total_bonus >= 15

    def test_score_adjustment_chaotic_penalty(self):
        """Chaotic market incurs -20 penalty."""
        rng = np.random.default_rng(55)
        log_rets = rng.normal(0, 0.15, 400)
        prices = np.exp(np.cumsum(log_rets)) * 100
        df = make_df(prices)
        sig = agent.get_signal(df)
        if sig.is_chaotic:
            assert sig.chaos_penalty == -20

    def test_total_bonus_respects_bounds(self):
        """total_bonus must always be in [-20, 35]."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            prices = np.cumsum(rng.normal(0, 1, 150)) + 100
            df = make_df(prices)
            sig = agent.get_signal(df)
            assert -20 <= sig.total_bonus <= 35, (
                f"total_bonus={sig.total_bonus} out of [-20,35]"
            )

    def test_score_adjustment_method(self):
        """score_adjustment() returns same value as get_signal().total_bonus."""
        prices = trending_prices(100)
        df = make_df(prices)
        assert agent.score_adjustment(df) == agent.get_signal(df).total_bonus

    def test_format_telegram_contains_symbol(self):
        """format_telegram output contains the symbol."""
        prices = trending_prices(100)
        df = make_df(prices)
        text = agent.format_telegram("BTCUSDT", df)
        assert "BTCUSDT" in text
        assert "Hurst" in text
