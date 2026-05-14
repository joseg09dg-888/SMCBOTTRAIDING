from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import date, timedelta

try:
    import httpx
except ImportError:
    httpx = None

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
WHALE_ALERT_URL = "https://api.whale-alert.io/v1/transactions"
LAST_HALVING_DATE = date(2024, 4, 20)
HALVING_CYCLE_DAYS = 1458  # ~4 years

CRYPTO_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"}


@dataclass
class FearGreedData:
    value: int    # 0-100
    label: str    # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: str


@dataclass
class MVRVData:
    ratio: float  # Market Value / Realized Value
    zone: str     # "undervalued"(<1), "neutral"(1-3.5), "overvalued"(>3.5)
    signal: str   # "buy_zone", "neutral", "sell_zone"


@dataclass
class HalvingCycleData:
    last_halving: str      # "2024-04-20"
    days_since: int
    cycle_pct: float       # 0.0-1.0 progress through 4-year cycle
    phase: str             # "accumulation", "expansion", "euphoria", "capitulation"
    next_halving_est: str  # estimated date


@dataclass
class OnChainSignal:
    symbol: str
    fear_greed: Optional[FearGreedData]
    mvrv: Optional[MVRVData]
    halving_cycle: HalvingCycleData   # always computed (pure date math)
    exchange_flow_bias: str           # "accumulation", "distribution", "neutral"
    whale_activity: str               # "buying", "selling", "neutral"
    funding_rate: float               # positive = longs paying, negative = shorts paying
    overall_bias: str                 # "bullish", "bearish", "neutral"
    score_bonus: int                  # 0-30
    summary: str


class OnChainAgent:
    """
    On-chain sentiment agent combining Fear & Greed index, MVRV ratio,
    BTC halving cycle position, exchange flows, and whale activity.
    Network-dependent calls (Fear & Greed) degrade gracefully to None.
    """

    def get_fear_greed(self) -> Optional[FearGreedData]:
        """Fetch from alternative.me; return None if network fails."""
        try:
            if httpx is None:
                return None
            response = httpx.get(FEAR_GREED_URL, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            entry = data["data"][0]
            return FearGreedData(
                value=int(entry["value"]),
                label=entry["value_classification"],
                timestamp=entry["timestamp"],
            )
        except Exception:
            return None

    def get_halving_cycle(self, as_of: date = None) -> HalvingCycleData:
        """
        Pure date math — always works without network.
        Phase boundaries:
          0-25%  → accumulation
          25-60% → expansion
          60-80% → euphoria
          80%+   → capitulation
        """
        if as_of is None:
            as_of = date.today()

        days_since = (as_of - LAST_HALVING_DATE).days
        # Handle edge case where as_of is before last halving
        if days_since < 0:
            days_since = 0

        cycle_pct = min(days_since / HALVING_CYCLE_DAYS, 1.0)

        if cycle_pct < 0.25:
            phase = "accumulation"
        elif cycle_pct < 0.60:
            phase = "expansion"
        elif cycle_pct < 0.80:
            phase = "euphoria"
        else:
            phase = "capitulation"

        next_halving = LAST_HALVING_DATE + timedelta(days=HALVING_CYCLE_DAYS)
        next_halving_est = next_halving.isoformat()

        return HalvingCycleData(
            last_halving=LAST_HALVING_DATE.isoformat(),
            days_since=days_since,
            cycle_pct=round(cycle_pct, 4),
            phase=phase,
            next_halving_est=next_halving_est,
        )

    def estimate_mvrv(self, price: float) -> MVRVData:
        """
        Simplified heuristic MVRV approximation.
        Real MVRV needs Glassnode API — this is a price-based fallback.
          price < 30000     → undervalued (ratio ~0.8)
          30000 - 150000    → neutral     (ratio ~2.0)
          > 150000          → overvalued  (ratio ~4.5)
        """
        if price < 30_000:
            ratio = 0.8
            zone = "undervalued"
            signal = "buy_zone"
        elif price <= 150_000:
            ratio = 2.0
            zone = "neutral"
            signal = "neutral"
        else:
            ratio = 4.5
            zone = "overvalued"
            signal = "sell_zone"

        return MVRVData(ratio=ratio, zone=zone, signal=signal)

    def get_signal(self, symbol: str, price: float = 0.0) -> OnChainSignal:
        """
        Combine all metrics; gracefully handle None for network-dependent ones.
        """
        fear_greed = self.get_fear_greed()
        halving_cycle = self.get_halving_cycle()
        mvrv = self.estimate_mvrv(price) if price > 0 else None

        # Derive overall bias from available data
        bullish_signals = 0
        bearish_signals = 0

        if fear_greed is not None:
            if fear_greed.value <= 25:
                bullish_signals += 1  # Extreme fear → contrarian buy
            elif fear_greed.value >= 75:
                bearish_signals += 1  # Extreme greed → contrarian sell

        if mvrv is not None:
            if mvrv.zone == "undervalued":
                bullish_signals += 1
            elif mvrv.zone == "overvalued":
                bearish_signals += 1

        if halving_cycle.phase in ("accumulation", "expansion"):
            bullish_signals += 1
        elif halving_cycle.phase == "capitulation":
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            overall_bias = "bullish"
        elif bearish_signals > bullish_signals:
            overall_bias = "bearish"
        else:
            overall_bias = "neutral"

        # Exchange flow and whale activity defaults (no live API in heuristic mode)
        exchange_flow_bias = "neutral"
        whale_activity = "neutral"
        funding_rate = 0.0

        score_bonus = self.score_adjustment(symbol, overall_bias, price)

        # Build summary
        fg_str = f"Fear&Greed: {fear_greed.value} ({fear_greed.label})" if fear_greed else "Fear&Greed: N/A"
        mvrv_str = f"MVRV: {mvrv.zone}" if mvrv else "MVRV: N/A"
        summary = (
            f"{symbol} | {fg_str} | {mvrv_str} | "
            f"Halving: {halving_cycle.phase} ({halving_cycle.cycle_pct*100:.1f}%) | "
            f"Bias: {overall_bias.upper()} | Bonus: +{score_bonus}"
        )

        return OnChainSignal(
            symbol=symbol,
            fear_greed=fear_greed,
            mvrv=mvrv,
            halving_cycle=halving_cycle,
            exchange_flow_bias=exchange_flow_bias,
            whale_activity=whale_activity,
            funding_rate=funding_rate,
            overall_bias=overall_bias,
            score_bonus=score_bonus,
            summary=summary,
        )

    def score_adjustment(self, symbol: str, bias: str, price: float = 0.0) -> int:
        """
        Only applies to crypto symbols (BTCUSDT, ETHUSDT, etc.).
        Bonus sources (max 30):
          +15  if Fear&Greed extreme fear AND bullish bias
          +10  if MVRV undervalued AND bullish
          +5   if halving cycle in expansion AND bullish
        Returns 0 for non-crypto symbols.
        """
        if symbol.upper() not in CRYPTO_SYMBOLS:
            return 0

        if bias != "bullish":
            return 0

        bonus = 0

        fear_greed = self.get_fear_greed()
        if fear_greed is not None and fear_greed.value <= 25:
            bonus += 15

        if price > 0:
            mvrv = self.estimate_mvrv(price)
            if mvrv.zone == "undervalued":
                bonus += 10

        halving = self.get_halving_cycle()
        if halving.phase == "expansion":
            bonus += 5

        return min(bonus, 30)

    def format_telegram(self, symbol: str, price: float = 0.0) -> str:
        """Structured output with all on-chain metrics."""
        signal = self.get_signal(symbol, price)
        hc = signal.halving_cycle

        fg_line = (
            f"Fear & Greed: {signal.fear_greed.value}/100 — {signal.fear_greed.label}"
            if signal.fear_greed
            else "Fear & Greed: N/A (network unavailable)"
        )
        mvrv_line = (
            f"MVRV: {signal.mvrv.ratio:.1f} ({signal.mvrv.zone}) → {signal.mvrv.signal}"
            if signal.mvrv
            else "MVRV: N/A"
        )

        lines = [
            f"ON-CHAIN SIGNAL — {symbol}",
            "",
            fg_line,
            mvrv_line,
            f"Halving Cycle: {hc.phase.upper()} | {hc.cycle_pct*100:.1f}% complete",
            f"  Days since halving: {hc.days_since}",
            f"  Next halving est: {hc.next_halving_est}",
            f"Exchange Flow: {signal.exchange_flow_bias}",
            f"Whale Activity: {signal.whale_activity}",
            f"Funding Rate: {signal.funding_rate:+.4f}",
            "",
            f"Overall Bias: {signal.overall_bias.upper()}",
            f"Score Bonus: +{signal.score_bonus}",
            "",
            signal.summary,
        ]
        return "\n".join(lines)
