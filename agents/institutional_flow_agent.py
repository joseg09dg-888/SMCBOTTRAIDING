"""
Module 3: Institutional Flow Agent
Fetches CFTC COT data and options flow to detect smart-money positioning.
All network calls are wrapped in try/except and return None gracefully.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class COTSnapshot:
    report_date: str
    symbol: str
    commercial_net: int       # commercial long - short (smart money)
    noncommercial_net: int    # speculator net
    retail_net: int           # small trader net
    commercial_bias: str      # "bullish", "bearish", "neutral"
    score_bonus: int          # 0-15


@dataclass
class OptionsFlowSnapshot:
    symbol: str
    call_volume: int
    put_volume: int
    put_call_ratio: float
    unusual_activity: bool
    bias: str              # "bullish" / "bearish" / "neutral"
    score_bonus: int       # 0-10


@dataclass
class InstitutionalSignal:
    cot: Optional[COTSnapshot]
    options: Optional[OptionsFlowSnapshot]
    total_bonus: int       # 0-25
    summary: str


class InstitutionalFlowAgent:
    """
    Combines CFTC COT report data with options flow to generate
    an institutional-sentiment score bonus (0-25 pts).
    All external fetches degrade gracefully to None when offline.
    """

    COT_URL = "https://www.cftc.gov/ddi/preliminary/aspx/fincomrep.aspx"
    OPTIONS_URL = "https://phx.unusualwhales.com/api/etf/{symbol}/flow"

    def __init__(self):
        # In-memory cache for last successful fetches
        self._cot_cache: Dict[str, COTSnapshot] = {}
        self._options_cache: Dict[str, OptionsFlowSnapshot] = {}

    # ------------------------------------------------------------------
    # COT (Commitments of Traders) helpers
    # ------------------------------------------------------------------

    def _compute_cot_bias(self, commercial_net: int) -> str:
        """Determine bias string from commercial net position."""
        if commercial_net > 0:
            return "bullish"
        elif commercial_net < 0:
            return "bearish"
        return "neutral"

    def _compute_cot_bonus(self, commercial_net: int) -> int:
        """
        +15 if commercial_net > 0 (smart money net long).
        +15 if commercial_net < 0 (smart money net short → bearish signal).
        0   if exactly 0 (neutral).
        """
        if commercial_net != 0:
            return 15
        return 0

    def get_cot_signal(self, symbol: str) -> Optional[COTSnapshot]:
        """
        Attempt to fetch CFTC COT data for *symbol*.
        Returns a COTSnapshot on success, or None if unavailable.
        Falls back to in-memory cache on network failure.
        """
        try:
            import requests  # optional dependency; may not be installed
            resp = requests.get(self.COT_URL, timeout=5)
            resp.raise_for_status()

            # --- Minimal parsing of CFTC CSV/HTML ---
            # The real CFTC endpoint returns a paginated HTML table.
            # We look for lines containing the symbol keyword and extract
            # commercial long/short columns (positions 9 and 10 in the
            # standard "legacy" CSV format).
            text = resp.text
            commercial_long = 0
            commercial_short = 0
            noncommercial_long = 0
            noncommercial_short = 0
            retail_long = 0
            retail_short = 0

            for line in text.splitlines():
                if symbol.upper() in line.upper():
                    parts = [p.strip().replace(",", "") for p in line.split(",")]
                    try:
                        # Legacy COT CSV columns (0-indexed):
                        # 9  = commercial long
                        # 10 = commercial short
                        # 5  = noncommercial long
                        # 6  = noncommercial short
                        # 12 = small trader long
                        # 13 = small trader short
                        if len(parts) > 13:
                            commercial_long = int(parts[9])
                            commercial_short = int(parts[10])
                            noncommercial_long = int(parts[5])
                            noncommercial_short = int(parts[6])
                            retail_long = int(parts[12])
                            retail_short = int(parts[13])
                    except (ValueError, IndexError):
                        pass
                    break

            commercial_net = commercial_long - commercial_short
            noncommercial_net = noncommercial_long - noncommercial_short
            retail_net = retail_long - retail_short
            bias = self._compute_cot_bias(commercial_net)
            bonus = self._compute_cot_bonus(commercial_net)

            snapshot = COTSnapshot(
                report_date=resp.headers.get("Last-Modified", "unknown"),
                symbol=symbol,
                commercial_net=commercial_net,
                noncommercial_net=noncommercial_net,
                retail_net=retail_net,
                commercial_bias=bias,
                score_bonus=bonus,
            )
            self._cot_cache[symbol] = snapshot
            return snapshot

        except Exception as exc:
            logger.warning("COT fetch failed for %s: %s — using cache", symbol, exc)
            return self._cot_cache.get(symbol)

    # ------------------------------------------------------------------
    # Options flow helpers
    # ------------------------------------------------------------------

    def _compute_options_bias(self, pcr: float) -> str:
        if pcr < 0.7:
            return "bullish"
        elif pcr > 1.3:
            return "bearish"
        return "neutral"

    def _compute_options_bonus(self, pcr: float) -> int:
        if pcr < 0.7 or pcr > 1.3:
            return 10
        return 0

    def get_options_signal(self, symbol: str) -> Optional[OptionsFlowSnapshot]:
        """
        Attempt to fetch options flow from unusualwhales.com free tier.
        Returns None gracefully if network is unavailable or data is missing.
        """
        try:
            import requests
            url = self.OPTIONS_URL.format(symbol=symbol)
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            call_volume = int(data.get("call_volume", 0))
            put_volume = int(data.get("put_volume", 0))
            total = call_volume + put_volume
            pcr = put_volume / call_volume if call_volume > 0 else 1.0
            unusual = data.get("unusual_activity", False)

            bias = self._compute_options_bias(pcr)
            bonus = self._compute_options_bonus(pcr)

            snapshot = OptionsFlowSnapshot(
                symbol=symbol,
                call_volume=call_volume,
                put_volume=put_volume,
                put_call_ratio=round(pcr, 4),
                unusual_activity=bool(unusual),
                bias=bias,
                score_bonus=bonus,
            )
            self._options_cache[symbol] = snapshot
            return snapshot

        except Exception as exc:
            logger.warning("Options fetch failed for %s: %s — using cache", symbol, exc)
            return self._options_cache.get(symbol)

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def get_combined_signal(self, symbol: str, bias: str) -> InstitutionalSignal:
        """
        Fetch COT + options and combine into a single InstitutionalSignal.
        *bias* is the directional bias from the calling strategy
        ("bullish" or "bearish") — used to align score bonuses.
        """
        cot = self.get_cot_signal(symbol)
        options = self.get_options_signal(symbol)

        cot_bonus = 0
        if cot is not None:
            # Award bonus only when institutional bias agrees with trade bias
            if cot.commercial_bias == bias:
                cot_bonus = cot.score_bonus

        opts_bonus = 0
        if options is not None:
            if options.bias == bias:
                opts_bonus = options.score_bonus

        total = min(cot_bonus + opts_bonus, 25)

        parts = []
        if cot is not None:
            parts.append(
                f"COT {cot.commercial_bias} (net {cot.commercial_net:+,})"
            )
        else:
            parts.append("COT: unavailable")

        if options is not None:
            parts.append(
                f"Options P/C={options.put_call_ratio:.2f} ({options.bias})"
            )
        else:
            parts.append("Options: unavailable")

        summary = " | ".join(parts) + f" → bonus +{total}"
        return InstitutionalSignal(
            cot=cot,
            options=options,
            total_bonus=total,
            summary=summary,
        )

    def score_adjustment(self, symbol: str, bias: str) -> int:
        """Return the 0-25 bonus score for *symbol* given *bias*."""
        signal = self.get_combined_signal(symbol, bias)
        return signal.total_bonus

    # ------------------------------------------------------------------
    # Telegram formatting
    # ------------------------------------------------------------------

    def format_telegram(self, symbol: str) -> str:
        """Return a human-readable Telegram string for the latest signal."""
        cot = self._cot_cache.get(symbol)
        options = self._options_cache.get(symbol)

        lines = [f"🏦 *Institutional Flow — {symbol}*"]

        if cot:
            lines.append(
                f"COT: commercial net `{cot.commercial_net:+,}` → "
                f"{cot.commercial_bias.upper()} (+{cot.score_bonus})"
            )
        else:
            lines.append("COT: no data")

        if options:
            ua = "⚡ inusual" if options.unusual_activity else ""
            lines.append(
                f"Options P/C=`{options.put_call_ratio:.2f}` → "
                f"{options.bias.upper()} (+{options.score_bonus}) {ua}"
            )
        else:
            lines.append("Options: no data")

        total = (cot.score_bonus if cot else 0) + (options.score_bonus if options else 0)
        total = min(total, 25)
        lines.append(f"Total bonus: `+{total}/25`")
        return "\n".join(lines)
