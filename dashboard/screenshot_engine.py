# dashboard/screenshot_engine.py
import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mplfinance as mpf
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


class TradeOutcome(Enum):
    WIN  = "win"
    LOSS = "loss"


@dataclass
class ChartAnnotation:
    symbol: str
    timeframe: str
    entry: float
    stop_loss: float
    take_profit: float
    score: int
    rr: float
    confidence: float
    trigger: str
    ob_zone: Optional[Tuple[float, float]] = None   # (low, high)
    fvg_zone: Optional[Tuple[float, float]] = None
    vwap: Optional[float] = None
    poc: Optional[float] = None


class ScreenshotEngine:
    """
    Generates annotated OHLCV chart images for Telegram.
    Uses mplfinance when available; falls back to simple matplotlib or
    a PNG placeholder when neither library is installed.
    """

    DARK_BG    = "#131722"
    GREEN      = "#26a69a"
    RED        = "#ef5350"
    WHITE      = "#d1d4dc"
    BLUE       = "#2962ff"
    ORANGE     = "#ff9800"
    CHART_SIZE = (14, 8)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def capture_entry(self, df: pd.DataFrame, annotation: "ChartAnnotation") -> bytes:
        """Capture a chart image at trade entry."""
        return self._render_chart(df, annotation, outcome=None)

    def capture_close(
        self,
        df: pd.DataFrame,
        annotation: "ChartAnnotation",
        pnl: float,
        pnl_pct: float,
        duration_min: int,
        outcome: "TradeOutcome",
        win_rate: float,
    ) -> bytes:
        """Capture a chart image when a trade is closed."""
        return self._render_chart(
            df, annotation,
            outcome=outcome,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_min=duration_min,
            win_rate=win_rate,
        )

    # --------------------------------------------------------------------------
    # Rendering internals
    # --------------------------------------------------------------------------

    def _render_chart(
        self,
        df: pd.DataFrame,
        annotation: "ChartAnnotation",
        outcome: Optional["TradeOutcome"] = None,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        duration_min: int = 0,
        win_rate: float = 0.0,
    ) -> bytes:
        """
        Main chart renderer.  Tries mplfinance first, then falls back to a
        simple matplotlib line chart, then to a static PNG placeholder.
        """
        if HAS_MPLFINANCE:
            return self._render_with_mplfinance(
                df, annotation, outcome, pnl, pnl_pct, duration_min, win_rate
            )
        return self._render_fallback(df, annotation, outcome)

    def _render_with_mplfinance(
        self,
        df: pd.DataFrame,
        annotation: "ChartAnnotation",
        outcome: Optional["TradeOutcome"] = None,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        duration_min: int = 0,
        win_rate: float = 0.0,
    ) -> bytes:
        """Full candlestick chart using mplfinance + matplotlib."""
        try:
            plot_df = df.copy()
            if "timestamp" in plot_df.columns:
                plot_df = plot_df.set_index("timestamp")
            plot_df.index = pd.DatetimeIndex(plot_df.index)

            mc = mpf.make_marketcolors(
                up=self.GREEN, down=self.RED,
                edge="inherit", wick="inherit",
                volume={"up": self.GREEN, "down": self.RED},
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                facecolor=self.DARK_BG,
                figcolor=self.DARK_BG,
                gridcolor="#2a2e39",
                gridstyle="--",
            )

            last = len(plot_df) - 1
            hlines = dict(
                hlines=[annotation.entry, annotation.stop_loss, annotation.take_profit],
                colors=[self.WHITE, self.RED, self.GREEN],
                linewidths=[1.5, 1.5, 1.5],
                linestyle="--",
            )

            fig, axes = mpf.plot(
                plot_df,
                type="candle",
                style=style,
                volume=True,
                returnfig=True,
                figsize=self.CHART_SIZE,
                hlines=hlines,
                title=(
                    f"\n{annotation.symbol} {annotation.timeframe} "
                    f"| Score: {annotation.score}/100"
                ),
            )

            ax = axes[0]

            # Order Block zone
            if annotation.ob_zone:
                rect = patches.Rectangle(
                    (0, annotation.ob_zone[0]),
                    last, annotation.ob_zone[1] - annotation.ob_zone[0],
                    linewidth=1, edgecolor=self.BLUE,
                    facecolor=self.BLUE, alpha=0.15,
                    transform=ax.get_yaxis_transform(),
                )
                ax.add_patch(rect)
                ax.text(
                    0.02, annotation.ob_zone[0], "OB",
                    color=self.BLUE, fontsize=8,
                    transform=ax.get_yaxis_transform(),
                )

            # FVG zone
            if annotation.fvg_zone:
                rect2 = patches.Rectangle(
                    (0, annotation.fvg_zone[0]),
                    last, annotation.fvg_zone[1] - annotation.fvg_zone[0],
                    linewidth=1, edgecolor=self.GREEN,
                    facecolor=self.GREEN, alpha=0.15,
                    transform=ax.get_yaxis_transform(),
                )
                ax.add_patch(rect2)

            # Outcome border
            if outcome == TradeOutcome.WIN:
                fig.patch.set_linewidth(4)
                fig.patch.set_edgecolor(self.GREEN)
            elif outcome == TradeOutcome.LOSS:
                fig.patch.set_linewidth(4)
                fig.patch.set_edgecolor(self.RED)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight",
                        facecolor=self.DARK_BG, dpi=120)
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        except Exception as e:
            logger.error(f"mplfinance render error: {e}")
            return self._render_fallback(df, annotation, outcome)

    def _render_fallback(
        self,
        df: pd.DataFrame,
        annotation: "ChartAnnotation",
        outcome: Optional["TradeOutcome"] = None,
    ) -> bytes:
        """
        Simple matplotlib line chart fallback.
        Returns b"PNG_PLACEHOLDER" if matplotlib is also unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.CHART_SIZE, facecolor=self.DARK_BG)
            ax.set_facecolor(self.DARK_BG)

            if not df.empty and "close" in df.columns:
                closes = df["close"].values
                x = range(len(closes))
                color = (
                    self.GREEN
                    if annotation.entry > df["close"].iloc[0]
                    else self.RED
                )
                ax.plot(x, closes, color=color, linewidth=1.5)
                ax.axhline(annotation.entry,      color=self.WHITE, linestyle="--", linewidth=1)
                ax.axhline(annotation.stop_loss,  color=self.RED,   linestyle="--", linewidth=1)
                ax.axhline(annotation.take_profit, color=self.GREEN, linestyle="--", linewidth=1)

            title = (
                f"{annotation.symbol} {annotation.timeframe} "
                f"| Score: {annotation.score}/100"
            )
            ax.set_title(title, color=self.WHITE, fontsize=12)
            ax.tick_params(colors=self.WHITE)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2e39")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight",
                        facecolor=self.DARK_BG, dpi=100)
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        except Exception as e:
            logger.error(f"Fallback render error: {e}")
            return b"PNG_PLACEHOLDER"

    # --------------------------------------------------------------------------
    # Caption builders
    # --------------------------------------------------------------------------

    def build_entry_caption(self, annotation: "ChartAnnotation") -> str:
        return (
            f"TRADE ABIERTO — {annotation.symbol} {annotation.timeframe}\n"
            f"Entrada: {annotation.entry} | SL: {annotation.stop_loss} "
            f"| TP: {annotation.take_profit}\n"
            f"Score: {annotation.score}/100 | R:R 1:{annotation.rr:.1f} "
            f"| Confianza: {annotation.confidence * 100:.0f}%\n"
            f"Setup: {annotation.trigger}"
        )

    def build_close_caption(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        duration_min: int,
        outcome: "TradeOutcome",
        win_rate: float,
    ) -> str:
        emoji = "WIN" if outcome == TradeOutcome.WIN else "LOSS"
        sign  = "+" if pnl >= 0 else ""
        hours = duration_min // 60
        mins  = duration_min % 60
        return (
            f"TRADE CERRADO [{emoji}] — {symbol}\n"
            f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)\n"
            f"Duracion: {hours}h {mins}min\n"
            f"Win rate acumulado: {win_rate:.1f}%"
        )
