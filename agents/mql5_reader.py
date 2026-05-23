"""
MQL5 RSS Reader — reads MQL5 articles, signals and calendar.
Extracts trading strategies using Claude API and saves summaries.
Runs as background task every 6 hours.
"""
import asyncio
import hashlib
import json
import logging
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "articles":  "https://rss.mql5.com/en/articles/rss.xml",
    "signals":   "https://rss.mql5.com/en/signals/rss.xml",
    "calendar":  "https://rss.mql5.com/en/calendar/rss.xml",
}

CACHE_FILE = Path(__file__).parent.parent / "memory" / "mql5_cache.json"
STRATEGIES_FILE = Path(__file__).parent.parent / "memory" / "mql5_strategies.json"
CHECK_INTERVAL_HOURS = 6


@dataclass
class MQL5Item:
    feed_type: str
    title: str
    link: str
    description: str
    pub_date: str
    item_id: str        # SHA256 of link — dedup key
    strategy_summary: str = ""
    extracted_at: str = ""


class MQL5Reader:
    """
    Reads MQL5 RSS feeds, extracts trading strategies with Claude API,
    and caches results locally. No ChromaDB required.
    """

    def __init__(self, anthropic_api_key: str = ""):
        self._api_key = anthropic_api_key
        self._seen_ids: set = self._load_cache()
        self._strategies: list = self._load_strategies()

    # ── Cache ─────────────────────────────────────────────────────────────

    def _load_cache(self) -> set:
        try:
            if CACHE_FILE.exists():
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                return set(data.get("seen_ids", []))
        except Exception:
            pass
        return set()

    def _save_cache(self):
        try:
            CACHE_FILE.parent.mkdir(exist_ok=True)
            CACHE_FILE.write_text(
                json.dumps({"seen_ids": list(self._seen_ids)}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_strategies(self) -> list:
        try:
            if STRATEGIES_FILE.exists():
                return json.loads(STRATEGIES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []

    def _save_strategies(self):
        try:
            STRATEGIES_FILE.parent.mkdir(exist_ok=True)
            STRATEGIES_FILE.write_text(
                json.dumps(self._strategies[-200:], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── RSS Fetch ─────────────────────────────────────────────────────────

    def fetch_feed(self, feed_type: str, url: str) -> list:
        """Fetch and parse RSS feed. Returns list of MQL5Item. Never raises."""
        items = []
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SMCBot/1.0 (trading research)"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                xml_data = r.read()

            root = ET.fromstring(xml_data)
            channel = root.find("channel")
            if channel is None:
                return items

            for entry in channel.findall("item"):
                title       = (entry.findtext("title") or "").strip()
                link        = (entry.findtext("link") or "").strip()
                description = (entry.findtext("description") or "").strip()
                pub_date    = (entry.findtext("pubDate") or "").strip()

                if not title or not link:
                    continue

                item_id = hashlib.sha256(link.encode()).hexdigest()[:16]
                items.append(MQL5Item(
                    feed_type=feed_type,
                    title=title,
                    link=link,
                    description=description[:500],
                    pub_date=pub_date,
                    item_id=item_id,
                ))
        except Exception as e:
            logger.debug(f"MQL5 fetch {feed_type}: {e}")
        return items

    # ── Claude API ────────────────────────────────────────────────────────

    def extract_strategy(self, item: MQL5Item) -> str:
        """Use Claude to extract trading strategy from article. Returns summary string."""
        if not self._api_key or "PEGA" in self._api_key:
            return ""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            prompt = (
                f"Article: {item.title}\n\n"
                f"Description: {item.description}\n\n"
                "Extract in 3 bullet points:\n"
                "1. Trading strategy/signal type\n"
                "2. Entry/exit rules (if mentioned)\n"
                "3. Applicable instruments and timeframes\n"
                "Be concise. If not trading-related, say 'Not a trading strategy'."
            )
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.debug(f"Claude extract error: {e}")
            return ""

    # ── Main scan ─────────────────────────────────────────────────────────

    def scan_once(self) -> list:
        """
        Fetch all feeds, filter new items, extract strategies for articles.
        Returns list of new MQL5Item with strategies populated.
        """
        new_items = []
        for feed_type, url in RSS_FEEDS.items():
            items = self.fetch_feed(feed_type, url)
            for item in items:
                if item.item_id in self._seen_ids:
                    continue
                self._seen_ids.add(item.item_id)

                # Extract strategy for article feed only (rate limit)
                if feed_type == "articles":
                    item.strategy_summary = self.extract_strategy(item)
                    item.extracted_at = datetime.now(timezone.utc).isoformat()

                new_items.append(item)
                self._strategies.append({
                    "feed":      feed_type,
                    "title":     item.title,
                    "link":      item.link,
                    "summary":   item.strategy_summary,
                    "date":      item.pub_date,
                    "saved_at":  datetime.now(timezone.utc).isoformat(),
                })

        self._save_cache()
        self._save_strategies()
        return new_items

    def get_recent_strategies(self, n: int = 10) -> list:
        """Return last n saved strategies."""
        return self._strategies[-n:]

    # ── MT5 news via terminal API ─────────────────────────────────────────

    def fetch_mt5_news(self, limit: int = 99) -> list:
        """
        Read news directly from running MT5 terminal via mt5.news_get().
        Returns list of MQL5Item. Works only when MT5 is connected.
        """
        items = []
        try:
            import MetaTrader5 as mt5
            news = mt5.news_get(limit)
            if news is None:
                return items
            for n in news:
                # MT5 news object has: time, category, source, title, body, url
                title   = getattr(n, "title",    "") or ""
                body    = getattr(n, "body",     "") or ""
                url     = getattr(n, "url",      "") or ""
                source  = getattr(n, "source",   "") or ""
                ts      = getattr(n, "time",     0)
                cat     = getattr(n, "category", "") or ""

                if not title:
                    continue

                link    = url or f"mt5://news/{ts}"
                item_id = hashlib.sha256(link.encode()).hexdigest()[:16]
                items.append(MQL5Item(
                    feed_type   = "mt5_news",
                    title       = title[:120],
                    link        = link,
                    description = body[:400],
                    pub_date    = str(ts),
                    item_id     = item_id,
                ))
        except Exception as e:
            logger.debug(f"MT5 news fetch error: {e}")
        return items

    def scan_mt5_news(self, limit: int = 99) -> list:
        """
        Fetch MT5 terminal news, extract strategies for new items.
        Returns list of new MQL5Item.
        """
        items    = self.fetch_mt5_news(limit)
        new_items = []
        for item in items:
            if item.item_id in self._seen_ids:
                continue
            self._seen_ids.add(item.item_id)
            # Extract strategy impact with Claude
            if self._api_key and "PEGA" not in self._api_key:
                item.strategy_summary = self.extract_strategy(item)
                item.extracted_at = datetime.now(timezone.utc).isoformat()
            new_items.append(item)
            self._strategies.append({
                "feed":     "mt5_news",
                "title":    item.title,
                "link":     item.link,
                "summary":  item.strategy_summary,
                "date":     item.pub_date,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            })
        self._save_cache()
        self._save_strategies()
        return new_items

    def format_telegram(self, items: list) -> str:
        """Format new items for Telegram notification."""
        if not items:
            return ""
        lines = ["<b>MQL5 NUEVOS ARTICULOS</b>", "━━━━━━━━━━━━━━━━━━━━"]
        for item in items[:5]:
            feed_emoji = {"articles": "📖", "signals": "📊", "calendar": "📅"}
            emoji = feed_emoji.get(item.feed_type, "🔔")
            lines.append(f"{emoji} <b>{item.title[:60]}</b>")
            if item.strategy_summary and "Not a trading" not in item.strategy_summary:
                short = item.strategy_summary[:150].replace("\n", " ")
                lines.append(f"   {short}")
        if len(items) > 5:
            lines.append(f"   ...y {len(items)-5} articulos mas")
        return "\n".join(lines)

    # ── Async background loop ─────────────────────────────────────────────

    async def run_loop(self, telegram_bot=None):
        """Background loop — scans every CHECK_INTERVAL_HOURS hours."""
        logger.info("MQL5 reader started")
        while True:
            try:
                loop = asyncio.get_event_loop()
                new_items = await loop.run_in_executor(None, self.scan_once)
                if new_items:
                    print(f"[MQL5] {len(new_items)} new items found")
                    if telegram_bot:
                        msg = self.format_telegram(new_items)
                        if msg:
                            try:
                                await telegram_bot.send_glint_alert(msg)
                            except Exception:
                                pass
                else:
                    logger.debug("MQL5: no new items")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"MQL5 loop error: {e}")
            await asyncio.sleep(CHECK_INTERVAL_HOURS * 3600)
