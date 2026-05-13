import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Graceful imports for optional dependencies
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_TRANSCRIPT_API = True
except ImportError:
    HAS_TRANSCRIPT_API = False

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class StrategyStyle(Enum):
    SMC          = "smc"
    WYCKOFF      = "wyckoff"
    PRICE_ACTION = "price_action"
    INSTITUTIONAL = "institutional"
    OTHER        = "other"


STYLE_CONDITIONS = {
    StrategyStyle.SMC:           ["trending", "momentum", "bos", "choch"],
    StrategyStyle.WYCKOFF:       ["ranging", "accumulation", "distribution"],
    StrategyStyle.PRICE_ACTION:  ["breakout", "pullback", "reversal"],
    StrategyStyle.INSTITUTIONAL: ["macro", "news", "fundamental"],
}


@dataclass
class ExtractedStrategy:
    title: str
    entry_rule: str
    exit_rule: str
    market_condition: str
    style: StrategyStyle
    channel: str
    confidence: float
    video_id: str = ""
    tags: List[str] = field(default_factory=list)


class StrategySelector:
    """Selects the most appropriate strategy for current market conditions."""

    CONDITION_STYLE_MAP = {
        "trending":     StrategyStyle.SMC,
        "momentum":     StrategyStyle.SMC,
        "ranging":      StrategyStyle.WYCKOFF,
        "accumulation": StrategyStyle.WYCKOFF,
        "distribution": StrategyStyle.WYCKOFF,
        "breakout":     StrategyStyle.PRICE_ACTION,
        "pullback":     StrategyStyle.PRICE_ACTION,
        "reversal":     StrategyStyle.PRICE_ACTION,
        "macro":        StrategyStyle.INSTITUTIONAL,
    }

    def select(
        self,
        strategies: List[ExtractedStrategy],
        market_condition: str,
    ) -> Optional[ExtractedStrategy]:
        if not strategies:
            return None

        preferred_style = self.CONDITION_STYLE_MAP.get(
            market_condition.lower(), StrategyStyle.SMC
        )

        # Filter by preferred style, sorted by confidence
        matching = sorted(
            [s for s in strategies if s.style == preferred_style],
            key=lambda s: s.confidence,
            reverse=True,
        )
        if matching:
            return matching[0]

        # Fallback: highest confidence overall
        return sorted(strategies, key=lambda s: s.confidence, reverse=True)[0]


EXTRACTION_PROMPT = """You are an expert in Smart Money Concepts (SMC) and trading strategies.
Analyze this YouTube video transcript and extract the main trading strategy.

Return ONLY valid JSON with this exact structure:
{
  "title": "Strategy name",
  "entry_rule": "Specific entry condition",
  "exit_rule": "Exit/TP/SL rules",
  "market_condition": "trending|ranging|breakout|accumulation|distribution|reversal",
  "style": "smc|wyckoff|price_action|institutional|other",
  "confidence": 0.85
}

If no clear strategy is found, return: {"error": "no_strategy"}

Transcript:
"""


class YouTubeTrainer:
    """
    Learns trading strategies from YouTube videos.
    Downloads transcripts -> extracts strategies with Claude -> stores in ChromaDB.
    """

    CHANNELS = {
        "InnerCircleTrader":    {"style": "smc",           "url_id": "UCpXKbJACcDqbVjM0JESwXoQ"},
        "SmartMoneyTechniques": {"style": "smc",           "url_id": "UCg2p-5gmNxEmC4P4qLvR7Mg"},
        "TheTraderNick":        {"style": "smc",           "url_id": "UCsOFwDU0qMtIj5WZ1WJBzSQ"},
        "TheRaynerTeo":         {"style": "price_action",  "url_id": "UCyLtE9E0Zt1jh1s5JCJ7Beg"},
        "TheChartGuys":         {"style": "price_action",  "url_id": "UCtt7GS_MTvbzFMFVj4c7ahw"},
        "WyckoffAnalytics":     {"style": "wyckoff",       "url_id": "UCzjODaFM1VHCbQrTFGDsTIg"},
        "AntonKreil":           {"style": "institutional", "url_id": "UCrWc3H-6A38RVGmFHxULiXg"},
        "AdamKhoo":             {"style": "price_action",  "url_id": "UCXMhHQQZQBMt2Zit6d8sFTA"},
        "TraderDante":          {"style": "wyckoff",       "url_id": "UCNYRQcMBW3FWLcmEAEoEasg"},
        "TopTradersUnplugged":  {"style": "institutional", "url_id": "UCVtMFrFKRU-6M2G40ZRinBw"},
    }

    def __init__(self, anthropic_api_key: str = "", db_path: str = "./memory/strategies_db"):
        self.api_key   = anthropic_api_key
        self.db_path   = db_path
        self.strategies: List[ExtractedStrategy] = []
        self.selector  = StrategySelector()
        self._db       = None
        self._client   = None

    def _get_transcript(self, video_id: str) -> Optional[str]:
        if not HAS_TRANSCRIPT_API:
            logger.warning("youtube-transcript-api not installed")
            return None
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "es"])
            return " ".join(t["text"] for t in transcript)
        except Exception as e:
            logger.warning(f"Transcript error for {video_id}: {e}")
            return None

    def _extract_with_claude(self, transcript: str, channel: str) -> Optional[str]:
        try:
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)

            # Truncate transcript to 8k chars for token efficiency
            text = transcript[:8000]
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT + text,
                }],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude extraction error: {e}")
            return None

    def _parse_strategy_json(self, json_text: str, channel: str, video_id: str) -> Optional[ExtractedStrategy]:
        try:
            # Extract JSON from possible markdown code blocks
            text = json_text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())
            if "error" in data:
                return None

            style_map = {s.value: s for s in StrategyStyle}
            style = style_map.get(data.get("style", "other"), StrategyStyle.OTHER)

            return ExtractedStrategy(
                title            = data.get("title", "Unknown Strategy"),
                entry_rule       = data.get("entry_rule", ""),
                exit_rule        = data.get("exit_rule", ""),
                market_condition = data.get("market_condition", "trending"),
                style            = style,
                channel          = channel,
                confidence       = float(data.get("confidence", 0.7)),
                video_id         = video_id,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"JSON parse failed: {e}")
            return None

    def process_video(self, video_id: str, channel: str) -> Optional[ExtractedStrategy]:
        transcript = self._get_transcript(video_id)
        if not transcript:
            return None
        raw = self._extract_with_claude(transcript, channel)
        if not raw:
            return None
        strategy = self._parse_strategy_json(raw, channel, video_id)
        if strategy:
            self.strategies.append(strategy)
            self._store_strategy(strategy)
        return strategy

    def _store_strategy(self, strategy: ExtractedStrategy):
        if not HAS_CHROMADB or self._db is None:
            return  # In-memory fallback is self.strategies list
        try:
            collection = self._db.get_or_create_collection("strategies")
            collection.add(
                documents=[f"{strategy.title}: {strategy.entry_rule}"],
                metadatas=[{
                    "channel":    strategy.channel,
                    "style":      strategy.style.value,
                    "condition":  strategy.market_condition,
                    "confidence": strategy.confidence,
                }],
                ids=[f"{strategy.channel}_{strategy.video_id}"],
            )
        except Exception as e:
            logger.warning(f"ChromaDB store failed: {e}")

    def get_best_strategy(self, market_condition: str) -> Optional[ExtractedStrategy]:
        return self.selector.select(self.strategies, market_condition)

    def summary(self) -> str:
        by_style: Dict[str, int] = {}
        for s in self.strategies:
            by_style[s.style.value] = by_style.get(s.style.value, 0) + 1
        total = len(self.strategies)
        lines = [f"YouTube Trainer: {total} strategies loaded"]
        for style, count in by_style.items():
            lines.append(f"  {style}: {count}")
        return "\n".join(lines)
