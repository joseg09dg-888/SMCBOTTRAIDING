"""Tests for MQL5 RSS reader agent."""
import pytest
import json
from unittest.mock import patch, MagicMock
from agents.mql5_reader import MQL5Reader, MQL5Item, RSS_FEEDS


# â”€â”€ RSS_FEEDS constant â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_rss_feeds_has_three_feeds():
    assert len(RSS_FEEDS) == 3
    assert "articles" in RSS_FEEDS
    assert "signals"  in RSS_FEEDS
    assert "calendar" in RSS_FEEDS

def test_rss_feeds_are_valid_urls():
    for name, url in RSS_FEEDS.items():
        assert url.startswith("https://rss.mql5.com"), f"{name} URL invalid"

# â”€â”€ MQL5Item â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_mql5item_fields():
    item = MQL5Item(
        feed_type="articles", title="Test", link="https://example.com",
        description="desc", pub_date="2026-05-22", item_id="abc123"
    )
    assert item.feed_type == "articles"
    assert item.strategy_summary == ""
    assert item.extracted_at == ""

# â”€â”€ MQL5Reader init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_reader_init_no_api_key():
    r = MQL5Reader(anthropic_api_key="")
    assert isinstance(r._seen_ids, set)
    assert isinstance(r._strategies, list)

def test_reader_cache_initially_empty(tmp_path, monkeypatch):
    import agents.mql5_reader as mod
    monkeypatch.setattr(mod, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(mod, "STRATEGIES_FILE", tmp_path / "strat.json")
    r = MQL5Reader()
    assert len(r._seen_ids) == 0
    assert len(r._strategies) == 0

# â”€â”€ fetch_feed (mocked) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SAMPLE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>SMC Trading Strategy Guide</title>
    <link>https://mql5.com/articles/123</link>
    <description>Order blocks and FVG analysis</description>
    <pubDate>Thu, 22 May 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>EURUSD Signal Analysis</title>
    <link>https://mql5.com/articles/124</link>
    <description>Bearish setup detected</description>
    <pubDate>Thu, 22 May 2026 11:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

def test_fetch_feed_parses_items(monkeypatch):
    import urllib.request
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_RSS
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: mock_response)
    r = MQL5Reader()
    items = r.fetch_feed("articles", "https://test.url")
    assert len(items) == 2
    assert items[0].title == "SMC Trading Strategy Guide"
    assert items[0].feed_type == "articles"

def test_fetch_feed_dedup_by_id(monkeypatch):
    import urllib.request
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_RSS
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: mock_response)
    r = MQL5Reader()
    items1 = r.fetch_feed("articles", "https://test.url")
    items2 = r.fetch_feed("articles", "https://test.url")
    # item_id is deterministic from link
    assert items1[0].item_id == items2[0].item_id

def test_fetch_feed_empty_on_network_error():
    r = MQL5Reader()
    items = r.fetch_feed("articles", "https://nonexistent.invalid.url")
    assert items == []

def test_fetch_feed_item_id_is_16_chars(monkeypatch):
    import urllib.request
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_RSS
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: mock_response)
    r = MQL5Reader()
    items = r.fetch_feed("articles", "https://test.url")
    assert len(items[0].item_id) == 16

# â”€â”€ scan_once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_scan_once_deduplicates(monkeypatch):
    import urllib.request, agents.mql5_reader as mod
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_RSS
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: mock_response)
    import tempfile; monkeypatch.setattr(mod, "CACHE_FILE", mod.Path(tempfile.mktemp(suffix="_c.json")))
    monkeypatch.setattr(mod, "STRATEGIES_FILE", mod.Path(tempfile.mktemp(suffix="_s.json")))

    r = MQL5Reader()
    first  = r.scan_once()
    second = r.scan_once()
    assert len(second) == 0  # already seen

def test_scan_once_returns_new_items(monkeypatch):
    import urllib.request, agents.mql5_reader as mod
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_RSS
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: mock_response)
    import tempfile; monkeypatch.setattr(mod, "CACHE_FILE", mod.Path(tempfile.mktemp(suffix="_cache.json")))
    monkeypatch.setattr(mod, "STRATEGIES_FILE", mod.Path(tempfile.mktemp(suffix="_strat.json")))

    r = MQL5Reader()
    items = r.scan_once()
    # 3 feeds x 2 items each = 6
    assert len(items) >= 2

# â”€â”€ extract_strategy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_extract_strategy_no_api_key():
    r = MQL5Reader(anthropic_api_key="")
    item = MQL5Item("articles", "Test", "https://x.com", "desc", "", "id1")
    result = r.extract_strategy(item)
    assert result == ""

def test_extract_strategy_placeholder_key():
    r = MQL5Reader(anthropic_api_key="PEGA_TU_KEY")
    item = MQL5Item("articles", "Test", "https://x.com", "desc", "", "id1")
    result = r.extract_strategy(item)
    assert result == ""

def test_extract_strategy_with_mock_api(monkeypatch):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="1. SMC strategy\n2. OB entry\n3. EURUSD H1")]
    mock_client.messages.create.return_value = mock_msg
    monkeypatch.setattr("anthropic.Anthropic", lambda **k: mock_client)

    r = MQL5Reader(anthropic_api_key="fake_key_123")
    item = MQL5Item("articles", "Test", "https://x.com", "desc", "", "id1")
    result = r.extract_strategy(item)
    assert "SMC strategy" in result

# â”€â”€ get_recent_strategies â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_get_recent_strategies_empty():
    r = MQL5Reader()
    assert r.get_recent_strategies(5) == []

def test_get_recent_strategies_respects_n():
    r = MQL5Reader()
    r._strategies = [{"title": f"s{i}"} for i in range(20)]
    result = r.get_recent_strategies(5)
    assert len(result) == 5

# â”€â”€ format_telegram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_format_telegram_empty():
    r = MQL5Reader()
    assert r.format_telegram([]) == ""

def test_format_telegram_with_items():
    r = MQL5Reader()
    item = MQL5Item("articles", "Order Block Strategy", "https://x.com",
                    "desc", "", "id1")
    item.strategy_summary = "1. SMC\n2. OB entry\n3. H1"
    msg = r.format_telegram([item])
    assert "MQL5" in msg or "ARTICULOS" in msg
    assert "Order Block" in msg

def test_format_telegram_html():
    r = MQL5Reader()
    item = MQL5Item("articles", "Test Title", "https://x.com", "desc", "", "id1")
    msg = r.format_telegram([item])
    assert "<b>" in msg

def test_format_telegram_max_5_items():
    r = MQL5Reader()
    items = [MQL5Item("articles", f"Title {i}", "https://x.com", "", "", f"id{i}") for i in range(10)]
    msg = r.format_telegram(items)
    assert "5 articulos mas" in msg

# â”€â”€ async run_loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@pytest.mark.asyncio
async def test_run_loop_cancels_cleanly(monkeypatch):
    import asyncio
    import agents.mql5_reader as mod
    monkeypatch.setattr(mod, "CHECK_INTERVAL_HOURS", 0.001)
    r = MQL5Reader()
    # Patch scan_once to return empty fast
    r.scan_once = lambda: []
    task = asyncio.create_task(r.run_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass  # expected



