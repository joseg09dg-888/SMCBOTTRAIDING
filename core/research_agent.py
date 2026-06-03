# core/research_agent.py
import re
import sqlite3
from typing import List, Optional
from memory.episodic_db import get_db, save_research

SMC_KEYWORDS = [
    "smart money", "smc", "order block", "ict", "liquidity",
    "fair value gap", "fvg", "choch", "bos", "market structure",
    "institutional", "imbalance", "sweep", "mitigation",
]

ARXIV_URL = (
    "https://export.arxiv.org/api/query"
    "?search_query=cat:q-fin.TR+AND+(SMC+OR+order+block+OR+ICT+OR+liquidity)"
    "&sortBy=submittedDate&sortOrder=descending&max_results=10"
)
MQL5_URL = "https://www.mql5.com/en/articles"


def _score_relevance(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in SMC_KEYWORDS if kw in text_lower)
    return min(1.0, hits * 0.25)


def _already_saved(url: str, conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT id FROM research WHERE url=?", (url,)).fetchone()
    return row is not None


class ResearchAgent:
    """
    Every 2 hours: fetch up to 5 new items from arXiv and MQL5,
    score relevance, and save those with relevance > 0.4 to research table.
    """

    RELEVANCE_THRESHOLD = 0.4

    def __init__(self, conn: sqlite3.Connection = None):
        self._conn = conn

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn or get_db()

    def _fetch_arxiv(self) -> List[dict]:
        try:
            import httpx
            resp = httpx.get(ARXIV_URL, timeout=10)
            if resp.status_code != 200:
                return []
            # Use xml.etree for robust parsing (arXiv Atom: id before title)
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
            except Exception:
                entries = []
            items = []
            for entry in entries[:5]:
                title   = (entry.findtext("atom:title", "", ns) or "").strip()
                summary = (entry.findtext("atom:summary", "", ns) or "").strip()[:500]
                url     = (entry.findtext("atom:id", "", ns) or "").strip()
                title   = re.sub(r"\s+", " ", title)
                summary = re.sub(r"\s+", " ", summary)
                score   = _score_relevance(f"{title} {summary}")
                if score >= self.RELEVANCE_THRESHOLD:
                    items.append({"source": "arxiv", "title": title,
                                  "summary": summary, "url": url.strip(),
                                  "relevance": score})
            return items
        except Exception as e:
            print(f"[RESEARCH] arxiv unavailable: {e}", flush=True)
            return []

    def _fetch_mql5(self) -> List[dict]:
        try:
            import httpx
            resp = httpx.get(MQL5_URL, timeout=10)
            if resp.status_code != 200:
                return []
            links = re.findall(
                r'href="(/articles/\d+)"[^>]*>(.*?)</a>', resp.text
            )
            items = []
            for href, title in links[:5]:
                title = re.sub(r"<[^>]+>", "", title).strip()
                if not title:
                    continue
                score = _score_relevance(title)
                if score >= self.RELEVANCE_THRESHOLD:
                    url = f"https://www.mql5.com{href}"
                    items.append({"source": "mql5", "title": title,
                                  "summary": f"MQL5 article: {title}",
                                  "url": url, "relevance": score})
            return items
        except Exception as e:
            print(f"[RESEARCH] mql5 unavailable: {e}", flush=True)
            return []

    def run_cycle(self):
        conn = self._get_conn()
        for source_items in [self._fetch_arxiv(), self._fetch_mql5()]:
            for item in source_items:
                if not _already_saved(item["url"], conn):
                    save_research(item, conn=conn)
                    print(
                        f"[RESEARCH] {item['source']} saved: {item['title'][:60]}",
                        flush=True,
                    )

    def get_top_research(self, n: int = 3, conn: sqlite3.Connection = None) -> list:
        c = conn or self._get_conn()
        rows = c.execute(
            "SELECT * FROM research WHERE applied=0 ORDER BY relevance DESC, id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]
