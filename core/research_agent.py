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

    # Conocimiento embebido de los 3 mejores traders — no necesita internet
    TOP_TRADER_RULES = [
        {
            "source": "druckenmiller",
            "title": "Druckenmiller: Ride winners, cut losers fast",
            "summary": (
                "Stanley Druckenmiller: nunca promedies perdedores. "
                "Cuando un trade funciona, agrega posicion (pyramid up). "
                "El tamano de posicion es la clave — una operacion grande en el momento "
                "correcto supera 10 operaciones mediocres. "
                "Regla: si D1+H4 confirman, aumentar volumen hasta 3x del normal."
            ),
            "url": "internal://druckenmiller/pyramid_winners",
            "relevance": 1.0,
        },
        {
            "source": "paul_tudor_jones",
            "title": "PTJ: 5:1 risk-reward, never risk more than 1% per trade",
            "summary": (
                "Paul Tudor Jones: busca siempre RR minimo 5:1. "
                "Si el mercado no da 5:1, no entras. "
                "Stop loss es sagrado — si lo tocas, sales SIN EXCEPCION. "
                "Regla: solo abrir swing si TP/SL >= 3.0 Y D1 confirma. "
                "PTJ opera principalmente en la direccion del trend semanal."
            ),
            "url": "internal://ptj/5to1_rr",
            "relevance": 1.0,
        },
        {
            "source": "george_soros",
            "title": "Soros: Reflexivity — trade the trend until it breaks",
            "summary": (
                "George Soros: el mercado crea sus propias tendencias (reflexividad). "
                "Cuando el momentum es fuerte, el mercado sigue mas de lo esperado. "
                "Regla: si H4 lleva 3+ velas consecutivas en la misma direccion, "
                "el proximo movimiento probablemente continua. "
                "Soros apuesta GRANDE en alta conviction — no trades timidos."
            ),
            "url": "internal://soros/reflexivity_momentum",
            "relevance": 1.0,
        },
        {
            "source": "ict_concepts",
            "title": "ICT: Kill zones — solo operar en ventanas institucionales",
            "summary": (
                "Inner Circle Trader (Michael Huddleston): los institucionales mueven "
                "el mercado en kill zones especificas. "
                "London Kill Zone: 02:00-05:00 UTC. NY Kill Zone: 12:00-15:00 UTC. "
                "Fuera de estas ventanas, los movimientos son ruido retail. "
                "Regla: scalps SOLO en kill zones, swings pueden mantenerse."
            ),
            "url": "internal://ict/kill_zones",
            "relevance": 0.95,
        },
        {
            "source": "druckenmiller",
            "title": "Druckenmiller: La macro manda — no operar contra D1",
            "summary": (
                "La tendencia diaria (D1) es la macro del retail. "
                "Nunca abrir un trade contra D1, sin importar cuanto prometa el M15. "
                "Si D1=SHORT, solo SELL. Si D1=LONG, solo BUY. Siempre."
            ),
            "url": "internal://druckenmiller/macro_d1",
            "relevance": 1.0,
        },
    ]

    _credit_fail_ts: float = 0.0  # cooldown 24h tras error de credito API

    def _generate_claude_insight(self) -> list:
        """Genera insight de trading usando Claude API cuando internet falla."""
        import time
        if time.time() - self.__class__._credit_fail_ts < 86400:
            return []  # 24h cooldown tras error de credito
        try:
            import anthropic, os, datetime
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            today = datetime.date.today().isoformat()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Fecha: {today}. Eres un sistema de trading SMC/ICT. "
                        "Dame UNA regla concreta que mejore el win rate en forex/indices hoy. "
                        "Formato: REGLA: [regla en 1 linea] | APLICA_A: [SCALP/SWING/AMBOS] | "
                        "CONDICION: [cuando aplicar] | ACCION: [que hacer exactamente]"
                    ),
                }],
            )
            text = msg.content[0].text if msg.content else ""
            if "REGLA:" in text:
                return [{
                    "source": "claude_insight",
                    "title": f"AI Trading Rule {today}",
                    "summary": text[:500],
                    "url": f"internal://claude/{today}",
                    "relevance": 0.85,
                }]
        except Exception as e:
            if "credit balance" in str(e).lower():
                self.__class__._credit_fail_ts = time.time()
                print(f"[RESEARCH] Claude API sin credito — pausando 24h", flush=True)
            else:
                print(f"[RESEARCH] claude insight error: {e}", flush=True)
        return []

    def run_cycle(self):
        conn = self._get_conn()
        # Primero: conocimiento embebido de top traders (siempre disponible)
        for item in self.TOP_TRADER_RULES:
            if not _already_saved(item["url"], conn):
                save_research(item, conn=conn)
                print(f"[RESEARCH] top-trader saved: {item['title'][:60]}", flush=True)
        # Segundo: insight diario de Claude API
        for item in self._generate_claude_insight():
            if not _already_saved(item["url"], conn):
                save_research(item, conn=conn)
                print(f"[RESEARCH] claude insight saved: {item['title'][:60]}", flush=True)
        # Tercero: intentar internet (no falla si no hay conexion)
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
