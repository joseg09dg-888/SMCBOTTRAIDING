"""
GlintBrowser — Playwright-based Glint connector.

Session flow:
  - memory/glint_session.json absent → opens visible Chromium, user logs in
    with Google once, cookies saved, window closes, continues headless.
  - memory/glint_session.json present → headless startup, cookies loaded.
  - Session expiry detected → re-opens visible window for re-login.

Polls DOM every 15 s. Same on_signal interface as GlintConnector.
"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Dict

from connectors.glint_connector import GlintSignal

GLINT_URL        = "https://glint.trade"
POLLING_INTERVAL = 15      # seconds
RETRY_INTERVAL   = 300     # 5 min before retrying after total failure
MAX_SEEN_IDS     = 1000
SESSION_FILE     = Path(__file__).parent.parent / "memory" / "glint_session.json"

# DOM selectors tried in order — first match wins
_SIGNAL_SELECTORS = [
    "[data-signal]",
    ".signal-card",
    ".signal-item",
    ".news-signal",
    ".feed-item",
    "[class*='SignalCard']",
    "[class*='signal-row']",
    "article",
]


class GlintBrowser:
    """
    Drop-in replacement for GlintConnector that uses Playwright instead of
    WebSocket. Same constructor signature and stats() / on_signal interface.
    """

    def __init__(
        self,
        ws_url: str,           # kept for interface compat with supervisor
        session_token: str,    # kept for interface compat
        email: str = "",
        on_signal: Optional[Callable[[GlintSignal], None]] = None,
        min_impact: str = "High",
    ):
        self.ws_url            = ws_url
        self.session_token     = session_token
        self.email             = email
        self.on_signal         = on_signal
        self.min_impact        = min_impact
        self.connected         = False
        self._signals_received = 0
        self._seen_ids: set    = set()
        self._offline          = False
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────

    async def connect(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("Glint en modo offline — instala playwright: pip install playwright")
            return

        while True:
            try:
                async with async_playwright() as p:
                    await self._run_session(p)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                if not self._offline:
                    print(f"Glint en modo offline - bot opera sin noticias ({e})")
                    self._offline = True
                await asyncio.sleep(RETRY_INTERVAL)

    # ── Session management ────────────────────────────────────────────────

    async def _run_session(self, p):
        saved = self._load_session()
        headless = saved is not None

        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()

        if saved:
            await context.add_cookies(saved["cookies"])

        page = await context.new_page()
        await page.goto(GLINT_URL, wait_until="domcontentloaded", timeout=30_000)

        if await self._needs_login(page):
            if headless:
                print("[Glint] Sesión expirada — abriendo ventana para re-login...")
                await browser.close()
                browser = await p.chromium.launch(headless=False)
                context  = await browser.new_context()
                page     = await context.new_page()
                await page.goto(GLINT_URL, wait_until="domcontentloaded", timeout=30_000)

            print("[Glint] Ventana abierta — haz login con tu cuenta de Google.")
            logged_in = await self._do_login(page)

            if not logged_in:
                self._offline = True
                await browser.close()
                return  # bot continues without Glint

            cookies = await context.cookies()
            self._save_session(cookies)
            print("[Glint] Sesión guardada — próximas veces entrará automático.")

            # Relaunch headless now that cookies are saved
            await browser.close()
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            fresh   = self._load_session()
            if fresh:
                await context.add_cookies(fresh["cookies"])
            page = await context.new_page()
            await page.goto(GLINT_URL, wait_until="domcontentloaded", timeout=30_000)

        self.connected = True
        self._offline  = False
        if headless:
            print("[Glint] Conectado (headless) — escaneando señales cada 15 s")
        else:
            print("[Glint] Conectado — escaneando señales cada 15 s")

        try:
            await self._polling_loop(page, context)
        finally:
            await browser.close()

    async def _needs_login(self, page) -> bool:
        url = page.url
        if any(kw in url for kw in ("login", "auth", "signin")):
            return True
        for sel in ("text=Continue with Google", "text=Sign in with Google",
                    "[data-provider='google']", "button[aria-label*='Google']"):
            try:
                el = await page.query_selector(sel)
                if el:
                    return True
            except Exception:
                pass
        return False

    async def _do_login(self, page) -> bool:
        """Returns True if login succeeded, False if timed out."""
        # Click Google button if visible
        for sel in ("text=Continue with Google", "text=Sign in with Google",
                    "[data-provider='google']"):
            try:
                el = await page.query_selector(sel)
                if el:
                    await el.click()
                    break
            except Exception:
                pass

        print("⏳ Tienes 2 minutos para hacer login con Google en la ventana...")
        try:
            # Wait until URL lands back on glint.trade (post-OAuth redirect)
            await page.wait_for_function(
                """() => {
                    const url = window.location.href;
                    return url.includes('glint.trade') &&
                           !url.includes('login') &&
                           !url.includes('signin') &&
                           !url.includes('auth');
                }""",
                timeout=120_000,
            )
            await asyncio.sleep(2)
            return True
        except Exception:
            print("[Glint] Tiempo agotado — continuando sin Glint (bot opera normal)")
            return False

    # ── Polling loop ──────────────────────────────────────────────────────

    async def _polling_loop(self, page, context):
        while True:
            try:
                signals = await self._extract_signals(page)
                self._dispatch(signals)

                await asyncio.sleep(POLLING_INTERVAL)

                await page.reload(wait_until="domcontentloaded", timeout=15_000)

                # Persist refreshed cookies
                cookies = await context.cookies()
                self._save_session(cookies)

                if await self._needs_login(page):
                    print("[Glint] Sesión expirada durante polling — reconectando...")
                    self.connected = False
                    break  # outer loop in connect() handles re-login

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[Glint] Error en ciclo de polling: {e}")
                await asyncio.sleep(POLLING_INTERVAL)

    # ── Signal extraction ─────────────────────────────────────────────────

    async def _extract_signals(self, page) -> list:
        signals = []
        try:
            items = []
            for sel in _SIGNAL_SELECTORS:
                found = await page.query_selector_all(sel)
                if found:
                    items = found
                    break

            for item in items[:20]:
                try:
                    text = (await item.inner_text()).strip()
                    if not text or len(text) < 10:
                        continue

                    sig_id   = await item.get_attribute("data-id") or f"glint-{hash(text[:60])}"
                    category = await item.get_attribute("data-category") or "Economics"
                    impact   = await item.get_attribute("data-impact") or self._infer_impact(text)
                    market   = await item.get_attribute("data-market") or ""
                    tier     = int(await item.get_attribute("data-tier") or 2)
                    relevance = float(await item.get_attribute("data-relevance") or 7.0)

                    signals.append(GlintSignal.from_dict({
                        "id":              sig_id,
                        "category":        category,
                        "impact":          impact,
                        "text":            text[:300],
                        "source_tier":     tier,
                        "relevance_score": relevance,
                        "matched_market":  market,
                        "timestamp":       datetime.now(timezone.utc).isoformat(),
                    }))
                except Exception:
                    continue
        except Exception:
            pass
        return signals

    def _infer_impact(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ("critical", "crash", "war", "attack", "emergency", "urgente")):
            return "Critical"
        if any(w in t for w in ("fed", "rate hike", "gdp", "major", "high", "important")):
            return "High"
        if any(w in t for w in ("medium", "moderate", "moderado")):
            return "Medium"
        return "Low"

    def _dispatch(self, signals: list):
        for signal in signals:
            if signal.signal_id in self._seen_ids:
                continue
            if len(self._seen_ids) >= MAX_SEEN_IDS:
                self._seen_ids.clear()
            self._seen_ids.add(signal.signal_id)
            self._signals_received += 1
            if self._should_process(signal) and self.on_signal:
                self.on_signal(signal)

    def _should_process(self, signal: GlintSignal) -> bool:
        impact_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        min_rank    = impact_rank.get(self.min_impact, 3)
        sig_rank    = impact_rank.get(signal.impact, 1)
        return sig_rank >= min_rank

    # ── Session persistence ───────────────────────────────────────────────

    def _load_session(self) -> Optional[Dict]:
        if not SESSION_FILE.exists():
            return None
        try:
            return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_session(self, cookies: list):
        try:
            SESSION_FILE.write_text(
                json.dumps(
                    {"cookies": cookies, "saved_at": datetime.now(timezone.utc).isoformat()},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[Glint] No se pudo guardar sesión: {e}")

    def stats(self) -> Dict:
        return {
            "connected":        self.connected,
            "signals_received": self._signals_received,
            "ws_url":           self.ws_url,
            "mode":             "browser",
            "session_saved":    SESSION_FILE.exists(),
        }
