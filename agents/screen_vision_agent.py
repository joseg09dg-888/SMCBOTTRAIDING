# agents/screen_vision_agent.py
"""
ScreenVisionAgent — Captura pantalla y analiza con Claude Vision API.
Todos los imports de pyautogui, mss, anthropic son opcionales (try/except).
"""

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Imports opcionales ────────────────────────────────────────────────────

try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from PIL import Image
    import io as _io
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

# PNG 1x1 rojo — fallback si Pillow no está disponible
_MINIMAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
)


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class ScreenCapture:
    image_bytes: bytes        # PNG bytes de la captura
    timestamp: datetime
    window_title: str         # "full" | "Binance" | "MetaTrader 5"
    width: int
    height: int

    def to_base64(self) -> str:
        return base64.b64encode(self.image_bytes).decode()


@dataclass
class VisionAnalysis:
    symbol: str                  # par detectado, e.g. "BTCUSDT"
    timeframe: str               # "1H", "4H", etc. "" si no detectado
    price: float                 # precio actual visible, 0.0 si no detectado
    market_structure: str        # "bullish", "bearish", "ranging", "unknown"
    order_blocks: list           # descripción de OBs visibles
    fvgs: list                   # descripción de FVGs visibles
    has_valid_setup: bool
    entry: float                 # 0.0 si no hay setup
    stop_loss: float             # 0.0 si no hay setup
    take_profit: float           # 0.0 si no hay setup
    visual_score: int            # 0-100
    raw_response: str            # respuesta original del modelo
    error: Optional[str] = None  # si falló el análisis


@dataclass
class VisionAlert:
    timestamp: datetime
    window: str
    analysis: VisionAnalysis
    alert_type: str              # "setup_detected", "strong_move", "pattern_complete"
    telegram_message: str


@dataclass
class MirrorSession:
    start_time: datetime
    actions_recorded: int
    patterns_learned: int
    is_active: bool


# ── Clase principal ───────────────────────────────────────────────────────

class ScreenVisionAgent:
    """
    Captura pantalla y analiza con Claude Vision API.
    Todos los imports de pyautogui, mss, anthropic son opcionales (try/except).
    Los tests usan MOCKS para todo lo externo.
    """

    VISION_PROMPT = """Eres un trader experto SMC. Analiza esta pantalla de trading y extrae:
1. Par y timeframe visible
2. Estructura de mercado (HH/HL/LH/LL)
3. Order Blocks visibles
4. FVGs presentes
5. Nivel de precio actual
6. Volumen visible
7. ¿Hay un setup SMC válido?
8. Si hay setup: entrada, SL, TP sugeridos
Responde SOLO en JSON con estas claves:
{"symbol": "BTCUSDT", "timeframe": "1H", "price": 67450.0, \
 "market_structure": "bullish", "order_blocks": ["bullish OB en 67000"],
 "fvgs": ["FVG alcista 67100-67200"], "has_valid_setup": true,
 "entry": 67100.0, "stop_loss": 66800.0, "take_profit": 67800.0,
 "visual_score": 82}"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        capture_interval: int = 5,
        telegram_bot=None,
        enabled: bool = True,
    ):
        self._api_key = api_key
        self._model = model
        self._capture_interval = capture_interval
        self._telegram = telegram_bot
        self._enabled = enabled
        self._is_running = False
        self._mirror_active = False
        self._mirror_session: Optional[MirrorSession] = None
        self._analysis_history: list = []
        self._last_capture: Optional[ScreenCapture] = None

    # ── Captura de pantalla ───────────────────────────────────────────────

    def capture_full_screen(self) -> Optional[ScreenCapture]:
        """
        Intenta captura con mss. Si mss no disponible → retorna None.
        Si mss disponible pero falla → retorna None.
        NUNCA lanza excepción.
        """
        if not HAS_MSS:
            return None
        try:
            with mss.MSS() as sct:
                monitor = sct.monitors[0]  # monitor 0 = all monitors combined
                screenshot = sct.grab(monitor)
                # Convertir a PNG bytes
                import io
                from PIL import Image as PILImage
                img = PILImage.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                return ScreenCapture(
                    image_bytes=image_bytes,
                    timestamp=datetime.now(timezone.utc),
                    window_title="full",
                    width=screenshot.width,
                    height=screenshot.height,
                )
        except Exception as e:
            logger.debug(f"capture_full_screen failed: {e}")
            return None

    def capture_window(self, window_title: str) -> Optional[ScreenCapture]:
        """
        Intenta encontrar y capturar ventana con ese título.
        Usa pyautogui.getWindowsWithTitle() si disponible.
        Si no → captura pantalla completa como fallback.
        NUNCA lanza excepción.
        """
        try:
            if HAS_PYAUTOGUI:
                windows = pyautogui.getWindowsWithTitle(window_title)
                if windows:
                    win = windows[0]
                    # Capturar region de la ventana con mss si disponible
                    if HAS_MSS:
                        try:
                            with mss.MSS() as sct:
                                region = {
                                    "top": win.top,
                                    "left": win.left,
                                    "width": win.width,
                                    "height": win.height,
                                }
                                screenshot = sct.grab(region)
                                import io
                                from PIL import Image as PILImage
                                img = PILImage.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                                buf = io.BytesIO()
                                img.save(buf, format="PNG")
                                image_bytes = buf.getvalue()
                                return ScreenCapture(
                                    image_bytes=image_bytes,
                                    timestamp=datetime.now(timezone.utc),
                                    window_title=window_title,
                                    width=win.width,
                                    height=win.height,
                                )
                        except Exception as e:
                            logger.debug(f"capture_window mss grab failed: {e}")
            # Fallback: captura pantalla completa
            return self.capture_full_screen()
        except Exception as e:
            logger.debug(f"capture_window failed: {e}")
            return None

    def create_mock_capture(self, width: int = 100, height: int = 100) -> ScreenCapture:
        """
        Crea una captura falsa para tests.
        image_bytes = bytes de PNG mínimo válido.
        window_title = "mock"
        """
        if HAS_PILLOW:
            try:
                import io
                img = Image.new("RGB", (width, height), color=(255, 0, 0))
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            except Exception:
                image_bytes = _MINIMAL_PNG
        else:
            image_bytes = _MINIMAL_PNG

        return ScreenCapture(
            image_bytes=image_bytes,
            timestamp=datetime.now(timezone.utc),
            window_title="mock",
            width=width,
            height=height,
        )

    # ── Análisis con Claude Vision ────────────────────────────────────────

    def _call_vision_api(self, capture: ScreenCapture) -> str:
        """
        Llama a Claude API con la imagen.
        Solo si anthropic está disponible Y self._api_key no está vacío.
        Retorna el texto de la respuesta.
        NUNCA lanza excepción — captura todo con try/except.
        """
        if not HAS_ANTHROPIC or not self._api_key:
            return ""
        try:
            client = _anthropic.Anthropic(api_key=self._api_key)
            image_b64 = capture.to_base64()
            message = client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": self.VISION_PROMPT,
                            },
                        ],
                    }
                ],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"_call_vision_api error: {e}")
            return ""

    def analyze_capture(self, capture: ScreenCapture) -> VisionAnalysis:
        """
        Envía la captura a Claude Vision API.
        Si api_key vacío o anthropic no disponible → retorna VisionAnalysis default con error.
        Parsea el JSON de la respuesta.
        Si el JSON es inválido → retorna VisionAnalysis con error="parse_error", raw_response=respuesta.
        NUNCA lanza excepción.
        """
        if not self._api_key:
            return VisionAnalysis(
                symbol="", timeframe="", price=0.0,
                market_structure="unknown", order_blocks=[], fvgs=[],
                has_valid_setup=False, entry=0.0, stop_loss=0.0, take_profit=0.0,
                visual_score=0, raw_response="",
                error="no_api_key",
            )

        try:
            raw = self._call_vision_api(capture)
            if not raw:
                return VisionAnalysis(
                    symbol="", timeframe="", price=0.0,
                    market_structure="unknown", order_blocks=[], fvgs=[],
                    has_valid_setup=False, entry=0.0, stop_loss=0.0, take_profit=0.0,
                    visual_score=0, raw_response=raw,
                    error="empty_response",
                )
            parsed = self.parse_vision_response(raw)
            if not parsed:
                return VisionAnalysis(
                    symbol="", timeframe="", price=0.0,
                    market_structure="unknown", order_blocks=[], fvgs=[],
                    has_valid_setup=False, entry=0.0, stop_loss=0.0, take_profit=0.0,
                    visual_score=0, raw_response=raw,
                    error="parse_error",
                )
            analysis = self.build_vision_analysis(parsed, raw)
            return analysis
        except Exception as e:
            logger.error(f"analyze_capture error: {e}")
            return VisionAnalysis(
                symbol="", timeframe="", price=0.0,
                market_structure="unknown", order_blocks=[], fvgs=[],
                has_valid_setup=False, entry=0.0, stop_loss=0.0, take_profit=0.0,
                visual_score=0, raw_response="",
                error=str(e),
            )

    def parse_vision_response(self, raw: str) -> dict:
        """
        Parsea el JSON de la respuesta de Claude.
        Busca el primer bloque JSON en el texto (puede haber texto antes/después).
        Si no encuentra JSON válido → retorna dict vacío {}.
        """
        if not raw:
            return {}
        # Intentar parsear directamente
        try:
            result = json.loads(raw.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        # Buscar el primer bloque JSON con regex
        # Busca el patrón {...} más externo
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, raw, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                continue
        # Intentar encontrar JSON de apertura/cierre de llaves balanceadas
        try:
            start = raw.index('{')
            depth = 0
            for i, ch in enumerate(raw[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start:i + 1]
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                        break
        except (ValueError, json.JSONDecodeError):
            pass
        return {}

    def build_vision_analysis(self, parsed: dict, raw: str) -> VisionAnalysis:
        """
        Construye VisionAnalysis desde el dict parseado.
        Usa .get() con defaults seguros para todas las claves.
        """
        return VisionAnalysis(
            symbol=parsed.get("symbol", ""),
            timeframe=parsed.get("timeframe", ""),
            price=float(parsed.get("price", 0.0)),
            market_structure=parsed.get("market_structure", "unknown"),
            order_blocks=list(parsed.get("order_blocks", [])),
            fvgs=list(parsed.get("fvgs", [])),
            has_valid_setup=bool(parsed.get("has_valid_setup", False)),
            entry=float(parsed.get("entry", 0.0)),
            stop_loss=float(parsed.get("stop_loss", 0.0)),
            take_profit=float(parsed.get("take_profit", 0.0)),
            visual_score=int(parsed.get("visual_score", 0)),
            raw_response=raw,
            error=None,
        )

    # ── Loop principal ────────────────────────────────────────────────────

    async def run_vision_loop(self, window_title: str = "full"):
        """
        Bucle asyncio que cada self._capture_interval segundos:
        1. Captura pantalla
        2. Analiza con Claude Vision
        3. Guarda en _analysis_history (max 100 entries)
        4. Si análisis tiene setup válido con visual_score > 70 → genera alerta
        Se detiene cuando self._is_running = False.
        """
        while self._is_running:
            try:
                # 1. Capturar
                if window_title == "full":
                    capture = self.capture_full_screen()
                else:
                    capture = self.capture_window(window_title)

                if capture is not None:
                    # 2. Analizar
                    analysis = self.analyze_capture(capture)
                    # 3. Guardar historial (max 100)
                    self._analysis_history.append(analysis)
                    if len(self._analysis_history) > 100:
                        self._analysis_history = self._analysis_history[-100:]
                    self._last_capture = capture
                    # 4. Alerta si setup válido con score > 70
                    if analysis.has_valid_setup and analysis.visual_score > 70:
                        await self.send_alert(analysis, window_title)
            except Exception as e:
                logger.error(f"run_vision_loop cycle error: {e}")
            await asyncio.sleep(self._capture_interval)

    def start(self):
        """Marca self._is_running = True."""
        self._is_running = True

    def stop(self):
        """Marca self._is_running = False."""
        self._is_running = False

    # ── Modo espejo ───────────────────────────────────────────────────────

    def start_mirror_mode(self) -> MirrorSession:
        """Inicia MirrorSession, marca self._mirror_active = True."""
        self._mirror_session = MirrorSession(
            start_time=datetime.now(timezone.utc),
            actions_recorded=0,
            patterns_learned=0,
            is_active=True,
        )
        self._mirror_active = True
        return self._mirror_session

    def stop_mirror_mode(self) -> Optional[MirrorSession]:
        """Detiene mirror mode, retorna el MirrorSession completado."""
        if self._mirror_session is None:
            return None
        self._mirror_session.is_active = False
        self._mirror_active = False
        return self._mirror_session

    def record_mirror_action(self, action_description: str):
        """Incrementa actions_recorded en el MirrorSession activo."""
        if self._mirror_session is not None and self._mirror_active:
            self._mirror_session.actions_recorded += 1
            logger.debug(f"Mirror action recorded: {action_description}")

    # ── Alertas ───────────────────────────────────────────────────────────

    def build_alert_message(self, analysis: VisionAnalysis, window: str) -> str:
        """
        Construye el mensaje de alerta para Telegram.
        """
        setup_str = "detectado" if analysis.has_valid_setup else "no detectado"
        return (
            f"DETECCION VISUAL — {analysis.symbol}\n"
            f"Vi: {analysis.market_structure} en {analysis.timeframe}\n"
            f"Precio: ${analysis.price}\n"
            f"Setup: {setup_str}\n"
            f"Score visual: {analysis.visual_score}/100"
        )

    async def send_alert(self, analysis: VisionAnalysis, window: str):
        """
        Si self._telegram disponible → llama send_glint_alert con build_alert_message.
        NUNCA lanza excepción.
        """
        if self._telegram is None:
            return
        try:
            msg = self.build_alert_message(analysis, window)
            await self._telegram.send_glint_alert(msg)
        except Exception as e:
            logger.error(f"send_alert error: {e}")

    # ── Estado y toggle ───────────────────────────────────────────────────

    def toggle(self) -> bool:
        """Alterna self._enabled. Retorna nuevo valor."""
        self._enabled = not self._enabled
        return self._enabled

    def get_status_message(self) -> str:
        """Mensaje de estado legible para Telegram."""
        state = "ACTIVA" if self._enabled else "DESACTIVADA"
        running = "corriendo" if self._is_running else "detenida"
        mirror = "ON" if self._mirror_active else "OFF"
        history_count = len(self._analysis_history)
        last = self._analysis_history[-1].symbol if self._analysis_history else "ninguno"
        return (
            f"Vision de Pantalla: {state}\n"
            f"Loop: {running}\n"
            f"Modo espejo: {mirror}\n"
            f"Analisis guardados: {history_count}\n"
            f"Ultimo simbolo detectado: {last}"
        )

    def get_last_analysis(self) -> Optional[VisionAnalysis]:
        """Retorna el último análisis o None."""
        if not self._analysis_history:
            return None
        return self._analysis_history[-1]

    def get_analysis_history(self, n: int = 10) -> list:
        """Últimas n análisis."""
        return self._analysis_history[-n:] if self._analysis_history else []
