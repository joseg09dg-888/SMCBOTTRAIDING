from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd

from smc.structure import MarketStructure, StructureResult
from smc.orderblocks import OrderBlockDetector, FVGDetector
from smc.volume_profile import VolumeProfile, AnchoredVWAP
from core.config import config


@dataclass
class SMCAnalysis:
    symbol: str
    timeframe: str
    structure: StructureResult
    order_blocks: List[Dict]
    fvgs: List[Dict]
    volume_profile: Dict
    vwap: float
    bias: str
    poi_zones: List[Dict] = field(default_factory=list)
    entry_scenarios: List[str] = field(default_factory=list)
    risk_levels: Dict = field(default_factory=dict)


class SMCAnalysisAgent:
    """
    Combines SMC technical analysis with Claude API to generate
    TDA-format checklist analysis with context, POI zones, scenarios,
    and risk management — following the SMC methodology.
    """

    SYSTEM_PROMPT = """Eres un analista experto en Smart Money Concepts (SMC).
Tu metodología:
1. Contexto & Estructura: Acumulación, Distribución o Tendencia (HH, HL, LH, LL)
2. Liquidity Sweeps: dónde el Smart Money barre stops de minoristas
3. Order Blocks & FVG: zonas Oferta/Demanda y Fair Value Gaps
4. Volumen & VWAP: confirmar interés institucional con POC, VAH, VAL y VWAP

REGLAS:
- Sé preciso y directo. No uses párrafos largos.
- Usa checklists con ✅/❌.
- Recuerda siempre: "Si no hay setup claro, no se opera."
- Formato:
  📊 CONTEXTO DE MERCADO
  🎯 ZONAS DE INTERÉS (POI)
  📈 ESCENARIOS
  ⚠️ GESTIÓN DE RIESGO (Entrada | SL | TP)
  ✅ CHECKLIST DE ENTRADA"""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _run_technical_analysis(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> SMCAnalysis:
        ms = MarketStructure(df)
        structure = ms.analyze()

        ob_det = OrderBlockDetector(df)
        fvg_det = FVGDetector(df)
        bull_obs  = ob_det.find_bullish_obs()
        bear_obs  = ob_det.find_bearish_obs()
        bull_fvgs = fvg_det.find_bullish_fvg()
        bear_fvgs = fvg_det.find_bearish_fvg()

        vp = VolumeProfile(df)
        vol_profile = vp.calculate()

        anchor = max(0, len(df) - 50)
        av = AnchoredVWAP(df, anchor_index=anchor)
        vwap_series = av.calculate()
        current_vwap = float(vwap_series[-1]) if vwap_series else 0.0

        poi_zones: List[Dict] = []
        for ob in (bull_obs + bear_obs)[-3:]:
            poi_zones.append({
                "type": ob["type"],
                "high": ob["zone_high"],
                "low":  ob["zone_low"],
                "strength": ob.get("strength", 0),
            })
        for fvg in (bull_fvgs + bear_fvgs)[-3:]:
            poi_zones.append({
                "type":     fvg["type"],
                "high":     fvg["gap_high"],
                "low":      fvg["gap_low"],
                "midpoint": fvg["midpoint"],
            })

        return SMCAnalysis(
            symbol        = symbol,
            timeframe     = timeframe,
            structure     = structure,
            order_blocks  = bull_obs + bear_obs,
            fvgs          = bull_fvgs + bear_fvgs,
            volume_profile= vol_profile,
            vwap          = current_vwap,
            bias          = structure.bias,
            poi_zones     = poi_zones,
        )

    def _build_checklist(self, analysis: SMCAnalysis) -> str:
        s  = analysis.structure
        vp = analysis.volume_profile

        has_trend  = s.structure_type.value in ("bullish_trend", "bearish_trend")
        has_ob     = len(analysis.order_blocks) > 0
        has_fvg    = len(analysis.fvgs) > 0
        bias_clear = analysis.bias != "neutral"

        lines = [
            f"{'✅' if has_trend else '❌'} Estructura: {s.structure_type.value}",
            f"{'✅' if has_ob else '❌'} Order Blocks: {len(analysis.order_blocks)} zonas",
            f"{'✅' if has_fvg else '❌'} FVG: {len(analysis.fvgs)} gaps",
            f"{'✅' if bias_clear else '❌'} Bias: {analysis.bias.upper()}",
            f"✅ POC: {vp['poc']:.5f} | VAH: {vp['vah']:.5f} | VAL: {vp['val']:.5f}",
            f"✅ VWAP: {analysis.vwap:.5f}",
        ]
        score = sum([has_trend, has_ob, has_fvg, bias_clear])
        lines.append(
            f"\n{'🟢 SETUP VÁLIDO' if score >= 3 else '🔴 NO HAY SETUP — No se opera'} ({score}/4)"
        )
        return "\n".join(lines)

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        glint_context: Optional[str] = None,
    ) -> str:
        analysis = self._run_technical_analysis(df, symbol, timeframe)
        checklist = self._build_checklist(analysis)

        technical_summary = (
            f"Symbol: {symbol} | TF: {timeframe}\n"
            f"Estructura: {analysis.structure.structure_type.value}\n"
            f"HH:{analysis.structure.higher_highs} HL:{analysis.structure.higher_lows} "
            f"LH:{analysis.structure.lower_highs} LL:{analysis.structure.lower_lows}\n"
            f"Order Blocks: {len(analysis.order_blocks)} | FVGs: {len(analysis.fvgs)}\n"
            f"POC: {analysis.volume_profile['poc']} | VWAP: {analysis.vwap:.5f}\n"
            f"POI Zones: {analysis.poi_zones}\n"
            f"Checklist:\n{checklist}"
        )
        glint_section = (
            f"\nContexto Glint (tiempo real):\n{glint_context}" if glint_context else ""
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"{technical_summary}{glint_section}\n\nGenera el análisis SMC en formato TDA.",
            }],
        )
        return response.content[0].text
