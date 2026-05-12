from dataclasses import dataclass, field
from typing import List
from enum import Enum


class HitoStatus(Enum):
    PENDIENTE  = "⏳"
    EN_PROCESO = "🔄"
    COMPLETADO = "✅"
    FALLADO    = "❌"


@dataclass
class Hito:
    nombre: str
    descripcion: str
    criterio: str
    status: HitoStatus = HitoStatus.PENDIENTE
    progreso: float = 0.0


TOP_10_TRADERS_SMC = [
    "George Soros — Macro global, rupturas de estructura HTF",
    "Paul Tudor Jones — Gestión de riesgo, momentum SMC",
    "Jesse Livermore — Acción del precio, liquidez",
    "Ray Dalio — Correlaciones macro, portafolio",
    "Stanley Druckenmiller — Macro + timing de entrada",
    "Ed Seykota — Sistemas cuantitativos, tendencias",
    "Larry Williams — Volume Profile, futuros",
    "Linda Bradford Raschke — Scalping con SMC, momentum",
    "Cathie Wood — Análisis sectorial crypto/tech",
    "Michael Marcus — Risk/Reward, paciencia",
]

LIBROS_OBLIGATORIOS = [
    "Trading in the Zone — Mark Douglas",
    "Market Wizards — Jack Schwager",
    "The Disciplined Trader — Mark Douglas",
    "Technical Analysis of Financial Markets — John Murphy",
    "Smart Money Concepts — ICT Mentorship (PDFs)",
]

CURRICULUM: List[Hito] = [
    Hito(
        "Leer 5 libros SMC",
        "Procesar PDFs con análisis por capítulo",
        "5 libros procesados con resumen de estrategias",
    ),
    Hito(
        "Estudiar Top 10 Traders",
        "Analizar estrategias de los 10 mejores traders",
        "10 perfiles completos en knowledge base",
    ),
    Hito(
        "500 trades en demo",
        "Operar en Binance Testnet y MT5 Demo",
        "500 trades con win rate > 55%",
    ),
    Hito(
        "Backtest 6 meses",
        "Backtesting con datos históricos de todos los mercados",
        "Profit factor > 1.5 en backtest",
    ),
    Hito(
        "Gestión de riesgo aprobada",
        "Nunca romper reglas de riesgo",
        "0 violaciones de stop loss en demo",
    ),
    Hito(
        "Integración Glint",
        "Leer señales en tiempo real y correlacionar con SMC",
        "50 señales procesadas y correlacionadas",
    ),
]


def print_curriculum_status():
    completados = sum(1 for h in CURRICULUM if h.status == HitoStatus.COMPLETADO)
    total = len(CURRICULUM)
    print(f"\nCURRICULUM DE ENTRENAMIENTO ({completados}/{total} completados)\n")
    for h in CURRICULUM:
        status_str = h.status.name
        print(f"[{status_str}] {h.nombre}")
        print(f"   Criterio: {h.criterio}")
        print(f"   Progreso: {h.progreso*100:.0f}%\n")
    if completados == total:
        print("ENTRENAMIENTO COMPLETO. Bot listo para operar en real.")
    else:
        print(f"Completar {total - completados} hitos mas antes de operar en real.")


if __name__ == "__main__":
    print_curriculum_status()
