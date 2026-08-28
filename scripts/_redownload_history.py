import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.historical_agent import HistoricalDataAgent


async def main():
    agent = HistoricalDataAgent()
    try:
        results = await agent.download_all(on_progress=print)
        print("\n=== RESULTADOS ===")
        for symbol, rows in results.items():
            print(f"{symbol}: {rows} filas")
    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(main())
