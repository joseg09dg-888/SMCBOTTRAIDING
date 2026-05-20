"""PASO 1: Verificar datos reales de Binance."""
import os, sys
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()
from binance.client import Client
from datetime import datetime

client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)
info = client.get_account()
balances = [b for b in info['balances'] if float(b['free']) > 0]
print('Balances reales:')
for b in balances[:8]:
    print(f"  {b['asset']}: {b['free']}")

orders = client.get_open_orders()
print(f'Ordenes abiertas: {len(orders)}')

btc_price = float(client.get_symbol_ticker(symbol='BTCUSDT')['price'])
print(f'BTC precio actual: ${btc_price:,.2f}')

today_ms = int(datetime.now().replace(hour=0,minute=0,second=0,microsecond=0).timestamp()*1000)
total_trades = 0
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT']:
    try:
        trades = client.get_my_trades(symbol=sym, startTime=today_ms, limit=50)
        total_trades += len(trades)
        if trades: print(f'  {sym}: {len(trades)} trades hoy')
    except: pass
print(f'Total trades hoy: {total_trades}')
