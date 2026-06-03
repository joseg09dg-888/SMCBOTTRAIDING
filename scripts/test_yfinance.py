"""Test yfinance forex data."""
import yfinance as yf
pairs = ['EURUSD=X', 'GBPUSD=X', 'XAUUSD=X', 'USDJPY=X']
for p in pairs:
    try:
        t = yf.Ticker(p)
        df = t.history(period='2d', interval='1h')
        if not df.empty:
            print(f'{p}: close={df["Close"].iloc[-1]:.5f} bars={len(df)}')
        else:
            print(f'{p}: no data')
    except Exception as e:
        print(f'{p}: error {e}')
