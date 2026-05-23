FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Encoding fix — no garbled chars in logs
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
# MT5 not available on Linux — bot uses yfinance for forex
ENV MT5_DEMO=true
ENV BINANCE_TESTNET=true

CMD ["python", "-u", "startup.py", "--auto", "--capital", "1000"]
