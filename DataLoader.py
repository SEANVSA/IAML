import yfinance as yf
import numpy as np
# Daftar aset: mix saham & crypto
##tickers = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "TRX-USD", "DOGE-USD", "ADA-USD", "BCH-USD", "XMR-USD", "LINK-USD", "XLM-USD", "ZEC-USD", "LTC-USD", "AVAX-USD","HBAR-USD", "SHIB-USD", "DOT-USD", "AAVE-USD", "NEAR-USD", "ALGO-USD", "ICP-USD", "FET-USD", "TON-USD", "PEP-USD"]
tickers = [
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "TRX-USD", "DOGE-USD", "ADA-USD", "BCH-USD", "XMR-USD", "LINK-USD", "XLM-USD", "ZEC-USD", "LTC-USD", "AVAX-USD",
    "HBAR-USD", "SHIB-USD", "DOT-USD", "AAVE-USD", "NEAR-USD", "ALGO-USD", "ICP-USD", "FET-USD", "TON-USD"
    ]
# Unduh harga penutupan 2 tahun terakhir
data = yf.download(tickers, period='3y', interval= "1mo")['Close']
data.dropna(inplace=True)

returns = (((data - data.shift(1))/ data.shift(1))*100).dropna()
mu = returns.mean()
cov = returns.cov()
print(cov)