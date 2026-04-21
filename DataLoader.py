import yfinance as yf
import numpy as np
import pandas as pd
# Daftar aset: mix saham & crypto
##tickers = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "TRX-USD", "DOGE-USD", "ADA-USD", "BCH-USD", "XMR-USD", "LINK-USD", "XLM-USD", "ZEC-USD", "LTC-USD", "AVAX-USD","HBAR-USD", "SHIB-USD", "DOT-USD", "AAVE-USD", "NEAR-USD", "ALGO-USD", "ICP-USD", "FET-USD", "TON-USD", "PEP-USD"]
top25 = [
        {"name": "Bitcoin",        "symbol": "BTC",   "ticker": "BTC-USD"},
        #{"name": "Ethereum",       "symbol": "ETH",   "ticker": "ETH-USD"},
        #{"name": "Tether",         "symbol": "USDT",  "ticker": "USDT-USD"},
        #{"name": "BNB",            "symbol": "BNB",   "ticker": "BNB-USD"},
        {"name": "XRP",            "symbol": "XRP",   "ticker": "XRP-USD"},
        #{"name": "USD Coin",       "symbol": "USDC",  "ticker": "USDC-USD"},
        {"name": "Solana",         "symbol": "SOL",   "ticker": "SOL-USD"},
        #{"name": "Dogecoin",       "symbol": "DOGE",  "ticker": "DOGE-USD"},
        #{"name": "TRON",           "symbol": "TRX",   "ticker": "TRX-USD"},
        #{"name": "Cardano",        "symbol": "ADA",   "ticker": "ADA-USD"},
        #{"name": "Avalanche",      "symbol": "AVAX",  "ticker": "AVAX-USD"},
        #{"name": "Shiba Inu",      "symbol": "SHIB",  "ticker": "SHIB-USD"},
        #{"name": "Chainlink",      "symbol": "LINK",  "ticker": "LINK-USD"},
        #{"name": "Polkadot",       "symbol": "DOT",   "ticker": "DOT-USD"},
        #{"name": "Bitcoin Cash",   "symbol": "BCH",   "ticker": "BCH-USD"},
        #{"name": "NEAR Protocol",  "symbol": "NEAR",  "ticker": "NEAR-USD"},
        #{"name": "Litecoin",       "symbol": "LTC",   "ticker": "LTC-USD"},
        #{"name": "Toncoin",        "symbol": "TON",   "ticker": "TON-USD"},
        {"name": "Dai",            "symbol": "DAI",   "ticker": "DAI-USD"},
        #{"name": "Internet Computer", "symbol": "ICP", "ticker": "ICP-USD"},
        #{"name": "Ethereum Classic", "symbol": "ETC", "ticker": "ETC-USD"},
        #{"name": "Fetch",          "symbol": "FET",   "ticker": "FET-USD"},
        #{"name": "Stellar",        "symbol": "XLM",   "ticker": "XLM-USD"},
        #{"name": "Monero",         "symbol": "XMR",   "ticker": "XMR-USD"},
        #{"name": "Filecoin",       "symbol": "FIL",   "ticker": "FIL-USD"},
]

tickers = [c["ticker"] for c in top25]
# Unduh harga penutupan 2 tahun terakhir
data = yf.download(tickers, period='3y', interval= "1d")['Close']
data.dropna(inplace=True)

returns = np.log(data / data.shift(1)).dropna()
rf = 0.05 # Risk-free rate 5% p.a.


# ... [Kode unduh data Anda sebelumnya ada di sini] ...

# KOREKSI: Karena interval='1mo' (Bulanan), dalam setahun ada 12 bulan
annual_factor = 252

# Ekspektasi Return dan Covariance yang disetahunkan
mu = returns.mean() * annual_factor
cov = returns.cov() * annual_factor

# ==========================================
# CARA MENGHITUNG VOLATILITAS PER CRYPTO
# ==========================================

# Volatilitas adalah Standar Deviasi.
# Untuk disetahunkan, kalikan dengan akar kuadrat (np.sqrt) dari annual_factor
volatility = returns.std() * np.sqrt(annual_factor)

# Opsional: Mengubahnya menjadi DataFrame agar rapi dan format persentase (%)
volatility_df = pd.DataFrame({
    'Return Tahunan (%)': mu * 100,
    'Volatilitas/Risiko (%)': volatility * 100
})

# Mari kita urutkan dari koin yang paling berisiko (volatilitas tertinggi)
volatility_df = volatility_df.sort_values(by='Volatilitas/Risiko (%)', ascending=False)

print("Tabel Return vs Volatilitas Kripto (Setahun):")
print(volatility_df)

def portfolio_score(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                    rf: float = 0.05, lam: float = 0.5) -> float:
    """
    Fungsi objektif custom berbasis implementasi vektor:

        f(w) = (wᵀμ - rf) / √(wᵀΣw)  -  λ · wᵀh

    Di mana:
        wᵀμ        = expected return portofolio (dot product)
        √(wᵀΣw)    = volatilitas portofolio (quadratic form)
        h          = vektor konsentrasi HHI = w² (element-wise)
        wᵀh        = wᵀ(w²) = Σ wᵢ³  → penalty portofolio terkonsentrasi
        λ          = bobot penalti diversifikasi

    Semua operasi menggunakan numpy vector/matrix operations.
    """
    # Return portofolio: skalar dari dot product
    port_return = w @ mu                            # wᵀμ

    # Volatilitas portofolio: quadratic form
    port_var    = w @ cov @ w                       # wᵀΣw
    port_vol    = np.sqrt(np.maximum(port_var, 1e-10))

    # Diversification penalty: HHI concentration vector
    h           = w ** 2                            # vektor konsentrasi
    concentration = w @ h                          # wᵀh = Σ wᵢ³

    score = (port_return - rf) / port_vol  -  lam * concentration
    return score
#BTC XRP SOL DAI
pos = np.array([0.30,0.07,0.03,0.60])
print(portfolio_score(pos, mu, cov, rf, 0.5))