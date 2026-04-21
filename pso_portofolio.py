import numpy as np
import yfinance as yf

top25 = [
        {"name": "Bitcoin",        "symbol": "BTC",   "ticker": "BTC-USD"},
        {"name": "Ethereum",       "symbol": "ETH",   "ticker": "ETH-USD"},
        {"name": "Tether",         "symbol": "USDT",  "ticker": "USDT-USD"},
        {"name": "BNB",            "symbol": "BNB",   "ticker": "BNB-USD"},
        {"name": "XRP",            "symbol": "XRP",   "ticker": "XRP-USD"},
        {"name": "USD Coin",       "symbol": "USDC",  "ticker": "USDC-USD"},
        {"name": "Solana",         "symbol": "SOL",   "ticker": "SOL-USD"},
        {"name": "Dogecoin",       "symbol": "DOGE",  "ticker": "DOGE-USD"},
        {"name": "TRON",           "symbol": "TRX",   "ticker": "TRX-USD"},
        {"name": "Cardano",        "symbol": "ADA",   "ticker": "ADA-USD"},
        {"name": "Avalanche",      "symbol": "AVAX",  "ticker": "AVAX-USD"},
        {"name": "Shiba Inu",      "symbol": "SHIB",  "ticker": "SHIB-USD"},
        {"name": "Chainlink",      "symbol": "LINK",  "ticker": "LINK-USD"},
        {"name": "Polkadot",       "symbol": "DOT",   "ticker": "DOT-USD"},
        {"name": "Bitcoin Cash",   "symbol": "BCH",   "ticker": "BCH-USD"},
        {"name": "NEAR Protocol",  "symbol": "NEAR",  "ticker": "NEAR-USD"},
        {"name": "Litecoin",       "symbol": "LTC",   "ticker": "LTC-USD"},
        {"name": "Toncoin",        "symbol": "TON",   "ticker": "TON-USD"},
        {"name": "Dai",            "symbol": "DAI",   "ticker": "DAI-USD"},
        {"name": "Internet Computer", "symbol": "ICP", "ticker": "ICP-USD"},
        {"name": "Ethereum Classic", "symbol": "ETC", "ticker": "ETC-USD"},
        {"name": "Fetch",          "symbol": "FET",   "ticker": "FET-USD"},
        {"name": "Stellar",        "symbol": "XLM",   "ticker": "XLM-USD"},
        {"name": "Monero",         "symbol": "XMR",   "ticker": "XMR-USD"},
        {"name": "Filecoin",       "symbol": "FIL",   "ticker": "FIL-USD"},
]
# ─────────────────────────────────────────────
# 1. DOWNLOAD HISTORICAL DATA VIA YFINANCE
# ─────────────────────────────────────────────

def load_returns(tickers: list[str], period: str = "3y") -> tuple[np.ndarray, np.ndarray, list[str]]:

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    print(raw)

    threshold = int(len(raw) * 0.80)
    raw = raw.dropna(axis=1, thresh=threshold).dropna()

    valid_tickers = list(raw.columns)

    # Daily log return: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(raw / raw.shift(1)).dropna()

    # Annualisasi (252 hari trading)
    mu  = log_returns.mean().values * 252          # shape: (n,)
    cov = log_returns.cov().values  * 252          # shape: (n, n)

    return mu, cov, valid_tickers


# ─────────────────────────────────────────────
# 2. FUNGSI OBJEKTIF CUSTOM — FULL VECTOR
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 3. NORMALISASI BOBOT (CONSTRAINT HANDLING)
# ─────────────────────────────────────────────

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Pastikan semua bobot ≥ 0 dan Σwᵢ = 1.
        wᵢ' = max(0, wᵢ) / Σ max(0, wⱼ)
    """
    x = np.maximum(x, 0.0)        # clip negatif → 0
    total = x.sum()
    if total < 1e-10:
        # fallback: distribusi merata
        x = np.ones_like(x)
        total = x.sum()
    return x / total


# ─────────────────────────────────────────────
# 4. PSO OPTIMIZER
# ─────────────────────────────────────────────

def pso_portfolio(mu: np.ndarray, cov: np.ndarray,
                  rf: float       = 0.05,
                  lam: float      = 0.5,
                  n_particles: int = 100,
                  n_iter: int      = 300,
                  w_max: float     = 0.9,
                  w_min: float     = 0.4,
                  c1: float        = 2.0,
                  c2: float        = 2.0,
                  seed: int        = 42,
                  verbose: bool    = True) -> tuple[np.ndarray, float, list[float]]:
    """
    PSO untuk optimasi portofolio.

    Update velocity (vektor):
        vᵢ(t+1) = ω·vᵢ(t) + c₁·r₁·(pBestᵢ − xᵢ(t)) + c₂·r₂·(gBest − xᵢ(t))

    Update posisi:
        xᵢ(t+1) = xᵢ(t) + vᵢ(t+1)

    Setelah update, normalisasi posisi agar memenuhi constraint.

    Args:
        verbose : True  → print progress tiap iterasi
                  False → silent, hanya cetak ringkasan akhir

    Returns:
        best_weights  : vektor bobot optimal
        best_score    : nilai objektif terbaik
        history       : list score per iterasi (convergence curve)
    """
    np.random.seed(seed)
    n = len(mu)

    # ── Inisialisasi populasi ──────────────────────────────────
    # Dirichlet distribution → langsung sum=1 dari awal
    pos = np.random.dirichlet(np.ones(n), n_particles)           # (P, n)

    vel = np.random.uniform(-0.1, 0.1, (n_particles, n))         # (P, n)
    print(pos[0])
    print(pos[1])
    print(pos[2])
    # ── Evaluasi fitness awal ──────────────────────────────────
    fitness   = np.array([portfolio_score(pos[k], mu, cov, rf, lam) for k in range(n_particles)])           # (P,)
    print(len(fitness))
    pBest     = pos.copy()                                        # (P, n)
    pBest_f   = fitness.copy()                                    # (P,)

    best_idx  = np.argmax(pBest_f)
    gBest     = pBest[best_idx].copy()                            # (n,)
    gBest_f   = pBest_f[best_idx]

    history   = [gBest_f]

    # ── Iterasi PSO ────────────────────────────────────────────
    for t in range(n_iter):

        # Inertia weight: linear decay ω dari w_max → w_min
        omega = w_max - (w_max - w_min) * t / n_iter

        # Komponen stokastik (scalar per iterasi — broadcast ke seluruh swarm)
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Update velocity (operasi vektor broadcast):
        #   vᵢ(t+1) = ω·vᵢ + c₁·r₁·(pBestᵢ - xᵢ) + c₂·r₂·(gBest - xᵢ)
        cognitive = c1 * r1 * (pBest - pos)                      # (P, n)
        social    = c2 * r2 * (gBest  - pos)                     # (P, n)
        vel       = omega * vel + cognitive + social              # (P, n)

        # Update posisi + normalisasi constraint
        pos = pos + vel                                           # (P, n)
        pos = np.array([normalize(pos[k]) for k in range(n_particles)])

        # Evaluasi fitness
        fitness = np.array([portfolio_score(pos[k], mu, cov, rf, lam)
                            for k in range(n_particles)])

        # Update pBest (per partikel)
        improved        = fitness > pBest_f                       # (P,) bool mask
        pBest[improved] = pos[improved]
        pBest_f[improved] = fitness[improved]

        # Update gBest
        max_idx = np.argmax(fitness)
        if fitness[max_idx] > gBest_f:
            gBest_f = fitness[max_idx]
            gBest   = pos[max_idx].copy()

        history.append(gBest_f)

        if verbose:
            print(f"  Iter {t+1:>4}/{n_iter}  |  ω={omega:.4f}  |  gBest Score={gBest_f:.6f}")

    if verbose:
        print(f"  {'─'*50}")
        print(f"  PSO selesai. Score terbaik: {gBest_f:.6f}")

    return gBest, gBest_f, history


# ─────────────────────────────────────────────
# 5. HASIL & ANALISIS
# ─────────────────────────────────────────────

def portfolio_metrics(w: np.ndarray, mu: np.ndarray,
                      cov: np.ndarray, rf: float = 0.05) -> dict:
    """Hitung metrik portofolio dari vektor bobot optimal."""
    port_return = w @ mu
    port_var    = w @ cov @ w
    port_vol    = np.sqrt(port_var)
    sharpe      = (port_return - rf) / port_vol
    hhi         = w @ (w ** 2)          # concentration index

    return {
        "expected_return": port_return,
        "volatility":      port_vol,
        "sharpe_ratio":    sharpe,
        "hhi_concentration": hhi,
        "custom_score":    portfolio_score(w, mu, cov, rf)
    }


def print_results(tickers: list[str], weights: np.ndarray,
                  metrics: dict, score: float) -> None:
    sep = "─" * 46

    print(f"\n{'═'*46}")
    print(f"  PSO PORTFOLIO OPTIMIZATION — HASIL AKHIR")
    print(f"{'═'*46}")

    print(f"\n{'ALOKASI BOBOT OPTIMAL':}")
    print(sep)
    sorted_idx = np.argsort(weights)[::-1]
    for i in sorted_idx:
        bar = "█" * int(weights[i] * 40)
        print(f"  {tickers[i]:<12}  {weights[i]:>6.2%}  {bar}")

    print(f"\n{sep}")
    print(f"  {'METRIK PORTOFOLIO'}")
    print(sep)
    print(f"  Expected Return (ann.)  : {metrics['expected_return']:>8.2%}")
    print(f"  Portfolio Volatility    : {metrics['volatility']:>8.2%}")
    print(f"  Sharpe Ratio            : {metrics['sharpe_ratio']:>8.4f}")
    print(f"  HHI Concentration       : {metrics['hhi_concentration']:>8.4f}")
    print(f"  Custom Objective Score  : {score:>8.4f}")
    print(f"{'═'*46}\n")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    RF          = 0.00   # risk-free rate 0% for crypto
    LAMBDA      = 0.5    # diversification penalty weight
    VERBOSE     = True   # True → print tiap iterasi | False → silent

    # 1. Ambil data 25 coin
    print("[ 1/4 ] Mengambil data 25 coin")
    coins   = top25
    tickers = [c["ticker"] for c in coins]
    names   = {c["ticker"]: c["name"] for c in coins}
    print(f"        Ditemukan: {', '.join([c['symbol'] for c in coins])}")

    # 2. Download historical data via yfinance
    print("[ 2/4 ] Mengunduh data historis 2 tahun (yfinance)...")
    mu, cov, valid_tickers = load_returns(tickers, period="3y")
    valid_names = [names.get(t, t) for t in valid_tickers]
    print(f"        Aset valid: {len(valid_tickers)} dari {len(tickers)}")
    print(f"        Tickers   : {', '.join(valid_tickers)}")

    # 3. Jalankan PSO
    print("[ 3/4 ] Menjalankan PSO (100 partikel × 300 iterasi)...")
    best_w, best_score, history = pso_portfolio(
        mu, cov,
        rf          = RF,
        lam         = LAMBDA,
        n_particles = 100,
        n_iter      = 300,
        verbose     = VERBOSE
    )
    print(f"        Konvergen pada score: {best_score:.4f}")

    # 4. Tampilkan hasil
    print("[ 4/4 ] Menghitung metrik portofolio...\n")
    metrics = portfolio_metrics(best_w, mu, cov, rf=RF)
    print_results(valid_tickers, best_w, metrics, best_score)