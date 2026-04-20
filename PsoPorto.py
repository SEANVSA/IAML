import numpy as np
import DataLoader
def sharpe_ratio(weights, mu, cov, rf=0.05):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(weights @ cov @ weights)
    return (port_return - rf) / port_vol
def normalize(x):
    x = np.clip(x, 0, None) # hapus bobot negatif
    return x / x.sum() # normalisasi → sum = 1
def pso_portfolio(mu, cov, rf=0.05, n_particles=100, n_iter=300):
    n = len(mu)
    # Hyperparameters PSO
    w_max, w_min = 0.9, 0.4  # inertia weight (linear decay)
    c1 = c2 = 2.0  # cognitive & social coefficients
    # Inisialisasi populasi
    pos = np.random.dirichlet(np.ones(n), n_particles)  # sum=1 dari awal
    vel = np.random.uniform(-0.1, 0.1, (n_particles, n))
    # Evaluasi awal
    fitness = np.array([sharpe_ratio(p, mu, cov, rf) for p in pos])
    pBest = pos.copy()
    pBest_f = fitness.copy()
    gBest = pBest[np.argmax(pBest_f)]
    gBest_f = pBest_f.max()
    history = []
    for t in range(n_iter):
        # Inertia weight decay
        omega = w_max - (w_max - w_min) * t / n_iter
        r1, r2 = np.random.rand(2)

        # Update velocity
        vel = (omega * vel + c1 * r1 * (pBest - pos) + c2 * r2 * (gBest - pos))

        # Update posisi & normalisasi
        pos = np.array([normalize(pos[k] + vel[k]) for k in range(n_particles)])

        # Evaluasi fitness
        fitness = np.array([sharpe_ratio(p, mu, cov, rf) for p in pos])

        # Update pBest & gBest
        improved = fitness > pBest_f
        pBest[improved] = pos[improved]
        pBest_f[improved] = fitness[improved]
        if fitness.max() > gBest_f:
            gBest_f = fitness.max()
        gBest = pos[np.argmax(fitness)].copy()
        history.append(gBest_f)

    return gBest, gBest_f, history
# ── Jalankan ──────────────────────────────────────

best_w, best_sharpe, conv = pso_portfolio(DataLoader.mu, DataLoader.cov, DataLoader.rf)
for asset, w in zip(DataLoader.tickers, best_w):
    print(f"{asset}: {w:.1%}")
print(f"Sharpe Ratio: {best_sharpe:.4f}")