"""Discrete tau test of the candidate remedies on the time-of-day-corrected PCs: AR(1) at
3/6/12 h and AR(2)/AR(3) at 3 h, each against a 20-replicate bootstrap envelope. None
passes (`docs/model_design.md` §5.6). Needs `07_leading_pcs_by_utc_hour.py` first.

Run from the repo root: `poetry run python notebooks/probes/07_tau_remedies.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import numpy as np
SP = str(RESULTS)  # cache directory; was the session scratchpad
z_all = np.load(f"{SP}/wsh_pcs400.npy")
T = z_all.shape[0]
slot = ((np.arange(T) * 3) % 24) // 3
comp = np.stack([z_all[slot == s].mean(0) for s in range(8)])
z_all = z_all - comp[slot]
z_all -= z_all.mean(0)

def embed(z, p):
    """Rows t = p-1..T-1: [z_t, z_{t-1}, ..., z_{t-p+1}]."""
    return np.concatenate([z[p - 1 - j: len(z) - j] for j in range(p)], axis=1)

def fit(z, p, lag):
    x = embed(z, p)
    past, fut = x[:-lag], x[lag:]
    coef = np.linalg.lstsq(past, fut[:, : z.shape[1]], rcond=None)[0]  # predict current block only
    n, N = z.shape[1], x.shape[1]
    G = np.zeros((N, N)); G[:n] = coef.T
    if p > 1:  # shift register (only valid for lag == 1)
        G[n:, :-n] = np.eye(N - n)
    resid = fut[:, :n] - past @ coef
    S = np.zeros((N, N)); S[:n, :n] = resid.T @ resid / len(resid)
    return G, S

def ratios(z, p, lag, test_mult):
    G, S = fit(z, p, lag)
    x = embed(z, p); n = z.shape[1]
    out = []
    for m in test_mult:
        Gm = np.linalg.matrix_power(G, m)
        pred_cov = np.zeros_like(S); Gj = np.eye(len(G))
        for _ in range(m):
            pred_cov += Gj @ S @ Gj.T; Gj = G @ Gj
        err = x[m * lag:, :n] - (x[: -m * lag] @ Gm.T)[:, :n]
        out.append(np.mean(np.sum(err**2, 1)) / np.trace(pred_cov[:n, :n]))
    return np.array(out), G, S

def simulate(G, S, n, T, rng, burn=500):
    N = len(G); L = np.linalg.cholesky(S[:n, :n] + 1e-12 * np.eye(n))
    x = np.zeros(N); out = np.empty((T, n))
    for t in range(T + burn):
        x = G @ x; x[:n] += L @ rng.standard_normal(n)
        if t >= burn: out[t - burn] = x[:n]
    return out

def simulate_lagged(G, S, n, lag, T, rng):
    """AR(1) fitted at `lag` snapshots: simulate on the lag grid, interleave `lag` independent chains."""
    m = int(np.ceil(T / lag)); chains = [simulate(G, S, n, m, rng) for _ in range(lag)]
    out = np.empty((m * lag, n))
    for k in range(lag): out[k::lag] = chains[k]
    return out[:T]

rng = np.random.default_rng(0)
NREP = 20
for k in (20, 100):
    z = z_all[:, :k]
    print(f"===== top {k} PCs (hour-of-day mean removed), T={T}")
    variants = [("AR(1) tau0=3h", 1, 1, [1, 2, 4, 8, 16]),
                ("AR(1) tau0=6h", 1, 2, [1, 2, 4, 8]),
                ("AR(1) tau0=12h", 1, 4, [1, 2, 4]),
                ("AR(2) tau0=3h", 2, 1, [1, 2, 4, 8, 16]),
                ("AR(3) tau0=3h", 3, 1, [1, 2, 4, 8, 16])]
    for name, p, lag, mult in variants:
        if k == 100 and p == 3: continue
        r, G, S = ratios(z, p, lag, mult)
        reps = []
        for _ in range(NREP):
            zs = simulate(G, S, k, T, rng) if lag == 1 else simulate_lagged(G, S, k, lag, T, rng)
            reps.append(ratios(zs, p, lag, mult)[0])
        reps = np.array(reps)
        lags_h = [m * lag * 3 for m in mult]
        cells = "  ".join(f"{h:2d}h {v:.3f} [{lo:.3f},{hi:.3f}]{'*' if (v < lo or v > hi) else ' '}"
                          for h, v, lo, hi in zip(lags_h, r, reps.min(0), reps.max(0)))
        print(f"  {name:15s} {cells}")
