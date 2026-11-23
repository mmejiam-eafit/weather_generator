"""Long-lag ACF of the leading PCs against AR(1), the variance share at periods over 3/10/30
days, and the tau test after high-passing: the slow-variability cause of the tau-test
failure (`docs/model_design.md` §5.6). Executes the top of `07_tau_remedies.py`.

Run from the repo root: `poetry run python notebooks/probes/07_slow_variability.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import numpy as np, sys
SP = str(RESULTS)  # cache directory; was the session scratchpad
exec(open(HERE / "07_tau_remedies.py").read().split("rng = np.random.default_rng(0)")[0])  # reuse fit/ratios/simulate + de-diurnal z_all

def acf(z, lags):
    z = z - z.mean(0); v = np.sum(z * z)
    return np.array([np.sum(z[l:] * z[:-l]) / v * len(z) / (len(z) - l) for l in lags])

lags = [1, 2, 4, 8, 16, 32, 64, 128, 256]
labels = ["3h", "6h", "12h", "1d", "2d", "4d", "8d", "16d", "32d"]
z20 = z_all[:, :20]
a = acf(z20, lags)
print("top-20 PCs, energy-weighted ACF:  " + "  ".join(f"{l}:{v:.3f}" for l, v in zip(labels, a)))
print("AR(1)-implied acf(3h)^k:          " + "  ".join(f"{l}:{a[0]**k:.3f}" for l, k in zip(labels, lags)))
# spectral share of slow variability
F = np.fft.rfft(z20, axis=0); P = np.sum(np.abs(F)**2, 1); f = np.fft.rfftfreq(T, d=3/24)  # cycles/day
for cut in (1/30, 1/10, 1/3):
    print(f"  share of top-20 variance at periods > {1/cut:4.0f} d: {P[(f > 0) & (f < cut)].sum() / P[f > 0].sum():.3f}")

def highpass(z, days):
    F = np.fft.rfft(z, axis=0); f = np.fft.rfftfreq(len(z), d=3/24); F[f < 1/days] = 0
    return np.fft.irfft(F, n=len(z), axis=0)

rng = np.random.default_rng(1)
for k in (20, 100):
    for days in (None, 30, 10, 3):
        z = z_all[:, :k] if days is None else highpass(z_all[:, :k], days)
        mult = [1, 2, 4, 8, 16]
        r, G, S = ratios(z, 1, 1, mult)
        reps = np.array([ratios(simulate(G, S, k, T, rng), 1, 1, mult)[0] for _ in range(20)])
        cells = "  ".join(f"{m*3:2d}h {v:.3f} [{lo:.3f},{hi:.3f}]{'*' if (v < lo or v > hi) else ' '}"
                          for m, v, lo, hi in zip(mult, r, reps.min(0), reps.max(0)))
        tag = "no filter" if days is None else f"high-pass {days}d"
        print(f"  top {k:3d} AR(1) 3h, {tag:13s} {cells}")
