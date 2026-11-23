"""Time-of-day mean's share of each leading PC (PC1 86%, PC2 71%; top-20 15% against a 0.2%
null), periodogram lines at 24/12/8/6 h, and the tau test with and without that mean
(`docs/open_questions.md` §7.2, `docs/correlated_variables.md` §4.4). Needs
`07_leading_pcs_by_utc_hour.py` first.

Run from the repo root: `poetry run python notebooks/probes/07_time_of_day_share.py`.

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
from train.linear_inverse import tau_test
SP = str(RESULTS)  # cache directory; was the session scratchpad
z = np.load(f"{SP}/wsh_pcs400.npy")
T = z.shape[0]
hours = (np.arange(T) * 3) % 24
slot = hours // 3
composite = np.stack([z[slot == s].mean(axis=0) for s in range(8)])  # (8, n)
det = composite[slot]
r2 = np.sum(det**2, axis=0) / np.sum((z - z.mean(0))**2, axis=0)
print("hour-of-day mean: fraction of each PC's variance (PC1..PC12):")
print(np.round(r2[:12], 3))
for k in (5, 20, 100):
    print(f"  top {k}: hour-of-day mean explains {np.sum(det[:, :k]**2)/np.sum(z[:, :k]**2):.3f} of their energy")
# significance: same statistic with slot labels circularly shifted by a random whole number of days?? -> use label permutation by day blocks
rng = np.random.default_rng(0)
null = []
days = T // 8
for _ in range(200):
    perm_slot = np.concatenate([rng.permutation(8) for _ in range(days)])
    comp = np.stack([z[perm_slot == s, :20].mean(axis=0) for s in range(8)])
    null.append(np.sum(comp[perm_slot]**2) / np.sum(z[:, :20]**2))
print(f"  null (within-day shuffled hour labels), top 20: max {max(null):.4f}, mean {np.mean(null):.4f}")
# spectrum peak check on top-20 energy: periodogram at 24h, 12h, 8h
f = np.fft.rfftfreq(T, d=3.0)  # cycles per hour
P = np.sum(np.abs(np.fft.rfft(z[:, :20] - z[:, :20].mean(0), axis=0))**2, axis=1)
for period in (24.0, 12.0, 8.0, 6.0):
    i = np.argmin(np.abs(f - 1 / period))
    nb = np.r_[P[i-12:i-2], P[i+3:i+13]]
    print(f"  period {period:4.0f} h: periodogram {P[i]:.3e}, neighbours median {np.median(nb):.3e}, ratio {P[i]/np.median(nb):.1f}")
dt = 3 * 3600.0
lags = [1, 2, 4, 8, 16]
for k in (20, 100):
    a = tau_test(z[:, :k], 1, lags, dt)
    b = tau_test(z[:, :k] - det[:, :k], 1, lags, dt)
    print(f"top {k}: ratio raw      {np.round(a.error_variance_ratio, 3)}  drift {np.round(a.operator_drift, 3)}")
    print(f"top {k}: ratio de-diurnal {np.round(b.error_variance_ratio, 3)}  drift {np.round(b.operator_drift, 3)}")
