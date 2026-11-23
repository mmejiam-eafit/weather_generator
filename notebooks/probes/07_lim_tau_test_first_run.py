"""First tau test on the production wsh 40% Gaussianized core, with 5-95% bootstrap
percentile bands (before the stage switched to min-max envelopes).

Superseded by `notebooks/07_lim_tau_test_wsh_core.py`, which runs the pipeline stage.
Reads task 07's cached production core under `results/task07/gaussianize/`. Ran without a memory cap; it loads the 2.5 GB core.

Run from the repo root: `poetry run python notebooks/probes/07_lim_tau_test_first_run.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import sys
import time

import numpy as np


from train.linear_inverse import fit_lim, group_pcs, simulate_lim, tau_test  # noqa: E402

CORE = RESULTS / "wsh_r248x576_gaussianized_core.npy"
DT = 3 * 3600.0
LAGS = [1, 2, 4, 8, 16]
N_BOOT = 20

t0 = time.time()
core = np.load(CORE, mmap_mode="r")
for rank in (20, 100):
    group = group_pcs({"wsh": np.asarray(core)}, ranks={"wsh": rank})
    z = group.pcs
    fit = fit_lim(z, lag=1, dt=DT)
    real = tau_test(z, training_lag=1, test_lags=LAGS, dt=DT)
    rng = np.random.default_rng(0)
    boot_drift, boot_ratio = [], []
    for _ in range(N_BOOT):
        sim = simulate_lim(fit, z[0], z.shape[0], rng)
        b = tau_test(sim, training_lag=1, test_lags=LAGS, dt=DT)
        boot_drift.append(b.operator_drift)
        boot_ratio.append(b.error_variance_ratio)
    bd, br = np.array(boot_drift), np.array(boot_ratio)
    print(f"\n=== r={rank} PCs: explained variance {group.explained_variance['wsh']:.3f}, "
          f"Q min eig {fit.q_min_eigenvalue:.3e}, logm imag frac {fit.logm_imag_fraction:.2e}")
    print("lag  drift(real)  drift(boot 5-95%)      ratio(real)  ratio(boot 5-95%)")
    for i, lag in enumerate(LAGS):
        print(f"{lag:>3}  {real.operator_drift[i]:>10.3f}   "
              f"[{np.percentile(bd[:, i], 5):.3f}, {np.percentile(bd[:, i], 95):.3f}]"
              f"        {real.error_variance_ratio[i]:>8.3f}   "
              f"[{np.percentile(br[:, i], 5):.3f}, {np.percentile(br[:, i], 95):.3f}]")
print(f"\nelapsed {time.time() - t0:.0f}s")
