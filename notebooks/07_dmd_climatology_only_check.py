"""Task 07: sanity check for the "does removing climatology starve DMD of
the low-rank structure it's good at" hypothesis raised while interpreting
`07_dmd_rank_sweep.py`'s results.

Mechanism under test: `ClimatologyRemover` fits, per cell, a linear
combination of exactly 6 deterministic time-basis functions for `wsh`
(intercept, linear trend, annual cos/sin, semi-annual cos/sin -- `wsh` is
not a diurnal variable, see `_DEFAULT_DIURNAL_VARIABLES`). Regardless of
how many spatial cells or Tucker/DMD modes are involved, that signal's
*time* dependence lives entirely inside a 6-dimensional subspace by
construction. So if the "anomaly is hard because climatology was removed"
story is right, projecting the climatology signal ALONE through the same
Tucker factors and checking its own singular-value spectrum should show
essentially 100% of its variance captured by a handful of modes (~6-8) --
in sharp contrast to the real (anomaly) core, which needed ~1287 of 1754
possible modes for 90% (`07_dmd_rank_sweep.py`'s companion finding).

This is a confirmatory check on the MECHANISM only, not a proposal to
change the pipeline -- see the discussion this script's result is meant to
inform in `docs/open_questions.md` Sec 5.3 (added once results are in hand).

Run directly: `poetry run python notebooks/07_dmd_climatology_only_check.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import scipy.linalg
from tensorly.tenalg import multi_mode_dot

from train.climatology import _days_since_epoch, _harmonic_design_matrix
from train.dmd import dmd_fit

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"
N_TIMES = 2192  # 2024-01-01 -> 2024-10-01 at 3h cadence, era 2 (274 days * 8/day)
DT_SECONDS = 3.0 * 3600.0


def main() -> None:
    clim = joblib.load(RESULTS_DIR / "wsh_climatology.joblib")
    npz = np.load(RESULTS_DIR / "anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz")
    factor_lat = npz["factor_lat"]  # (621, 124)
    factor_lon = npz["factor_lon"]  # (1440, 288)
    r_lat, r_lon = factor_lat.shape[1], factor_lon.shape[1]

    times = np.datetime64("2024-01-01T00:00:00") + np.arange(N_TIMES) * np.timedelta64(3, "h")
    print(f"time range: {times[0]} -> {times[-1]}, T={N_TIMES}")

    # Climatology is, by construction, a linear combination of just 6
    # deterministic time-basis functions per cell (intercept, trend,
    # annual/semi-annual cos-sin -- wsh is not diurnal). Exploit that
    # directly instead of materializing a (T, 621, 1440) physical-grid
    # array (~15.7GB, ~31GB+ transient inside transform() -- OOM-killed
    # the first version of this script, exit 137): project the small
    # (n_terms, 621, 1440) coefficient array through the Tucker factors
    # FIRST, then expand over time only at the already-small (124, 288)
    # core resolution.
    t_days = _days_since_epoch(times, clim.t_train_start_)
    delta_tau_years = clim._delta_tau_years(times)
    design = _harmonic_design_matrix(t_days, include_diurnal=False, delta_tau_years=delta_tau_years)  # (T, n_terms)

    coeffs = clim.coeffs_["wsh"]  # (n_terms, 621, 1440)
    # Land cells get NaN coefficients by construction (fit on all-NaN-at-
    # land data -- see ClimatologyRemover.fit's docstring). The real
    # pipeline never hits this directly: land ends up zero (not NaN) in
    # z_full right before Tucker fitting (07_placement2_tucker_on_
    # anomalies.py's final np.where(isnan, 0.0, z_full)), so the Tucker
    # factors themselves were fit treating land as a zero contribution.
    # Projecting the raw coefficients here needs the same zero-fill first,
    # or NaN propagates through multi_mode_dot and contaminates every
    # output cell (confirmed: this is what caused "array must not contain
    # infs or NaNs" out of scipy's SVD).
    n_land = int(np.isnan(coeffs[0]).sum())
    coeffs = np.where(np.isnan(coeffs), 0.0, coeffs)
    print(f"climatology coefficients shape {coeffs.shape} ({coeffs.nbytes / 1e6:.1f} MB), "
          f"{n_land} land cells zero-filled")

    # Same orthonormal-factor projection convention as tucker_transform
    # (train/tucker.py) and 07_placement2_tucker_on_anomalies.py's own fit.
    coeffs_core = multi_mode_dot(coeffs, [factor_lat.T, factor_lon.T], modes=[1, 2])  # (n_terms, r_lat, r_lon)
    clim_core = np.einsum("tc,c...->t...", design, coeffs_core)  # (T, r_lat, r_lon)
    print(f"climatology-only core shape {clim_core.shape} ({clim_core.nbytes / 1e6:.1f} MB)")

    n_state = r_lat * r_lon
    snaps = clim_core.reshape(N_TIMES, n_state).T  # (n_state, T)
    snaps_centered = snaps - snaps.mean(axis=1, keepdims=True)
    total_var = np.sum(snaps_centered**2)
    # gesdd (numpy's default LAPACK driver) fails to converge on this
    # matrix -- expected: it's essentially exact-rank-6 by construction,
    # a known pathological case for divide-and-conquer SVD. gesvd is the
    # older, slower, more numerically robust driver; fine here since the
    # SVD-relevant dimension (T=2192) is small regardless.
    _, S, _ = scipy.linalg.svd(snaps_centered, full_matrices=False, lapack_driver="gesvd")
    cum_var = np.cumsum(S**2) / total_var
    print("\nCumulative variance explained by leading r POD modes of the CLIMATOLOGY-ONLY core:")
    for r in [1, 2, 4, 6, 8, 10, 15, 20, 40]:
        if r <= len(S):
            print(f"  r={r:3d}  {100 * cum_var[r - 1]:.6f}%")
    r90 = int(np.searchsorted(cum_var, 0.90)) + 1
    r9999 = int(np.searchsorted(cum_var, 0.9999)) + 1
    print(f"  r needed for 90% variance:   {r90}")
    print(f"  r needed for 99.99% variance: {r9999}")
    print("  (real anomaly core needed r~1287 of 1754 for 90% -- see 07_dmd_rank_sweep.py)")

    print("\nDMD fit on the climatology-only core, small r_dmd candidates:")
    rng = np.random.default_rng(42)
    for r_dmd in [4, 6, 8, 12, 20]:
        fit = dmd_fit(clim_core, r_dmd=r_dmd, dt=DT_SECONDS, rng=rng, variant="exact")
        # In-sample projection reconstruction: how much of the (already
        # tiny, ~6-dim) temporal signal does an r_dmd-mode Phi subspace
        # capture. Phi @ pinv(Phi) is the projector onto that subspace.
        recon_flat = (fit.Phi @ (np.linalg.pinv(fit.Phi) @ snaps)).real  # (n_state, T)
        err_pct = 100.0 * np.linalg.norm(recon_flat - snaps) / np.linalg.norm(snaps)
        n_growing = int(np.sum(np.abs(fit.Lambda) > 1.0 + 1e-6))
        period_days = sorted(
            {
                f"{2 * np.pi / abs(lc.imag) / 86400:.1f}d" if abs(lc.imag) > 1e-8 else "static/trend"
                for lc in fit.lam_cont
            }
        )
        print(
            f"  r_dmd={r_dmd:3d}  in-sample recon err={err_pct:.4f}%  n_growing={n_growing}  "
            f"periods~{period_days}"
        )


if __name__ == "__main__":
    main()
