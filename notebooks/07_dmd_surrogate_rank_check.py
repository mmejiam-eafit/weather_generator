"""Task 07: phase-randomized surrogate test for the DMD rank sweep.

Answers: is the ~69% of Gaussianized-core variance DMD can't reach by
r_dmd=200 (`notebooks/07_dmd_rank_sweep.py`,
`tasks/07-tucker-dmd-rank-selection.md` §6) real recoverable structure, or
already statistically indistinguishable from correlated Gaussian noise --
exactly what mode-OU is designed to model? Full reviewed design:
`tasks/07-tucker-dmd-rank-selection.md` §8.

Method: build N_SURROGATES independent Prichard-Theiler multivariate FFT
surrogates of the full Gaussianized core (the same random phase rotation
applied to every one of the 35,712 state channels at each frequency,
preserving the complete complex cross-spectrum -- i.e. all auto- and
cross-covariance at every lag -- while destroying any structure beyond
second-order/linear statistics). Run the exact same per-rank DMD sweep
(`07_dmd_rank_sweep.py`) on each surrogate and compare against the real
core's own sweep.

Reading the results: the POD variance-explained curve is a sanity check
on the surrogate code, NOT evidence -- by Parseval's theorem it is
expected to closely match the real curve regardless of whether real
recoverable structure exists, since POD is determined by exactly the
cross-spectral quantity this surrogate construction preserves exactly.
The held-out multi-step DMD error is the load-bearing comparison: a
linear predictor is Bayes-optimal for a genuinely Gaussian/linear process,
so the surrogate error band traces the covariance-implied error floor.
Real error at/inside that band at every rank means the anomaly's rank
requirement is already fully explained by its own second-order
statistics; real error meaningfully below the band at some rank
quantifies real recoverable structure at that rank.

Memory discipline: process one surrogate fully (build -> fit every rank
-> record scalar metrics -> discard the (T, n_state) arrays) before
starting the next, so peak memory stays close to a couple of extra copies
of the core rather than accumulating N_SURROGATES copies simultaneously.

Run directly (same cached core as `07_dmd_rank_sweep.py`):
`poetry run python notebooks/07_dmd_surrogate_rank_check.py --core_path
results/task07/gaussianize/anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz`.
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import scipy.linalg
import xarray as xr

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import dmd_fit

R_DMD_CANDIDATES = [5, 10, 20, 40, 50, 80, 100, 150, 200]
TRAIN_FRACTION = 0.8
N_BINS = 500
SEED = 42
N_SURROGATES = 20
# Same short-lead rationale as `07_dmd_rank_sweep.py`: |Lambda| < 1 for
# every existing candidate, so a full-holdout-range trajectory underflows
# to numerical zero regardless of rank -- short leads ask the useful
# question instead.
LEAD_TIMES = [1, 6, 12, 24, 48]


def _wrap(core: np.ndarray) -> xr.Dataset:
    """Wrap a Tucker core as a synthetic Dataset for `GaussianCopula`."""
    return xr.Dataset({"core": (("time", "core_lat", "core_lon"), core)})


def _unwrap(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["core"].values)


def make_phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Prichard-Theiler multivariate FFT surrogate.

    One random phase per frequency bin, applied identically to every
    channel (not independently per channel): for channels i,j,
    `X_i'(f)*conj(X_j'(f)) = X_i(f)*e^{i*theta}*conj(X_j(f))*e^{-i*theta}
    = X_i(f)*conj(X_j(f))`, so the full complex cross-spectrum -- and
    therefore every auto- and cross-covariance at every lag -- is
    preserved exactly, while higher-order/nonlinear structure is
    destroyed. Safe to apply after Gaussianization: since each channel's
    marginal is already ~N(0,1), this does not suffer the marginal-drift
    pitfall that motivates AAFT/IAAFT for raw (non-Gaussian) data.

    Parameters
    ----------
    x : (T, n_state) float64
    rng : np.random.Generator

    Returns
    -------
    (T, n_state) float64
    """
    n_times = x.shape[0]
    spectrum = np.fft.rfft(x, axis=0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[0])
    phases[0] = 0.0  # DC is already real for a real signal; keep it that way.
    if n_times % 2 == 0:
        phases[-1] = 0.0  # Nyquist bin, same reason, only present for even T.
    rotation = np.exp(1j * phases)[:, None]
    surrogate = np.fft.irfft(spectrum * rotation, n=n_times, axis=0)
    return np.asarray(surrogate, dtype=np.float64)


def pod_variance_explained(x: np.ndarray, ranks: list[int]) -> list[float]:
    """Cumulative POD variance explained at each candidate rank.

    `compute_uv=False` avoids ever forming the (extremely large) full U/V
    factors -- only singular values are computed. Defaults to `gesdd`
    (fast); `07_dmd_climatology_only_check.py` needed `gesvd` only because
    that core was exactly rank ~6 and degenerate, which does not apply to
    a full-rank anomaly or a full-rank surrogate.
    """
    n_state = int(np.prod(x.shape[1:]))
    flat = x.reshape(x.shape[0], n_state)
    try:
        singular_values = scipy.linalg.svd(flat, compute_uv=False, lapack_driver="gesdd")
    except np.linalg.LinAlgError:
        singular_values = scipy.linalg.svd(flat, compute_uv=False, lapack_driver="gesvd")
    power = np.asarray(singular_values, dtype=np.float64) ** 2
    cumulative = np.cumsum(power)
    total = float(cumulative[-1])
    return [float(cumulative[min(r, len(cumulative)) - 1] / total) for r in ranks]


def dmd_sweep_metrics(
    gauss_train: np.ndarray,
    gauss_holdout: np.ndarray,
    r_lat: int,
    r_lon: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """The `07_dmd_rank_sweep.py` per-rank loop, factored out for reuse here."""
    rows: list[dict[str, object]] = []
    for r_dmd in R_DMD_CANDIDATES:
        t0 = time.time()
        fit = dmd_fit(
            gauss_train, r_dmd=r_dmd, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact"
        )

        n_growing = int(np.sum(np.abs(fit.Lambda) > 1.0 + 1e-6))
        max_lambda = float(np.abs(fit.Lambda).max())
        phi_cond = float(np.linalg.cond(fit.Phi))

        # Corrected anchor (see `07_dmd_rank_sweep.py`'s own extensive
        # comment on this): project the MEASURED end-of-training state,
        # advance by Lambda**lead for small leads -- not the deterministic
        # trajectory from the fit's own t=0, which has already decayed to
        # numerical zero by n_train ~= 1754 regardless of r_dmd.
        a_ref = np.linalg.pinv(fit.Phi) @ gauss_train[-1].reshape(-1)
        leads_arr = np.array(LEAD_TIMES)
        a_pred_leads = a_ref[:, None] * (fit.Lambda[:, None] ** leads_arr[None, :])
        recon_leads_raw = fit.Phi @ a_pred_leads

        imag_real_ratio = float(
            np.linalg.norm(recon_leads_raw.imag) / max(np.linalg.norm(recon_leads_raw.real), 1e-300)
        )
        # .T before .reshape -- see `07_dmd_rank_sweep.py`: same element
        # count either way, wrong layout fails silently without this.
        recon_leads = recon_leads_raw.real.T.reshape(len(LEAD_TIMES), r_lat, r_lon)

        lead_errors: dict[str, object] = {}
        for i, lead in enumerate(LEAD_TIMES):
            actual_at_lead = gauss_holdout[lead - 1]
            lead_errors[f"err_pct_lead{lead}"] = float(
                100.0
                * np.linalg.norm(recon_leads[i] - actual_at_lead)
                / np.linalg.norm(actual_at_lead)
            )

        elapsed = time.time() - t0
        rows.append(
            {
                "r_dmd": r_dmd,
                "variant": "exact",
                "n_growing": n_growing,
                "max_lambda": max_lambda,
                "phi_cond": phi_cond,
                **lead_errors,
                "imag_real_ratio": imag_real_ratio,
                "elapsed_s": f"{elapsed:.2f}",
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--core_path", type=str, required=True)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    p.add_argument("--tag", type=str, default="wsh_r124x288_202401-202410")
    p.add_argument("--n_surrogates", type=int, default=N_SURROGATES)
    args = p.parse_args()

    core_path = Path(args.core_path)
    npz = np.load(core_path)
    core = npz["core"].astype(np.float64)
    n_times, r_lat, r_lon = core.shape
    n_state = r_lat * r_lon
    n_train = int(round(n_times * TRAIN_FRACTION))
    n_holdout = n_times - n_train
    print(
        f"core shape {core.shape}, n_state={n_state}, " f"n_train={n_train}, n_holdout={n_holdout}"
    )

    rng = np.random.default_rng(SEED)

    copula = GaussianCopula(n_bins=N_BINS)
    copula.fit(_wrap(core[:n_train]))
    gauss_train = _unwrap(copula.transform(_wrap(core[:n_train])))
    gauss_holdout = _unwrap(copula.transform(_wrap(core[n_train:])))
    # Build the surrogate from the full concatenated series, not
    # train/holdout independently: avoids a second wraparound
    # discontinuity and gives holdout the same frequency resolution as
    # train. No leakage -- no parameters are fit on this, it's a
    # data-augmentation transform of already-legitimate gauss_train/holdout.
    gauss_full = np.concatenate([gauss_train, gauss_holdout], axis=0)
    del core, npz
    gc.collect()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"dmd_surrogate_rank_check_{args.tag}.csv"

    lead_fieldnames = [f"err_pct_lead{lead}" for lead in LEAD_TIMES]
    fieldnames = [
        "source",
        "surrogate_id",
        "r_dmd",
        "variant",
        "n_train",
        "n_holdout",
        "core_path",
        "n_growing",
        "max_lambda",
        "phi_cond",
        *lead_fieldnames,
        "imag_real_ratio",
        "pod_var_explained",
        "elapsed_s",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("=== real data ===", file=sys.stderr)
        pod_real = pod_variance_explained(gauss_train, R_DMD_CANDIDATES)
        real_rows = dmd_sweep_metrics(gauss_train, gauss_holdout, r_lat, r_lon, rng)
        for row, pod in zip(real_rows, pod_real):
            writer.writerow(
                {
                    "source": "real",
                    "surrogate_id": -1,
                    "n_train": n_train,
                    "n_holdout": n_holdout,
                    "core_path": str(core_path),
                    "pod_var_explained": pod,
                    **row,
                }
            )
            print(f"  r_dmd={row['r_dmd']:>4d}  pod={pod:.4f}  {row}", file=sys.stderr)
        f.flush()

        for surrogate_id in range(args.n_surrogates):
            t_sur = time.time()
            surrogate_flat = make_phase_randomized_surrogate(
                gauss_full.reshape(n_times, n_state), rng
            )

            # Self-check on the phase-randomization code itself (not a
            # scientific finding): per-channel variance must survive
            # exactly, up to FFT round-trip float error.
            np.testing.assert_allclose(
                np.var(surrogate_flat, axis=0),
                np.var(gauss_full.reshape(n_times, n_state), axis=0),
                rtol=1e-6,
                atol=1e-8,
            )

            surrogate = surrogate_flat.reshape(n_times, r_lat, r_lon)
            sur_train = surrogate[:n_train]
            sur_holdout = surrogate[n_train:]

            pod_sur = pod_variance_explained(sur_train, R_DMD_CANDIDATES)
            sur_rows = dmd_sweep_metrics(sur_train, sur_holdout, r_lat, r_lon, rng)
            for row, pod in zip(sur_rows, pod_sur):
                writer.writerow(
                    {
                        "source": "surrogate",
                        "surrogate_id": surrogate_id,
                        "n_train": n_train,
                        "n_holdout": n_holdout,
                        "core_path": str(core_path),
                        "pod_var_explained": pod,
                        **row,
                    }
                )
            f.flush()
            print(
                f"surrogate {surrogate_id:>2d}/{args.n_surrogates}: done "
                f"({time.time() - t_sur:.1f}s)",
                file=sys.stderr,
            )

            del surrogate_flat, surrogate, sur_train, sur_holdout
            gc.collect()

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
