"""Task 07: DMD rank (`r_dmd`) selection under the now-canonical
Gaussianization placement (ii) — Tucker on climatology-removed/standardized
anomalies, then Gaussianize the Tucker core, then DMD (see approved plan,
`docs/open_questions.md` §4.1).

Closes the DMD half of task 07 (`docs/open_questions.md` §5.3): the task
doc recommended `r_dmd~=20` while its own notebook's sweep concluded
`r_dmd=10`, and that sweep ran on an 800-dimensional, un-Gaussianized,
single-season Tucker core `(20,40)` -- nothing like the current pipeline.
This script fits DMD on a real Gaussianized core at the one Tucker rank
currently trusted (20%, `r_lat=124, r_lon=288`, era 2, 9-month window
2024-01-01 -> 2024-10-01, `T=2192`).

Scope note: the existing "cond(Phi) tracks held-out quality" finding
(`open_questions.md` Sec 5.3, 2026-08-16) was established with DMD fit on a
*raw, non-Gaussianized* Tucker core (that experiment Gaussianized the DMD
*residual* afterward, testing placement iii). Whether cond(Phi) still
tracks quality once DMD's own input has already been through placement
(ii)'s Gaussianization is untested here, not assumed -- Gaussianization is
copula-invariant (Sklar's theorem) but not Pearson/covariance-structure-
invariant, and DMD is SVD-driven. Because this script reads both the
in-sample stability diagnostics (n_growing, max_lambda, phi_cond) and the
held-out reconstruction error off the SAME per-rank fit, the sweep directly
shows whether they still track together under a Gaussianized input.

Both the copula and DMD are fit on the TRAIN portion only (80% of the
window), never on the whole core before splitting -- fitting the copula on
the whole window first would leak the holdout period's own values into its
per-cell LUT, a second leakage channel distinct from (and avoidable unlike)
the Tucker spatial factors', which were fit on the whole window when the
core was produced (mild: modes=[1,2] keeps time full-rank, so this leaks
only "the spatial basis saw holdout-period variance", not holdout
dynamics).

`fit_mode_ou` is explicitly out of scope: it is a downstream fit that
happens after `r_dmd` is chosen, not part of choosing it.

Run directly (after the source core has been scp'd down from the VM):
`poetry run python notebooks/07_dmd_rank_sweep.py --core_path
results/task07/gaussianize/anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz`.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import xarray as xr

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import dmd_fit

R_DMD_CANDIDATES = [5, 10, 20, 40, 50, 80, 100, 150, 200]
TRAIN_FRACTION = 0.8
N_BINS = 500
SEED = 42
# Steps ahead into the holdout window (3h cadence: 3h, 18h, 36h, 3d, 6d).
# NOT the full holdout range -- with every |lambda| < 1, extrapolating
# b0*lambda**t all the way to t=n_train (~1754) underflows to numerical
# zero regardless of r_dmd (verified: max |lambda|**1754 ~ 3e-17, below
# float64 eps), making a full-range aggregate error read exactly 100% at
# every candidate and testing nothing. Short lead times ask the actually
# useful question: does a larger r_dmd extend how long the deterministic
# component stays informative.
LEAD_TIMES = [1, 6, 12, 24, 48]


def _wrap(core: np.ndarray) -> xr.Dataset:
    """Wrap a Tucker core as a synthetic Dataset for `GaussianCopula`.

    `check_canonical` only pattern-matches dim-NAME suffixes
    (`endswith("_lat")`/`"_lon"`), never coordinate values, and
    `GaussianCopula` never reads coords either -- so no time coordinate or
    real grid is needed, just names.
    """
    return xr.Dataset({"core": (("time", "core_lat", "core_lon"), core)})


def _unwrap(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["core"].values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--core_path", type=str, required=True)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    p.add_argument("--tag", type=str, default="wsh_r124x288_202401-202410")
    args = p.parse_args()

    core_path = Path(args.core_path)
    npz = np.load(core_path)
    core = npz["core"].astype(np.float64)
    n_times, r_lat, r_lon = core.shape
    n_state = r_lat * r_lon
    n_train = int(round(n_times * TRAIN_FRACTION))
    n_holdout = n_times - n_train
    print(
        f"core shape {core.shape}, n_state={n_state}, "
        f"n_train={n_train}, n_holdout={n_holdout}"
    )

    core_train = core[:n_train]
    core_holdout = core[n_train:]

    copula = GaussianCopula(n_bins=N_BINS)
    copula.fit(_wrap(core_train))
    gauss_train = _unwrap(copula.transform(_wrap(core_train)))
    gauss_holdout = _unwrap(copula.transform(_wrap(core_holdout)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"dmd_rank_sweep_{args.tag}.csv"
    lambda_path = out_dir / f"dmd_rank_sweep_{args.tag}_lambda.npz"

    lead_fieldnames = [f"err_pct_lead{lead}" for lead in LEAD_TIMES]
    fieldnames = [
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
        "elapsed_s",
    ]
    lambdas: dict[str, np.ndarray] = {}

    rng = np.random.default_rng(SEED)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r_dmd in R_DMD_CANDIDATES:
            t0 = time.time()
            fit = dmd_fit(gauss_train, r_dmd=r_dmd, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact")

            n_growing = int(np.sum(np.abs(fit.Lambda) > 1.0 + 1e-6))
            max_lambda = float(np.abs(fit.Lambda).max())
            phi_cond = float(np.linalg.cond(fit.Phi))

            # NOT b0*Lambda**t from t=0: that's anchored at the START of
            # training, so even at lead=1 the exponent is still ~n_train
            # (1754+), already decayed to numerical zero (verified: this
            # is what actually produced the flat 100% error on the first
            # attempt at this fix, since a small LEAD doesn't shrink an
            # exponent that's dominated by n_train). Standard one-step(s)-
            # forward DMD evaluation instead: project the ACTUAL (measured)
            # state at the end of training onto the modes, then advance
            # that measured amplitude forward by Lambda**lead for small
            # lead -- the same "measured amplitude" operation `fit_mode_ou`
            # already uses (`a_t = pinv(Phi) @ snapshots`), just evaluated
            # at one reference point and propagated forward instead of
            # compared against the deterministic trajectory at every step.
            a_ref = np.linalg.pinv(fit.Phi) @ gauss_train[-1].reshape(-1)  # (r_dmd,)
            leads_arr = np.array(LEAD_TIMES)
            a_pred_leads = a_ref[:, None] * (fit.Lambda[:, None] ** leads_arr[None, :])
            recon_leads_raw = fit.Phi @ a_pred_leads  # (n_state, n_leads)

            imag_real_ratio = float(
                np.linalg.norm(recon_leads_raw.imag) / max(np.linalg.norm(recon_leads_raw.real), 1e-300)
            )
            # .T before .reshape: recon_leads_raw is (n_state, n_leads), one
            # COLUMN per lead time, matching dmd_fit's own
            # core.reshape(n_times, n_state).T convention. Reshaping
            # directly without transposing has the same element count so
            # it fails SILENTLY (wrong layout, not an error).
            recon_leads = recon_leads_raw.real.T.reshape(len(LEAD_TIMES), r_lat, r_lon)

            lead_errors = {}
            for i, lead in enumerate(LEAD_TIMES):
                actual_at_lead = gauss_holdout[lead - 1]
                lead_errors[f"err_pct_lead{lead}"] = float(
                    100.0
                    * np.linalg.norm(recon_leads[i] - actual_at_lead)
                    / np.linalg.norm(actual_at_lead)
                )

            elapsed = time.time() - t0
            row = {
                "r_dmd": r_dmd,
                "variant": "exact",
                "n_train": n_train,
                "n_holdout": n_holdout,
                "core_path": str(core_path),
                "n_growing": n_growing,
                "max_lambda": max_lambda,
                "phi_cond": phi_cond,
                **lead_errors,
                "imag_real_ratio": imag_real_ratio,
                "elapsed_s": f"{elapsed:.2f}",
            }
            writer.writerow(row)
            f.flush()
            lambdas[f"r_dmd_{r_dmd}"] = fit.Lambda
            lead_str = "  ".join(f"L{lead}={lead_errors[f'err_pct_lead{lead}']:.2f}%" for lead in LEAD_TIMES)
            print(
                f"r_dmd={r_dmd:>4d}  n_growing={n_growing:>3d}  max|lambda|={max_lambda:.6f}  "
                f"cond(Phi)={phi_cond:.4e}  {lead_str}  "
                f"imag/real={imag_real_ratio:.4e}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )

    np.savez_compressed(lambda_path, **lambdas)
    print(f"Wrote {csv_path}")
    print(f"Wrote {lambda_path}")


if __name__ == "__main__":
    main()
