"""Task 07: real-data walk-forward cross-validation of the DMD rank finding.

Full design and rationale: `dmd_rank_cv_plan.md` (repo root). Follows up on
two prior experiments (`tasks/07-tucker-dmd-rank-selection.md` §8) that
both rested on a single 80/20 train/holdout split of the one cached real
core: a phase-randomized surrogate test and a DMD+OU generative fidelity
check, both suggesting the anomaly's DMD rank plateaus around r_dmd=
50-100. This script checks whether that conclusion replicates across
several genuinely different real held-out windows -- no synthetic
surrogate at all, only real data, sliced differently.

Method: `sklearn.model_selection.TimeSeriesSplit(n_splits=8,
test_size=190)` (expanding training window, non-overlapping sequential
real test blocks). Per fold: `GaussianCopula` fit ONCE on that fold's
train slice (never per-rank -- refitting it per rank would cost ~9x for
byte-identical results, since the copula does not depend on r_dmd); then
per `r_dmd`, `dmd_fit` on the Gaussianized train slice, and a ROLLING-
anchor held-out evaluation across the fold's entire test block (not just
the single last-training-snapshot anchor `07_dmd_rank_sweep.py` uses --
with only one fold that was the only option; with 8 real folds, rolling
the anchor through every valid start in each ~190-step test block turns
1 sample/fold into ~142, at the cost of one cheap matrix-vector product
per anchor, no new fits). Within-fold spread is descriptive; the median
per (fold, r_dmd, lead) is the unit compared *across* folds.

Also computes a per-fold POD/variance-capture curve (cheap, no DMD fit
needed) as a complementary, independent check of whether the "modes
needed for variance" finding is itself stable across real seasons.

Memory-capped at 25 GiB (`RLIMIT_AS`, matching the technique already
validated in this task, §4, for exactly this purpose): an over-budget
allocation raises a clean, catchable `MemoryError` instead of the OS
OOM-killer taking down an unrelated process. Not expected to trigger --
every array here is far smaller than the two prior scripts' already-
validated ~1GB footprints -- this is a safety net, not an expected limit.

Run directly (same cached core as the other 07_dmd_*.py scripts):
`poetry run python notebooks/07_dmd_rank_cv_check.py --core_path
results/task07/gaussianize/anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz`.
"""

import argparse
import csv
import gc
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import scipy.linalg
import xarray as xr
from sklearn.model_selection import TimeSeriesSplit

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import dmd_fit

R_DMD_CANDIDATES = [5, 10, 20, 40, 50, 80, 100, 150, 200]
N_BINS = 500
SEED = 42
N_FOLDS = 8
TEST_SIZE = 190
LEAD_TIMES = [1, 6, 12, 24, 48]
MEMORY_CAP_BYTES = 25 * 1024**3  # 25 GiB


def _apply_memory_cap() -> None:
    """Cap this process's virtual address space at `MEMORY_CAP_BYTES`.

    Matches the RLIMIT_AS-capped-process technique already validated in
    this task (`tasks/07-tucker-dmd-rank-selection.md` §4) for exactly
    this purpose. Applied directly to this process (not a child
    subprocess): this script is already a standalone `poetry run python`
    process, not a shared interactive kernel, so there's no separate
    process to isolate it from.
    """
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))


def _wrap(core: np.ndarray) -> xr.Dataset:
    """Wrap a Tucker core as a synthetic Dataset for `GaussianCopula`."""
    return xr.Dataset({"core": (("time", "core_lat", "core_lon"), core)})


def _unwrap(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["core"].values)


def pod_variance_explained(x: np.ndarray, ranks: list[int]) -> list[float]:
    """Cumulative POD variance explained at each candidate rank.

    Same pattern as `07_dmd_surrogate_rank_check.py`: `compute_uv=False`
    avoids ever forming the full U/V factors; default `gesdd` (fast) is
    appropriate here since no fold's train slice is degenerate/near-exact
    rank the way the climatology-only core was.
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


def rolling_anchor_errors(
    gauss_train: np.ndarray,
    gauss_test: np.ndarray,
    r_dmd: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Fit DMD on `gauss_train`; evaluate held-out error at every valid
    rolling anchor in `gauss_test`, for every lead in `LEAD_TIMES`.

    Fully vectorized across anchors (a matrix-matrix product, not a
    Python loop of matrix-vector products): projects every anchor at
    once via `pinv(Phi)`, advances all of them by `Lambda**lead`
    simultaneously, and compares against the fixed shifted-by-`lead`
    slice of `gauss_test`. The anchor range [0, n_anchors) is the SAME
    for every lead (not lead-dependent), so all leads are compared on an
    identical anchor set within a fold.

    Returns a dict with n_growing, max_lambda, phi_cond, n_anchors, and
    per-lead median/IQR error, plus the DMDFit's own dt-independent
    diagnostics -- everything one CSV row for this (fold, r_dmd) needs.
    """
    n_test, r_lat, r_lon = gauss_test.shape
    n_state = r_lat * r_lon
    n_anchors = n_test - max(LEAD_TIMES)

    fit = dmd_fit(gauss_train, r_dmd=r_dmd, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact")
    n_growing = int(np.sum(np.abs(fit.Lambda) > 1.0 + 1e-6))
    max_lambda = float(np.abs(fit.Lambda).max())
    phi_cond = float(np.linalg.cond(fit.Phi))

    test_flat = gauss_test.reshape(n_test, n_state)  # (n_test, n_state)
    phi_pinv = np.linalg.pinv(fit.Phi)  # (r_dmd, n_state)
    a_ref_all = phi_pinv @ test_flat.T  # (r_dmd, n_test) -- every anchor's projection at once
    a_ref_valid = a_ref_all[:, :n_anchors]  # anchors 0..n_anchors-1, shared across all leads

    result: dict[str, object] = {
        "n_growing": n_growing,
        "max_lambda": max_lambda,
        "phi_cond": phi_cond,
        "n_anchors": n_anchors,
    }
    for lead in LEAD_TIMES:
        a_pred = a_ref_valid * (fit.Lambda[:, None] ** lead)  # (r_dmd, n_anchors)
        recon = (fit.Phi @ a_pred).real  # (n_state, n_anchors)
        targets = test_flat[lead : lead + n_anchors].T  # (n_state, n_anchors)
        errors = 100.0 * np.linalg.norm(recon - targets, axis=0) / np.linalg.norm(targets, axis=0)
        result[f"median_err_pct_lead{lead}"] = float(np.median(errors))
        result[f"iqr_err_pct_lead{lead}"] = float(
            np.percentile(errors, 75) - np.percentile(errors, 25)
        )
    return result


def main() -> None:
    _apply_memory_cap()

    p = argparse.ArgumentParser()
    p.add_argument("--core_path", type=str, required=True)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    p.add_argument("--tag", type=str, default="wsh_r124x288_202401-202410")
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--test_size", type=int, default=TEST_SIZE)
    p.add_argument("--r_dmd_candidates", type=int, nargs="+", default=R_DMD_CANDIDATES)
    args = p.parse_args()

    core_path = Path(args.core_path)
    npz = np.load(core_path)
    core = npz["core"].astype(np.float64)
    n_times, r_lat, r_lon = core.shape
    print(
        f"core shape {core.shape}, memory cap {MEMORY_CAP_BYTES / 1024**3:.0f} GiB, "
        f"n_folds={args.n_folds}, test_size={args.test_size}"
    )
    del npz
    gc.collect()

    splitter = TimeSeriesSplit(n_splits=args.n_folds, test_size=args.test_size)
    folds = list(splitter.split(np.arange(n_times)))
    for i, (train_idx, test_idx) in enumerate(folds):
        print(
            f"  fold {i}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"train=[0:{train_idx[-1] + 1}) test=[{test_idx[0]}:{test_idx[-1] + 1})"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"dmd_rank_cv_check_{args.tag}.csv"

    lead_fieldnames = [
        f"{stat}_err_pct_lead{lead}" for lead in LEAD_TIMES for stat in ("median", "iqr")
    ]
    fieldnames = [
        "fold",
        "n_train",
        "n_test",
        "r_dmd",
        "n_growing",
        "max_lambda",
        "phi_cond",
        "n_anchors",
        "pod_var_explained",
        *lead_fieldnames,
        "elapsed_s",
    ]

    rng = np.random.default_rng(SEED)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                t_fold = time.time()
                copula = GaussianCopula(n_bins=N_BINS)
                copula.fit(_wrap(core[train_idx]))
                gauss_train = _unwrap(copula.transform(_wrap(core[train_idx])))
                gauss_test = _unwrap(copula.transform(_wrap(core[test_idx])))

                pod_curve = pod_variance_explained(gauss_train, args.r_dmd_candidates)

                for r_dmd, pod in zip(args.r_dmd_candidates, pod_curve):
                    t0 = time.time()
                    metrics = rolling_anchor_errors(gauss_train, gauss_test, r_dmd, rng)
                    elapsed = time.time() - t0
                    writer.writerow(
                        {
                            "fold": fold_idx,
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                            "r_dmd": r_dmd,
                            "pod_var_explained": pod,
                            "elapsed_s": f"{elapsed:.2f}",
                            **metrics,
                        }
                    )
                    f.flush()
                    print(
                        f"fold={fold_idx} r_dmd={r_dmd:>4d} pod={pod:.4f} "
                        f"n_growing={metrics['n_growing']} "
                        f"median_err_lead1={metrics['median_err_pct_lead1']:.2f}% "
                        f"median_err_lead48={metrics['median_err_pct_lead48']:.2f}% "
                        f"({elapsed:.1f}s)",
                        file=sys.stderr,
                    )

                del copula, gauss_train, gauss_test
                gc.collect()
                print(f"fold {fold_idx} done ({time.time() - t_fold:.1f}s)", file=sys.stderr)
        except MemoryError:
            print(
                f"KILLED: exceeded {MEMORY_CAP_BYTES / 1024**3:.0f} GiB memory cap "
                f"at fold={fold_idx}, r_dmd={r_dmd}. Partial results up to the last "
                f"flushed row are in {csv_path}.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
