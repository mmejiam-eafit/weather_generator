"""Task 07 Gaussianization round, placement-(ii) comparison: Tucker-compress
the climatology-removed + standardized anomalies directly, WITHOUT per-cell
Gaussianization first, then compare reconstruction error against the
placement-(i) sweep (`07_gaussianized_rank_sweep.py`) at the same rank.

Motivation (see `docs/open_questions.md` §4.1 and
`docs/gaussian_copula_and_marginals.md` §2.1-2.4, §6.1): per-cell marginal
Gaussianization is provably copula-neutral (Sklar's theorem -- a monotone
transform applied independently per cell cannot change the rank-based
dependence structure). But Tucker is a linear/SVD tool that only sees
second-moment (Pearson/covariance) structure, which is NOT copula-invariant.
So placement (i) (Gaussianize, then Tucker -- what
`07_gaussianized_rank_sweep.py` measured) can need a much higher linear rank
than placement (ii) (Tucker first, then Gaussianize only the much smaller
core) even though the true spatial dependence is identical in both. This
script tests the first half of that claim directly: does Tucker at the SAME
rank reconstruct the un-Gaussianized anomalies far better than it
reconstructed the Gaussianized field?

Reuses the fitted `ClimatologyRemover`/`PerCellStandardizer` from
`07_gaussianize_fit.py` (NOT the copula -- this script never loads it) and
the same checkpointed masked-Tucker ALS loop as `07_gaussianized_rank_sweep.py`
(`svd="randomized_svd"`, `svd_mask_repeats=0`, `modes=[1,2]`, chunked
`init=(core,factors)` resume). Error is reported as a percentage in
standardized-anomaly space (pre-Gaussianization), directly comparable in
*rank* (not in absolute error value -- different space) to the placement-(i)
run at the same `--r_lat --r_lon`.

No chunking needed for the transform step here: without the copula, the
memory profile is just climatology + standardize, each ~2-4x the tensor
(per `07_gaussianized_rank_sweep.py`'s module docstring) -- comfortably
affordable in one shot for a ~7.8GB tensor.

Run directly: `poetry run python notebooks/07_placement2_tucker_on_anomalies.py
--r_lat 31 --r_lon 72 --total_iters 200 --t_start 2024-01-01 --t_end 2024-10-01
--transformers_dir results/task07/gaussianize --ocean_mask_path
results/task07/gaussianize/era2_ocean_mask.npy`.
"""

import argparse
import csv
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import tensorly as tl
import xarray as xr
from tensorly.tenalg import multi_mode_dot

from core.config import canonical_dims
from train.ingest import load_training_window
from train.masked_tucker import partial_tucker_lowmem
from train.masking import height_log_energy

tl.set_backend("numpy")

SEED = 42
SPATIAL_MODES = [1, 2]  # (time, lat, lon) -- time (mode 0) stays full-rank
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def ocean_only_sentinel_fill(
    x: np.ndarray, ocean_mask: np.ndarray, sentinel_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill ocean NaN with sentinel_value; LEAVE LAND NaN untouched.

    `ocean_mask` follows `land_mask_from_sigwh`'s convention: True == ocean.
    Same helper as `07_gaussianized_rank_sweep.py`; duplicated rather than
    imported, matching this task's existing pattern of small script-local
    helpers.
    """
    nan_mask = np.isnan(x)
    ocean_3d = np.broadcast_to(ocean_mask[None, :, :], x.shape)
    filled = np.where(nan_mask & ocean_3d, sentinel_value, x)
    tucker_mask = ocean_3d.astype(np.int8)
    n_ocean_nan = int(np.sum(nan_mask & ocean_3d))
    return filled, tucker_mask, n_ocean_nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r_lat", type=int, required=True)
    p.add_argument("--r_lon", type=int, required=True)
    p.add_argument("--total_iters", type=int, required=True)
    p.add_argument("--checkpoint_every", type=int, default=12)
    p.add_argument("--t_start", type=str, required=True, help="ISO date, e.g. 2024-01-01")
    p.add_argument("--t_end", type=str, required=True, help="ISO date, e.g. 2024-10-01")
    p.add_argument("--variable", type=str, default="wsh")
    p.add_argument(
        "--transformers_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    p.add_argument(
        "--ocean_mask_path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "results"
            / "task07"
            / "gaussianize"
            / "era2_ocean_mask.npy"
        ),
    )
    p.add_argument(
        "--early_stop_tol",
        type=float,
        default=1e-5,
        help=(
            "Stop once a checkpoint's improvement in reconstruction error "
            "(fractional -- err_pct/100, the SAME scale as the ALS's own "
            "internal `tol` below, not the printed *100 percent scale) "
            "falls below this value. Set to 0 or negative to disable."
        ),
    )
    p.add_argument("--mem_cap_gb", type=float, default=58.0)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    args = p.parse_args()

    cap_bytes = int(args.mem_cap_gb * (1024**3))
    resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    t_start = parse_date(args.t_start)
    t_end = parse_date(args.t_end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_tag = f"{t_start:%Y%m}-{t_end:%Y%m}"

    transformed_cache = out_dir / f"{args.variable}_std_anomalies_{window_tag}.npy"
    mask_cache = out_dir / f"{args.variable}_tucker_mask_{window_tag}.npy"

    if transformed_cache.exists() and mask_cache.exists():
        print(f"Reusing cached standardized-anomaly tensor: {transformed_cache}")
        z_full = np.load(transformed_cache)
        mask_int8 = np.load(mask_cache)
    else:
        ocean_mask = np.load(args.ocean_mask_path)
        transformers_dir = Path(args.transformers_dir)
        clim = joblib.load(transformers_dir / f"{args.variable}_climatology.joblib")
        std = joblib.load(transformers_dir / f"{args.variable}_standardize.joblib")

        print(f"Loading {args.variable} for {t_start} -> {t_end}...")
        ds = load_training_window(DATA_DIR, t_start, t_end, [args.variable])
        raw = ds[args.variable].transpose(*canonical_dims("wave")).values.astype(np.float32)
        time_coord = ds["time"].values
        del ds

        filled, tucker_mask, n_ocean_nan = ocean_only_sentinel_fill(
            raw, ocean_mask, sentinel_value=0.0
        )
        del raw
        x = height_log_energy(filled, floor_m=0.01)
        del filled
        print(f"n_ocean_nan={n_ocean_nan}")

        lat_dim, lon_dim = canonical_dims("wave")[1], canonical_dims("wave")[2]
        ds_x = xr.Dataset(
            {args.variable: (("time", lat_dim, lon_dim), x)}, coords={"time": time_coord}
        )
        del x  # xr.Dataset wraps x's buffer directly (no copy) -- ds_x is now the only handle

        print("Transforming (climatology + standardize, no copula, whole window at once)...")
        t0 = time.time()
        anomalies = clim.transform(ds_x)
        del ds_x
        standardized = std.transform(anomalies)
        del anomalies
        z_full = standardized[args.variable].values.astype(np.float32)
        del standardized
        print(f"  done in {time.time() - t0:.1f}s")

        z_full = np.where(np.isnan(z_full), 0.0, z_full)
        mask_int8 = tucker_mask

        np.save(transformed_cache, z_full)
        np.save(mask_cache, mask_int8)
        print(f"Cached standardized-anomaly tensor to {transformed_cache}")

    rank = [args.r_lat, args.r_lon]
    tag = (
        f"anomalies_{args.variable}_r{args.r_lat}x{args.r_lon}"
        f"_checkpointed{args.total_iters}_{window_tag}"
    )
    checkpoint_path = out_dir / f"{tag}_checkpoints.csv"
    with open(checkpoint_path, "w", newline="") as f:
        csv.writer(f).writerow(["iteration", "true_err_pct", "elapsed_s_cumulative"])

    init = "svd"
    done_iters = 0
    err_pct = float("nan")
    prev_err_pct = float("nan")
    t0 = time.time()
    while done_iters < args.total_iters:
        chunk = min(args.checkpoint_every, args.total_iters - done_iters)
        (core, factors), _ = partial_tucker_lowmem(
            z_full,
            modes=SPATIAL_MODES,
            rank=rank,
            init=init,
            tol=1e-5,
            n_iter_max=chunk,
            random_state=SEED,
            mask=mask_int8,
            svd="randomized_svd",
            svd_mask_repeats=0,
        )
        init = (core, factors)
        done_iters += chunk
        elapsed = time.time() - t0
        recon = multi_mode_dot(core, factors, modes=SPATIAL_MODES)
        err_pct = 100.0 * float(
            np.linalg.norm((recon - z_full) * mask_int8) / np.linalg.norm(z_full * mask_int8)
        )
        with open(checkpoint_path, "a", newline="") as f:
            csv.writer(f).writerow([done_iters, f"{err_pct:.4f}", f"{elapsed:.1f}"])
        print(
            f"CHECKPOINT rank=({args.r_lat},{args.r_lon}) iter={done_iters} "
            f"true_err_pct={err_pct:.4f}% elapsed_s={elapsed:.1f}"
        )

        # Save every checkpoint, not just on final completion -- a process
        # killed early (wall-clock timeout, manual stop) otherwise loses
        # `core`/`factors` entirely, since Python variables don't survive
        # process death. Overwrites the same path each time, so this is
        # always the single latest snapshot, not an accumulating series.
        tucker_path = out_dir / f"{tag}_tucker.npz"
        np.savez_compressed(tucker_path, core=core, factor_lat=factors[0], factor_lon=factors[1])
        print(f"Saved Tucker factors to {tucker_path} (checkpoint iter={done_iters})")

        if args.early_stop_tol > 0 and not np.isnan(prev_err_pct):
            delta_frac = (prev_err_pct - err_pct) / 100.0
            if delta_frac < args.early_stop_tol:
                print(
                    f"EARLY STOP: checkpoint delta {delta_frac:.2e} (fractional) < "
                    f"tol {args.early_stop_tol:.2e} at iter={done_iters}"
                )
                break
        prev_err_pct = err_pct

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(
        f"RESULT placement2-anomalies-checkpointed {args.variable} rank=({args.r_lat},{args.r_lon}) "
        f"err_pct={err_pct:.4f}% peak_rss_gb={peak_rss_gb:.2f} elapsed_s={elapsed:.1f} "
        f"n_iters={done_iters}"
    )


if __name__ == "__main__":
    main()
