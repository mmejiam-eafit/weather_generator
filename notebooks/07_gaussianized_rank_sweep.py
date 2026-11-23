"""Task 07 Gaussianization round, step 3: transform + checkpointed Tucker
sweep on Gaussianized wsh, era 2 (approved plan, Phase B design +
execution order steps 6-7).

Loads the transformers `07_gaussianize_fit.py` fit on the strided 2-year
window, transforms a ~9-month era-2 window (full resolution, T~2184) through
them, then runs the same checkpointed masked-Tucker sweep established for
the (non-Gaussianized) era-3 runs this session -- `svd="randomized_svd"`,
`svd_mask_repeats=0`, `modes=[1,2]` (canonical (time, lat, lon); time stays
full-rank), chunked `init=(core,factors)` resume -- with error now measured
in copula space and reported as a percentage, per the plan's rebased metric.

**The copula transform, not Tucker, is what forces chunking.** Measured
peak for `GaussianCopula.transform` is ~`max(5L + 2Q, L + 12Q)`, where
`L = n_bins*n_cells*8` and `Q = T*n_cells*8`. At 894,240 cells with
`n_bins=500`, `L` alone is 3.58GB, so one T~2184 call would need ~190GB.
`ClimatologyRemover`/`PerCellStandardizer` are cheap by comparison (~2-4x
the tensor) and run on the whole window at once; only the copula step is
chunked over time (`--transform_chunk`, default 250 -> ~23GB peak for that
step, comfortably under a 62GB VM). All three transformers support
fit-on-one-window/transform-another (confirmed this session: each
recomputes its per-call state -- climatology re-evaluates the harmonic at
the *transform* window's own timestamps against its stored origins -- from
the window it's actually given), so this is a correct composition, not an
approximation.

The transformed+masked tensor is cached to `--out_dir` after the first run
at a given window, so re-running at a different rank doesn't repeat the
(comparatively slow) copula transform.

Secondary note for the reconstruction GIFs: `GaussianCopula.inverse_transform`
is lossy (clips probabilities to [1e-6, 1-1e-6], saturating z at ~+/-4.75,
and linearly extrapolates off the outermost LUT bin) -- a physical-space
error after inverting conflates Tucker truncation with that inversion loss;
the primary metric here stays in copula space for that reason. Run directly:
`poetry run python notebooks/07_gaussianized_rank_sweep.py --r_lat N --r_lon N
--total_iters N --t_start YYYY-MM-DD --t_end YYYY-MM-DD
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
    Same helper as `07_gaussianize_fit.py`; duplicated rather than imported,
    matching this task's existing pattern of small script-local helpers
    (e.g. `07_crop_shortwindow_control.py` importing only `log_result`
    from its sibling, not the whole module).
    """
    nan_mask = np.isnan(x)
    ocean_3d = np.broadcast_to(ocean_mask[None, :, :], x.shape)
    filled = np.where(nan_mask & ocean_3d, sentinel_value, x)
    tucker_mask = ocean_3d.astype(np.int8)
    n_ocean_nan = int(np.sum(nan_mask & ocean_3d))
    return filled, tucker_mask, n_ocean_nan


def transform_chunked(
    clim, std, copula, ds: xr.Dataset, variable: str, chunk_size: int
) -> np.ndarray:
    """ClimatologyRemover/PerCellStandardizer on the whole window at once
    (modest memory multiple); GaussianCopula.transform time-chunked (see
    module docstring for why -- it's the actual memory wall).

    Explicitly drops each stage's input once its output exists: the caller
    must not keep its own reference to `ds` across this call (construct it
    as an inline argument expression, don't bind it to a variable) -- as
    long as any reference to a same-size object survives on the call
    stack, Python can't reclaim it, and holding `ds`/`anomalies`/
    `standardized` simultaneously (~3x the tensor) on top of even one
    chunked `copula.transform` call's own fixed per-call overhead (the LUT
    itself, `n_bins*n_cells` -- constant regardless of chunk size, unlike
    the query-side cost chunking actually reduces) is what first blew the
    memory cap here, not the chunking design itself.
    """
    anomalies = clim.transform(ds)
    del ds
    standardized = std.transform(anomalies)
    del anomalies
    n_times = standardized[variable].shape[0]
    out = np.empty(standardized[variable].shape, dtype=np.float32)
    for start in range(0, n_times, chunk_size):
        end = min(start + chunk_size, n_times)
        chunk_ds = standardized.isel(time=slice(start, end))
        gaussianized_chunk = copula.transform(chunk_ds)
        out[start:end] = gaussianized_chunk[variable].values.astype(np.float32)
        print(f"    transformed chunk [{start}:{end}] of {n_times}", file=sys.stderr)
    del standardized
    return out


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
    p.add_argument("--transform_chunk", type=int, default=250)
    p.add_argument("--mem_cap_gb", type=float, default=55.0)
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

    transformed_cache = out_dir / f"{args.variable}_gaussianized_{window_tag}.npy"
    mask_cache = out_dir / f"{args.variable}_tucker_mask_{window_tag}.npy"

    if transformed_cache.exists() and mask_cache.exists():
        print(f"Reusing cached transformed tensor: {transformed_cache}")
        z_full = np.load(transformed_cache)
        mask_int8 = np.load(mask_cache)
    else:
        ocean_mask = np.load(args.ocean_mask_path)
        transformers_dir = Path(args.transformers_dir)
        clim = joblib.load(transformers_dir / f"{args.variable}_climatology.joblib")
        std = joblib.load(transformers_dir / f"{args.variable}_standardize.joblib")
        copula = joblib.load(transformers_dir / f"{args.variable}_copula.joblib")

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

        print("Transforming (chunked copula)...")
        t0 = time.time()
        # ds_x necessarily stays alive in this frame for the whole call (Python
        # can't release an argument the caller still names) -- transform_chunked
        # drops ITS OWN internal copies (ds/anomalies) as it goes instead of also
        # holding all three at once, which is what actually blew the cap here.
        z_full = transform_chunked(clim, std, copula, ds_x, args.variable, args.transform_chunk)
        del ds_x
        print(f"  done in {time.time() - t0:.1f}s")

        z_full = np.where(np.isnan(z_full), 0.0, z_full)
        mask_int8 = tucker_mask

        np.save(transformed_cache, z_full)
        np.save(mask_cache, mask_int8)
        print(f"Cached transformed tensor to {transformed_cache}")

    rank = [args.r_lat, args.r_lon]
    tag = (
        f"gaussianized_{args.variable}_r{args.r_lat}x{args.r_lon}"
        f"_checkpointed{args.total_iters}_{window_tag}"
    )
    checkpoint_path = out_dir / f"{tag}_checkpoints.csv"
    with open(checkpoint_path, "w", newline="") as f:
        csv.writer(f).writerow(["iteration", "true_err_pct", "elapsed_s_cumulative"])

    init = "svd"
    done_iters = 0
    err_pct = float("nan")
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

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(
        f"RESULT gaussianized-checkpointed {args.variable} rank=({args.r_lat},{args.r_lon}) "
        f"err_pct={err_pct:.4f}% peak_rss_gb={peak_rss_gb:.2f} elapsed_s={elapsed:.1f} "
        f"n_iters={done_iters}"
    )

    tucker_path = out_dir / f"{tag}_tucker.npz"
    np.savez_compressed(tucker_path, core=core, factor_lat=factors[0], factor_lon=factors[1])
    print(f"Saved Tucker factors to {tucker_path}")


if __name__ == "__main__":
    main()
