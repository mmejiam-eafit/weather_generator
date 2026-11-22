"""Task 07 Gaussianization round, step 2: fit ClimatologyRemover ->
PerCellStandardizer -> GaussianCopula on a strided 2-year era-2 window
(approved plan, Phase B design + execution order step 5).

Composition (see the plan and `notebooks/07_gaussianize_fit.py`'s sibling
smoke test that validated this ordering on a crop first):
    raw wsh (time, lat, lon)
      -> sentinel-fill OCEAN NaN only, LAND STAYS NaN (land must stay NaN
         for the transformers below -- ClimatologyRemover/
         PerCellStandardizer/GaussianCopula all handle an all-NaN column
         correctly, but zero-filling land would hand PerCellStandardizer
         a constant column -> sigma=0)
      -> height_log_energy
      -> ClimatologyRemover.fit -> PerCellStandardizer.fit -> GaussianCopula.fit

Why strided: fitting climatology on the FULL era-2 span at 3-hourly
cadence would need `load_training_window` to hold ~43GB per variable
(float16 on disk upcast to float64, no stride/chunking support in
`load_training_window` itself). Loading month-by-month and slicing each
month's array with `--stride` after it's in RAM keeps peak memory to one
month at a time (~826MB at float32) while still covering the full 2-year
span the fit needs to see a real annual cycle. Default stride=4 (12-hourly)
gives ~1,512 fit snapshots at 5.4GB total, resident only once (accumulated
via `np.concatenate`, then discarded once fitting starts).

Peak memory during fitting itself (measured this session): ~2-4x the
5.4GB fit tensor per transformer stage, well within a 62GB VM.
`GaussianCopula.fit` is the slow step (~10 min/variable at 894,240 cells
-- `np.nanquantile(axis=0)` is a Python-level per-cell loop), not a memory
problem -- the copula's *transform* is the memory wall, handled by
`07_gaussianized_rank_sweep.py`'s time-chunking, not here.

Requires `notebooks/07_quantify_era2_ocean_mask.py` to have already run
(reads its saved ocean mask rather than recomputing it). Run directly:
`poetry run python notebooks/07_gaussianize_fit.py --ocean_mask_path
results/task07/gaussianize/era2_ocean_mask.npy`.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import xarray as xr

from core.config import canonical_dims
from train.climatology import ClimatologyRemover
from train.copula import GaussianCopula
from train.ingest import load_training_window
from train.masking import height_log_energy
from train.standardize import PerCellStandardizer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ERA2_START = datetime(2023, 10, 2, tzinfo=timezone.utc)
ERA2_END = datetime(2025, 10, 27, tzinfo=timezone.utc)

_MONTHS = [
    (2023, 10), (2023, 11), (2023, 12),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5), (2024, 6),
    (2024, 7), (2024, 8), (2024, 9), (2024, 10), (2024, 11), (2024, 12),
    (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5), (2025, 6),
    (2025, 7), (2025, 8), (2025, 9), (2025, 10),
]


def month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (
        datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if month == 12
        else datetime(year, month + 1, 1, tzinfo=timezone.utc)
    )
    return max(start, ERA2_START), min(end, ERA2_END)


def ocean_only_sentinel_fill(
    x: np.ndarray, ocean_mask: np.ndarray, sentinel_value: float = 0.0
) -> tuple[np.ndarray, int]:
    """Fill ocean NaN with sentinel_value; LEAVE LAND NaN untouched.

    `ocean_mask` follows `land_mask_from_sigwh`'s convention: True == ocean.

    Distinct from `train.masking.apply_partition_sentinel_policy`, which
    also zero-fills land -- wrong here, since the transformers below must
    see land as NaN (see module docstring).
    """
    nan_mask = np.isnan(x)
    ocean_3d = np.broadcast_to(ocean_mask[None, :, :], x.shape)
    filled = np.where(nan_mask & ocean_3d, sentinel_value, x)
    n_ocean_nan = int(np.sum(nan_mask & ocean_3d))
    return filled, n_ocean_nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variable", type=str, default="wsh")
    p.add_argument("--stride", type=int, default=4, help="snapshots per fit sample (4 = 12-hourly)")
    p.add_argument("--n_bins", type=int, default=500)
    p.add_argument("--ocean_mask_path", type=str, required=True)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    args = p.parse_args()

    ocean_mask = np.load(args.ocean_mask_path)
    print(
        f"ocean mask: {ocean_mask.sum()}/{ocean_mask.size} cells "
        f"({100 * ocean_mask.mean():.2f}%)"
    )

    chunks = []
    time_chunks = []
    t0 = time.time()
    for year, month in _MONTHS:
        t_start, t_end = month_bounds(year, month)
        if t_start >= t_end:
            continue
        ds = load_training_window(DATA_DIR, t_start, t_end, [args.variable])
        arr = ds[args.variable].transpose(*canonical_dims("wave")).values.astype(np.float32)
        times = ds["time"].values
        del ds
        arr_strided = arr[:: args.stride]
        times_strided = times[:: args.stride]
        del arr

        filled, n_ocean_nan = ocean_only_sentinel_fill(arr_strided, ocean_mask, sentinel_value=0.0)
        del arr_strided
        x = height_log_energy(filled, floor_m=0.01)
        del filled
        chunks.append(x)
        time_chunks.append(times_strided)
        print(
            f"{year}-{month:02d}: {x.shape[0]} strided snapshots, n_ocean_nan={n_ocean_nan}, "
            f"elapsed={time.time() - t0:.1f}s",
            file=sys.stderr,
        )

    x_full = np.concatenate(chunks, axis=0)
    time_full = np.concatenate(time_chunks, axis=0)
    del chunks, time_chunks
    print(f"fit tensor shape: {x_full.shape}, {x_full.nbytes / 1e9:.2f} GB")

    lat_dim, lon_dim = canonical_dims("wave")[1], canonical_dims("wave")[2]
    ds_fit = xr.Dataset(
        {args.variable: (("time", lat_dim, lon_dim), x_full)}, coords={"time": time_full}
    )
    del x_full

    print("Fitting ClimatologyRemover...")
    t1 = time.time()
    clim = ClimatologyRemover()
    clim.fit(ds_fit)
    anomalies = clim.transform(ds_fit)
    print(f"  done in {time.time() - t1:.1f}s")

    print("Fitting PerCellStandardizer...")
    t1 = time.time()
    std = PerCellStandardizer()
    std.fit(anomalies)
    standardized = std.transform(anomalies)
    print(f"  done in {time.time() - t1:.1f}s")

    print("Fitting GaussianCopula (slow step: per-cell nanquantile)...")
    t1 = time.time()
    copula = GaussianCopula(n_bins=args.n_bins)
    copula.fit(standardized)
    print(f"  done in {time.time() - t1:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clim, out_dir / f"{args.variable}_climatology.joblib")
    joblib.dump(std, out_dir / f"{args.variable}_standardize.joblib")
    joblib.dump(copula, out_dir / f"{args.variable}_copula.joblib")
    print(f"Saved fitted transformers to {out_dir}")
    print(f"RESULT gaussianize-fit {args.variable} n_fit_snapshots={ds_fit.sizes['time']} "
          f"elapsed_s={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
