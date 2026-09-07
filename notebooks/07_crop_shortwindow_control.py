"""Task 07 follow-on: crop+short-window control experiment (plan §2, step 1).

Isolates whether the full grid's much higher reconstruction error at
matched rank fractions (vs. `07_tucker_dmd_rank_selection.ipynb`'s crop
sweep) is a genuine spatial-heterogeneity effect or an artifact of the
12-day/T=96 window the full-grid attempts used (the crop sweep itself runs
on a 6-month window). Reruns the existing 280x230 crop
(`CROP = (-55.75, 14.0, 211.0, 268.25)`, from
`07_tucker_dmd_rank_selection.ipynb`) restricted to the same 12-day window
as `07_fullgrid_rank_sweep_svdinit.py`, at the same small rank (10,10), so
the two results are directly comparable. Run directly:
`poetry run python notebooks/07_crop_shortwindow_control.py --r_lat N
--r_lon N --n_iter_max N`.
"""

import argparse
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker

from train.ingest import load_training_window
from train.masking import apply_partition_sentinel_policy, height_log_energy, land_mask_from_sigwh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_fullgrid = import_module("07_fullgrid_rank_sweep_svdinit")
log_result = _fullgrid.log_result

tl.set_backend("numpy")

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
T_START = datetime(2025, 11, 1, tzinfo=timezone.utc)
T_END = datetime(2025, 11, 13, tzinfo=timezone.utc)
CROP = (-55.75, 14.0, 211.0, 268.25)  # lat_min, lat_max, lon_min, lon_max -- 280x230, 0% NaN for sigwh


def load_var(name):
    ds = load_training_window(DATA_DIR, T_START, T_END, [name])
    lat_min, lat_max, lon_min, lon_max = CROP
    ds = ds.sel(wave_lat=slice(lat_min, lat_max), wave_lon=slice(lon_min, lon_max))
    arr = ds[name].transpose("wave_lat", "wave_lon", "time").values.astype(np.float32)
    del ds
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r_lat", type=int, required=True)
    p.add_argument("--r_lon", type=int, required=True)
    p.add_argument("--mem_cap_gb", type=float, default=26.0)
    p.add_argument("--n_iter_max", type=int, default=40)
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "task07"))
    args = p.parse_args()

    cap_bytes = int(args.mem_cap_gb * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    sigwh = load_var("sigwh")
    land_mask = land_mask_from_sigwh(sigwh)
    del sigwh

    wsh = load_var("wsh")
    sentinel = apply_partition_sentinel_policy(wsh, land_mask, sentinel_value=0.0)
    del wsh
    x = height_log_energy(sentinel.filled, floor_m=0.01)
    mask = sentinel.tucker_mask

    t0 = time.time()
    (core, factors), rec_errors = partial_tucker(
        x, modes=[0, 1, 2], rank=[args.r_lat, args.r_lon, x.shape[-1]],
        init="svd", tol=1e-5, n_iter_max=args.n_iter_max, random_state=SEED, mask=mask,
    )
    elapsed = time.time() - t0
    recon = tl.tucker_to_tensor((core, factors))
    err = np.linalg.norm((recon - x) * mask) / np.linalg.norm(x * mask)

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    n_iters = len(rec_errors) if rec_errors is not None else -1
    print(f"RESULT crop-shortwindow-control wsh rank=({args.r_lat},{args.r_lon}) err={err:.4f} "
          f"peak_rss_gb={peak_rss_gb:.2f} elapsed_s={elapsed:.1f} n_iters={n_iters}")

    tag = f"cropcontrol_r{args.r_lat}x{args.r_lon}_iter{args.n_iter_max}"
    log_result(
        args.out_dir, tag, "crop-shortwindow-control-wsh", args.r_lat, args.r_lon,
        "svd", args.n_iter_max, elapsed, peak_rss_gb, rec_errors, err,
    )


if __name__ == "__main__":
    main()
