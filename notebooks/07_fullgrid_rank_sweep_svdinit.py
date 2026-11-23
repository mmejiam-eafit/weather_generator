"""Task 07 follow-on: full (721x1440) wave-grid Tucker rank sweep, `init="svd"`.

Companion to `07_fullgrid_rank_sweep_randominit.py` -- built after the
random-init variant, at the same rank (10,10), spent 300+ iterations
without converging in any tractable time. Incremental `n_iter_max` timing
(5/10/20 iterations, rank (10,10)) found the cost is front-loaded: ~500s
one-time SVD-initialization step, then only ~10-20s/iteration after, with
error already near its floor by iteration ~10 (0.7107->0.7068->0.7039).
This is why `n_iter_max` is exposed as a CLI flag here (unlike the
random-init script's hardcoded 300) -- bound it explicitly rather than
assume `tol=1e-5` will trigger early stopping on this problem. Investigation
paused mid-run at rank (144,288) (`n_iter_max=30`, ~16min elapsed when
killed) -- not yet resolved into task 07's own Conclusion; also still open
whether the full grid's higher error at matched rank fractions vs. the
crop sweep in `07_tucker_dmd_rank_selection.ipynb` is a genuine spatial-
heterogeneity effect or a T=96-vs-T=1448 window-length confound (planned
next step: rerun the crop at T=96 to isolate it). Run directly:
`poetry run python notebooks/07_fullgrid_rank_sweep_svdinit.py --r_lat N
--r_lon N --n_iter_max N`.
"""

import argparse
import csv
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot

from core.config import canonical_dims
from train.ingest import load_training_window
from train.masking import apply_partition_sentinel_policy, height_log_energy, land_mask_from_sigwh

tl.set_backend("numpy")

SEED = 42
SPATIAL_MODES = [1, 2]  # (time, lat, lon) -- time (mode 0) stays full-rank
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
T_START = datetime(2025, 11, 1, tzinfo=timezone.utc)
T_END = datetime(2025, 11, 13, tzinfo=timezone.utc)
CSV_FIELDS = [
    "timestamp", "tag", "config", "r_lat", "r_lon", "init", "n_iter_max",
    "elapsed_s", "peak_rss_gb", "n_iters", "final_err",
]


def log_result(out_dir, tag, config, r_lat, r_lon, init, n_iter_max, elapsed, peak_rss_gb, rec_errors, final_err):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rec_errors is not None:
        np.save(out_dir / f"{tag}_rec_errors.npy", np.asarray(rec_errors))
    csv_path = out_dir / "log.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": tag, "config": config, "r_lat": r_lat, "r_lon": r_lon,
            "init": init, "n_iter_max": n_iter_max, "elapsed_s": f"{elapsed:.1f}",
            "peak_rss_gb": f"{peak_rss_gb:.2f}",
            "n_iters": len(rec_errors) if rec_errors is not None else -1,
            "final_err": f"{final_err:.4f}",
        })


def load_var(name):
    ds = load_training_window(DATA_DIR, T_START, T_END, [name])
    arr = ds[name].transpose(*canonical_dims("wave")).values.astype(np.float32)
    del ds
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r_lat", type=int, required=True)
    p.add_argument("--r_lon", type=int, required=True)
    p.add_argument("--mem_cap_gb", type=float, default=26.0)
    p.add_argument("--n_iter_max", type=int, default=300)
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
        x, modes=SPATIAL_MODES, rank=[args.r_lat, args.r_lon],
        init="svd", tol=1e-5, n_iter_max=args.n_iter_max, random_state=SEED, mask=mask,
    )
    elapsed = time.time() - t0
    recon = multi_mode_dot(core, factors, modes=SPATIAL_MODES)
    err = np.linalg.norm((recon - x) * mask) / np.linalg.norm(x * mask)

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    n_iters = len(rec_errors) if rec_errors is not None else -1
    print(f"RESULT fullgrid-svdinit wsh rank=({args.r_lat},{args.r_lon}) err={err:.4f} "
          f"peak_rss_gb={peak_rss_gb:.2f} elapsed_s={elapsed:.1f} n_iters={n_iters}")

    tag = f"fullgrid_r{args.r_lat}x{args.r_lon}_iter{args.n_iter_max}"
    log_result(
        args.out_dir, tag, "fullgrid-svdinit-wsh", args.r_lat, args.r_lon,
        "svd", args.n_iter_max, elapsed, peak_rss_gb, rec_errors, err,
    )


if __name__ == "__main__":
    main()
