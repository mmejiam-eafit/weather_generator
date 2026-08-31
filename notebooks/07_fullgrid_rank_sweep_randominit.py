"""Task 07 follow-on: full (721x1440) wave-grid Tucker rank sweep, `init="random"`.

Standalone script, not a notebook cell -- masked HOOI on the full grid runs
long enough (~114s/iter observed at rank (10,10), `n_iter_max=300` default)
that an isolated, killable subprocess with an RLIMIT_AS memory cap is safer
than blocking a live kernel. Confirmed the full-grid fit is *runtime*-bound,
not memory-bound (peak RSS ~3-6GB vs. crop's ~12GB) -- see
`07_fullgrid_rank_sweep_svdinit.py` for the faster-converging variant this
finding motivated. Investigation paused mid-sweep (rank (144,288), 20% of
full grid, requested to compare against the crop-based sweep in
`07_tucker_dmd_rank_selection.ipynb`) -- not yet resolved into task 07's
own Conclusion. Run directly: `poetry run python
notebooks/07_fullgrid_rank_sweep_randominit.py --r_lat N --r_lon N`.
"""

import argparse
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker

from train.ingest import load_training_window
from train.masking import apply_partition_sentinel_policy, height_log_energy, land_mask_from_sigwh

tl.set_backend("numpy")

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
T_START = datetime(2025, 11, 1, tzinfo=timezone.utc)
T_END = datetime(2025, 11, 13, tzinfo=timezone.utc)  # 12 days, ~96 3-hourly steps -- full grid, no crop


def load_var(name):
    ds = load_training_window(DATA_DIR, T_START, T_END, [name])
    arr = ds[name].transpose("wave_lat", "wave_lon", "time").values.astype(np.float32)
    del ds
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r_lat", type=int, required=True)
    p.add_argument("--r_lon", type=int, required=True)
    p.add_argument("--mem_cap_gb", type=float, default=26.0)
    args = p.parse_args()

    cap_bytes = int(args.mem_cap_gb * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    sigwh = load_var("sigwh")
    print(f"full grid shape: {sigwh.shape}", flush=True)
    land_mask = land_mask_from_sigwh(sigwh)
    print(f"land_mask ocean frac: {land_mask.mean():.4f}", flush=True)
    del sigwh

    wsh = load_var("wsh")
    sentinel = apply_partition_sentinel_policy(wsh, land_mask, sentinel_value=0.0)
    del wsh
    x = height_log_energy(sentinel.filled, floor_m=0.01)
    mask = sentinel.tucker_mask

    (core, factors), _ = partial_tucker(
        x, modes=[0, 1, 2], rank=[args.r_lat, args.r_lon, x.shape[-1]],
        init="random", tol=1e-5, n_iter_max=300, random_state=SEED, mask=mask,
    )
    recon = tl.tucker_to_tensor((core, factors))
    err = np.linalg.norm((recon - x) * mask) / np.linalg.norm(x * mask)

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"RESULT fullgrid wsh rank=({args.r_lat},{args.r_lon}) err={err:.4f} peak_rss_gb={peak_rss_gb:.2f}")


if __name__ == "__main__":
    main()
