"""Task 07 follow-on: checkpointed full-grid Tucker rank sweep.

`partial_tucker` (tensorly 0.9.0) has no callback hook, and its own internal
`rec_errors` curve is unusable under masking -- it bounces off zero once
`norm(core)` exceeds the once-computed `norm_tensor` (self-imputation of
masked cells lets `core`'s norm grow past that fixed baseline; the formula's
`abs()` then reflects the curve back upward). See
`results/task07/` and the corresponding chart for the full diagnosis.

This script checkpoints the *trustworthy* metric instead -- the same
post-hoc, masked-only `err = ||(recon-x)*mask|| / ||x*mask||` used for every
final result this task has reported -- every `--checkpoint_every` iterations,
by calling `partial_tucker` in chunks and feeding each chunk's returned
`(core, factors)` back in as `init` for the next. This resumes the HOOI
iteration exactly (deterministic given a fixed `random_state`) without
repaying the one-time SVD-init cost more than once.

Uses `svd="randomized_svd"` rather than tensorly's default `"truncated_svd"`.
The default always computes a *full* economy SVD of the unfolded tensor
before slicing to the requested rank -- fine at short windows, but at the
full 9-month window (T~2184) the mode-0/1 unfold during *initialization*
(before any factor exists to contract the other axes down) is
(721, 1440*2184): materializing that decomposition needs 17GB+ for one
factor alone and OOMs even a 62GB VM. `randomized_svd` (Halko et al.)
never materializes the full decomposition -- only the per-iteration HOOI
updates are cheap regardless of `svd=`, since by then the tensor has
already been contracted through the other factors to a much smaller
unfold.

Also passes `svd_mask_repeats=0`. Independent of the SVD method,
`svd_interface`'s mask-imputation *polish* -- which repeats (default 5x)
`matrix = matrix*mask + (U@St@V)*(1-mask)` on the still-uncontracted
(721, 1440*2184)-shaped unfold during init -- needs several full-size
temporaries at once (>50GB) and OOMs the VM's 62GB regardless of the cap.
This polish is redundant here: `x` is already sentinel-filled by
`apply_partition_sentinel_policy` before it ever reaches `partial_tucker`,
and the *main* HOOI loop already re-imputes masked cells from the current
core/factors estimate every single iteration (600x over a full run) --
losing 5 extra rounds of it at init only, before any factors even exist,
is a rounding error by comparison.

Decomposes only `modes=[0, 1]` (lat, lon) -- NOT `modes=[0, 1, 2]` as
earlier full-grid scripts did. Passing mode 2 (time) with `rank[2] ==
x.shape[-1]` (the full length, since we never intend to compress time)
asked HOOI for a *full-rank* factor on that axis every iteration, which is
a mathematical no-op (a full-rank orthonormal factor changes nothing on
reconstruction -- confirmed empirically: identical error with 2 vs 3
factors on a synthetic case) but not a computational no-op: it's a
near-full SVD of a (lat*lon, T) matrix. At T=96 this was cheap enough to
go unnoticed; at T~2184 it forced `numpy.linalg.qr` to upcast a
(1,038,240, 2184) array to float64 (17GB) inside the "randomized" SVD's
own power iteration, since requesting essentially-full rank defeats the
whole point of a randomized/truncated method. Dropping mode 2 entirely
(tensorly's `multi_mode_dot` leaves un-listed modes untouched on
reconstruction) removes this cost completely with no change in result.

Also casts `mask` to `int8` and drops the `sentinel` object once `x` and
`mask` are extracted from it. Every main-loop iteration reconstructs the
full tensor once (`tensor*mask + recon*(1-mask)`, needed regardless of any
of the fixes above -- this is the actual per-iteration self-imputation
step, not incidental waste), which needs several 8.45GB (721x1440x2184,
float32) temporaries alive at once; at T~2184 that recurring cost is what
finally exceeds the VM's 62GB even after every fix above. `tucker_mask` is
just a 0/1 indicator (`apply_partition_sentinel_policy` stores it as
float32 only because tensorly's own `1 - mask` can't operate on a bool
array); storing it as `int8` instead costs 1 byte/element instead of 4
with no change to any `mask`-involving arithmetic (numpy promotes int8 to
match the float32 operand it's combined with). `sentinel.filled` is a
redundant copy of what became `x` and was never freed.

Saves the final `core`/`factors` to `{tag}_tucker.npz` at the end of the
run. Earlier runs only ever persisted the scalar error curve -- `core` and
`factors` (the only inputs `tl.tucker_to_tensor` needs to reconstruct the
full tensor) lived in-process and were discarded on exit, so there was no
way to reconstruct and visually compare against the real data after the
fact. Both arrays are small relative to the raw tensor (e.g. at 20% rank,
`core` is ~362MB vs. the 8.45GB raw `x`), so saving them is effectively
free next to the run's own compute cost. Run directly:
`poetry run python notebooks/07_fullgrid_rank_sweep_checkpointed.py --r_lat N
--r_lon N --total_iters N --checkpoint_every 12`.
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

from train.ingest import load_training_window
from train.masking import apply_partition_sentinel_policy, height_log_energy, land_mask_from_sigwh

tl.set_backend("numpy")

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_T_START = "2025-11-01"
DEFAULT_T_END = "2025-11-13"


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def load_var(name, t_start, t_end):
    ds = load_training_window(DATA_DIR, t_start, t_end, [name])
    arr = ds[name].transpose("wave_lat", "wave_lon", "time").values.astype(np.float32)
    del ds
    return arr


def true_masked_err(x, mask, core, factors):
    recon = tl.tucker_to_tensor((core, factors))
    return float(np.linalg.norm((recon - x) * mask) / np.linalg.norm(x * mask))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r_lat", type=int, required=True)
    p.add_argument("--r_lon", type=int, required=True)
    p.add_argument("--mem_cap_gb", type=float, default=26.0)
    p.add_argument("--total_iters", type=int, required=True)
    p.add_argument("--checkpoint_every", type=int, default=12)
    p.add_argument("--t_start", type=str, default=DEFAULT_T_START, help="ISO date, e.g. 2025-11-01")
    p.add_argument("--t_end", type=str, default=DEFAULT_T_END, help="ISO date, e.g. 2025-11-13")
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "task07"))
    args = p.parse_args()
    t_start = parse_date(args.t_start)
    t_end = parse_date(args.t_end)

    cap_bytes = int(args.mem_cap_gb * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    sigwh = load_var("sigwh", t_start, t_end)
    land_mask = land_mask_from_sigwh(sigwh)
    del sigwh

    wsh = load_var("wsh", t_start, t_end)
    sentinel = apply_partition_sentinel_policy(wsh, land_mask, sentinel_value=0.0)
    del wsh
    x = height_log_energy(sentinel.filled, floor_m=0.01)
    mask = sentinel.tucker_mask.astype(np.int8)
    del sentinel
    rank = [args.r_lat, args.r_lon]

    window_tag = f"{t_start:%Y%m}-{t_end:%Y%m}"
    tag = f"fullgrid_r{args.r_lat}x{args.r_lon}_checkpointed{args.total_iters}_{window_tag}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{tag}_checkpoints.csv"
    with open(checkpoint_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "true_err", "elapsed_s_cumulative"])

    init = "svd"
    done_iters = 0
    t0 = time.time()
    while done_iters < args.total_iters:
        chunk = min(args.checkpoint_every, args.total_iters - done_iters)
        (core, factors), _ = partial_tucker(
            x, modes=[0, 1], rank=rank,
            init=init, tol=1e-5, n_iter_max=chunk, random_state=SEED, mask=mask,
            svd="randomized_svd", svd_mask_repeats=0,
        )
        init = (core, factors)
        done_iters += chunk
        elapsed = time.time() - t0
        err = true_masked_err(x, mask, core, factors)
        with open(checkpoint_path, "a", newline="") as f:
            csv.writer(f).writerow([done_iters, f"{err:.5f}", f"{elapsed:.1f}"])
        print(f"CHECKPOINT rank=({args.r_lat},{args.r_lon}) iter={done_iters} "
              f"true_err={err:.4f} elapsed_s={elapsed:.1f}")

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"RESULT fullgrid-checkpointed wsh rank=({args.r_lat},{args.r_lon}) err={err:.4f} "
          f"peak_rss_gb={peak_rss_gb:.2f} elapsed_s={elapsed:.1f} n_iters={done_iters}")

    tucker_path = out_dir / f"{tag}_tucker.npz"
    np.savez_compressed(tucker_path, core=core, factor_lat=factors[0], factor_lon=factors[1])
    print(f"Saved Tucker factors to {tucker_path}")


if __name__ == "__main__":
    main()
