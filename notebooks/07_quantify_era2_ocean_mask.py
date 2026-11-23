"""Task 07 Gaussianization round, step 1: quantify the ocean mask over the
full 2-year era-2 span for `wsh` (approved plan, Phase B execution order
step 3).

`land_mask_from_sigwh` defaults to `valid_frac=1.0` -- any cell NaN even
once counts as land. Over 2 years of seasonal ice migration this excludes
materially more area than era 3's ~9 months did; the ocean-cell count here
is part of this experiment's definition, not incidental.

Computed month-by-month (25 months, clipped to era 2's actual boundaries:
2023-10-02 -> 2025-10-27) with a running logical-AND, so no more than one
month's `sigwh` (~621x1440x~240 float32, ~826MB) is ever resident at once
-- `load_training_window` has no stride/chunking of its own (upcasts
float16 -> float64 and loads the whole requested window eagerly), and the
full unstrided span is needed here (unlike the fit window in
`07_gaussianize_fit.py`, which strides): missing a transient NaN at a
skipped timestep would let a cell through as "ocean" that later produces
NaN during the actual Tucker window, exactly the leak this task's
composition is designed to avoid.

Saves the mask to `--out_dir` for `07_gaussianize_fit.py` and
`07_gaussianized_rank_sweep.py` to load rather than recomputing. Run
directly: `poetry run python notebooks/07_quantify_era2_ocean_mask.py`.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from core.config import canonical_dims
from train.ingest import load_training_window
from train.masking import land_mask_from_sigwh

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
    """Calendar-month bounds, clipped to era 2's actual span at the edges."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (
        datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if month == 12
        else datetime(year, month + 1, 1, tzinfo=timezone.utc)
    )
    return max(start, ERA2_START), min(end, ERA2_END)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    args = p.parse_args()

    running_ocean_mask = None
    total_snapshots = 0

    for year, month in _MONTHS:
        t_start, t_end = month_bounds(year, month)
        if t_start >= t_end:
            continue
        ds = load_training_window(DATA_DIR, t_start, t_end, ["sigwh"])
        sigwh = ds["sigwh"].transpose(*canonical_dims("wave")).values.astype(np.float32)
        del ds
        n_t = sigwh.shape[0]
        total_snapshots += n_t
        month_mask = land_mask_from_sigwh(sigwh, valid_frac=1.0)  # True == ocean
        del sigwh
        running_ocean_mask = (
            month_mask if running_ocean_mask is None else running_ocean_mask & month_mask
        )
        ocean_so_far = int(running_ocean_mask.sum())
        print(
            f"{year}-{month:02d}: {n_t} snapshots, cumulative ocean cells so far: "
            f"{ocean_so_far} ({100 * ocean_so_far / running_ocean_mask.size:.2f}%)",
            file=sys.stderr,
        )

    assert running_ocean_mask is not None
    n_ocean = int(running_ocean_mask.sum())
    n_total = running_ocean_mask.size
    print(f"Total snapshots processed: {total_snapshots}")
    print(
        f"RESULT era2-ocean-mask valid_frac=1.0 n_ocean={n_ocean} n_total={n_total} "
        f"ocean_pct={100 * n_ocean / n_total:.2f} shape={running_ocean_mask.shape}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "era2_ocean_mask.npy"
    np.save(out_path, running_ocean_mask)
    print(f"Saved ocean mask to {out_path}")


if __name__ == "__main__":
    main()
