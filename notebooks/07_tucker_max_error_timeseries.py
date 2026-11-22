"""Task 07: per-frame max |error| across the 9-month window, all three ranks.

Answers a specific follow-up to the skew/kurtosis findings (§5.1,
`docs/tucker_decomposition.md`): those measures are driven by a small
number of extreme cells -- this checks whether the WORST cell per snapshot
(over the full ocean mask, not just the coastal band) itself behaves with
real temporal structure (persists/clusters -- i.e. real storm events) or
looks like unstructured noise from snapshot to snapshot.

Two diagnostics, not just a plot to eyeball:
1. Lag-1 autocorrelation of the max-|error| time series, per rank -- the
   standard test for "is this structured or i.i.d. noise." Near 0 = no
   persistence frame-to-frame (consistent with noise); large and positive
   = the worst cell's magnitude persists across consecutive snapshots
   (consistent with real, multi-snapshot-duration events like storms).
2. The distribution of those 548 per-frame max values, per rank (a
   histogram) -- shows whether it has a stable, describable shape or not.

Uses the same 548-frame true-field cache and per-frame on-the-fly
reconstruction as `07_tucker_rank_timelapse_gif.py` (memory-conscious --
never holds all frames' reconstructions at once).

Run directly: `poetry run python notebooks/07_tucker_max_error_timeseries.py`.
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
WINDOW_START = datetime(2024, 1, 1)

RANK_PATHS = {
    "20%": GAUSS_DIR / "anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz",
    "30%": GAUSS_DIR / "anomalies_wsh_r186x432_checkpointed80_202401-202410_tucker.npz",
    "40%": GAUSS_DIR / "anomalies_wsh_r248x576_checkpointed80_202401-202410_tucker.npz",
}
RANK_COLORS = {"20%": "tab:blue", "30%": "tab:orange", "40%": "tab:green"}


def lag1_autocorr(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def main() -> None:
    true_bundle = np.load(GAUSS_DIR / "timelapse_true_slices.npz")
    time_indices, true_slices = true_bundle["time_indices"], true_bundle["true"]
    n_frames = len(time_indices)

    ocean_mask = np.load(GAUSS_DIR / "era2_ocean_mask.npy")
    land = ~ocean_mask

    ranks = {label: (np.load(p)["core"], np.load(p)["factor_lat"], np.load(p)["factor_lon"]) for label, p in RANK_PATHS.items()}

    max_err = {label: np.empty(n_frames, dtype=np.float64) for label in ranks}
    for i, t in enumerate(time_indices):
        tf = np.where(land, np.nan, true_slices[i])
        for label, (core, fl, fo) in ranks.items():
            recon = np.where(land, np.nan, fl @ core[int(t)] @ fo.T)
            max_err[label][i] = np.nanmax(np.abs(tf - recon))

    dates = [WINDOW_START + timedelta(hours=3 * int(t)) for t in time_indices]

    print("Per-rank max-|error| time series stats:")
    for label, vals in max_err.items():
        ac1 = lag1_autocorr(vals)
        print(
            f"  {label}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} "
            f"max={vals.max():.3f} lag-1 autocorr={ac1:.3f}"
        )

    fig, (ax_ts, ax_hist) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    for label in ranks:
        ax_ts.plot(dates, max_err[label], label=label, color=RANK_COLORS[label], linewidth=1)
    ax_ts.set_ylabel("max |error| over ocean\n(standardized-anomaly units)")
    ax_ts.set_title("Worst-cell error per snapshot, across the 9-month window")
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)

    bins = np.linspace(
        min(v.min() for v in max_err.values()), max(v.max() for v in max_err.values()), 40
    )
    for label in ranks:
        ax_hist.hist(max_err[label], bins=bins, alpha=0.5, label=label, color=RANK_COLORS[label])
    ax_hist.set_xlabel("max |error| over ocean (standardized-anomaly units)")
    ax_hist.set_ylabel("count (out of 548 snapshots)")
    ax_hist.set_title("Distribution of the per-snapshot worst-cell error")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    out_path = GAUSS_DIR / "tucker_max_error_timeseries.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
