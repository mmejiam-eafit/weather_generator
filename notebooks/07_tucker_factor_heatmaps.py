"""Task 07: render a saved Tucker bundle's two factor matrices as static heatmaps.

Loads `factor_lat` and `factor_lon` from a `(core, factor_lat, factor_lon)` bundle
written by the rank-sweep scripts (`07_placement2_tucker_on_anomalies.py` /
`07_fullgrid_rank_sweep_checkpointed.py`) and saves one PNG per factor. Each factor
has one row per grid line and one column per Tucker mode (orthonormal columns), so
the image shows which latitudes / longitudes each mode loads on: low modes are
smooth, high modes oscillate. `core` (~1 GB) is never read -- `np.load` on an
`.npz` is lazy per key.

The diverging color scale is symmetric about zero. Its limit is the 99.5th
percentile of |loading| so a few large entries do not wash out the rest. The
sign of a mode is arbitrary (the decomposition is only defined up to a sign flip
per column), so read the pattern and not the colour polarity.

Latitude is labelled with era 2's wave-grid origin (`lat0=-75.0`, 621 rows, true
range [-75, 80]), which is what the 2024 training window uses. `core.config.
grid_coordinates` would label this grid [-90, 65] -- see its module docstring.

Run directly: `poetry run python notebooks/07_tucker_factor_heatmaps.py
--tucker_npz results/task07/gaussianize/anomalies_wsh_r248x576_checkpointed300_202401-202410_tucker.npz
--label 40pct`.
"""

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

ERA2_LAT_AT_ROW0 = -75.0
GRID_STEP_DEG = 0.25
LOADING_CLIP_PERCENTILE = 99.5
# Axes must be >= 1440 px tall (factor_lon rows) so no cell is resampled: blending
# the alternating +/- high-mode entries paints purple, which is not in the colormap.
SAVE_DPI = 300


def plot_factor(
    factor: np.ndarray,
    axis_name: str,
    origin_deg: float,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Save one factor matrix (rows = grid lines, columns = modes) as a heatmap."""
    n_rows, n_modes = factor.shape
    f64 = factor.astype(np.float64)
    vmax = float(np.percentile(np.abs(f64), LOADING_CLIP_PERCENTILE))

    lo = origin_deg - GRID_STEP_DEG / 2
    hi = origin_deg + (n_rows - 1) * GRID_STEP_DEG + GRID_STEP_DEG / 2
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    im = ax.imshow(
        f64,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, n_modes + 0.5, lo, hi),
    )
    ax.set_xlabel("Tucker mode (factor column)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"{axis_name} loading (clipped at ±{vmax:.3f}, p{LOADING_CLIP_PERCENTILE:g})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tucker_npz", type=str, required=True)
    p.add_argument(
        "--label", type=str, required=True, help="e.g. '40pct', used in filenames/titles"
    )
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    tucker = np.load(args.tucker_npz)
    factor_lat, factor_lon = tucker["factor_lat"], tucker["factor_lon"]

    for name, factor in (("factor_lat", factor_lat), ("factor_lon", factor_lon)):
        f64 = factor.astype(np.float64)
        ortho_err = float(np.max(np.abs(f64.T @ f64 - np.eye(f64.shape[1]))))
        print(f"RESULT {name} shape={factor.shape} max|FtF - I|={ortho_err:.2e}")

    out_dir = Path(args.out_dir)
    plot_factor(
        factor_lat,
        "latitude",
        ERA2_LAT_AT_ROW0,
        "Latitude (°N)",
        f"Tucker rank {args.label}: factor_lat {factor_lat.shape} (wsh, era-2 grid)",
        out_dir / f"tucker_factor_lat_{args.label}.png",
    )
    plot_factor(
        factor_lon,
        "longitude",
        0.0,
        "Longitude (°E)",
        f"Tucker rank {args.label}: factor_lon {factor_lon.shape} (wsh, era-2 grid)",
        out_dir / f"tucker_factor_lon_{args.label}.png",
    )


if __name__ == "__main__":
    main()
