"""Task 07: animate the Tucker reconstruction as the matrix product that defines it.

For a bundle `(core, factor_lat, factor_lon)` with time as the undecomposed mode 0
(`modes=[1, 2]` throughout this task), the reconstructed frame at time index `t` is

    frame[t] = factor_lat @ core[t] @ factor_lon.T
               (621x248)    (248x576)   (576x1440)   ->   (621x1440)

Each GIF frame draws that equation left to right, with every matrix at the same
inches-per-index scale so the relative sizes are true: the two factors are fixed
for the whole animation, only `core[t]` changes, and the reconstructed map on the
right follows it. Land cells (per the static era-2 ocean mask) are shown in flat
gray, not as data, so a calm-sea value near zero is never confused with land.

The factor panels and the core are drawn nearest-neighbour (each shown pixel is a
real entry, as in `07_tucker_factor_heatmaps.py`); the map uses matplotlib's
default antialiased resampling like the sibling timelapse scripts. Factors are
orthonormal-column and stored float32; the product is computed in float64.

Time index `t` is 3-hourly from 2024-01-01 00:00 UTC (274 days x 8 = 2192 frames),
so `--stride 4` gives the same 00/12 UTC cadence as `07_tucker_rank_timelapse_gif.py`.
Only the selected core slices are kept in memory (the full core is ~1.25 GB).

Run directly: `poetry run python notebooks/07_tucker_factor_product_gif.py --label 40pct`
(defaults to the 248x576 bundle in `results/task07/gaussianize/`).
"""

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DEFAULT_TUCKER = GAUSS_DIR / "anomalies_wsh_r248x576_checkpointed300_202401-202410_tucker.npz"

WINDOW_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP_HOURS = 3
ERA2_LAT_AT_ROW0 = -75.0
GRID_STEP_DEG = 0.25

FIG_W_IN = 16.0
FIG_H_IN = 4.5
MARGIN_IN = 0.55
GAP_IN = 0.5
BOTTOM_IN = 1.3
LOADING_CLIP_PERCENTILE = 99.5
SCALE_SAMPLE_FRAMES = 60
INK_SECONDARY = "#52514e"
LAND_GRAY = "#b8b8b8"


def reconstruct(core_t: np.ndarray, factor_lat: np.ndarray, factor_lon: np.ndarray) -> np.ndarray:
    """Return `factor_lat @ core_t @ factor_lon.T` in float64 (see docs/model_design.md §5.5)."""
    return np.asarray(factor_lat @ core_t.astype(np.float64) @ factor_lon.T, dtype=np.float64)


def symmetric_limit(values: np.ndarray, percentile: float) -> float:
    """Return the `percentile` of |values|, ignoring NaN, as a symmetric color limit."""
    return float(np.nanpercentile(np.abs(values), percentile))


def add_colorbar(fig: Figure, im: AxesImage, ax: Axes, vmax: float) -> None:
    """Attach a slim horizontal colorbar under `ax` with ticks at -vmax, 0, +vmax."""
    box = ax.get_position()
    cax = fig.add_axes((box.x0, box.y0 - 0.72 / FIG_H_IN, box.width, 0.11 / FIG_H_IN))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-vmax, 0.0, vmax])
    cbar.ax.set_xticklabels([f"{-vmax:.3g}", "0", f"{vmax:.3g}"])
    cbar.ax.tick_params(labelsize=7, length=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tucker_npz", type=str, default=str(DEFAULT_TUCKER))
    p.add_argument("--ocean_mask_path", type=str, default=str(GAUSS_DIR / "era2_ocean_mask.npy"))
    p.add_argument("--label", type=str, required=True, help="e.g. '40pct', used in filename/title")
    p.add_argument("--stride", type=int, default=4, help="take every Nth 3-hourly time index")
    p.add_argument("--max_frames", type=int, default=None, help="cap frame count (quick tests)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--out_dir", type=str, default=str(GAUSS_DIR))
    args = p.parse_args()

    bundle = np.load(args.tucker_npz)
    factor_lat = bundle["factor_lat"].astype(np.float64)
    factor_lon = bundle["factor_lon"].astype(np.float64)
    n_time = bundle["core"].shape[0]
    time_indices = np.arange(0, n_time, args.stride)[: args.max_frames]
    core_sel = bundle["core"][time_indices]
    n_frames = len(time_indices)
    n_lat, r_lat = factor_lat.shape
    n_lon, r_lon = factor_lon.shape
    print(
        f"n_frames={n_frames} (of {n_time}) factor_lat={factor_lat.shape} factor_lon={factor_lon.shape}"
    )

    ocean = np.load(args.ocean_mask_path)
    if ocean.shape != (n_lat, n_lon):
        raise ValueError(f"ocean mask {ocean.shape} does not match factors {(n_lat, n_lon)}")

    def frame_map(i: int) -> np.ndarray:
        return np.where(ocean, reconstruct(core_sel[i], factor_lat, factor_lon), np.nan)

    sample = np.linspace(0, n_frames - 1, min(SCALE_SAMPLE_FRAMES, n_frames)).astype(int)
    map_vmax = symmetric_limit(np.stack([frame_map(i) for i in sample]), 99)
    core_vmax = symmetric_limit(core_sel[sample], 99)
    lat_vmax = symmetric_limit(factor_lat, LOADING_CLIP_PERCENTILE)
    lon_vmax = symmetric_limit(factor_lon, LOADING_CLIP_PERCENTILE)
    print(
        f"vmax: map={map_vmax:.3f} core={core_vmax:.3f} "
        f"factor_lat={lat_vmax:.3f} factor_lon={lon_vmax:.3f}"
    )

    cmap: Colormap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(LAND_GRAY)

    # True relative sizes: one shared inches-per-index scale for every panel.
    usable_w = FIG_W_IN - 2 * MARGIN_IN - 3 * GAP_IN
    scale = usable_w / (r_lat + r_lon + n_lon + n_lon)
    shapes = [(n_lat, r_lat), (r_lat, r_lon), (r_lon, n_lon), (n_lat, n_lon)]
    x = MARGIN_IN
    boxes = []
    for rows, cols in shapes:
        boxes.append(
            (x / FIG_W_IN, BOTTOM_IN / FIG_H_IN, cols * scale / FIG_W_IN, rows * scale / FIG_H_IN)
        )
        x += cols * scale + GAP_IN

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax_lat, ax_core, ax_lon, ax_map = (fig.add_axes(b) for b in boxes)

    def show(
        ax: Axes, data: np.ndarray, vmax: float, origin: Literal["upper", "lower"], nearest: bool
    ) -> AxesImage:
        return ax.imshow(
            data,
            origin=origin,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest" if nearest else None,
        )

    lat_ticks = [-60, -30, 0, 30, 60]
    lon_ticks = [0, 90, 180, 270, 360]
    lat_extent = (
        ERA2_LAT_AT_ROW0 - GRID_STEP_DEG / 2,
        ERA2_LAT_AT_ROW0 + (n_lat - 0.5) * GRID_STEP_DEG,
    )
    lon_extent = (-GRID_STEP_DEG / 2, (n_lon - 0.5) * GRID_STEP_DEG)

    im_lat = show(ax_lat, factor_lat, lat_vmax, "lower", True)
    im_lat.set_extent((0.5, r_lat + 0.5, *lat_extent))
    ax_lat.set_yticks(lat_ticks)
    ax_lat.set_ylabel("Latitude (°N)", fontsize=8)

    im_core = show(ax_core, core_sel[0], core_vmax, "upper", True)
    ax_core.set_xticks([])
    ax_core.set_yticks([])

    im_lon = show(ax_lon, factor_lon.T, lon_vmax, "upper", True)
    im_lon.set_extent((*lon_extent, r_lon + 0.5, 0.5))
    ax_lon.set_xticks(lon_ticks)
    ax_lon.set_yticks([])
    ax_lon.set_xlabel("Longitude (°E)", fontsize=8)

    im_map = show(ax_map, frame_map(0), map_vmax, "lower", False)
    im_map.set_extent((*lon_extent, *lat_extent))
    ax_map.set_xticks(lon_ticks)
    ax_map.set_yticks(lat_ticks)
    ax_map.set_xlabel("Longitude (°E)", fontsize=8)

    panels = [
        (ax_lat, "factor_lat", (n_lat, r_lat), "fixed", im_lat, lat_vmax),
        (ax_core, "core[t]", (r_lat, r_lon), "changes", im_core, core_vmax),
        (ax_lon, "factor_lonᵀ", (r_lon, n_lon), "fixed", im_lon, lon_vmax),
        (ax_map, "frame[t]", (n_lat, n_lon), "reconstructed", im_map, map_vmax),
    ]
    for ax, name, (rows, cols), note, im, vmax in panels:
        ax.set_title(f"{name}  ({rows}×{cols}) — {note}", fontsize=9)
        ax.tick_params(labelsize=7, length=2)
        add_colorbar(fig, im, ax, vmax)

    mid_y = (BOTTOM_IN + n_lat * scale / 2) / FIG_H_IN
    for left_box, right_box, symbol in zip(boxes[:-1], boxes[1:], ["×", "×", "="]):
        mid_x = (left_box[0] + left_box[2] + right_box[0]) / 2
        fig.text(mid_x, mid_y, symbol, ha="center", va="center", fontsize=24, color=INK_SECONDARY)

    fig.text(
        0.5,
        0.975,
        f"Tucker rank {args.label} ({r_lat}×{r_lon}): frame[t] = factor_lat · core[t] · factor_lonᵀ"
        "   (wsh standardized anomaly; gray = land)",
        ha="center",
        va="top",
        fontsize=11,
    )
    stamp = fig.text(0.5, 0.905, "", ha="center", va="top", fontsize=10, color=INK_SECONDARY)

    def update(i: int) -> list[Artist]:
        t = int(time_indices[i])
        when = WINDOW_START + timedelta(hours=STEP_HOURS * t)
        im_core.set_data(core_sel[i])
        im_map.set_data(frame_map(i))
        stamp.set_text(f"{when:%Y-%m-%d %H:%M} UTC — t = {t}  (frame {i + 1}/{n_frames})")
        return [im_core, im_map, stamp]

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tucker_factor_product_{args.label}.gif"
    ani.save(out_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
