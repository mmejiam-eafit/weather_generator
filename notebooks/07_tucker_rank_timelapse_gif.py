"""Task 07: animated true-vs-reconstruction timelapse across Tucker ranks.

Builds a GIF over the full 9-month training window (2024-01-01 to
2024-10-01), sampled at 00 and 12 UTC each day (2/day, 548 frames total --
`i % 8 in (0, 4)` over the window's 3-hourly indices, since 274 days x 8
samples/day = 2192 = T exactly). Two rows per frame:

- Row 1: True field, then the 20%/30%/40% Tucker reconstructions (same
  color scale across all four, so intensity is directly comparable).
- Row 2: a date/per-frame-RMSE info panel, then the 20%/30%/40% error maps
  (true - reconstructed; shared color scale across all three ranks).

The point of animating this (vs. the earlier two-timestep static check,
`07_tucker_reconstruction_visual_check.py`) is to see how the error
pattern MOVES with the weather -- the coastline trace stays fixed frame to
frame (land doesn't move), while the open-ocean edge-tracing error follows
storm/eddy boundaries as they evolve. That residual is illustrative of the
high-spatial-frequency content a low-rank Tucker fit cannot represent at
any single rank -- conceptually the same kind of "left over" signal that
downstream DMD (dynamics) and the OU residual model (stochastic noise)
exist to handle, though note this is illustrative, not a literal statement
about what DMD/OU operate on (they run on the Gaussianized *core* after
Tucker, not on this raw spatial residual directly).

Reconstruction is done per-frame, on the fly, from each rank's already-
loaded (core, factor_lat, factor_lon) -- not precomputed/stored for all
548 frames at once, to keep memory low (each frame's reconstruction is one
small matrix triple-product; only `true_slices` and the three compact
cores are held in memory for the whole run).

Run directly: `poetry run python notebooks/07_tucker_rank_timelapse_gif.py`
(all paths default to what's already in `results/task07/gaussianize/`).
"""

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
WINDOW_START = datetime(2024, 1, 1)


def reconstruct(core: np.ndarray, factor_lat: np.ndarray, factor_lon: np.ndarray, t: int) -> np.ndarray:
    """Single-timestep Tucker-2 reconstruction: factor_lat @ core[t] @ factor_lon.T."""
    return factor_lat @ core[t] @ factor_lon.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--true_slices_npz", type=str, default=str(GAUSS_DIR / "timelapse_true_slices.npz"))
    p.add_argument(
        "--tucker_20pct", type=str,
        default=str(GAUSS_DIR / "anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz"),
    )
    p.add_argument(
        "--tucker_30pct", type=str,
        default=str(GAUSS_DIR / "anomalies_wsh_r186x432_checkpointed80_202401-202410_tucker.npz"),
    )
    p.add_argument(
        "--tucker_40pct", type=str,
        default=str(GAUSS_DIR / "anomalies_wsh_r248x576_checkpointed80_202401-202410_tucker.npz"),
    )
    p.add_argument("--ocean_mask_path", type=str, default=str(GAUSS_DIR / "era2_ocean_mask.npy"))
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=45, help="lower keeps the GIF file size manageable")
    p.add_argument("--percentile_stride", type=int, default=9, help="subsample stride for fixing color scales")
    p.add_argument("--out_dir", type=str, default=str(GAUSS_DIR))
    args = p.parse_args()

    true_bundle = np.load(args.true_slices_npz)
    time_indices, true_slices = true_bundle["time_indices"], true_bundle["true"]
    n_frames = len(time_indices)
    print(f"n_frames={n_frames}")

    ocean_mask = np.load(args.ocean_mask_path)
    land = ~ocean_mask

    ranks = {}
    for label, path in [("20pct", args.tucker_20pct), ("30pct", args.tucker_30pct), ("40pct", args.tucker_40pct)]:
        z = np.load(path)
        ranks[label] = (z["core"], z["factor_lat"], z["factor_lon"])

    def masked_true(i: int) -> np.ndarray:
        return np.where(land, np.nan, true_slices[i])

    def masked_recon(label: str, t: int) -> np.ndarray:
        core, factor_lat, factor_lon = ranks[label]
        return np.where(land, np.nan, reconstruct(core, factor_lat, factor_lon, t))

    # Fixed color scales from a strided subsample (not all 548 frames, to
    # keep this cheap) -- percentile-based, per the established convention
    # (raw max is dominated by a few extreme storm pixels and washes out
    # the plot).
    sample_i = list(range(0, n_frames, args.percentile_stride))
    field_vals, error_vals = [], []
    for i in sample_i:
        t = int(time_indices[i])
        tf = masked_true(i)
        field_vals.append(tf)
        for label in ranks:
            error_vals.append(tf - masked_recon(label, t))
    field_absmax = float(np.nanpercentile(np.abs(field_vals), 99))
    error_absmax = float(np.nanpercentile(np.abs(error_vals), 99))
    print(f"field_absmax={field_absmax:.3f} error_absmax={error_absmax:.3f} (from {len(sample_i)}-frame subsample)")
    del field_vals, error_vals

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    (ax_true, ax_20, ax_30, ax_40) = axes[0]
    (ax_info, ax_20e, ax_30e, ax_40e) = axes[1]

    field_axes = [ax_true, ax_20, ax_30, ax_40]
    field_titles = ["True", "20% recon", "30% recon", "40% recon"]
    error_axes = [ax_20e, ax_30e, ax_40e]
    error_labels = ["20pct", "30pct", "40pct"]
    error_titles = ["20% error", "30% error", "40% error"]

    init_field = masked_true(0)
    field_ims = []
    for ax, title in zip(field_axes, field_titles):
        im = ax.imshow(init_field, origin="lower", cmap="RdBu_r", vmin=-field_absmax, vmax=field_absmax, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        field_ims.append(im)
    fig.colorbar(field_ims[0], ax=field_axes, shrink=0.7, location="right", pad=0.01)

    ax_info.axis("off")
    info_text = ax_info.text(0.05, 0.5, "", fontsize=11, va="center", family="monospace")

    error_ims = []
    for ax, title in zip(error_axes, error_titles):
        im = ax.imshow(
            np.full_like(init_field, np.nan), origin="lower", cmap="RdBu_r",
            vmin=-error_absmax, vmax=error_absmax, aspect="auto",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        error_ims.append(im)
    fig.colorbar(error_ims[0], ax=error_axes, shrink=0.7, location="right", pad=0.01)

    fig.suptitle("Tucker rank timelapse: wsh, standardized log-energy anomaly space")

    def update(i: int):
        t = int(time_indices[i])
        date = WINDOW_START + timedelta(hours=3 * t)
        tf = masked_true(i)
        field_ims[0].set_data(tf)
        rmses = []
        for field_im, error_im, label in zip(field_ims[1:], error_ims, error_labels):
            recon = masked_recon(label, t)
            field_im.set_data(recon)
            err = tf - recon
            error_im.set_data(err)
            rmses.append(float(np.sqrt(np.nanmean(err**2))))
        info_text.set_text(
            f"{date:%Y-%m-%d %H:%M} UTC\nframe {i + 1}/{n_frames}\n\n"
            f"RMSE (this frame):\n 20%: {rmses[0]:.3f}\n 30%: {rmses[1]:.3f}\n 40%: {rmses[2]:.3f}"
        )
        return [*field_ims, *error_ims, info_text]

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_path = Path(args.out_dir) / "tucker_rank_timelapse.gif"
    ani.save(out_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
