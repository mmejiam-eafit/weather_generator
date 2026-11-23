"""Task 07: animated true-field vs. raw-core timelapse, any rank.

Direct empirical answer to "is the core just the same map at lower
resolution?" (it isn't -- see `tasks/07-tucker-dmd-rank-selection.md` §5
for the static two-frame check and the energy-concentration numbers).
Same 548-frame window as `07_tucker_rank_timelapse_gif.py` (00/12 UTC
daily, 2024-01-01 to 2024-10-01), animating the true field alongside the
raw core[t] side by side so the lack of resemblance -- and the core's
near-static, noise-like appearance frame to frame -- is directly visible
rather than asserted from two static snapshots.

Run directly: `poetry run python notebooks/07_tucker_core_timelapse_gif.py`.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--true_slices_npz", type=str, default=str(GAUSS_DIR / "timelapse_true_slices.npz"))
    p.add_argument(
        "--tucker_npz", type=str,
        default=str(GAUSS_DIR / "anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz"),
    )
    p.add_argument("--ocean_mask_path", type=str, default=str(GAUSS_DIR / "era2_ocean_mask.npy"))
    p.add_argument("--label", type=str, default="20pct", help="e.g. '30pct', used in the output filename/title")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=70)
    p.add_argument("--percentile_stride", type=int, default=9)
    p.add_argument("--out_dir", type=str, default=str(GAUSS_DIR))
    args = p.parse_args()

    true_bundle = np.load(args.true_slices_npz)
    time_indices, true_slices = true_bundle["time_indices"], true_bundle["true"]
    n_frames = len(time_indices)
    print(f"n_frames={n_frames}")

    ocean_mask = np.load(args.ocean_mask_path)
    land = ~ocean_mask

    z = np.load(args.tucker_npz)
    core = z["core"]

    def masked_true(i: int) -> np.ndarray:
        return np.where(land, np.nan, true_slices[i])

    sample_i = list(range(0, n_frames, args.percentile_stride))
    field_absmax = float(np.nanpercentile(np.abs([masked_true(i) for i in sample_i]), 99))
    core_absmax = float(np.percentile(np.abs([core[int(time_indices[i])] for i in sample_i]), 99))
    print(f"field_absmax={field_absmax:.3f} core_absmax={core_absmax:.3f} (from {len(sample_i)}-frame subsample)")

    fig, (ax_true, ax_core) = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)

    im_true = ax_true.imshow(
        masked_true(0), origin="lower", cmap="RdBu_r", vmin=-field_absmax, vmax=field_absmax, aspect="auto"
    )
    ax_true.set_title("True field (621x1440)")
    ax_true.set_xticks([])
    ax_true.set_yticks([])
    fig.colorbar(im_true, ax=ax_true, shrink=0.8)

    im_core = ax_core.imshow(
        core[int(time_indices[0])], origin="lower", cmap="RdBu_r", vmin=-core_absmax, vmax=core_absmax, aspect="auto"
    )
    ax_core.set_title(f"Raw core[t] ({args.label}, {core.shape[1]}x{core.shape[2]}) -- NOT a coarsened map")
    ax_core.set_xticks([])
    ax_core.set_yticks([])
    fig.colorbar(im_core, ax=ax_core, shrink=0.8)

    suptitle = fig.suptitle("")

    def update(i: int):
        t = int(time_indices[i])
        date = WINDOW_START + timedelta(hours=3 * t)
        im_true.set_data(masked_true(i))
        im_core.set_data(core[t])
        suptitle.set_text(f"{date:%Y-%m-%d %H:%M} UTC -- frame {i + 1}/{n_frames}")
        return [im_true, im_core, suptitle]

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_path = Path(args.out_dir) / f"tucker_core_timelapse_{args.label}.gif"
    ani.save(out_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
